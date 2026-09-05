# SPDX-License-Identifier: Apache-2.0
"""The Breeze TTS 2 backbone as an SGLang model (M2 of the port).

What SGLang runs: the Qwen3 backbone (28 layers, hidden 2048, 16 heads / 8 KV,
head_dim 128, ~1.7B) over prompt embeddings that preprocessing already merged
(text-encoder features + reference-frame codebook embeddings), with a
2051+1-way lm_head that predicts the FIRST codebook of the next 12.5 Hz frame
(the +1 is the backbone's EOS). The decode-step input is not a token: it is
the embedding SUM of the previous frame's 16 codebooks (breeze-tts's
BreezeBackboneModelEmbeddings), so like Fish S2-Pro this model keeps its own
per-request frame state and treats the token SGLang hands it as a trigger.

After the backbone's logits for a step, `_decode_codebooks` samples codebook 0,
runs the depth decoder (12 layers, d=1024) for the other 15 codebooks
conditioned on the backbone's last hidden state — batched across every running
request — records the frame, and prepares the next step's input embedding.
The depth decoder and the audio-embedding table are eager HF modules loaded
from the checkpoint in `setup_breeze_decode`; only the backbone is SGLang.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional, Tuple

import torch
from torch import Tensor, nn

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import make_layers

# Identical Qwen3 math (qk-norm, GQA, RoPE) — reuse the S2 layer implementation.
from sglang_omni.models.fishaudio_s2_pro.sglang_model import (
    S2ProDecoderLayer,
    _default_weight_loader,
)

logger = logging.getLogger(__name__)


class BreezeSGLangBackbone(nn.Module):
    def __init__(self, config: Any = None, quant_config: Any = None) -> None:
        super().__init__()
        bb = config.backbone_config if hasattr(config, "backbone_config") else config
        get = (lambda k, d=None: (bb.get(k, d) if isinstance(bb, dict) else getattr(bb, k, d)))
        self.hidden_size = int(get("hidden_size", 2048))
        self.num_layers = int(get("num_hidden_layers", 28))
        self.num_codebooks = int(getattr(config, "audio_num_codebooks", 16))
        self.codebook_size = int(getattr(config, "audio_vocab_size", 2051))
        self.audio_embed_size = int(getattr(config, "audio_embed_size", self.hidden_size) or self.hidden_size)
        self.eos_token_id = self.codebook_size          # lm_head's extra class
        self.layers = make_layers(
            self.num_layers,
            lambda idx, prefix: S2ProDecoderLayer(
                hidden_size=self.hidden_size,
                intermediate_size=int(get("intermediate_size", 6144)),
                num_heads=int(get("num_attention_heads", 16)),
                num_kv_heads=int(get("num_key_value_heads", 8)),
                head_dim=int(get("head_dim", 128)),
                layer_id=idx,
                rope_base=float(get("rope_theta", 1000000.0)),
                max_position_embeddings=int(get("max_position_embeddings", 40960)),
                rms_norm_eps=float(get("rms_norm_eps", 1e-6)),
                qk_norm=True,
            ),
        )
        from sglang.srt.layers.layernorm import RMSNorm
        self.norm = RMSNorm(self.hidden_size, eps=float(get("rms_norm_eps", 1e-6)))
        self.start_layer, self.end_layer = 0, self.num_layers
        # first-codebook head: codebook_size + 1 (EOS)
        self.lm_head = ParallelLMHead(self.codebook_size + 1, self.hidden_size)
        # audio embedding table: num_codebooks * codebook_size rows, summed per frame
        self.embed_audio_tokens = nn.Embedding(self.num_codebooks * self.codebook_size, self.audio_embed_size)
        self.audio_embeds_projector = (
            nn.Linear(self.audio_embed_size, self.hidden_size, bias=False)
            if self.audio_embed_size != self.hidden_size else None)
        self.register_buffer("audio_tokens_offsets",
                             torch.arange(self.num_codebooks) * self.codebook_size, persistent=False)
        # decode state (setup_breeze_decode): the eager depth decoder + per-slot frames
        self.depth_decoder: Any = None
        self._decode_ready = False
        self._last_frame: Tensor | None = None       # [max_bs, num_codebooks]
        self._has_frame: Tensor | None = None        # [max_bs] bool: a frame exists for this slot
        self._frame_sink: list[Any] = []             # per-slot request data (output_codes lists)
        self._sampling: dict[str, Tensor] = {}

    # ------------------------------------------------------------------ setup
    def setup_breeze_decode(self, *, depth_decoder: Any, max_batch_size: int, device: str) -> None:
        self.depth_decoder = depth_decoder.to(device).eval()
        self._last_frame = torch.zeros(max_batch_size, self.num_codebooks, dtype=torch.long, device=device)
        self._has_frame = torch.zeros(max_batch_size, dtype=torch.bool, device=device)
        self._frame_sink = [None] * max_batch_size
        self._decode_ready = True

    def frame_embeds(self, frames: Tensor) -> Tensor:
        """[bs, num_codebooks] codes → [bs, hidden]: the sum of the codebook embeddings."""
        e = self.embed_audio_tokens(frames + self.audio_tokens_offsets)
        if self.audio_embeds_projector is not None:
            e = self.audio_embeds_projector(e)
        return e.sum(dim=1)

    # ---------------------------------------------------------------- forward
    def forward(self, input_ids: Tensor, positions: Tensor, forward_batch: ForwardBatch,
                input_embeds: Optional[Tensor] = None) -> LogitsProcessorOutput:
        if input_embeds is None and forward_batch.input_embeds is not None:
            input_embeds = forward_batch.input_embeds
        if input_embeds is not None:
            hidden_states = input_embeds                # the merged prompt (prefill)
        else:
            # decode: the previous frame's embedding, not the token SGLang sampled
            bs = input_ids.shape[0]
            assert self._decode_ready, "setup_breeze_decode() before decoding"
            hidden_states = self.frame_embeds(self._last_frame[:bs]).to(self.lm_head.weight.dtype)
        residual = None
        for layer_idx in range(self.start_layer, self.end_layer):
            hidden_states, residual = self.layers[layer_idx](positions, hidden_states, forward_batch, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        if forward_batch.forward_mode.is_extend():
            last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            hidden_states = hidden_states[last_index]
        logits = self.lm_head(hidden_states)
        if self._decode_ready:
            self._decode_codebooks(logits, hidden_states)
        return LogitsProcessorOutput(next_token_logits=logits, hidden_states=hidden_states)

    @torch.no_grad()
    def _decode_codebooks(self, logits: Tensor, hidden_states: Tensor) -> None:
        """Sample codebook 0, run the depth decoder for codebooks 1..15 (batched
        over the running requests), record the frame, stage the next input."""
        bs = logits.shape[0]
        cb0_logits = logits[:, : self.codebook_size].float()      # EOS is decided by SGLang from the full logits
        temperature = self._sampling.get("temperature")
        if temperature is not None:
            cb0_logits = cb0_logits / temperature[:bs].unsqueeze(1).clamp(min=1e-3)
        cb0 = torch.multinomial(torch.softmax(cb0_logits, dim=-1), 1).squeeze(1)   # [bs]
        frame = torch.empty(bs, self.num_codebooks, dtype=torch.long, device=logits.device)
        frame[:, 0] = cb0
        seq = cb0.unsqueeze(1)                                                       # [bs, 1]
        for step in range(1, self.num_codebooks):
            out = self.depth_decoder(input_ids=seq, backbone_last_hidden_state=hidden_states,
                                     use_cache=False, return_dict=True)
            step_logits = out.logits[:, -1, :].float()
            step_logits[:, self.codebook_size - 1] = -float("inf")                 # pad id, never sampled
            tok = torch.multinomial(torch.softmax(step_logits, dim=-1), 1)          # [bs, 1]
            frame[:, step] = tok.squeeze(1)
            seq = torch.cat([seq, tok], dim=1)
        self._last_frame[:bs] = frame
        self._has_frame[:bs] = True
        for i in range(bs):
            sink = self._frame_sink[i]
            if sink is not None:
                sink.output_codes.append(frame[i].detach().cpu())

    # ---------------------------------------------------------------- weights
    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> None:
        """Checkpoint names → this module. Only the backbone, its audio
        embeddings and the lm_head live here; the depth decoder, text encoder
        and codec are loaded eagerly by the stages."""
        params = dict(self.named_parameters())
        stacked = {}   # fused qkv / gate_up assembled from the per-projection tensors
        loaded: set[str] = set()
        for name, w in weights:
            if name.startswith("backbone_model.layers."):
                rest = name[len("backbone_model.layers."):]
                idx, sub = rest.split(".", 1)
                if sub.startswith("self_attn.") and any(p in sub for p in ("q_proj", "k_proj", "v_proj")):
                    stacked.setdefault((idx, "qkv"), {})[sub.split(".")[1]] = w
                    continue
                if sub.startswith("mlp.") and any(p in sub for p in ("gate_proj", "up_proj")):
                    stacked.setdefault((idx, "gate_up"), {})[sub.split(".")[1]] = w
                    continue
                target = f"layers.{idx}.{sub}"             # o_proj, down_proj, both layernorms, q_norm/k_norm
            elif name == "backbone_model.norm.weight":
                target = "norm.weight"
            elif name in ("backbone_model.embed_tokens.embed_audio_tokens.weight",
                          "depth_decoder.model.embed_tokens.weight"):
                # tie_codebooks_embeddings: the checkpoint stores the table once, under the depth decoder
                if "embed_audio_tokens.weight" in loaded:
                    continue
                target = "embed_audio_tokens.weight"
            elif name == "backbone_model.embed_tokens.audio_embeds_projector.weight":
                target = "audio_embeds_projector.weight"
            elif name == "lm_head.weight":
                target = "lm_head.weight"
            else:
                continue     # text_encoder.*, depth_decoder.* (rest), codec_model.*, embed_text_tokens.*, text_encoder_proj
            loaded.add(target)
            if target not in params:
                logger.debug("breeze: no parameter for %s (from %s)", target, name)
                continue
            _default_weight_loader(params[target], w)
        for (idx, kind), parts in stacked.items():
            if kind == "qkv":
                fused = torch.cat([parts["q_proj"], parts["k_proj"], parts["v_proj"]], dim=0)
                _default_weight_loader(params[f"layers.{idx}.self_attn.qkv_proj.weight"], fused)
            else:
                fused = torch.cat([parts["gate_proj"], parts["up_proj"]], dim=0)
                _default_weight_loader(params[f"layers.{idx}.mlp.gate_up_proj.weight"], fused)
            loaded.add(f"layers.{idx}.{kind}")
        expected = {"norm.weight", "lm_head.weight", "embed_audio_tokens.weight"}
        missing = expected - loaded
        if missing:
            raise ValueError(f"breeze backbone: checkpoint did not provide {sorted(missing)}")


EntryClass = BreezeSGLangBackbone
