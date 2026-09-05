# SPDX-License-Identifier: Apache-2.0
"""The Breeze TTS 2 backbone as an SGLang model (M2 of the port).

What SGLang runs: the Qwen3 backbone (28 layers, hidden 2048, 16 heads / 8 KV,
head_dim 128) over prompt embeddings that preprocessing already merged
(text-encoder features + reference-frame codebook embeddings), with a
2051+1-way lm_head that predicts the FIRST codebook of the next 12.5 Hz frame
(class 2051 is the backbone's EOS). The decode-step input is not a token: it is
the SUM of the previous frame's 16 codebook embeddings (breeze-tts's
BreezeBackboneModelEmbeddings), so like Fish S2-Pro this model keeps per-slot
frame state on the GPU and ignores the token SGLang hands it.

Per step, `_decode_codebooks` samples codebook 0 (or EOS) from the backbone's
logits the way the reference runtime does (temperature / top-k, reserved ids
2048..2050 suppressed, repetition penalty over the codebook-0 history), then
runs the eager depth decoder (12 layers, d=1024) for codebooks 1..15,
conditioned on the backbone's last hidden state — batched across every running
request — and stages the frame. The model runner copies the frame into the
request, overrides SGLang's sampled token with ours (so stop / EOS logic
agrees), and re-syncs each slot before the next step because batch rows move.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional, Tuple

import torch
from torch import Tensor, nn

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# Identical Qwen3 math (qk-norm, GQA, RoPE) — reuse the S2 layer implementation.
from sglang_omni.models.fishaudio_s2_pro.sglang_model import (
    S2ProDecoderLayer,
    _default_weight_loader,
)
from sglang_omni.vendor.sglang.utils import make_layers

logger = logging.getLogger(__name__)

REP_HISTORY_LEN = 64          # codebook-0 tokens the repetition penalty looks back over
GRAPH_TOP_K = 64              # fixed top-k width (the reference samples with top_k=50)
_NEG_INF = -float("inf")
_DEBUG_STEPS = bool(__import__("os").environ.get("BREEZE_DEBUG_STEPS"))     # log row-0 logits per step
_DEBUG_TIMING = bool(__import__("os").environ.get("BREEZE_DEBUG_TIMING"))   # backbone vs depth-decoder ms per step


def _cfg(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class BreezeForConditionalGeneration(nn.Module):
    """Named after the HF architecture so the registry resolves it."""

    def __init__(self, config: Any = None, quant_config: Any = None, prefix: str = "") -> None:
        super().__init__()
        del quant_config, prefix
        bb = _cfg(config, "backbone_config") or config
        self.hidden_size = int(_cfg(bb, "hidden_size", 2048))
        self.num_layers = int(_cfg(bb, "num_hidden_layers", 28))
        self.num_codebooks = int(_cfg(config, "num_codebooks", _cfg(config, "audio_num_codebooks", 16)))
        self.codebook_size = int(_cfg(config, "vocab_size", _cfg(config, "audio_vocab_size", 2051)))
        self.codec_codebook_size = 2048                          # ids >= this (below EOS) are reserved
        self.pad_token_id = int(_cfg(config, "codebook_pad_token_id", 2050))
        self.eos_token_id = self.codebook_size                   # lm_head's extra class
        self.audio_embed_size = int(_cfg(config, "audio_embed_size", 0) or self.hidden_size)
        self.layers = make_layers(
            self.num_layers,
            lambda idx, prefix: S2ProDecoderLayer(
                hidden_size=self.hidden_size,
                intermediate_size=int(_cfg(bb, "intermediate_size", 6144)),
                num_heads=int(_cfg(bb, "num_attention_heads", 16)),
                num_kv_heads=int(_cfg(bb, "num_key_value_heads", 8)),
                head_dim=int(_cfg(bb, "head_dim", 128)),
                layer_id=idx,
                rope_base=float(_cfg(bb, "rope_theta", 1000000.0)),
                max_position_embeddings=int(_cfg(bb, "max_position_embeddings", 40960)),
                rms_norm_eps=float(_cfg(bb, "rms_norm_eps", 1e-6)),
                qk_norm=True,
            ),
            prefix="layers",
        )
        # The S2 attention builds its RoPE in Fish's interleaved (GPT-J) style; Breeze's backbone
        # is HF Qwen3, which rotates the two halves (NeoX style). Same weights, different rotation.
        from sglang.srt.layers.rotary_embedding import get_rope
        head_dim = int(_cfg(bb, "head_dim", 128))
        for layer in self.layers:
            layer.self_attn.rotary_emb = get_rope(
                head_dim,
                rotary_dim=head_dim,
                max_position=int(_cfg(bb, "max_position_embeddings", 40960)),
                base=float(_cfg(bb, "rope_theta", 1000000.0)),
                is_neox_style=True,
            )
        from sglang.srt.layers.layernorm import RMSNorm
        self.norm = RMSNorm(self.hidden_size, eps=float(_cfg(bb, "rms_norm_eps", 1e-6)))
        self.start_layer, self.end_layer = 0, self.num_layers
        self.lm_head = ParallelLMHead(self.codebook_size + 1, self.hidden_size)
        # codebook embedding table: num_codebooks * codebook_size rows, summed per frame
        self.embed_audio_tokens = nn.Embedding(self.num_codebooks * self.codebook_size, self.audio_embed_size)
        self.audio_embeds_projector = (
            nn.Linear(self.audio_embed_size, self.hidden_size, bias=False)
            if self.audio_embed_size != self.hidden_size else None)
        self.register_buffer("audio_tokens_offsets",
                             torch.arange(self.num_codebooks) * self.codebook_size, persistent=False)
        # decode state (setup_breeze_decode)
        self.depth_decoder: Any = None
        self._decode_ready = False
        self._timing: dict = {}
        self._t_start = 0.0
        self._captured_logged = False

    # ------------------------------------------------------------------ setup
    def setup_breeze_decode(self, *, depth_decoder: Any, max_batch_size: int, device: str) -> None:
        """Attach the eager depth decoder and allocate per-slot GPU buffers."""
        dev = torch.device(device)
        self.depth_decoder = depth_decoder.to(dev).eval()
        n = int(max_batch_size)
        self._last_frame = torch.zeros(n, self.num_codebooks, dtype=torch.long, device=dev)   # next step's input
        self._out_frame = torch.zeros(n, self.num_codebooks, dtype=torch.long, device=dev)    # this step's frame
        self._out_token = torch.full((n,), self.eos_token_id, dtype=torch.long, device=dev)   # cb0 or EOS
        self._temperature = torch.full((n,), 0.9, device=dev)
        self._top_k = torch.full((n,), 50, dtype=torch.long, device=dev)
        self._depth_temperature = torch.full((n,), 0.9, device=dev)
        self._depth_top_k = torch.full((n,), 50, dtype=torch.long, device=dev)
        self._rep_penalty = torch.full((n,), 1.1, device=dev)
        self._prev_tokens = torch.zeros(n, REP_HISTORY_LEN, dtype=torch.long, device=dev)
        self._prev_count = torch.zeros(n, dtype=torch.long, device=dev)
        self._rep_positions = torch.arange(REP_HISTORY_LEN, device=dev)
        # reserved ids (codec_codebook_size .. codebook_size-1, i.e. 2048..2050) are never sampled
        bias = torch.zeros(self.codebook_size + 1, device=dev)
        bias[self.codec_codebook_size:self.codebook_size] = _NEG_INF
        self._backbone_bias = bias
        self._depth_bias = bias[: self.codebook_size].clone()
        self._depth_graph = None
        if not __import__("os").environ.get("BREEZE_DEPTH_EAGER"):
            from .depth_graph import DepthDecoderGraph
            try:
                graph = DepthDecoderGraph(self.depth_decoder, num_codebooks=self.num_codebooks,
                                          hidden_size=self.hidden_size, sample_fn=self._sample,
                                          depth_bias=self._depth_bias, device=dev, max_batch_size=n)
                graph.capture()
                self._depth_graph = graph
            except Exception:
                logger.exception("breeze depth decoder: CUDA graph capture failed; staying eager")
        self._decode_ready = True

    @property
    def decode_max_batch_size(self) -> int:
        return int(self._last_frame.shape[0]) if self._decode_ready else 0

    def frame_embeds(self, frames: Tensor) -> Tensor:
        """[bs, num_codebooks] codes → [bs, hidden]: the sum of the codebook embeddings."""
        e = self.embed_audio_tokens(frames + self.audio_tokens_offsets)
        if self.audio_embeds_projector is not None:
            e = self.audio_embeds_projector(e)
        return e.sum(dim=1)

    # ---------------------------------------------------------------- forward
    def forward(self, input_ids: Tensor, positions: Tensor, forward_batch: ForwardBatch,
                input_embeds: Optional[Tensor] = None) -> LogitsProcessorOutput:
        timing = _DEBUG_TIMING and not torch.cuda.is_current_stream_capturing()
        if timing:
            torch.cuda.synchronize()
            self._t_start = __import__("time").perf_counter()
        if input_embeds is None and forward_batch.input_embeds is not None:
            input_embeds = forward_batch.input_embeds
        if input_embeds is not None:
            hidden_states = input_embeds                # the merged prompt (prefill)
        else:
            # decode: the previous frame's embedding, not the token SGLang sampled
            assert self._decode_ready, "setup_breeze_decode() before decoding"
            bs = input_ids.shape[0]
            hidden_states = self.frame_embeds(self._last_frame[:bs]).to(self.lm_head.weight.dtype)
        residual = None
        taps = {}
        for layer_idx in range(self.start_layer, self.end_layer):
            hidden_states, residual = self.layers[layer_idx](positions, hidden_states, forward_batch, residual)
            if _DEBUG_STEPS and layer_idx in (0, 1, 2, self.end_layer - 1):
                taps[layer_idx] = (hidden_states + residual)          # the residual stream after this layer
        hidden_states, _ = self.norm(hidden_states, residual)
        if forward_batch.forward_mode.is_extend():
            last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            hidden_states = hidden_states[last_index]
            if _DEBUG_STEPS:
                for i, t in taps.items():
                    v = t[last_index][0].float()
                    logger.info("breeze tap layer%d last-pos: mean|x|=%.4f first4=%s", i, float(v.abs().mean()),
                                [round(float(x), 4) for x in v[:4]])
                v = hidden_states[0].float()
                logger.info("breeze tap final-norm last-pos: mean|x|=%.4f first4=%s", float(v.abs().mean()),
                            [round(float(x), 4) for x in v[:4]])
                logger.info("breeze tap prefill positions: n=%d first=%s last=%s", int(positions.numel()),
                            int(positions[0]), int(positions[-1]))
        # ParallelLMHead refuses direct calls (its weight is meant for SGLang's sampler); the
        # padded rows (2052 → 2112) are dropped so SGLang sees exactly codebook_size + 1 classes.
        logits = torch.nn.functional.linear(hidden_states, self.lm_head.weight)[:, : self.codebook_size + 1]
        if self._decode_ready:
            if timing:
                torch.cuda.synchronize()
                t_mid = __import__("time").perf_counter()
            self._decode_codebooks(logits, hidden_states)
            if timing:
                torch.cuda.synchronize()
                t_end = __import__("time").perf_counter()
                self._timing_log(forward_batch, int(logits.shape[0]), t_mid - self._t_start, t_end - t_mid)
        return LogitsProcessorOutput(next_token_logits=logits, hidden_states=hidden_states)

    def _timing_log(self, forward_batch: ForwardBatch, bs: int, backbone_s: float, depth_s: float) -> None:
        mode = "extend" if forward_batch.forward_mode.is_extend() else "decode"
        acc = self._timing.setdefault((mode, bs), [0, 0.0, 0.0])
        acc[0] += 1
        acc[1] += backbone_s
        acc[2] += depth_s
        if acc[0] % 25 == 0:
            logger.info("breeze timing %s bs=%d n=%d: backbone %.1f ms, depth decoder %.1f ms (per step)",
                        mode, bs, acc[0], 1000 * acc[1] / acc[0], 1000 * acc[2] / acc[0])

    # --------------------------------------------------------------- sampling
    def _sample(self, logits: Tensor, temperature: Tensor, top_k: Tensor) -> Tensor:
        """Per-row temperature + top-k sampling on float logits [bs, V]. A fixed
        top-k width keeps it CUDA-graph safe (no host syncs); per-row top_k is a
        mask inside that width, top_k <= 0 means the full width."""
        k_max = min(GRAPH_TOP_K, logits.shape[-1])
        topk_vals, topk_idx = torch.topk(logits, k_max, dim=-1)
        pos = torch.arange(k_max, device=logits.device).unsqueeze(0)
        k_eff = torch.where(top_k > 0, top_k.clamp(max=k_max), torch.full_like(top_k, k_max))
        topk_vals = topk_vals.masked_fill(pos >= k_eff.unsqueeze(1), _NEG_INF)
        probs = torch.softmax(topk_vals / temperature.clamp(min=1e-5).unsqueeze(1), dim=-1)
        choice = torch.multinomial(probs, 1)
        return topk_idx.gather(-1, choice).squeeze(-1)

    @torch.no_grad()
    def _decode_codebooks(self, logits: Tensor, hidden_states: Tensor) -> None:
        bs = logits.shape[0]
        lg = logits[:, : self.codebook_size + 1].float() + self._backbone_bias   # drop the lm_head's vocab padding
        # repetition penalty over the codebook-0 history (reference: sample_logits with token_history)
        prev = self._prev_tokens[:bs]
        count = self._prev_count[:bs]
        scores = torch.gather(lg, -1, prev)
        pen = self._rep_penalty[:bs].unsqueeze(1)
        penalized = torch.where(scores < 0, scores * pen, scores / pen)
        valid = self._rep_positions.unsqueeze(0) < count.unsqueeze(1)
        lg.scatter_(-1, prev, torch.where(valid, penalized, scores))
        token = self._sample(lg, self._temperature[:bs], self._top_k[:bs])            # cb0 or EOS
        if _DEBUG_STEPS and not torch.cuda.is_current_stream_capturing():
            raw = logits[0, : self.codebook_size + 1].float()
            top = torch.topk(raw, 5)
            logger.info("breeze step row0 n=%d top5=%s eos_logit=%.2f chosen=%d",
                        int(self._prev_count[0].item()), [(int(i), round(float(v), 2)) for v, i in zip(top.values, top.indices)],
                        float(raw[self.eos_token_id]), int(token[0]))
        is_eos = token == self.eos_token_id
        cb0 = torch.where(is_eos, torch.zeros_like(token), token)
        # depth decoder: [dummy, cb0, .., cb_{k-1}] → codebook k at the last position (one CUDA
        # graph replay per frame when captured; the eager loop otherwise)
        depth_hidden = hidden_states.to(self.depth_decoder.dtype if hasattr(self.depth_decoder, "dtype") else torch.bfloat16)
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and not self._captured_logged:
            self._captured_logged = True
            logger.info("breeze: codebook sampling + depth-decoder loop are being captured inside SGLang's decode graph (bs=%d)", bs)
        if self._depth_graph is not None and not capturing:
            frame = self._depth_graph.run(depth_hidden, cb0, self._depth_temperature[:bs], self._depth_top_k[:bs])
        else:
            # eager loop: also what SGLang's CUDA-graph capture records (a graph cannot replay inside a capture)
            seq = torch.cat([torch.zeros(bs, 1, dtype=torch.long, device=logits.device), cb0.unsqueeze(1)], dim=1)
            for _ in range(1, self.num_codebooks):
                out = self.depth_decoder(input_ids=seq, backbone_last_hidden_state=depth_hidden, use_cache=False,
                                         cache_position=torch.arange(seq.shape[1], device=seq.device), return_dict=True)
                step_logits = out.logits[:, -1, :].float() + self._depth_bias
                tok = self._sample(step_logits, self._depth_temperature[:bs], self._depth_top_k[:bs])
                seq = torch.cat([seq, tok.unsqueeze(1)], dim=1)
            frame = seq[:, 1:]
        self._out_frame[:bs] = frame
        self._out_token[:bs] = token
        self._last_frame[:bs] = frame

    # ---------------------------------------------------------------- weights
    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> None:
        """Checkpoint names → this module. Only the backbone, its audio
        embeddings and the lm_head live here; the depth decoder, text encoder
        and codec are loaded eagerly by the stages."""
        params = dict(self.named_parameters())
        # per-projection checkpoint tensors → shards of sglang's fused layers. The S2 layer keeps its
        # MLP projections directly on the layer (gate_up_proj / down_proj, no `mlp.`); resolve either.
        fused = {
            "self_attn.q_proj.weight": ("self_attn.qkv_proj.weight", "q"),
            "self_attn.k_proj.weight": ("self_attn.qkv_proj.weight", "k"),
            "self_attn.v_proj.weight": ("self_attn.qkv_proj.weight", "v"),
            "mlp.gate_proj.weight": ("mlp.gate_up_proj.weight", 0),
            "mlp.up_proj.weight": ("mlp.gate_up_proj.weight", 1),
        }

        def resolve(idx: str, sub: str) -> str | None:
            for cand in (f"layers.{idx}.{sub}", f"layers.{idx}.{sub.replace('mlp.', '', 1)}"):
                if cand in params:
                    return cand
            return None

        loaded: set[str] = set()
        for name, w in weights:
            if name.startswith("backbone_model.layers."):
                rest = name[len("backbone_model.layers."):]
                idx, sub = rest.split(".", 1)
                if sub in fused:
                    target_sub, shard = fused[sub]
                    target = resolve(idx, target_sub)
                    if target is None:
                        raise KeyError(f"breeze backbone: no parameter for {name} (tried {target_sub})")
                    param = params[target]
                    param.weight_loader(param, w, shard)
                    loaded.add(target)
                    continue
                target = resolve(idx, sub)                 # o_proj, down_proj, both layernorms, q_norm/k_norm
                if target is None:
                    raise KeyError(f"breeze backbone: no parameter for {name}; sample: {sorted(params)[:8]}")
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
            if target not in params:
                logger.debug("breeze: no parameter for %s (from %s)", target, name)
                continue
            param = params[target]
            if target == "lm_head.weight" and param.shape[0] > w.shape[0]:
                # ParallelLMHead pads the vocab to a multiple of 64 (2052 → 2112); the extra rows stay zero
                param.data[: w.shape[0]].copy_(w.to(param.dtype))
                param.data[w.shape[0]:].zero_()
            else:
                weight_loader = getattr(param, "weight_loader", _default_weight_loader)
                weight_loader(param, w)
            loaded.add(target)
        last = str(self.num_layers - 1)
        expected = {"norm.weight", "lm_head.weight", "embed_audio_tokens.weight",
                    resolve(last, "self_attn.qkv_proj.weight"), resolve(last, "mlp.gate_up_proj.weight")}
        missing = expected - loaded
        if missing:
            raise ValueError(f"breeze backbone: checkpoint did not provide {sorted(missing)}")


# The registry keys on the HF architecture name.
BreezeSGLangBackbone = BreezeForConditionalGeneration
EntryClass = BreezeForConditionalGeneration
