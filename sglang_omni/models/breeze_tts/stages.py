# SPDX-License-Identifier: Apache-2.0
"""Stage executors for Breeze TTS 2: preprocessing → tts_engine → vocoder."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import load_state as _load_pipeline_state
from sglang_omni.scheduling.pipeline_state import store_state
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_checkpoint

from .payload_types import BreezeState
from .tokenizer import BreezePromptAdapter, BreezeReference, breeze_src

logger = logging.getLogger(__name__)


def load_breeze_model(checkpoint_dir: str, device: str):
    """The reference HF model (breeze-tts's BreezeForConditionalGeneration). The
    stages use its text encoder, audio embeddings, depth decoder and codec; the
    SGLang backbone loads its own weights. ~6 GB bf16; a later milestone loads
    only the pieces each stage needs."""
    breeze_src()
    from models.breeze import BreezeForConditionalGeneration  # breeze-tts checkout
    from models.breeze_config import BreezeConfig
    import json
    from safetensors.torch import load_file
    from transformers.initialization import no_init_weights

    # Built eagerly rather than through from_pretrained: transformers 5 constructs
    # on the meta device and leaves this custom model's non-persistent buffers
    # (codebook offsets, the text encoder's rotary inv_freq) uninitialized.
    config = BreezeConfig.from_pretrained(checkpoint_dir)
    with no_init_weights():
        model = BreezeForConditionalGeneration(config)
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        shards = sorted(set(json.load(open(index_path))["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(os.path.join(checkpoint_dir, shard)))
    missing, unexpected = model.load_state_dict(state, strict=False)
    tied = {"backbone_model.embed_tokens.embed_audio_tokens.weight"}
    if unexpected or (set(missing) - tied):
        raise ValueError(f"breeze checkpoint mismatch: missing {sorted(set(missing) - tied)[:5]} unexpected {list(unexpected)[:5]}")
    _tie_audio_embeddings(model)
    return model.to(device=device, dtype=torch.bfloat16).eval()


def _tie_audio_embeddings(model: Any) -> None:
    """The checkpoint stores the audio-codebook table once (under the depth
    decoder) and breeze-tts ties the backbone's copy to it in `_tie_weights`;
    transformers 5's loader no longer honours that hook, leaving the backbone
    table newly initialized — tie it here (weights are identical objects, as
    the reference runtime has them under 4.57)."""
    if not getattr(model.config, "tie_codebooks_embeddings", True):
        return
    backbone = model.backbone_model.embed_tokens.embed_audio_tokens
    depth = model.depth_decoder.model.embed_tokens
    if backbone.weight.data_ptr() != depth.weight.data_ptr():
        if tuple(backbone.weight.shape) != tuple(depth.weight.shape):
            raise ValueError(f"cannot tie audio embeddings: backbone {tuple(backbone.weight.shape)} vs depth decoder {tuple(depth.weight.shape)}")
        backbone.weight = depth.weight


def load_state(payload: StagePayload) -> BreezeState:
    return _load_pipeline_state(payload, BreezeState)


@torch.no_grad()
def build_prompt_embeds(model: Any, adapter: BreezePromptAdapter, request: dict[str, Any],
                        device: str) -> torch.Tensor:
    """[prompt_len, hidden] — the backbone's prompt exactly as breeze-tts
    builds it: text through the T5Gemma2 encoder + projection, reference audio
    through Mimi's encoder into per-frame codebook embeddings."""
    inputs = adapter.prepare_inputs(request)
    merged = model._merge_input_ids_with_input_values(
        input_ids=inputs["input_ids"],
        input_values=inputs.get("input_values"),
        text_ids_mask=inputs["text_ids_mask"],
        text_ids_len=inputs["text_ids_len"],
        attention_mask=inputs.get("attention_mask"),
    )
    return merged_embeds(merged)[0]


def merged_embeds(merged: Any) -> torch.Tensor:
    """breeze-tts's merge returns a dict in some revisions and the tensor in others."""
    if isinstance(merged, dict):
        return merged["inputs_embeds"]
    return merged


def load_audio_tokenizer(checkpoint_dir: str, device: str):
    """The bundled Qwen3-TTS 12.5 Hz audio tokenizer: it encodes references AND
    decodes generated frames — the reference runtime never decodes through the
    checkpoint's Mimi (that codec is the training-side encoder), so neither do we."""
    breeze_src()
    from qwen_tts import Qwen3TTSTokenizer
    bundled = os.path.join(checkpoint_dir, "audio_tokenizer")
    if not os.path.isdir(bundled):
        raise FileNotFoundError(f"Breeze checkpoint has no audio_tokenizer/ at {bundled}")
    return Qwen3TTSTokenizer.from_pretrained(bundled, device_map=device)


@torch.no_grad()
def decode_frames(audio_tokenizer: Any, frames: torch.Tensor) -> tuple[torch.Tensor, int]:
    """[T, num_codebooks] codes → ([samples] float32, sample_rate) through the
    audio tokenizer's decoder, the way breeze-tts's server does it."""
    codes = frames.to(torch.long)
    wavs, sample_rate = audio_tokenizer.decode({"audio_codes": codes})
    wav = wavs[0]
    audio = torch.as_tensor(wav, dtype=torch.float32).reshape(-1).cpu()
    return audio, int(sample_rate)


def _reference_from_payload(ref: dict[str, Any], scratch_dir: str) -> BreezeReference | None:
    """A reference given as a path, or as raw samples + rate (uploaded) which
    are written to a wav — breeze-tts's audio tokenizer reads files."""
    text = ref.get("text", "")
    path = ref.get("audio_path")
    if path:
        return BreezeReference(audio_path=path, text=text)
    samples = ref.get("audio")
    if samples is not None:
        import hashlib
        import soundfile as sf
        wav = torch.as_tensor(samples, dtype=torch.float32).reshape(-1).numpy()
        sr = int(ref.get("sample_rate", 24000))
        digest = hashlib.sha1(wav.tobytes() + str(sr).encode()).hexdigest()[:16]
        path = os.path.join(scratch_dir, f"ref-{digest}.wav")
        if not os.path.exists(path):
            sf.write(path, wav, sr)
        return BreezeReference(audio_path=path, text=text)
    return None


def _install_reference_code_cache(adapter: BreezePromptAdapter, max_items: int = 64) -> None:
    """Reference wav → codes is ~0.7 s per request on the audio tokenizer and
    the same few voices come back all day: memoize breeze_infer.templates'
    encoder by path (+ mtime/size), like S2's ReferenceEncodeService."""
    import collections
    import threading
    templates = adapter._templates
    original = templates._encode_prompt_audio
    if getattr(original, "_breeze_cached", False):
        return
    cache: "collections.OrderedDict[tuple, torch.Tensor]" = collections.OrderedDict()
    guard = threading.Lock()

    def cached(audio_tokenizer: Any, audio_path: Any) -> torch.Tensor:
        try:
            st = os.stat(audio_path)
            key = (str(audio_path), st.st_mtime_ns, st.st_size)
        except OSError:
            return original(audio_tokenizer, audio_path)
        with guard:
            hit = cache.get(key)
            if hit is not None:
                cache.move_to_end(key)
                return hit
        codes = original(audio_tokenizer, audio_path)
        with guard:
            cache[key] = codes
            while len(cache) > max_items:
                cache.popitem(last=False)
        return codes

    cached._breeze_cached = True
    templates._encode_prompt_audio = cached


def create_preprocessing_executor(model_path: str, *, max_concurrency: int = 4, device: str = "cuda:0"):
    """Threaded preprocessing on one GPU copy of the model's encoders."""
    import tempfile
    from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
    checkpoint_dir = _resolve_checkpoint(model_path)
    model = load_breeze_model(checkpoint_dir, device)
    adapter = BreezePromptAdapter(checkpoint_dir, model, device)
    _install_reference_code_cache(adapter)
    scratch_dir = tempfile.mkdtemp(prefix="breeze-refs-")
    lock = __import__("threading").Lock()          # one GPU model, many workers
    # The engine shares this GPU and keeps it busy with back-to-back graph replays; a
    # high-priority stream lets a prompt's text-encoder work interleave instead of queueing.
    stream = torch.cuda.Stream(device=device, priority=-1) if device.startswith("cuda") else None

    def _preprocess(payload: StagePayload) -> StagePayload:
        import time
        t0 = time.perf_counter()
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}
        if isinstance(inputs, str):
            inputs = {"text": inputs}
        refs = inputs.get("references") or []
        reference = _reference_from_payload(refs[0], scratch_dir) if refs else None
        request = adapter.build_request(text=inputs.get("text", ""), reference=reference,
                                        instruction=inputs.get("instruction"),
                                        speaker=inputs.get("speaker", "S0"),
                                        request_id=payload.request_id)
        # One prompt at a time on the GPU: without the lock a burst of prompts contends with the
        # engine's decode steps and everything slows (merge 0.4 s → 3 s, engine step +50%).
        with lock, (torch.cuda.stream(stream) if stream is not None else __import__("contextlib").nullcontext()):
            t1 = time.perf_counter()
            inputs_t = adapter.prepare_inputs(request)                      # reference codes + token ids
            t2 = time.perf_counter()
            merged = model._merge_input_ids_with_input_values(
                input_ids=inputs_t["input_ids"], input_values=inputs_t.get("input_values"),
                text_ids_mask=inputs_t["text_ids_mask"], text_ids_len=inputs_t["text_ids_len"],
                attention_mask=inputs_t.get("attention_mask"))
            embeds = merged_embeds(merged)[0].to(torch.bfloat16).cpu()
            if stream is not None:
                stream.synchronize()
            t3 = time.perf_counter()
        state = BreezeState(
            prefill_embeds=embeds,
            prompt_len=int(embeds.shape[0]),
            max_new_tokens=int(params.get("max_new_tokens", 1024)),
            temperature=float(params.get("temperature", 0.9)),
            top_p=float(params.get("top_p", 1.0)),
            top_k=int(params.get("top_k", 50)),
            repetition_penalty=float(params.get("repetition_penalty", 1.1)),
            seed=params.get("seed"),
            preprocess_encode_s=t2 - t1,
            preprocess_merge_s=t3 - t2,
            preprocess_time_s=time.perf_counter() - t0,
        )
        return store_state(payload, state)

    return ThreadedSimpleScheduler(_preprocess, max_concurrency=max(1, int(max_concurrency)))


def create_sglang_tts_engine_executor(model_path: str, *, device: str = "cuda", max_new_tokens: int = 1024,
                                      server_args_overrides: dict[str, Any] | None = None):
    """Returns the OmniScheduler for the Breeze backbone stage."""
    from .engine_builder import BreezeEngineBuilder
    return BreezeEngineBuilder(max_new_tokens=max_new_tokens).build(
        model_path, device=device, server_args_overrides=server_args_overrides)


class BreezeVocoderScheduler(StreamingSimpleScheduler):
    """M1: whole-utterance decode through the audio tokenizer, batched.
    Streaming per-chunk decode is M3 (the S2 streaming_vocoder is the template)."""

    def __init__(self, audio_tokenizer: Any, *, device: str, max_batch_size: int = 8, max_batch_wait_ms: int = 2):
        self._audio_tokenizer = audio_tokenizer
        self._device = torch.device(device)
        super().__init__(self._vocode_payload, batch_compute_fn=self._vocode_payloads,
                         max_batch_size=max_batch_size, max_batch_wait_ms=max_batch_wait_ms)

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        return False

    def validate_non_streaming_payload(self, payload: StagePayload) -> None:
        self._validate(payload)

    def _validate(self, payload: StagePayload) -> BreezeState:
        state = load_state(payload)
        if state.output_codes is None or state.output_codes.ndim != 2 or state.output_codes.shape[0] == 0:
            raise ValueError(f"Request {payload.request_id}: Breeze generated no audio frames")
        return state

    def _vocode_payload(self, payload: StagePayload) -> StagePayload:
        return self._vocode_payloads([payload])[0]

    def _vocode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        import time
        states = [self._validate(p) for p in payloads]
        t0 = time.perf_counter()
        # one batched decode for the whole group (the tokenizer pads internally)
        wavs, sample_rate = self._audio_tokenizer.decode([{"audio_codes": s.output_codes.to(torch.long)} for s in states])
        vocode_s = (time.perf_counter() - t0) / max(len(states), 1)
        out: list[StagePayload] = []
        for payload, state, wav in zip(payloads, states, wavs):
            audio = torch.as_tensor(wav, dtype=torch.float32).reshape(-1).cpu()
            state.sample_rate = int(sample_rate)
            frames = int(state.output_codes.shape[0])
            eng = state.engine_time_s or 0.0
            logger.info("breeze timing %s: %d frames → %.2f s (%s) | preprocess %.2fs (encode %.2f, merge %.2f) | engine %.2fs = %.0f ms/frame | vocode %.2fs",
                        payload.request_id, frames, audio.shape[-1] / state.sample_rate, state.finish_reason,
                        state.preprocess_time_s or 0.0, state.preprocess_encode_s or 0.0, state.preprocess_merge_s or 0.0,
                        eng, 1000.0 * eng / max(frames, 1), vocode_s)
            state.audio_samples = audio
            done = store_state(payload, state)
            # what the client reads off the terminal payload (same keys the S2 vocoder emits;
            # plain lists — the payload is serialized across processes)
            done.data["audio_data"] = audio.tolist()
            done.data["sample_rate"] = int(state.sample_rate)
            done.data["modality"] = "audio"
            done.data["usage"] = {"prompt_tokens": int(state.prompt_tokens or 0),
                                  "completion_tokens": frames, "total_tokens": int(state.prompt_tokens or 0) + frames}
            if state.finish_reason is not None:
                done.data["finish_reason"] = state.finish_reason
            out.append(done)
        return out


def create_vocoder_executor(model_path: str, *, device: str | None = None, gpu_id: int | None = None,
                            max_batch_size: int = 8, max_batch_wait_ms: int = 2):
    if device is None:
        device = f"cuda:{gpu_id}" if gpu_id is not None else "cpu"
    audio_tokenizer = load_audio_tokenizer(_resolve_checkpoint(model_path), device)
    return BreezeVocoderScheduler(audio_tokenizer, device=device, max_batch_size=max_batch_size,
                                  max_batch_wait_ms=max_batch_wait_ms)
