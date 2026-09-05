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
    model = BreezeForConditionalGeneration.from_pretrained(checkpoint_dir, torch_dtype=torch.bfloat16)
    return model.to(device).eval()


def load_state(payload: StagePayload) -> BreezeState:
    return _load_pipeline_state(payload, BreezeState)


@torch.no_grad()
def build_prompt_embeds(model: Any, adapter: BreezePromptAdapter, request: dict[str, Any],
                        device: str) -> torch.Tensor:
    """[prompt_len, hidden] — the backbone's prompt exactly as breeze-tts
    builds it: text through the T5Gemma2 encoder + projection, reference audio
    through Mimi's encoder into per-frame codebook embeddings."""
    inputs = adapter.prepare_inputs(request, codec=model.codec_model, device=device)
    merged = model._merge_input_ids_with_input_values(
        input_ids=inputs["input_ids"],
        input_values=inputs.get("input_values"),
        text_ids_mask=inputs["text_ids_mask"],
        text_ids_len=inputs["text_ids_len"],
        attention_mask=inputs.get("attention_mask"),
    )
    return merged["inputs_embeds"][0]


@torch.no_grad()
def decode_frames(model: Any, frames: torch.Tensor) -> torch.Tensor:
    """[T, num_codebooks] Mimi codes → [samples] float32 at 24 kHz."""
    codes = frames.to(model.codec_model.device).T.unsqueeze(0)      # [1, num_codebooks, T]
    out = model.codec_model.decode(audio_codes=codes)
    audio = out.audio_values if hasattr(out, "audio_values") else out[0]
    return audio.reshape(-1).float().cpu()


def _reference_from_payload(ref: dict[str, Any]) -> BreezeReference | None:
    """A reference given as a path, or as raw samples + rate (uploaded)."""
    text = ref.get("text", "")
    path = ref.get("audio_path")
    if path:
        import torchaudio
        wav, sr = torchaudio.load(path)
        return BreezeReference(audio=wav.mean(0), sample_rate=int(sr), text=text)
    samples = ref.get("audio")
    if samples is not None:
        wav = torch.as_tensor(samples, dtype=torch.float32)
        return BreezeReference(audio=wav, sample_rate=int(ref.get("sample_rate", 24000)), text=text)
    return None


def create_preprocessing_executor(model_path: str, *, max_concurrency: int = 4, device: str = "cuda:0"):
    """Threaded preprocessing on one GPU copy of the model's encoders."""
    from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
    checkpoint_dir = _resolve_checkpoint(model_path)
    model = load_breeze_model(checkpoint_dir, device)
    adapter = BreezePromptAdapter(checkpoint_dir)
    lock = __import__("threading").Lock()          # one GPU model, many workers

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}
        if isinstance(inputs, str):
            inputs = {"text": inputs}
        refs = inputs.get("references") or []
        reference = _reference_from_payload(refs[0]) if refs else None
        request = adapter.build_request(text=inputs.get("text", ""), reference=reference,
                                        instruction=inputs.get("instruction"),
                                        speaker=inputs.get("speaker", "S1"))
        with lock:
            embeds = build_prompt_embeds(model, adapter, request, device).to(torch.bfloat16).cpu()
        state = BreezeState(
            prefill_embeds=embeds,
            prompt_len=int(embeds.shape[0]),
            max_new_tokens=int(params.get("max_new_tokens", 1024)),
            temperature=float(params.get("temperature", 0.8)),
            top_p=float(params.get("top_p", 0.95)),
            top_k=int(params.get("top_k", 50)),
            repetition_penalty=float(params.get("repetition_penalty", 1.0)),
            seed=params.get("seed"),
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
    """M1: whole-utterance Mimi decode, batched. Streaming per-frame decode
    with overlap is M3 (the S2 streaming_vocoder is the template)."""

    def __init__(self, model: Any, *, device: str, max_batch_size: int = 8, max_batch_wait_ms: int = 2):
        self._model = model
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
        out: list[StagePayload] = []
        for payload in payloads:
            state = self._validate(payload)
            audio = decode_frames(self._model, state.output_codes)
            state.audio_samples = audio
            out.append(store_state(payload, state))
        return out


def create_vocoder_executor(model_path: str, *, device: str | None = None, gpu_id: int | None = None,
                            max_batch_size: int = 8, max_batch_wait_ms: int = 2):
    if device is None:
        device = f"cuda:{gpu_id}" if gpu_id is not None else "cpu"
    model = load_breeze_model(_resolve_checkpoint(model_path), device)
    return BreezeVocoderScheduler(model, device=device, max_batch_size=max_batch_size,
                                  max_batch_wait_ms=max_batch_wait_ms)
