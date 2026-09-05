# SPDX-License-Identifier: Apache-2.0
"""StagePayload <-> SGLang request adapters for Breeze TTS 2."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.proto import StagePayload

from .payload_types import BreezeState


@dataclass
class BreezeSGLangRequestData(SGLangARRequestData):
    """Per-request state for the Breeze backbone stage."""

    num_codebooks: int = 16
    codebook_size: int = 2051
    backbone_eos_token_id: int = 2051
    prompt_len: int = 0
    seed: int | None = None
    # Frames the model has produced so far, [num_codebooks] each — appended by
    # sglang_model._decode_codebooks, drained by the stream output builder.
    output_codes: list[torch.Tensor] = field(default_factory=list)
    streamed_frames: int = 0
    engine_start_s: float | None = None


def build_sglang_tts_request(
    state: BreezeState, request_id: str = ""
) -> BreezeSGLangRequestData:
    """A Req whose prompt is ALREADY EMBEDDED (prefill_input_embeds): the ids
    are placeholders that only give SGLang the prompt length for scheduling and
    KV allocation. The backbone predicts the first codebook of the next frame;
    its EOS is the lm_head's extra class (config.vocab_size)."""
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    embeds = state.prefill_embeds
    if not isinstance(embeds, torch.Tensor):
        embeds = torch.as_tensor(embeds)
    prompt_len = int(embeds.shape[0])
    placeholder_ids = [0] * prompt_len
    eos = int(state.backbone_eos_token_id)
    # The backbone's vocabulary is the first codebook (+1 for EOS).
    vocab_size = int(state.codebook_size) + 1
    sampling_params = SamplingParams(
        max_new_tokens=int(state.max_new_tokens),
        temperature=float(state.temperature),
        top_p=float(state.top_p),
        top_k=int(state.top_k),
        repetition_penalty=float(state.repetition_penalty),
        stop_token_ids=[eos],
    )
    sampling_params.verify(vocab_size)
    req = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=placeholder_ids,
        sampling_params=sampling_params,
        vocab_size=vocab_size,
        eos_token_ids={eos},
    )
    req._input_embeds_are_projected = True
    return BreezeSGLangRequestData(
        input_ids=torch.tensor(placeholder_ids, dtype=torch.long),
        req=req,
        prefill_input_embeds=embeds.to(torch.bfloat16),
        input_embeds_are_projected=True,
        num_codebooks=int(state.num_codebooks),
        codebook_size=int(state.codebook_size),
        backbone_eos_token_id=eos,
        prompt_len=prompt_len,
        max_new_tokens=int(state.max_new_tokens),
        temperature=float(state.temperature),
        top_p=float(state.top_p),
        top_k=int(state.top_k),
        repetition_penalty=float(state.repetition_penalty),
        seed=state.seed,
    )


def apply_tts_result(state: BreezeState, result: BreezeSGLangRequestData) -> None:
    if not result.output_codes:
        raise ValueError(f"Request {result.req.rid}: Breeze generated no audio frames")
    state.output_codes = torch.stack(result.output_codes, dim=0)   # [T, num_codebooks]
    state.completion_tokens = int(state.output_codes.shape[0])
    state.prompt_tokens = int(result.prompt_len)
    state.finish_reason = result.finish_reason or "stop"


def make_tts_scheduler_adapters(*, max_new_tokens_cap: int | None = None,
                                context_length: int | None = None):
    """Build StagePayload <-> scheduler adapters for the Breeze backbone."""

    def request_builder(payload: StagePayload) -> BreezeSGLangRequestData:
        state = BreezeState.from_dict(payload.data)
        if max_new_tokens_cap is not None:
            state.max_new_tokens = min(int(state.max_new_tokens), int(max_new_tokens_cap))
        if context_length is not None:
            state.max_new_tokens = min(
                int(state.max_new_tokens), max(int(context_length) - 1 - int(state.prompt_len), 1))
        req_data = build_sglang_tts_request(state, request_id=payload.request_id)
        req_data.engine_start_s = time.perf_counter()
        req_data.stage_payload = payload
        return req_data

    def result_adapter(data: BreezeSGLangRequestData) -> StagePayload:
        payload = data.stage_payload
        state = BreezeState.from_dict(payload.data)
        apply_tts_result(state, data)
        if data.engine_start_s:
            state.engine_time_s = time.perf_counter() - data.engine_start_s
        return StagePayload(request_id=payload.request_id, request=payload.request,
                            data=state.to_dict())

    def stream_output_builder(data: BreezeSGLangRequestData) -> dict[str, Any] | None:
        """New frames since the last call, for the streaming vocoder (M3)."""
        n = len(data.output_codes)
        if n <= data.streamed_frames:
            return None
        chunk = torch.stack(data.output_codes[data.streamed_frames:n], dim=0)
        data.streamed_frames = n
        return {"codes": chunk, "frame_offset": n - int(chunk.shape[0])}

    return request_builder, result_adapter, stream_output_builder
