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
    # breeze-tts prepare_inputs tensors (CPU) when the engine embeds the prompt itself
    prompt_inputs: dict[str, Any] | None = None
    # Frames the model has produced so far, [num_codebooks] each (CPU) —
    # appended by the model runner, drained by the stream output builder.
    output_codes: list[torch.Tensor] = field(default_factory=list)
    last_frame: torch.Tensor | None = None          # device view of the newest frame: the next step's input
    pending_frame: torch.Tensor | None = None       # newest frame not yet copied to output_codes (device)
    cb0_history: list[int] = field(default_factory=list)   # for the repetition penalty
    depth_temperature: float = 0.9
    depth_top_k: int = 50
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
    prompt_inputs = None
    if embeds is not None:
        if not isinstance(embeds, torch.Tensor):
            embeds = torch.as_tensor(embeds)
        embeds = embeds.to(torch.bfloat16)
        prompt_len = int(embeds.shape[0])
    else:
        # the engine embeds the prompt at prefill (model_runner) from these
        prompt_inputs = {
            "input_ids": torch.as_tensor(state.input_ids, dtype=torch.long),
            "text_ids_mask": torch.as_tensor(state.text_ids_mask, dtype=torch.bool),
            "text_ids_len": torch.as_tensor(state.text_ids_len, dtype=torch.long),
            "input_values": (torch.as_tensor(state.input_values, dtype=torch.long)
                             if state.input_values is not None else None),
        }
        prompt_len = int(prompt_inputs["input_ids"].shape[0])
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
    sampling_params.normalize(None)     # initializes stop_strs / stop-string state the scheduler reads
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
        prefill_input_embeds=embeds,
        prompt_inputs=prompt_inputs,
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
    # the runner copies frames to the host one step late; a request that just finished
    # may still hold its last frame on the device (an EOS step stages no frame)
    if result.pending_frame is not None:
        last_token = result.req.output_ids[-1] if getattr(result.req, "output_ids", None) else None
        if last_token != result.backbone_eos_token_id:
            result.output_codes.append(result.pending_frame.cpu())
        result.pending_frame = None
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

    def stream_output_builder(request_id: str, data: BreezeSGLangRequestData, req_output: Any) -> list:
        """Per-step hook from the OmniScheduler. M2 vocodes whole utterances, so
        nothing streams yet; M3 returns OutgoingMessage(type="stream",
        target="vocoder") chunks of new frames here (see the S2 builder)."""
        del request_id, req_output
        data.streamed_frames = len(data.output_codes)
        return []

    return request_builder, result_adapter, stream_output_builder
