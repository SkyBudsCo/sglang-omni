# SPDX-License-Identifier: Apache-2.0
"""Breeze TTS 2 pipeline state definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class BreezeState(DeclarativeStateBase):
    """Per-request pipeline state for Breeze TTS 2."""

    sample_rate: int = 24000

    # -- From preprocessing ------------------------------------------------
    # The backbone's prompt, already embedded: text-encoder features and the
    # reference frames' codebook embeddings merged by breeze-tts's own
    # _merge_input_ids_with_input_values. [prompt_len, hidden] bf16.
    prefill_embeds: Any | None = wire(None, codec="tensor_restore")
    prompt_len: int = 0
    num_codebooks: int = 16
    codebook_size: int = 2051
    backbone_eos_token_id: int = 2051     # config.vocab_size: the lm_head's extra class

    # -- Generation params -------------------------------------------------
    max_new_tokens: int = 1024            # frames (12.5 Hz → 80 s cap)
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
    seed: int | None = None

    # -- From TTS engine ---------------------------------------------------
    output_codes: Any | None = wire(None, codec="tensor_restore")   # [T_frames, num_codebooks]
    finish_reason: str | None = None

    # -- Timing (seconds) --------------------------------------------------
    preprocess_time_s: float | None = None
    preprocess_encode_s: float | None = None       # reference audio → codes
    preprocess_merge_s: float | None = None        # text encoder + merge
    engine_time_s: float | None = None

    # -- From vocoder ------------------------------------------------------
    audio_samples: Any | None = wire(None, codec="tensor_list")
