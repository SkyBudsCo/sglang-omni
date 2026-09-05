# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Breeze TTS 2."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

_PKG = "sglang_omni.models.breeze_tts"


class BreezePipelineConfig(PipelineConfig):
    """3-stage TTS pipeline: preprocessing → tts_engine → vocoder.

    preprocessing (CPU threads + one GPU copy of the text encoder / codec
    encoder): text → T5Gemma2 features, reference wav → Mimi codes, both merged
    into the backbone's prompt embeddings exactly as breeze-tts does.
    tts_engine (SGLang): the Qwen3 backbone + the depth decoder per frame.
    vocoder: Mimi frames → 24 kHz PCM.
    """

    architecture: ClassVar[str] = "BreezeForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": EngineStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="preprocessing",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            next="tts_engine",
        ),
        EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory=FactoryArgs(device="cuda:0", max_new_tokens=1024),
            gpu=0,
            next="vocoder",
            # M3 adds stream_to=["vocoder"] with a streaming Mimi decoder.
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            gpu=0,
            terminal=True,
        ),
    ]

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = BreezePipelineConfig
