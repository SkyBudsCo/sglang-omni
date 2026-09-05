# SPDX-License-Identifier: Apache-2.0
"""Prompt construction for Breeze TTS 2 — delegated to breeze-tts's own templates.

breeze-tts (github.com/breezeblue-ai/breeze-tts, Apache-2.0) already defines
the prompt layout: a `[speaker]` prefix, the reference transcript, one
`<|AUDIO|>` placeholder per reference frame, an optional
`<ins_bos>instruction<ins_eos>` block, then the target text — and
`prepare_inputs` turns that into input_ids / text_ids_mask / text_ids_len /
input_values. Reusing it keeps this port bit-compatible with the reference
runtime; vendoring is the upstream TODO (BREEZE_TTS_SRC points at the checkout
until then).
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from typing import Any

import torch

DEFAULT_BREEZE_SRC = "/workspace/breeze-tts"


def breeze_src() -> str:
    """The breeze-tts checkout, importable. Set BREEZE_TTS_SRC to override."""
    src = os.environ.get("BREEZE_TTS_SRC", DEFAULT_BREEZE_SRC)
    if not os.path.isdir(os.path.join(src, "breeze_infer")):
        raise RuntimeError(
            f"breeze-tts checkout not found at {src} — clone "
            "github.com/breezeblue-ai/breeze-tts and set BREEZE_TTS_SRC"
        )
    if src not in sys.path:
        sys.path.insert(0, src)
    _tolerate_auto_registration()
    return src


def _tolerate_auto_registration() -> None:
    """breeze-tts (pinned to transformers 4.57) registers T5Gemma configs
    that newer transformers already ship — the same classes, so re-registering
    is harmless; make its `AutoConfig.register(...)` calls `exist_ok`."""
    from transformers import AutoConfig, AutoModel
    for auto in (AutoConfig, AutoModel):
        original = auto.register
        if getattr(original, "_breeze_tolerant", False):
            continue

        def register(*args, _original=original, **kwargs):
            kwargs["exist_ok"] = True
            return _original(*args, **kwargs)

        register._breeze_tolerant = True
        auto.register = staticmethod(register) if isinstance(auto.__dict__.get("register"), staticmethod) else register


@dataclass
class BreezeReference:
    """A reference voice: its waveform (mono, any rate — resampled by the
    codec front end) and the exact transcript, both required by Breeze."""

    audio: torch.Tensor          # [samples] float32
    sample_rate: int
    text: str


class BreezePromptAdapter:
    """(text, reference, instruction) → the tensors breeze-tts's runtime feeds
    its backbone, via breeze_infer.templates.prepare_inputs."""

    def __init__(self, checkpoint_dir: str) -> None:
        breeze_src()
        self._templates = importlib.import_module("breeze_infer.templates")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    def build_request(
        self,
        *,
        text: str,
        reference: BreezeReference | None,
        instruction: str | None = None,
        speaker: str = "S1",
    ) -> dict[str, Any]:
        """The request dict breeze_infer.templates understands."""
        req: dict[str, Any] = {"text": text, "speaker": speaker}
        if reference is not None:
            req["ref_text"] = reference.text
            req["ref_audio"] = reference.audio
            req["ref_sample_rate"] = reference.sample_rate
        if instruction:
            req["instruction"] = instruction
        return req

    def prepare_inputs(self, request: dict[str, Any], codec: Any, device: str) -> dict[str, Any]:
        """input_ids / text_ids_mask / text_ids_len / input_values for one
        request, exactly as the reference runtime builds them."""
        return self._templates.prepare_inputs(
            request, tokenizer=self.tokenizer, codec=codec, device=device
        )
