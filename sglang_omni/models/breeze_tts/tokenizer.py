# SPDX-License-Identifier: Apache-2.0
"""Prompt construction for Breeze TTS 2 — delegated to breeze-tts's own templates.

breeze-tts (github.com/breezeblue-ai/breeze-tts, Apache-2.0) already defines
the prompt layout its server uses: `[S0]` speaker prefix, the reference
transcript, one `<|AUDIO|>` placeholder per reference frame (+ `<|audio_eos|>`),
then `<ins_bos>instruction<ins_eos>` and the target text — template
"ref_edit_tata" with a reference, "tts_instruction" without — and
`breeze_infer.templates.prepare_inputs` turns that into input_ids /
attention_mask / text_ids_mask / text_ids_len / input_values (reference codes
from the bundled Qwen3-TTS audio tokenizer). Reusing it keeps this port
bit-compatible with the reference runtime; vendoring is the upstream TODO
(BREEZE_TTS_SRC points at the checkout until then).
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from typing import Any

DEFAULT_BREEZE_SRC = "/workspace/breeze-tts"
DEFAULT_INSTRUCTION = "Speak clearly and naturally."     # breeze_infer.api's Form default
DEFAULT_SPEAKER = "S0"


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
    _shim_transformers_5()
    return src


def _shim_transformers_5() -> None:
    """breeze-tts and its qwen-tts audio tokenizer are pinned to transformers
    4.57; under 5.x three things moved: `no_init_weights` (modeling_utils →
    initialization), `check_model_inputs` (a `@check_model_inputs()` factory
    became a plain decorator) and the T5Gemma configs breeze registers already
    ship — the same classes, so re-registering is harmless once `exist_ok` is
    forced. Vendoring both (the upstream TODO) retires this."""
    import transformers.modeling_utils as modeling_utils
    if not hasattr(modeling_utils, "no_init_weights"):
        from transformers.initialization import no_init_weights
        modeling_utils.no_init_weights = no_init_weights
    import transformers.modeling_rope_utils as rope_utils
    if "default" not in rope_utils.ROPE_INIT_FUNCTIONS:      # 5.x dropped the key; qwen-tts looks it up
        rope_utils.ROPE_INIT_FUNCTIONS["default"] = _default_rope_parameters
    import inspect
    import transformers.masking_utils as masking_utils
    for fn_name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        original_mask = getattr(masking_utils, fn_name, None)
        if original_mask is None or getattr(original_mask, "_breeze_tolerant", False):
            continue
        accepted = set(inspect.signature(original_mask).parameters)

        def make_mask(*args, _original=original_mask, _accepted=accepted, **kwargs):
            # 4.57 called (config, input_embeds, attention_mask, cache_position, past_key_values,
            # position_ids); 5.x renamed input_embeds and derives positions differently.
            if "input_embeds" in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            cache_position = kwargs.get("cache_position")
            if "position_ids" in _accepted and kwargs.get("position_ids") is None and cache_position is not None:
                kwargs["position_ids"] = cache_position.unsqueeze(0)
            kwargs = {k: v for k, v in kwargs.items() if k in _accepted}
            return _original(*args, **kwargs)

        make_mask._breeze_tolerant = True
        setattr(masking_utils, fn_name, make_mask)
    import transformers.utils.generic as generic
    original_check = generic.check_model_inputs
    if not getattr(original_check, "_breeze_tolerant", False):

        def check_model_inputs(func=None, **kwargs):
            if func is None:                       # 4.57 form: @check_model_inputs()
                return lambda f: original_check(f)
            return original_check(func)

        check_model_inputs._breeze_tolerant = True
        generic.check_model_inputs = check_model_inputs
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


def _default_rope_parameters(config: Any, device: Any = None, seq_len: int | None = None, **_: Any):
    """transformers 4.57's `_compute_default_rope_parameters`: plain RoPE
    inverse frequencies from the config's rope_theta / head_dim."""
    import torch
    del seq_len
    base = float(getattr(config, "rope_theta", 10000.0))
    partial = float(getattr(config, "partial_rotary_factor", 1.0))
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


@dataclass
class BreezeReference:
    """A reference voice: a wav on disk (any rate — the audio tokenizer
    resamples) and its exact transcript, both required by Breeze."""

    audio_path: str
    text: str


@dataclass
class BreezeRequest:
    """One prompt as breeze_infer.templates understands it."""

    request: dict[str, Any]
    template: str
    reference: BreezeReference | None = field(default=None, repr=False)


class BreezePromptAdapter:
    """(text, reference, instruction) → the tensors breeze-tts's runtime feeds
    its backbone, via breeze_infer.templates.prepare_inputs on the same
    tokenizer + audio tokenizer the reference server loads."""

    def __init__(self, checkpoint_dir: str, model: Any, device: str) -> None:
        breeze_src()
        self._templates = importlib.import_module("breeze_infer.templates")
        self.model = model
        from transformers import AutoTokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, fix_mistral_regex=False)
        except TypeError:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        from qwen_tts import Qwen3TTSTokenizer
        bundled = os.path.join(checkpoint_dir, "audio_tokenizer")
        if not os.path.isdir(bundled):
            raise FileNotFoundError(f"Breeze checkpoint has no audio_tokenizer/ at {bundled}")
        self.audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(bundled, device_map=device)

    def build_request(
        self,
        *,
        text: str,
        reference: BreezeReference | None,
        instruction: str | None = None,
        speaker: str = DEFAULT_SPEAKER,
        request_id: str | None = None,
    ) -> BreezeRequest:
        req: dict[str, Any] = {
            "id": request_id or "sglang-omni",
            "text": text,
            "instruction": instruction or DEFAULT_INSTRUCTION,
            "speaker": speaker,
        }
        template = "tts_instruction"
        if reference is not None:
            req["ref_audio_path"] = reference.audio_path
            req["ref_text"] = reference.text
            template = "ref_edit_tata"
        return BreezeRequest(request=req, template=template, reference=reference)

    def prepare_inputs(self, req: BreezeRequest) -> dict[str, Any]:
        """input_ids / attention_mask / text_ids_mask / text_ids_len /
        input_values for one request, exactly as the reference runtime builds
        them. Single branch (no classifier-free guidance) in M1."""
        return self.prepare_inputs_batch([req], req.template)

    def prepare_inputs_batch(self, reqs: list[BreezeRequest], template: str) -> dict[str, Any]:
        """The same for several requests sharing a template: a left-padded batch
        (attention_mask marks the real tokens)."""
        return self._templates.prepare_inputs(
            self.tokenizer,
            self.audio_tokenizer,
            self.model,
            [r.request for r in reqs],
            self._templates.get_template(template),
            guidance_scale=1.0,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
