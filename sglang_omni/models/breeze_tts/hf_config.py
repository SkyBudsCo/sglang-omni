# SPDX-License-Identifier: Apache-2.0
"""HuggingFace config registration for Breeze TTS 2 checkpoints.

The checkpoint's config.json has model_type "breeze" with nested backbone /
depth-decoder / text-encoder / codec configs; breeze-tts's own BreezeConfig
(models/breeze_config.py in the checkout) registers all of them with
AutoConfig. SGLang's ModelConfig loads the config through AutoConfig, so this
must run in the model worker before ModelConfig.from_server_args."""

from __future__ import annotations

BREEZE_MODEL_ARCH_OVERRIDE = "BreezeForConditionalGeneration"

_registered = False


def register_breeze_hf_config() -> None:
    global _registered
    if _registered:
        return
    from .tokenizer import breeze_src
    breeze_src()                      # sys.path + the transformers-5 shims
    import models.breeze_config  # noqa: F401  (AutoConfig.register("breeze", BreezeConfig) and its sub-configs)
    _registered = True
