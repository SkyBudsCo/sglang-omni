# SPDX-License-Identifier: Apache-2.0
"""Breeze TTS 2 (BreezeBlue) model support for sglang-omni.

Breeze is a dual-AR TTS in the same family as Fish S2-Pro: a Qwen3 backbone
predicts the first RVQ codebook of each 12.5 Hz Mimi frame, a small depth
decoder predicts the other 15 codebooks conditioned on the backbone's hidden
state, and the Mimi codec turns frames into 24 kHz audio. The T5Gemma2 text
encoder and the reference-audio codes are one-shot work: they are merged into
the prompt embeddings in preprocessing and handed to the SGLang backbone as
projected input embeds, so the engine stage is a plain Qwen3 stack with
continuous batching, paged KV and RadixAttention — the throughput the
reference PyTorch runtime (one request per process) leaves on the table.

Contributed by SkyBuds (commercial Breeze licensee); see docs/models/breeze_tts.md.
"""

from sglang_omni.models.model_capabilities import ModelCapabilities

from . import config

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=True,
    supports_batch_vocoder=True,
    supports_streaming_vocoder=False,    # M3: per-frame Mimi streaming decode
    supports_cuda_graph=False,           # M4
    supports_torch_compile=False,        # M4
    supports_breakable_prefill_cuda_graph=False,
)

__all__ = ["CAPABILITIES", "config"]
