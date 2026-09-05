# SPDX-License-Identifier: Apache-2.0
"""SGLang engine construction for the Breeze TTS 2 backbone stage."""

from __future__ import annotations

import importlib
import os
from typing import Any

from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.vendor.sglang.server_args import override_server_args

from . import request_builders


class BreezeEngineBuilder(TtsEngineBuilder):
    model_name = "Breeze TTS 2"
    model_arch_override = "BreezeForConditionalGeneration"
    context_length = 2048          # the checkpoint's top-level max_position_embeddings; prompt + up to 750 frames

    def __init__(self, *, max_new_tokens: int) -> None:
        self.max_new_tokens = max_new_tokens
        self._stream_output_builder = None

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir
        from .hf_config import register_breeze_hf_config
        register_breeze_hf_config()

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        del dtype
        max_bs = int(os.environ.get("BREEZE_SGL_MAX_BS", "32"))
        return {
            "max_running_requests": max_bs,
            # SGLang captures the decode step per batch bucket; the codebook sampling and the
            # depth-decoder loop run inside that capture (sglang_model._decode_codebooks).
            "disable_cuda_graph": bool(os.environ.get("BREEZE_SGL_NO_GRAPH")),
            "cuda_graph_max_bs": max_bs,
            # Prompts arrive as embeddings behind placeholder ids (all zeros): with the radix
            # cache on, a second request of the same length would "hit" the first one's prefix
            # and reuse its KV — another voice, another text. No prefix sharing here.
            "disable_radix_cache": True,
            "mem_fraction_static": float(os.environ.get("BREEZE_SGL_MEM_FRACTION", "0.35")),
            "chunked_prefill_size": 4096,
            "dtype": "bfloat16",
            "enable_torch_compile": False,
            "random_seed": int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF,
        }

    def customize_server_args(self, server_args: Any) -> None:
        override_server_args(server_args, "sglang_omni.breeze_tts.runtime_defaults",
                             disable_overlap_schedule=True)

    def setup_model(self, *, model_worker: Any, checkpoint_dir: str, device: str, gpu_id: int,
                    server_args: Any) -> None:
        """Give the SGLang backbone its eager companion: the depth decoder
        (codebooks 1..15). The audio embedding table the two share is loaded
        by the backbone's load_weights from the depth decoder's checkpoint key."""
        del gpu_id
        from .stages import load_breeze_model
        model = model_worker.model_runner.model
        hf = load_breeze_model(checkpoint_dir, device)
        model.setup_breeze_decode(depth_decoder=hf.depth_decoder,
                                  max_batch_size=int(server_args.max_running_requests), device=device)
        del hf.codec_model                                     # the vocoder decodes through the audio tokenizer
        model.hf_prompt_model = hf                             # text encoder + merge: prompts are embedded at prefill

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        mod = importlib.import_module("sglang_omni.models.breeze_tts.model_runner")
        return mod.BreezeModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        request_builder, result_adapter, self._stream_output_builder = (
            request_builders.make_tts_scheduler_adapters(
                max_new_tokens_cap=self.max_new_tokens, context_length=self.context_length))
        return request_builder, result_adapter

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {"stream_output_builder": self._stream_output_builder}
