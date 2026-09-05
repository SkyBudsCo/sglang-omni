# SPDX-License-Identifier: Apache-2.0
"""M1 tests for the Breeze TTS 2 plugin: the pipeline is discoverable and the
preprocessing / vocoder stages agree with the reference runtime bit for bit.
GPU tests need the checkpoint (BREEZE_TTS_MODEL) and the breeze-tts checkout
(BREEZE_TTS_SRC); they skip otherwise."""

import os

import pytest
import torch

MODEL = os.environ.get("BREEZE_TTS_MODEL", "/workspace/models/breeze-tts-2")
gpu = pytest.mark.skipif(not (torch.cuda.is_available() and os.path.isdir(MODEL)),
                         reason="needs a GPU and the Breeze checkpoint")


def test_pipeline_config_is_registered():
    from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
    from sglang_omni.models.breeze_tts.config import BreezePipelineConfig
    assert "BreezeForConditionalGeneration" in PIPELINE_CONFIG_REGISTRY.get_supported_archs()
    cfg = PIPELINE_CONFIG_REGISTRY.get_config("BreezeForConditionalGeneration")
    assert cfg is BreezePipelineConfig
    names = [s.name for s in BreezePipelineConfig.model_fields["stages"].default]
    assert names == ["preprocessing", "tts_engine", "vocoder"]


def test_state_round_trips():
    from sglang_omni.models.breeze_tts.payload_types import BreezeState
    s = BreezeState(prefill_embeds=torch.zeros(7, 2048, dtype=torch.bfloat16), prompt_len=7)
    back = BreezeState.from_dict(s.to_dict())
    assert back.prompt_len == 7 and tuple(back.prefill_embeds.shape) == (7, 2048)
    assert back.num_codebooks == 16 and back.backbone_eos_token_id == 2051


def test_request_builder_hands_sglang_the_embeds():
    from sglang_omni.models.breeze_tts.payload_types import BreezeState
    from sglang_omni.models.breeze_tts.request_builders import build_sglang_tts_request
    s = BreezeState(prefill_embeds=torch.zeros(5, 2048), prompt_len=5, max_new_tokens=40)
    rd = build_sglang_tts_request(s, request_id="r1")
    assert rd.prompt_len == 5 and rd.input_embeds_are_projected
    assert rd.prefill_input_embeds.dtype == torch.bfloat16
    assert set(rd.req.sampling_params.stop_token_ids) == {2051}
    assert len(rd.req.origin_input_ids) == 5


@gpu
def test_prompt_embeds_match_the_reference_runtime():
    """preprocessing's merged prompt == breeze-tts's _merge_input_ids_with_input_values."""
    from sglang_omni.models.breeze_tts.stages import load_breeze_model, build_prompt_embeds
    from sglang_omni.models.breeze_tts.tokenizer import BreezePromptAdapter, BreezeReference
    import torchaudio
    model = load_breeze_model(MODEL, "cuda")
    adapter = BreezePromptAdapter(MODEL)
    ref_dir = os.environ.get("BREEZE_TEST_REF", "/workspace/self-hosted/services/tts/references/ash-final")
    wav, sr = torchaudio.load(os.path.join(ref_dir, "sample.wav"))
    ref = BreezeReference(audio=wav.mean(0), sample_rate=sr, text=open(os.path.join(ref_dir, "sample.lab")).read().strip())
    req = adapter.build_request(text="Hi there, it's me.", reference=ref)
    embeds = build_prompt_embeds(model, adapter, req, "cuda")
    assert embeds.ndim == 2 and embeds.shape[1] == model.config.hidden_size
    # the reference path, called directly
    inputs = adapter.prepare_inputs(req, codec=model.codec_model, device="cuda")
    ref_embeds = model._merge_input_ids_with_input_values(
        input_ids=inputs["input_ids"], input_values=inputs.get("input_values"),
        text_ids_mask=inputs["text_ids_mask"], text_ids_len=inputs["text_ids_len"],
        attention_mask=inputs["attention_mask"])["inputs_embeds"][0]
    assert torch.equal(embeds.to(ref_embeds.dtype), ref_embeds)


@gpu
def test_vocoder_decodes_frames_like_the_codec():
    from sglang_omni.models.breeze_tts.stages import load_breeze_model, decode_frames
    model = load_breeze_model(MODEL, "cuda")
    frames = torch.randint(0, 2048, (25, 16), device="cuda")        # 2 s of 12.5 Hz frames
    audio = decode_frames(model, frames)
    assert audio.ndim == 1 and 24000 * 1.8 < audio.numel() < 24000 * 2.2
