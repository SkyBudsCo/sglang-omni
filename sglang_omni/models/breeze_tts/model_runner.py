# SPDX-License-Identifier: Apache-2.0
"""Model runner for the Breeze backbone.

Prefill: hands SGLang the pre-embedded prompt. Every step: re-syncs each batch
row's slot on the model (sampling params, codebook-0 history, the last frame —
rows move between steps), and after the forward copies the frame the model
staged into the request and overrides SGLang's sampled token with the model's
codebook-0 / EOS choice so scheduling, stop and streaming logic agree."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner

from .sglang_model import REP_HISTORY_LEN


class BreezeModelRunner(ModelRunner):
    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)

    # ------------------------------------------------------------------ hooks
    def before_prefill(self, forward_batch, schedule_batch, requests):
        del schedule_batch
        self._sync_rows(requests)
        forward_batch.input_embeds = self._build_prefill_input_embeds(forward_batch, requests)

    def before_decode(self, forward_batch, schedule_batch, requests, *, is_lookahead: bool = False):
        del schedule_batch, is_lookahead, forward_batch
        self._sync_rows(requests)

    def post_prefill(self, result, forward_batch, schedule_batch, requests):
        del forward_batch, schedule_batch
        self._collect(result, requests)

    def post_decode(self, result, forward_batch, schedule_batch, requests):
        del forward_batch, schedule_batch
        self._collect(result, requests)

    # --------------------------------------------------------------- per row
    def _sync_rows(self, requests: list) -> None:
        model = self.model
        for row, sched_req in enumerate(requests):
            data = sched_req.data
            model._temperature[row] = float(data.temperature)
            model._top_k[row] = int(data.top_k)
            model._depth_temperature[row] = float(getattr(data, "depth_temperature", 0.9))
            model._depth_top_k[row] = int(getattr(data, "depth_top_k", 50))
            model._rep_penalty[row] = float(data.repetition_penalty)
            history = data.cb0_history[-REP_HISTORY_LEN:]
            n = len(history)
            if n:
                model._prev_tokens[row, :n] = torch.as_tensor(history, dtype=torch.long,
                                                              device=model._prev_tokens.device)
            model._prev_count[row] = n
            if data.last_frame is not None:
                model._last_frame[row].copy_(data.last_frame.to(model._last_frame.device))

    def _collect(self, result: Any, requests: list) -> None:
        """After a forward: the model's codebook-0 / EOS choice replaces
        SGLang's sample; complete frames go to the request."""
        bs = len(requests)
        if bs == 0:
            return
        model = self.model
        tokens = model._out_token[:bs]
        result.next_token_ids = tokens.clone()
        frames = model._out_frame[:bs]
        token_list = tokens.tolist()
        for row, sched_req in enumerate(requests):
            data = sched_req.data
            if getattr(data.req, "inflight_middle_chunks", 0) > 0:
                continue                      # chunked prefill: no frame until the last chunk
            token = token_list[row]
            if token == model.eos_token_id:
                continue
            frame = frames[row].clone()
            data.last_frame = frame
            data.cb0_history.append(int(token))
            data.output_codes.append(frame.cpu())

    def _build_prefill_input_embeds(self, forward_batch: Any, requests: list) -> torch.Tensor:
        """Concatenate each request's prefill_input_embeds over the range SGLang
        is extending this step (chunked prefill / prefix cache aware)."""
        input_ids = forward_batch.input_ids
        device = input_ids.device
        pieces: list[torch.Tensor] = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            start = len(req.prefix_indices)
            length = int(req.extend_range.length)
            emb = data.prefill_input_embeds
            if emb is None:
                raise ValueError(f"Request {req.rid}: no prefill_input_embeds")
            pieces.append(emb[start:start + length].to(device=device, dtype=torch.bfloat16))
        out = torch.cat(pieces, dim=0)
        if out.shape[0] != input_ids.shape[0]:
            raise ValueError(f"prefill embeds {out.shape[0]} rows vs {input_ids.shape[0]} ids")
        return out
