# SPDX-License-Identifier: Apache-2.0
"""Model runner for the Breeze backbone: hands SGLang the pre-embedded prompt
on prefill and, on every decode step, tells the model which request sits in
which batch row so the previous frame's embedding — not a token — is the input."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner


class BreezeModelRunner(ModelRunner):
    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)

    def before_prefill(self, forward_batch, schedule_batch, requests):
        del schedule_batch
        self._sync_rows(requests)
        forward_batch.input_embeds = self._build_prefill_input_embeds(forward_batch, requests)

    def before_decode(self, forward_batch, schedule_batch, requests, *, is_lookahead: bool = False):
        del schedule_batch, is_lookahead, forward_batch
        self._sync_rows(requests)

    def post_prefill(self, result, forward_batch, schedule_batch, requests):
        del result, forward_batch, schedule_batch, requests

    def post_decode(self, result, forward_batch, schedule_batch, requests):
        del result, forward_batch, schedule_batch, requests

    def _sync_rows(self, requests: list) -> None:
        """Row i of the batch ↔ request i: sampling params, the frame sink the
        model appends to, and the last frame (the next step's input)."""
        model = self.model
        for row, sched_req in enumerate(requests):
            data = sched_req.data
            model._frame_sink[row] = data
            if data.output_codes:
                model._last_frame[row].copy_(data.output_codes[-1].to(model._last_frame.device))
                model._has_frame[row] = True
            else:
                model._has_frame[row] = False
            temp = model._sampling.setdefault(
                "temperature", torch.ones(model._last_frame.shape[0], device=model._last_frame.device))
            temp[row] = float(data.temperature or 1.0)

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
