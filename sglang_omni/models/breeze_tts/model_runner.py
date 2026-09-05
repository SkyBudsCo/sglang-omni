# SPDX-License-Identifier: Apache-2.0
"""Model runner for the Breeze backbone.

Prefill: hands SGLang the pre-embedded prompt. Every step: re-syncs the batch's
slots on the model (sampling params, codebook-0 history, the last frame — rows
move between steps) with a handful of batched copies, and after the forward
overrides SGLang's sampled token with the model's codebook-0 / EOS choice.

Frames never make the CPU wait for the GPU inside a step: the newest frame stays
on the device (the next step's input is a device→device gather), and the copy
of tokens + frames to pinned host memory is asynchronous and consumed one step
late, so SGLang's CPU-side scheduling of step t+1 overlaps the GPU's step t.
A request's final pending frame is flushed by apply_tts_result."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner

from .sglang_model import REP_HISTORY_LEN

logger = logging.getLogger(__name__)
_DEBUG_TIMING = bool(os.environ.get("BREEZE_DEBUG_TIMING"))


class BreezeModelRunner(ModelRunner):
    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self._t_last_decode: float | None = None
        self._acc: dict[int, list[float]] = {}          # bs → [n, period, sync, collect]
        self._pending: tuple | None = None              # (event, requests, bs)
        self._pin_tokens: torch.Tensor | None = None
        self._pin_frames: torch.Tensor | None = None

    # ------------------------------------------------------------------ hooks
    def before_prefill(self, forward_batch, schedule_batch, requests):
        del schedule_batch
        self._sync_rows(requests)
        forward_batch.input_embeds = self._build_prefill_input_embeds(forward_batch, requests)

    def before_decode(self, forward_batch, schedule_batch, requests, *, is_lookahead: bool = False):
        del schedule_batch, is_lookahead, forward_batch
        if _DEBUG_TIMING:
            now = time.perf_counter()
            acc = self._acc.setdefault(len(requests), [0, 0.0, 0.0, 0.0])
            if self._t_last_decode is not None:
                acc[1] += now - self._t_last_decode
            self._t_last_decode = now
            self._sync_rows(requests)
            acc[2] += time.perf_counter() - now
        else:
            self._sync_rows(requests)

    def post_prefill(self, result, forward_batch, schedule_batch, requests):
        del forward_batch, schedule_batch
        self._collect(result, requests)

    def post_decode(self, result, forward_batch, schedule_batch, requests):
        del forward_batch, schedule_batch
        if _DEBUG_TIMING:
            t0 = time.perf_counter()
            self._collect(result, requests)
            acc = self._acc.setdefault(len(requests), [0, 0.0, 0.0, 0.0])
            acc[3] += time.perf_counter() - t0
            acc[0] += 1
            if acc[0] % 50 == 0:
                n = acc[0]
                logger.info("breeze runner bs=%d n=%d: step period %.1f ms | sync_rows %.1f ms | collect %.1f ms",
                            len(requests), n, 1000 * acc[1] / max(n - 1, 1), 1000 * acc[2] / n, 1000 * acc[3] / n)
        else:
            self._collect(result, requests)

    # --------------------------------------------------------------- batched
    def _ensure_pinned(self) -> None:
        if self._pin_tokens is None:
            n = self.model._last_frame.shape[0]
            self._pin_tokens = torch.zeros(n, dtype=torch.long).pin_memory()
            self._pin_frames = torch.zeros(n, self.model.num_codebooks, dtype=torch.long).pin_memory()

    def _sync_rows(self, requests: list) -> None:
        """Row i of the batch ↔ request i: a few batched copies, no host sync."""
        self._flush_pending()
        bs = len(requests)
        if bs == 0:
            return
        model = self.model
        dev = model._last_frame.device
        temps, top_ks, d_temps, d_top_ks, pens, counts = [], [], [], [], [], []
        prev = torch.zeros(bs, REP_HISTORY_LEN, dtype=torch.long)
        last_rows: list[torch.Tensor] = []
        zero_frame = None
        for row, sched_req in enumerate(requests):
            data = sched_req.data
            temps.append(float(data.temperature))
            top_ks.append(int(data.top_k))
            d_temps.append(float(getattr(data, "depth_temperature", 0.9)))
            d_top_ks.append(int(getattr(data, "depth_top_k", 50)))
            pens.append(float(data.repetition_penalty))
            history = data.cb0_history[-REP_HISTORY_LEN:]
            counts.append(len(history))
            if history:
                prev[row, : len(history)] = torch.tensor(history, dtype=torch.long)
            if data.last_frame is not None:
                last_rows.append(data.last_frame)
            else:
                if zero_frame is None:
                    zero_frame = torch.zeros(model.num_codebooks, dtype=torch.long, device=dev)
                last_rows.append(zero_frame)
        model._temperature[:bs].copy_(torch.tensor(temps), non_blocking=True)
        model._top_k[:bs].copy_(torch.tensor(top_ks), non_blocking=True)
        model._depth_temperature[:bs].copy_(torch.tensor(d_temps), non_blocking=True)
        model._depth_top_k[:bs].copy_(torch.tensor(d_top_ks), non_blocking=True)
        model._rep_penalty[:bs].copy_(torch.tensor(pens), non_blocking=True)
        model._prev_count[:bs].copy_(torch.tensor(counts), non_blocking=True)
        model._prev_tokens[:bs].copy_(prev.to(dev, non_blocking=True))
        model._last_frame[:bs].copy_(torch.stack(last_rows))          # device→device

    def _collect(self, result: Any, requests: list) -> None:
        """After a forward: the model's codebook-0 / EOS choice replaces SGLang's
        sample (device tensor, no sync); frames + tokens start an async copy to
        pinned memory that _flush_pending consumes at the next hook."""
        bs = len(requests)
        if bs == 0:
            return
        model = self.model
        self._flush_pending()
        self._ensure_pinned()
        tokens = model._out_token[:bs]
        result.next_token_ids = tokens.clone()
        frames = model._out_frame[:bs].clone()
        for row, sched_req in enumerate(requests):
            data = sched_req.data
            data.pending_frame = frames[row]          # device view: candidate next input + last frame
            data.last_frame = frames[row]
        self._pin_tokens[:bs].copy_(tokens, non_blocking=True)
        self._pin_frames[:bs].copy_(frames, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._pending = (event, list(requests), bs)

    def _flush_pending(self) -> None:
        """Distribute the previous step's tokens/frames (host copies) to the requests."""
        if self._pending is None:
            return
        event, requests, bs = self._pending
        self._pending = None
        event.synchronize()
        token_list = self._pin_tokens[:bs].tolist()
        frames_cpu = self._pin_frames[:bs].clone()
        eos = self.model.eos_token_id
        for row, sched_req in enumerate(requests):
            data = sched_req.data
            data.pending_frame = None
            if getattr(data.req, "inflight_middle_chunks", 0) > 0:
                continue                      # chunked prefill: no frame until the last chunk
            token = token_list[row]
            if token == eos:
                data.last_frame = None
                continue
            data.cb0_history.append(int(token))
            data.output_codes.append(frames_cpu[row])

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
