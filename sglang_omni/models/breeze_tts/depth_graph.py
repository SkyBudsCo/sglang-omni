# SPDX-License-Identifier: Apache-2.0
"""The depth decoder's 15-codebook loop as one CUDA graph per batch bucket.

Eager, the loop is 15 sequential HF forwards (12 layers each) per frame:
~117 ms of kernel launches at any batch size — 1.5× slower than real time on
its own. Captured once per bucket (static shapes: the token sequence grows
1 → 16 the same way every frame), a replay is a few milliseconds. Sampling
(temperature / top-k / multinomial) is inside the graph; torch keeps the CUDA
RNG graph-safe by advancing the Philox offset on every replay.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

DEFAULT_BUCKETS = (1, 2, 4, 8, 16, 32)


class DepthDecoderGraph:
    def __init__(
        self,
        depth_decoder: Any,
        *,
        num_codebooks: int,
        hidden_size: int,
        sample_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
        depth_bias: Tensor,
        device: torch.device,
        max_batch_size: int,
        buckets: tuple[int, ...] = DEFAULT_BUCKETS,
    ) -> None:
        self.depth_decoder = depth_decoder
        self.num_codebooks = int(num_codebooks)
        self.hidden_size = int(hidden_size)
        self.sample_fn = sample_fn
        self.depth_bias = depth_bias
        self.device = device
        self.buckets = tuple(b for b in buckets if b <= max_batch_size) or (max_batch_size,)
        if self.buckets[-1] < max_batch_size:
            self.buckets = self.buckets + (int(max_batch_size),)
        self.dtype = next(depth_decoder.parameters()).dtype
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._io: dict[int, dict[str, Tensor]] = {}
        self._pool = None
        self.enabled = False

    # ------------------------------------------------------------- the loop
    def _loop(self, hidden: Tensor, cb0: Tensor, temperature: Tensor, top_k: Tensor) -> Tensor:
        """[B, hidden], [B] → frame [B, num_codebooks]. Exactly the eager loop."""
        bs = cb0.shape[0]
        seq = torch.cat([torch.zeros(bs, 1, dtype=torch.long, device=cb0.device), cb0.unsqueeze(1)], dim=1)
        for _ in range(1, self.num_codebooks):
            # cache_position must be given: without it the codebooks head indexes its
            # weight with a CPU arange (a host→device copy, illegal under graph capture)
            cache_position = torch.arange(seq.shape[1], device=cb0.device)
            out = self.depth_decoder(input_ids=seq, backbone_last_hidden_state=hidden, use_cache=False,
                                     cache_position=cache_position, return_dict=True)
            step_logits = out.logits[:, -1, :].float() + self.depth_bias
            tok = self.sample_fn(step_logits, temperature, top_k)
            seq = torch.cat([seq, tok.unsqueeze(1)], dim=1)
        return seq[:, 1:]

    # --------------------------------------------------------------- capture
    def capture(self) -> None:
        self._pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream), torch.no_grad():
            for b in self.buckets:
                io = {
                    "hidden": torch.zeros(b, self.hidden_size, dtype=self.dtype, device=self.device),
                    "cb0": torch.zeros(b, dtype=torch.long, device=self.device),
                    "temperature": torch.full((b,), 0.9, device=self.device),
                    "top_k": torch.full((b,), 50, dtype=torch.long, device=self.device),
                    "frame": torch.zeros(b, self.num_codebooks, dtype=torch.long, device=self.device),
                }
                for _ in range(2):                                         # warm up (cuBLAS, allocator)
                    io["frame"].copy_(self._loop(io["hidden"], io["cb0"], io["temperature"], io["top_k"]))
                torch.cuda.synchronize(self.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self._pool, stream=stream):
                    io["frame"].copy_(self._loop(io["hidden"], io["cb0"], io["temperature"], io["top_k"]))
                self._graphs[b] = graph
                self._io[b] = io
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self.enabled = True
        logger.info("breeze depth decoder: CUDA graphs captured for batch buckets %s", list(self.buckets))

    # ------------------------------------------------------------------ run
    @torch.no_grad()
    def run(self, hidden: Tensor, cb0: Tensor, temperature: Tensor, top_k: Tensor) -> Tensor:
        bs = cb0.shape[0]
        if not self.enabled or bs > self.buckets[-1]:
            return self._loop(hidden.to(self.dtype), cb0, temperature, top_k)
        b = next(x for x in self.buckets if x >= bs)
        io = self._io[b]
        io["hidden"][:bs].copy_(hidden.to(self.dtype))
        io["cb0"][:bs].copy_(cb0)
        io["temperature"][:bs].copy_(temperature)
        io["top_k"][:bs].copy_(top_k)
        if b > bs:                                       # padded rows: valid inputs, outputs ignored
            io["hidden"][bs:].copy_(io["hidden"][0].expand(b - bs, -1))
            io["cb0"][bs:].copy_(io["cb0"][0].expand(b - bs))
            io["temperature"][bs:].copy_(io["temperature"][0].expand(b - bs))
            io["top_k"][bs:].copy_(io["top_k"][0].expand(b - bs))
        self._graphs[b].replay()
        return io["frame"][:bs].clone()
