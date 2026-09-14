# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.model_executor.models.personaplex.personaplex_mimi import (
    _MimiStreamingTransformer,
)
from vllm_omni.model_executor.models.personaplex.personaplex_temporal import (
    _RingKV,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def make_mimi_tf(batch: int = 2, context: int = 16) -> _MimiStreamingTransformer:
    torch.manual_seed(1234)
    tf = _MimiStreamingTransformer(num_layers=2, dim=32, num_heads=4, context=context)
    for p in tf.parameters():
        if p.dtype.is_floating_point:
            torch.nn.init.normal_(p, std=0.1)
    tf.streaming_init(batch)
    return tf


def test_batched_singleton_lockstep_steps_bit_parity() -> None:
    B, H, D, cap, T, steps = 3, 2, 4, 8, 2, 20
    kv = _RingKV(B, H, D, cap, torch.device("cpu"), torch.float32)
    rings = [_RingKV(1, H, D, cap, torch.device("cpu"), torch.float32) for _ in range(B)]
    for _ in range(steps):
        k, v = torch.randn(B, H, T, D), torch.randn(B, H, T, D)
        _, _, pos = kv.complete(k, v)
        for b in range(B):
            _, _, pos1 = rings[b].complete(k[b : b + 1], v[b : b + 1])
            assert torch.equal(pos[b], pos1[0])
            assert torch.equal(kv.cache[0][b], rings[b].cache[0][0])
            assert torch.equal(kv.end_offset[b], rings[b].end_offset[0])


def test_active_mask_bit_parity():
    B, H, D, T, capacity = 2, 2, 2, 4, 8
    kv = _RingKV(B, H, D, capacity, torch.device("cpu"), torch.float32)
    for _ in range(10):
        kv.complete(torch.randn(B, H, T, D), torch.randn(B, H, T, D))
    cache_before, off_before = kv.cache.clone(), kv.end_offset.clone()

    kv.complete(torch.randn(B, H, T, D), torch.randn(B, H, T, D), active=torch.tensor([False, False]))
    assert torch.equal(kv.cache, cache_before) and torch.equal(kv.end_offset, off_before)

    kv.complete(torch.randn(B, H, T, D), torch.randn(B, H, T, D), active=torch.tensor([True, False]))
    assert torch.equal(kv.cache[0][1], cache_before[0][1])
    assert kv.end_offset[0] == off_before[0] + T
    assert kv.end_offset[1] == off_before[1]


def test_mimi_transformer_rows_independency() -> None:
    torch.manual_seed(12345)
    stream_a1, stream_a2, stream_b = (torch.randn(30, 2, 32) for _ in range(3))
    x_ab = torch.stack([stream_a1, stream_b], dim=1)
    x_a2b = torch.stack([stream_a2, stream_b], dim=1)

    tf1, tf2 = make_mimi_tf(2), make_mimi_tf(2)
    out1 = [tf1.step(x_ab[t]) for t in range(30)]
    out2 = [tf2.step(x_a2b[t]) for t in range(30)]
    for o1, o2 in zip(out1, out2):
        assert not torch.equal(o1[0], o2[0])  # row 0: different input -> different output
        assert torch.equal(o1[1], o2[1])  # row 1: same input -> bit parity output
