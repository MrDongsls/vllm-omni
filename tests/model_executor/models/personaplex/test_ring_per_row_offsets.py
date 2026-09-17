# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.personaplex.personaplex_mimi import (
    PersonaPlexMimiCodec,
    _MimiStreamingTransformer,
    _normalize_active,
    _StreamConv1d,
    _StreamConvTr1d,
)
from vllm_omni.model_executor.models.personaplex.personaplex_temporal import (
    _RingKV,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SEED = 1234


def test_normalize_active_defaults_to_all_rows() -> None:
    all_active = torch.ones(2, dtype=torch.bool)

    assert _normalize_active(None, all_active) is all_active
    assert torch.equal(_normalize_active(torch.tensor([1, 0]), all_active), torch.tensor([True, False]))
    with pytest.raises(ValueError, match=r"active must have shape \(2,\)"):
        _normalize_active(torch.ones(1), all_active)


def make_mimi_tf(batch: int = 2, context: int = 16) -> _MimiStreamingTransformer:
    torch.manual_seed(SEED)
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
    active = torch.ones(B, dtype=torch.bool)
    singleton_active = torch.ones(1, dtype=torch.bool)
    for _ in range(steps):
        k, v = torch.randn(B, H, T, D), torch.randn(B, H, T, D)
        _, _, pos = kv.complete(k, v, active)
        for b in range(B):
            _, _, pos1 = rings[b].complete(k[b : b + 1], v[b : b + 1], singleton_active)
            assert torch.equal(pos[b], pos1[0])
            assert torch.equal(kv.cache[0][b], rings[b].cache[0][0])
            assert torch.equal(kv.end_offset[b], rings[b].end_offset[0])


def test_inactive_rows_do_not_advance_ring() -> None:
    B, H, D, T, capacity = 2, 2, 2, 4, 8
    kv = _RingKV(B, H, D, capacity, torch.device("cpu"), torch.float32)
    all_active = torch.ones(B, dtype=torch.bool)
    for _ in range(10):
        _, _, pos_before = kv.complete(torch.randn(B, H, T, D), torch.randn(B, H, T, D), all_active)
    off_before = kv.end_offset.clone()

    _, _, positions = kv.complete(torch.randn(B, H, T, D), torch.randn(B, H, T, D), active=torch.tensor([False, False]))
    assert torch.equal(positions, pos_before)
    assert torch.equal(kv.end_offset, off_before)

    kv.complete(torch.randn(B, H, T, D), torch.randn(B, H, T, D), active=torch.tensor([True, False]))
    assert kv.end_offset[0] == off_before[0] + T
    assert kv.end_offset[1] == off_before[1]


def test_mimi_transformer_rows_independency() -> None:
    torch.manual_seed(SEED + 1)
    stream_a1, stream_a2, stream_b = (torch.randn(30, 2, 32) for _ in range(3))
    x_ab = torch.stack([stream_a1, stream_b], dim=1)
    x_a2b = torch.stack([stream_a2, stream_b], dim=1)

    tf1, tf2 = make_mimi_tf(2), make_mimi_tf(2)
    active = torch.ones(2, dtype=torch.bool)
    out1 = [tf1.step(x_ab[t], active) for t in range(30)]
    out2 = [tf2.step(x_a2b[t], active) for t in range(30)]
    for o1, o2 in zip(out1, out2):
        assert not torch.equal(o1[0], o2[0])  # row 0: different input -> different output
        assert torch.equal(o1[1], o2[1])  # row 1: same input -> bit parity output


def test_mimi_transformer_active_rows_preserve_state() -> None:
    torch.manual_seed(SEED + 1)
    batched = make_mimi_tf(2)
    row0 = make_mimi_tf(1)
    row1 = make_mimi_tf(1)
    singleton_active = torch.ones(1, dtype=torch.bool)
    active_steps = [
        torch.tensor([True, True]),
        torch.tensor([True, False]),
        torch.tensor([False, False]),
        torch.tensor([True, True]),
    ]

    for active in active_steps:
        x = torch.randn(2, 2, 32)
        inactive_end_offsets = [kv.end_offset[~active].clone() for kv in batched._kv]
        inactive_offsets = batched._offset[~active].clone()
        output = batched.step(x, active=active)
        assert torch.equal(batched._offset[~active], inactive_offsets)
        for kv, end_offset in zip(batched._kv, inactive_end_offsets):
            assert torch.equal(kv.end_offset[~active], end_offset)
        if active[0]:
            expected0 = row0.step(x[0:1], singleton_active)
            # Batched and singleton execution can use different floating-point
            # reduction orders, so this is a numerical check. Logical state
            # preservation is asserted above; byte-identical codec output
            # remains an end-to-end gate.
            torch.testing.assert_close(output[0], expected0[0], rtol=1e-6, atol=1e-6)
        if active[1]:
            expected1 = row1.step(x[1:2], singleton_active)
            torch.testing.assert_close(output[1], expected1[0], rtol=1e-6, atol=1e-6)

    assert torch.equal(batched._offset, torch.tensor([6, 4]))


@pytest.mark.parametrize("kind", ["conv", "convtr"])
def test_mimi_conv_carry_inactive_rows_preserve_state(kind: str) -> None:
    torch.manual_seed(SEED + 2)
    if kind == "conv":
        conv = nn.Conv1d(1, 2, kernel_size=3, stride=2)
        streams = [_StreamConv1d(conv, pad_mode="replicate") for _ in range(3)]
        stage_kind, samples = "conv", 4
    else:
        conv = nn.ConvTranspose1d(1, 2, kernel_size=4, stride=2)
        streams = [_StreamConvTr1d(conv) for _ in range(3)]
        stage_kind, samples = "convtr", 2
    batched, row0, row1 = streams
    singleton_active = torch.ones(1, dtype=torch.bool)
    for stream, batch_size in ((batched, 2), (row0, 1), (row1, 1)):
        stream.reset(batch_size, torch.device("cpu"), torch.float32)

    for active in (
        torch.tensor([True, True]),
        torch.tensor([False, True]),
        torch.tensor([True, False]),
        torch.tensor([True, True]),
    ):
        x = torch.randn(2, 1, samples)
        carry = batched.prev if kind == "conv" else batched.partial
        inactive_carry = carry[~active].clone()
        inactive_fresh = batched._fresh[~active].clone()
        output = PersonaPlexMimiCodec._run_stages(x, [(stage_kind, batched)], active)

        carry = batched.prev if kind == "conv" else batched.partial
        assert torch.equal(carry[~active], inactive_carry)
        assert torch.equal(batched._fresh[~active], inactive_fresh)
        if active[0]:
            expected = PersonaPlexMimiCodec._run_stages(x[:1], [(stage_kind, row0)], singleton_active)
            # CPU kernels can reduce B=1 and B=2 in a different order. The
            # exact no-write assertions above are the state invariant; this
            # compares the active row's numerical streaming result.
            torch.testing.assert_close(output[:1], expected, rtol=1e-6, atol=1e-6)
        if active[1]:
            expected = PersonaPlexMimiCodec._run_stages(x[1:], [(stage_kind, row1)], singleton_active)
            torch.testing.assert_close(output[1:], expected, rtol=1e-6, atol=1e-6)
