# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

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
    PersonaPlexTemporalStreaming,
    _RingKV,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SEED = 1234


def _mask(*rows: bool) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.bool)


def _make_mimi_transformer(batch_size: int, context: int = 6) -> _MimiStreamingTransformer:
    torch.manual_seed(SEED)
    transformer = _MimiStreamingTransformer(num_layers=2, dim=32, num_heads=4, context=context)
    for parameter in transformer.parameters():
        if parameter.dtype.is_floating_point:
            nn.init.normal_(parameter, std=0.1)
    transformer.streaming_init(batch_size)
    return transformer


def _make_temporal(batch_size: int, context: int = 6) -> PersonaPlexTemporalStreaming:
    torch.manual_seed(SEED)
    temporal = PersonaPlexTemporalStreaming(
        dim=16,
        num_layers=2,
        num_heads=4,
        hidden=32,
        context=context,
        text_card=11,
    )
    for parameter in temporal.parameters():
        if parameter.dtype.is_floating_point:
            nn.init.normal_(parameter, std=0.1)
    temporal.streaming_init(batch_size)
    return temporal


def _assert_valid_ring_row_matches(
    batched: _RingKV,
    reference: _RingKV,
    batched_positions: torch.Tensor,
    reference_positions: torch.Tensor,
    row: int,
) -> None:
    assert torch.equal(batched_positions[row], reference_positions)
    assert torch.equal(batched.end_offset[row], reference.end_offset[0])

    # The active mask covers the physical write as well as offset advancement,
    # so the complete per-row ring state remains singleton-identical.
    assert torch.equal(batched.cache[:, row], reference.cache[:, 0])


def _stream_carry(stream: _StreamConv1d | _StreamConvTr1d) -> torch.Tensor:
    if isinstance(stream, _StreamConv1d):
        assert stream.prev is not None
        return stream.prev
    assert stream.partial is not None
    return stream.partial


class _IdentityCodecStage:
    def __init__(self) -> None:
        self.active: list[torch.Tensor] = []

    def reset(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        del batch_size, device, dtype

    def reset_slot(self, row: int) -> None:
        del row

    def __call__(self, x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        self.active.append(active.clone())
        return x


class _IdentityCodecTransformer:
    def __init__(self) -> None:
        self.active: list[torch.Tensor] = []

    def streaming_init(self, batch_size: int) -> None:
        del batch_size

    def reset_streaming(self) -> None:
        return None

    def reset_slot(self, row: int) -> None:
        del row

    def step(self, x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        self.active.append(active.clone())
        return x


class _ZeroCodecQuantizer:
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(8, x.shape[0], x.shape[-1], dtype=torch.long)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return torch.zeros(codes.shape[0], 1, codes.shape[-1])


def _make_codec_stub() -> PersonaPlexMimiCodec:
    codec = PersonaPlexMimiCodec.__new__(PersonaPlexMimiCodec)
    nn.Module.__init__(codec)
    codec.device = torch.device("cpu")
    codec.dtype = torch.float32
    codec.model = SimpleNamespace(quantizer=_ZeroCodecQuantizer())
    codec._enc_stages = []
    codec._dec_stages = []
    codec._downsample = _IdentityCodecStage()
    codec._upsample = _IdentityCodecStage()
    codec.encoder_transformer = _IdentityCodecTransformer()
    codec.decoder_transformer = _IdentityCodecTransformer()
    codec.streaming_init(batch_size=2)
    return codec


def test_normalize_active_defaults_to_all_rows() -> None:
    all_active = torch.ones(2, dtype=torch.bool)

    assert _normalize_active(None, all_active) is all_active
    normalized = _normalize_active(torch.tensor([1, 0]), all_active)
    assert torch.equal(normalized, _mask(True, False))
    assert normalized.dtype == torch.bool

    with pytest.raises(ValueError, match=r"active must have shape \(2,\)"):
        _normalize_active(torch.ones(1), all_active)


def test_codec_entrypoints_forward_active_mask() -> None:
    codec = _make_codec_stub()
    active = _mask(True, False)
    all_active = torch.ones_like(active)

    codec.encode_frame(torch.zeros(2, 1920), active)
    codec.decode_frame(torch.zeros(2, 8), active)
    codec.decode_frames(torch.zeros(2, 8, 3), active)
    codec.decode_frames(torch.zeros(2, 8, 3), active=None)

    assert torch.equal(torch.stack(codec._downsample.active), active[None])
    assert torch.equal(torch.stack(codec._upsample.active), torch.stack((active, active, all_active)))
    assert torch.equal(torch.stack(codec.encoder_transformer.active), active[None])
    assert torch.equal(torch.stack(codec.decoder_transformer.active), torch.stack((active, active, all_active)))


def test_ring_mixed_offsets_match_singleton_streams() -> None:
    batch_size, heads, head_dim, capacity, tokens = 3, 2, 3, 6, 2
    batched = _RingKV(batch_size, heads, head_dim, capacity, torch.device("cpu"), torch.float32)
    singletons = [_RingKV(1, heads, head_dim, capacity, torch.device("cpu"), torch.float32) for _ in range(batch_size)]
    reference_positions: list[torch.Tensor | None] = [None] * batch_size
    active_schedule = [
        _mask(True, True, True),
        _mask(True, False, False),
        _mask(False, False, True),
        _mask(True, True, False),
        _mask(False, True, True),
        _mask(True, False, True),
        _mask(True, True, True),
    ] * 2

    torch.manual_seed(SEED)
    for active in active_schedule:
        keys = torch.randn(batch_size, heads, tokens, head_dim)
        values = torch.randn_like(keys)
        _, _, positions = batched.complete(keys, values, active)

        for row, is_active in enumerate(active.tolist()):
            if is_active:
                _, _, singleton_positions = singletons[row].complete(
                    keys[row : row + 1],
                    values[row : row + 1],
                    _mask(True),
                )
                reference_positions[row] = singleton_positions[0].clone()

            assert reference_positions[row] is not None
            _assert_valid_ring_row_matches(
                batched,
                singletons[row],
                positions,
                reference_positions[row],
                row,
            )

    assert any(int(offset) > capacity for offset in batched.end_offset)
    assert not torch.equal(batched.end_offset[0], batched.end_offset[1])


def test_ring_slot_recycle_masks_old_history() -> None:
    ring = _RingKV(2, 1, 1, 8, torch.device("cpu"), torch.float32)
    for _ in range(4):
        ring.complete(torch.randn(2, 1, 2, 1), torch.randn(2, 1, 2, 1), _mask(True, True))

    old_end = ring.end_offset.clone()
    ring.reset_slot(1)
    _, _, positions = ring.complete(
        torch.randn(2, 1, 2, 1),
        torch.randn(2, 1, 2, 1),
        _mask(False, True),
    )
    visible = positions[1] >= 0
    assert torch.all(positions[1][visible] >= old_end[1])

    ring.reset_slot(0)
    ring.bump_slot_start(0)
    _, _, positions = ring.complete(
        torch.randn(2, 1, 2, 1),
        torch.randn(2, 1, 2, 1),
        _mask(True, False),
    )
    visible = positions[0] >= 0
    assert torch.all(positions[0][visible] >= old_end[0] + 1)


def test_mimi_transformer_mixed_offsets_match_singletons() -> None:
    batch_size, tokens, dim = 3, 2, 32
    batched = _make_mimi_transformer(batch_size)
    singletons = [_make_mimi_transformer(1) for _ in range(batch_size)]
    active_schedule = [
        _mask(True, True, True),
        _mask(True, False, False),
        _mask(False, False, True),
        _mask(True, True, False),
        _mask(False, True, True),
        _mask(True, False, True),
        _mask(True, True, True),
    ] * 2

    torch.manual_seed(SEED + 1)
    for active in active_schedule:
        inputs = torch.randn(batch_size, tokens, dim)
        offsets_before = batched._offset.clone()
        output = batched.step(inputs, active)

        for row, is_active in enumerate(active.tolist()):
            if is_active:
                expected = singletons[row].step(inputs[row : row + 1], _mask(True))
                torch.testing.assert_close(
                    output[row : row + 1],
                    expected,
                    rtol=1e-5,
                    atol=1e-6,
                )
                assert torch.equal(batched._offset[row], singletons[row]._offset[0])
                for batched_kv, singleton_kv in zip(batched._kv, singletons[row]._kv):
                    assert torch.equal(batched_kv.end_offset[row], singleton_kv.end_offset[0])
            else:
                assert torch.equal(batched._offset[row], offsets_before[row])
                for kv in batched._kv:
                    assert torch.equal(kv.end_offset[row], offsets_before[row])


@pytest.mark.parametrize("kind", ["conv", "convtr"])
def test_mimi_conv_carries_preserve_inactive_rows(kind: str) -> None:
    if kind == "conv":
        conv = nn.Conv1d(1, 2, kernel_size=3, stride=2)
        batched = _StreamConv1d(conv, pad_mode="replicate")
        row0 = _StreamConv1d(conv, pad_mode="replicate")
        row1 = _StreamConv1d(conv, pad_mode="replicate")
        samples = 4
    else:
        conv = nn.ConvTranspose1d(1, 2, kernel_size=4, stride=2)
        batched = _StreamConvTr1d(conv)
        row0 = _StreamConvTr1d(conv)
        row1 = _StreamConvTr1d(conv)
        samples = 2

    for stream, batch_size in ((batched, 2), (row0, 1), (row1, 1)):
        stream.reset(batch_size, torch.device("cpu"), torch.float32)

    singletons = [row0, row1]
    active_schedule = [
        _mask(False, False),
        _mask(True, False),
        _mask(True, True),
        _mask(False, True),
        _mask(True, True),
    ]

    torch.manual_seed(SEED + 2)
    for active in active_schedule:
        inputs = torch.randn(2, 1, samples)
        carry_before = _stream_carry(batched).clone()
        fresh_before = batched._fresh.clone()
        output = PersonaPlexMimiCodec._run_stages(inputs, [(kind, batched)], active)

        for row, is_active in enumerate(active.tolist()):
            if is_active:
                expected = PersonaPlexMimiCodec._run_stages(
                    inputs[row : row + 1],
                    [(kind, singletons[row])],
                    _mask(True),
                )
                torch.testing.assert_close(output[row : row + 1], expected, rtol=1e-5, atol=1e-6)
                assert torch.equal(_stream_carry(batched)[row], _stream_carry(singletons[row])[0])
                assert torch.equal(batched._fresh[row], singletons[row]._fresh[0])
            else:
                assert torch.equal(_stream_carry(batched)[row], carry_before[row])
                assert torch.equal(batched._fresh[row], fresh_before[row])


def test_temporal_streaming_mixed_offsets_match_singletons() -> None:
    batched = _make_temporal(2)
    row0, row1 = _make_temporal(1), _make_temporal(1)
    singletons = [row0, row1]
    with pytest.raises(ValueError, match=r"active must have shape \(2,\)"):
        batched.step(torch.zeros(2, 1, 16), torch.ones(1))

    active_schedule = [
        _mask(True, True),
        _mask(True, False),
        _mask(False, False),
        _mask(False, True),
        _mask(True, True),
        _mask(True, False),
        _mask(False, True),
        _mask(True, True),
    ]

    torch.manual_seed(SEED + 3)
    for active in active_schedule:
        inputs = torch.randn(2, 1, 16)
        offsets_before = batched._offset.clone()
        output, logits = batched.step(inputs, active)

        for row, is_active in enumerate(active.tolist()):
            if is_active:
                expected_output, expected_logits = singletons[row].step(inputs[row : row + 1], _mask(True))
                torch.testing.assert_close(output[row : row + 1], expected_output, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(logits[row : row + 1], expected_logits, rtol=1e-5, atol=1e-6)
                assert torch.equal(batched._offset[row], singletons[row]._offset[0])
                for batched_kv, singleton_kv in zip(batched._kv, singletons[row]._kv):
                    assert torch.equal(batched_kv.end_offset[row], singleton_kv.end_offset[0])
            else:
                assert torch.equal(batched._offset[row], offsets_before[row])
                for kv in batched._kv:
                    assert torch.equal(kv.end_offset[row], offsets_before[row])
