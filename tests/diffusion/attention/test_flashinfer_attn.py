# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.attention.backends import flashinfer_attn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flashinfer_attn import (
    FlashInferAttentionBackend,
    FlashInferAttentionImpl,
)
from vllm_omni.diffusion.attention.capabilities import (
    CompilationMode,
    ExecutionContext,
    MaskMode,
    SupportStatus,
)
from vllm_omni.diffusion.data import AttentionSpec, AttnQuantSpec

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _impl(*, causal: bool = False, backend_explicit: bool = False):
    # Avoid CUDA/wrapper init; these tests only cover mask validation.
    obj = FlashInferAttentionImpl.__new__(FlashInferAttentionImpl)
    obj.causal = causal
    obj.softmax_scale = 0.5
    obj.flashinfer_backend = "fa2"
    obj.device_capability = None
    obj.dtype_qk = None
    obj.dtype_vo = None
    obj.backend_explicit = backend_explicit
    obj._sdpa_fallback = None
    return obj


def _resolve(
    impl,
    *,
    context: ExecutionContext | None = None,
    attn_metadata: AttentionMetadata | None = None,
    dtype=torch.bfloat16,
):
    tensor = torch.empty((1, 2, 2, 8), dtype=dtype)
    return impl.resolve_execution_path(
        context or ExecutionContext(platform="cuda"),
        tensor,
        tensor,
        tensor,
        attn_metadata,
    )


def test_flashinfer_preconstruction_capability_is_conservative():
    result = FlashInferAttentionBackend.resolve_capabilities(ExecutionContext(platform="cuda", require_fullgraph=True))

    assert result.path == "unverified"
    assert result.support.status is SupportStatus.UNMIGRATED
    assert result.compilation_mode is CompilationMode.EAGER_ONLY
    assert (
        result.requested_support(ExecutionContext(platform="cuda", require_fullgraph=True)).status
        is SupportStatus.UNSUPPORTED
    )


def test_flashinfer_dense_path_uses_initialized_backend_and_stays_unmigrated():
    impl = _impl()
    result = _resolve(
        impl,
        context=ExecutionContext(platform="cuda", kernel_variant="cute-dsl"),
    )

    assert result.backend == "FLASHINFER_ATTN"
    assert result.path == "flashinfer_fa2_dense"
    assert result.kernel_variant == "fa2"
    assert result.support.status is SupportStatus.UNMIGRATED
    assert result.compilation_mode is CompilationMode.EAGER_ONLY


def _candidate_impl():
    impl = _impl()
    impl.flashinfer_backend = "cute-dsl"
    impl.device_capability = (12, 0)
    return impl


def test_flashinfer_exact_sm120_cute_dsl_path_is_supported_custom_op():
    impl = _candidate_impl()
    query = torch.empty((1, 4, 2, 128), dtype=torch.bfloat16)
    context = ExecutionContext(platform="cuda", require_fullgraph=True)

    result = impl.resolve_execution_path(context, query, query, query, None)

    assert result.path == "flashinfer_cute_dsl_dense"
    assert result.support.status is SupportStatus.SUPPORTED
    assert result.compilation_mode is CompilationMode.CUSTOM_OP
    assert result.requested_support(context).status is SupportStatus.SUPPORTED


@pytest.mark.parametrize(
    ("device_capability", "dtype", "causal"),
    [
        ((10, 0), torch.bfloat16, False),
        ((12, 0), torch.float16, False),
        ((12, 0), torch.bfloat16, True),
    ],
)
def test_flashinfer_unverified_near_miss_stays_unmigrated(device_capability, dtype, causal):
    impl = _candidate_impl()
    impl.device_capability = device_capability
    impl.causal = causal
    query = torch.empty((1, 4, 2, 128), dtype=dtype)
    context = ExecutionContext(platform="cuda", require_fullgraph=True)

    result = impl.resolve_execution_path(context, query, query, query, None)

    assert result.support.status is SupportStatus.UNMIGRATED
    assert result.compilation_mode is CompilationMode.EAGER_ONLY
    assert result.requested_support(context).status is SupportStatus.UNSUPPORTED


def test_flashinfer_unpublished_mask_is_runtime_dependent():
    metadata = AttentionMetadata(attn_mask=torch.ones((2, 2), dtype=torch.bool))

    result = _resolve(_impl(), attn_metadata=metadata)

    assert result.path == "runtime_mask_dependent"
    assert result.support.status is SupportStatus.UNMIGRATED


def test_flashinfer_published_mask_mode_is_preserved_in_context():
    metadata = AttentionMetadata(
        attn_mask=torch.ones((2, 2), dtype=torch.bool),
        extra={"attention_mask_mode": "padding"},
    )

    result = _resolve(_impl(), attn_metadata=metadata)

    assert result.path == "flashinfer_fa2_masked"
    assert result.support.status is SupportStatus.UNMIGRATED


def test_flashinfer_mask_mode_rejects_reserved_unknown_value():
    metadata = AttentionMetadata(
        attn_mask=torch.ones((2, 2), dtype=torch.bool),
        extra={"attention_mask_mode": MaskMode.UNKNOWN.value},
    )

    with pytest.raises(ValueError, match="reserved"):
        _resolve(_impl(), attn_metadata=metadata)


def test_flashinfer_rejects_float_mask_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    query = torch.randn(1, 2, 2, 8)
    metadata = AttentionMetadata(attn_mask=torch.zeros(2, 2))

    with pytest.raises(ValueError, match="boolean-only"):
        _impl(backend_explicit=True).forward_cuda(query, query, query, metadata)


def test_flashinfer_rejects_causal_custom_mask_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    query = torch.randn(1, 2, 2, 8)
    metadata = AttentionMetadata(attn_mask=torch.tensor([[True, False], [True, True]]))

    with pytest.raises(ValueError, match="causal=True"):
        _impl(causal=True, backend_explicit=True).forward_cuda(query, query, query, metadata)


def test_explicit_cute_dsl_rejects_custom_mask_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    impl = _impl(backend_explicit=True)
    impl.flashinfer_backend = "cute-dsl"
    query = torch.randn(1, 2, 2, 8)
    metadata = AttentionMetadata(attn_mask=torch.tensor([[True, False], [True, True]]))

    with pytest.raises(ValueError, match="cute-dsl"):
        impl.forward_cuda(query, query, query, metadata)


def test_auto_cute_dsl_falls_back_to_sdpa_for_custom_mask(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    impl = _impl(backend_explicit=False)
    impl.flashinfer_backend = "cute-dsl"
    impl._run_batch_prefill = lambda *args, **kwargs: pytest.fail("unexpected cute-dsl prefill")
    query = torch.randn(1, 2, 2, 8)
    metadata = AttentionMetadata(attn_mask=torch.tensor([[True, False], [True, True]]))

    output = impl.forward_cuda(query, query, query, metadata)

    assert output.shape == query.shape


def test_explicit_cute_dsl_is_not_mask_capable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))
    spec = AttentionSpec(backend="FLASHINFER_ATTN")

    assert FlashInferAttentionBackend.supports_attention_mask() is True
    assert FlashInferAttentionBackend.supports_attention_mask(spec) is False


def test_explicit_fa2_flashinfer_is_mask_capable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))
    spec = AttentionSpec(backend="FLASHINFER_ATTN", quant=AttnQuantSpec(flashinfer_backend="fa2"))

    assert FlashInferAttentionBackend.supports_attention_mask(spec) is True


def test_explicit_fa2_backend_kwargs_select_fa2_on_blackwell(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))
    spec = AttentionSpec(backend="FLASHINFER_ATTN", quant=AttnQuantSpec(flashinfer_backend="fa2"))
    kwargs = spec.backend_kwargs() or {}
    requested = (kwargs.get("quant") or {}).get("flashinfer_backend", "auto")

    assert kwargs == {"quant": {"flashinfer_backend": "fa2"}}
    assert FlashInferAttentionImpl._select_backend(requested) == "fa2"
    assert FlashInferAttentionBackend.supports_attention_mask(spec) is True


def test_auto_flashinfer_backend_kwargs_select_cute_dsl_on_blackwell(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))
    spec = AttentionSpec(backend="FLASHINFER_ATTN")
    kwargs = spec.backend_kwargs() or {}
    requested = (kwargs.get("quant") or {}).get("flashinfer_backend", "auto")

    assert "quant" not in kwargs
    assert FlashInferAttentionImpl._select_backend(requested) == "cute-dsl"
    assert FlashInferAttentionBackend.supports_attention_mask(spec) is False


def test_hopper_explicit_flashinfer_is_mask_capable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (9, 0))
    spec = AttentionSpec(backend="FLASHINFER_ATTN")

    assert FlashInferAttentionBackend.supports_attention_mask(spec) is True
