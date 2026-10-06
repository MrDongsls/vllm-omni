# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import functools

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends import flashinfer_attn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _inputs(*, value_head_dim: int = 128):
    query = torch.randn(6, 2, 128, dtype=torch.bfloat16)
    key = torch.randn(10, 2, 128, dtype=torch.bfloat16)
    value = torch.randn(10, 2, value_head_dim, dtype=torch.bfloat16)
    workspace = torch.empty(0, dtype=torch.uint8)
    seq_lens = torch.tensor([5, 5], dtype=torch.int32)
    qo_indptr = torch.tensor([0, 3, 6], dtype=torch.int32)
    kv_indptr = torch.tensor([0, 5, 10], dtype=torch.int32)
    return (
        query,
        key,
        value,
        workspace,
        seq_lens,
        qo_indptr,
        kv_indptr,
        3,
        5,
        128**-0.5,
        2,
    )


@pytest.fixture
def fake_cute_dsl_kernel(monkeypatch):
    calls = []

    def kernel(**kwargs):
        calls.append(kwargs)
        query = kwargs["query"]
        value = kwargs["value"]
        return query.new_empty((*query.shape[:-1], value.shape[-1]))

    monkeypatch.setattr(
        flashinfer_attn,
        "trtllm_ragged_attention_deepseek",
        kernel,
    )
    return calls


def test_flashinfer_cute_dsl_custom_op_schema_and_fake(fake_cute_dsl_kernel):
    op = torch.ops.vllm_omni.flashinfer_cute_dsl_attention.default

    for value_head_dim in (128, 64):
        result = torch.library.opcheck(
            op,
            _inputs(value_head_dim=value_head_dim),
            test_utils=("test_schema", "test_faketensor"),
        )

        assert result == {
            "test_schema": "SUCCESS",
            "test_faketensor": "SUCCESS",
        }


def test_flashinfer_cute_dsl_custom_op_preserves_inputs(fake_cute_dsl_kernel):
    args = _inputs()
    tensor_snapshots = [tensor.clone() for tensor in args[:7]]

    first = flashinfer_attn._flashinfer_cute_dsl_attention_op(*args)
    second = flashinfer_attn._flashinfer_cute_dsl_attention_op(*args)

    assert len(fake_cute_dsl_kernel) == 2
    assert first.shape == (6, 2, 128)
    assert first.dtype == torch.bfloat16
    assert first.device == args[0].device
    assert first.is_contiguous()
    assert first.data_ptr() != args[0].data_ptr()
    assert second.data_ptr() != first.data_ptr()
    for tensor, snapshot in zip(args[:7], tensor_snapshots, strict=True):
        torch.testing.assert_close(tensor, snapshot)

    call = fake_cute_dsl_kernel[0]
    assert call["workspace_buffer"] is args[3]
    assert call["seq_lens"] is args[4]
    assert call["cum_seq_lens_q"] is args[5]
    assert call["cum_seq_lens_kv"] is args[6]
    assert call["backend"] == "cute-dsl"
    assert call["is_causal"] is False
    assert call["return_lse"] is False


def test_flashinfer_cute_dsl_custom_op_reports_missing_kernel(monkeypatch):
    monkeypatch.setattr(
        flashinfer_attn,
        "trtllm_ragged_attention_deepseek",
        None,
    )

    with pytest.raises(RuntimeError, match="FlashInfer cute-dsl kernel is unavailable"):
        flashinfer_attn._flashinfer_cute_dsl_attention_op(*_inputs())


class _UnexpectedWrapper:
    def plan(self, *args, **kwargs):
        pytest.fail("candidate path unexpectedly planned the stateful wrapper")

    def run(self, *args, **kwargs):
        pytest.fail("candidate path unexpectedly ran the stateful wrapper")


class _RecordingWrapper:
    def __init__(self):
        self.plan_calls = 0
        self.run_calls = 0

    def plan(self, *args, **kwargs):
        self.plan_calls += 1

    def run(self, query, key, value, **kwargs):
        self.run_calls += 1
        return query.clone()


def _candidate_impl():
    impl = flashinfer_attn.FlashInferAttentionImpl.__new__(flashinfer_attn.FlashInferAttentionImpl)
    impl.causal = False
    impl.softmax_scale = 128**-0.5
    impl.device = torch.device("cpu")
    impl.dtype_qk = None
    impl.dtype_vo = None
    impl.flashinfer_backend = "cute-dsl"
    impl.backend_explicit = True
    impl._workspace = torch.empty(0, dtype=torch.uint8)
    impl._wrapper = _UnexpectedWrapper()
    impl._qo_indptr = None
    impl._kv_indptr = None
    impl._plan_key = None
    impl._sdpa_fallback = None
    return impl


@pytest.fixture
def executable_cute_dsl_kernel(monkeypatch, tmp_path):
    marker = tmp_path / "flashinfer-kernel"
    marker.write_text("loaded", encoding="utf-8")
    calls = []

    @functools.cache
    def load_kernel():
        with open(marker, encoding="utf-8") as handle:
            return handle.read()

    def kernel(**kwargs):
        calls.append(kwargs)
        assert load_kernel() == "loaded"
        batch_size = kwargs["batch_size"]
        query = kwargs["query"].reshape(batch_size, kwargs["max_q_len"], 2, 128)
        key = kwargs["key"].reshape(batch_size, kwargs["max_kv_len"], 2, 128)
        value = kwargs["value"].reshape(batch_size, kwargs["max_kv_len"], 2, 128)
        return (
            F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                scale=kwargs["bmm1_scale"],
            )
            .transpose(1, 2)
            .reshape_as(query)
            .reshape(-1, 2, 128)
            .contiguous()
        )

    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    monkeypatch.setattr(flashinfer_attn, "_is_cuda_execution_path", lambda *_tensors: True)
    monkeypatch.setattr(
        flashinfer_attn,
        "trtllm_ragged_attention_deepseek",
        kernel,
    )
    return calls


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_dense_cute_dsl_candidate_is_opaque_to_dynamic_fullgraph(
    executable_cute_dsl_kernel,
    noncontiguous,
):
    impl = _candidate_impl()
    compile_count = 0

    def counting_backend(graph_module, _example_inputs):
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    compiled = torch.compile(
        impl.forward_cuda,
        backend=counting_backend,
        fullgraph=True,
        dynamic=True,
    )

    for batch_size, query_length, kv_length in ((1, 3, 5), (2, 7, 4)):
        tensors = [
            torch.randn(batch_size, 2, length, 128, dtype=torch.bfloat16).transpose(1, 2)
            for length in (query_length, kv_length, kv_length)
        ]
        if not noncontiguous:
            tensors = [tensor.contiguous() for tensor in tensors]
        else:
            assert all(not tensor.is_contiguous() for tensor in tensors)
        query, key, value = tensors
        snapshots = [tensor.clone() for tensor in tensors]

        expected = impl.forward_cuda(query, key, value)
        actual = compiled(query, key, value)

        torch.testing.assert_close(actual, expected)
        for tensor, snapshot in zip(tensors, snapshots, strict=True):
            torch.testing.assert_close(tensor, snapshot)

    # Dynamo specializes the first batch-size-1 input, so the batch-size-2
    # case may produce a second full graph even with dynamic=True.
    assert 1 <= compile_count <= 2
    assert len(executable_cute_dsl_kernel) == 4


def test_neighboring_unverified_path_keeps_stateful_wrapper(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    monkeypatch.setattr(
        flashinfer_attn,
        "trtllm_ragged_attention_deepseek",
        lambda **_kwargs: pytest.fail("non-cute-dsl path unexpectedly used the custom op"),
    )
    impl = _candidate_impl()
    impl.flashinfer_backend = "fa2"
    impl._wrapper = _RecordingWrapper()
    query = torch.randn(1, 4, 2, 128, dtype=torch.bfloat16)

    output = impl.forward_cuda(query, query, query)

    torch.testing.assert_close(output, query)
    assert impl._wrapper.plan_calls == 1
    assert impl._wrapper.run_calls == 1


@pytest.mark.parametrize(
    ("attribute", "value", "tensor_dtype", "head_dim", "metadata"),
    [
        ("flashinfer_backend", "fa2", torch.bfloat16, 128, None),
        ("causal", True, torch.bfloat16, 128, None),
        ("dtype_qk", torch.float8_e4m3fn, torch.bfloat16, 128, None),
        ("flashinfer_backend", "cute-dsl", torch.float16, 128, None),
        ("flashinfer_backend", "cute-dsl", torch.bfloat16, 64, None),
        (
            "flashinfer_backend",
            "cute-dsl",
            torch.bfloat16,
            128,
            AttentionMetadata(attn_mask=torch.ones(4, 4, dtype=torch.bool)),
        ),
        (
            "flashinfer_backend",
            "cute-dsl",
            torch.bfloat16,
            128,
            AttentionMetadata(
                extra={
                    "cu_seqlens_q": torch.tensor([0, 4], dtype=torch.int32),
                    "cu_seqlens_k": torch.tensor([0, 4], dtype=torch.int32),
                    "max_seqlen_q": 4,
                    "max_seqlen_k": 4,
                }
            ),
        ),
        (
            "flashinfer_backend",
            "cute-dsl",
            torch.bfloat16,
            128,
            AttentionMetadata(full_attn_spans=[[(0, 4)]]),
        ),
    ],
)
def test_only_exact_dense_cute_dsl_path_is_custom_op_candidate(
    attribute,
    value,
    tensor_dtype,
    head_dim,
    metadata,
):
    impl = _candidate_impl()
    setattr(impl, attribute, value)
    query = torch.empty(1, 4, 2, head_dim, dtype=tensor_dtype)

    assert not impl._is_cute_dsl_custom_op_candidate(query, query, query, metadata)
