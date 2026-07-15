import functools
import importlib
import sys
import types

import pytest
import torch
from vllm.platforms import current_platform

SELECTOR_MODULE = "vllm_omni.diffusion.attention.selector"
FLASHINFER_MODULE = "vllm_omni.diffusion.attention.backends.flashinfer_attn"
FLASHINFER_CLS_PATH = f"{FLASHINFER_MODULE}.FlashInferAttentionBackend"


@pytest.fixture(autouse=True)
def _clear_selector_cache():
    """Reset the global functools.cache on _cached_get_backend_cls so tests
    don't leak resolved backends across (backend_name, head_size) keys."""
    selector = importlib.import_module(SELECTOR_MODULE)
    selector._cached_get_backend_cls.cache_clear()
    yield
    selector._cached_get_backend_cls.cache_clear()


def _install_fake_flashinfer(monkeypatch):
    """Install a fake 'flashinfer.prefill' module in sys.modules.

    The fake single_prefill_with_kv_cache wraps a functools.cache-decorated
    helper that touches disk via builtin open(), mirroring the structure of
    the real get_single_prefill_module. Dynamo always traces the raw body of
    a functools.cache_wrapped function regardless of cache hits, so this is
    the minimal fixture that reproduces the issue #4988 failure mode: any call
    path that bypasses the torch.library.custom_op boundary will crash under
    fullgraph=True.
    """

    @functools.cache
    def _get_single_prefill_module(*_args, **_kwargs):
        import tempfile

        with tempfile.NamedTemporaryFile() as f:
            with open(f.name, "rb") as fh:  # Dynamo cannot trace builtin open
                fh.read()
        return object()

    def _fake_single_prefill_with_kv_cache(query, key, value, **kwargs):
        _get_single_prefill_module(query.shape, key.shape)
        return query.clone()

    prefill_mod = types.ModuleType("flashinfer.prefill")
    prefill_mod.single_prefill_with_kv_cache = _fake_single_prefill_with_kv_cache
    prefill_mod.get_single_prefill_module = _get_single_prefill_module

    flashinfer_mod = types.ModuleType("flashinfer")
    flashinfer_mod.prefill = prefill_mod

    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer_mod)
    monkeypatch.setitem(sys.modules, "flashinfer.prefill", prefill_mod)

    return _fake_single_prefill_with_kv_cache


@pytest.fixture
def flashinfer_backend_cls(monkeypatch):
    """Force-select FlashInfer backend by name, skipping availability checks."""
    _install_fake_flashinfer(monkeypatch)

    mod = importlib.import_module(FLASHINFER_MODULE)
    importlib.reload(mod)
    assert mod.HAS_FLASHINFER, "fake flashinfer was not picked up"

    selector = importlib.import_module(SELECTOR_MODULE)
    backend_cls = selector._cached_get_backend_cls("FLASHINFER_ATTN", 64)
    yield backend_cls

    # teardown
    mod.HAS_FLASHINFER = False
    if hasattr(mod, "single_prefill_with_kv_cache"):
        del mod.single_prefill_with_kv_cache
    if hasattr(mod, "_flashinfer_prefill_op"):
        mod._flashinfer_prefill_op = None


@pytest.mark.core_model
@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_flashinfer_backend_compiles_with_fullgraph(flashinfer_backend_cls):
    """Regression test for #4988: the FlashInfer backend, resolved through
    the selector, must compile under fullgraph=True. The real JIT loader is
    wrapped in torch.library.custom_op, so Dynamo treats it as opaque and
    never inlines into the builtin open() call."""
    impl_cls = flashinfer_backend_cls.get_impl_cls()
    impl = impl_cls(num_heads=8, head_size=64, softmax_scale=1.0, causal=False)

    batch, seq, heads, head_dim = 1, 16, 8, 64
    query = torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    compiled = torch.compile(lambda q, k, v: impl.forward_cuda(q, k, v, attn_metadata=None), fullgraph=True)
    out = compiled(query, key, value)

    assert out.shape == query.shape
