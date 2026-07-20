import pytest

from vllm_omni.utils.dynamic_import import (
    load_callable,
)

# A real callable in the standard library — no vllm_omni import needed.
_VALID_PATH = "math.sqrt"
_CTX = "test context"


class TestLoadCallable:
    def test_valid_path_returns_callable(self):
        import math

        func = load_callable(_VALID_PATH, context=_CTX)
        assert func is math.sqrt

    def test_no_dot_raises_value_error(self):
        with pytest.raises(ValueError, match="nodot"):
            load_callable("nodot", context=_CTX)

    def test_empty_path_raises_value_error(self):
        with pytest.raises(ValueError):
            load_callable("", context=_CTX)

    def test_missing_module_raises_import_error(self):
        with pytest.raises(ImportError, match="nonexistent_module_xyz_abc"):
            load_callable("nonexistent_module_xyz_abc.some_func", context=_CTX)

    def test_missing_attr_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="nonexistent_func_xyz_abc"):
            load_callable("math.nonexistent_func_xyz_abc", context=_CTX)

    def test_not_callable_raises_type_error(self):
        with pytest.raises(TypeError, match="float"):
            load_callable("math.pi", context=_CTX)

    @pytest.mark.parametrize(
        "bad_path,exc_type",
        [
            ("nodot", ValueError),
            ("nonexistent_module_xyz_abc.func", ImportError),
            ("math.nonexistent_func_xyz_abc", AttributeError),
            ("math.pi", TypeError),
        ],
    )
    def test_context_appears_in_every_error_message(self, bad_path, exc_type):
        ctx = "stage 1 custom_process_input_func"
        with pytest.raises(exc_type, match="stage 1"):
            load_callable(bad_path, context=ctx)
