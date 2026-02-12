"""Tests for shared/utils/backend.py."""

from unittest.mock import patch

import pytest

from shared.utils.backend import (
    is_gpu_error,
    is_oom_error,
    is_cuda_arch_error,
    resolve_backend,
    check_cuda_available,
)


# ---------------------------------------------------------------------------
# Error classifiers (pure — no mocking needed)
# ---------------------------------------------------------------------------

class TestIsGpuError:
    @pytest.mark.parametrize("msg", [
        "CUDA error: out of memory",
        "RuntimeError: no kernel image is available",
        "nvrtc compilation failed",
        "libnvrtc.so not found",
        "GPU memory allocation failed",
        "cudaErrorNoKernel",
    ])
    def test_gpu_errors_detected(self, msg):
        assert is_gpu_error(RuntimeError(msg))

    @pytest.mark.parametrize("msg", [
        "FileNotFoundError: /tmp/data.npy",
        "ValueError: invalid literal",
        "Connection refused",
    ])
    def test_non_gpu_errors_rejected(self, msg):
        assert not is_gpu_error(RuntimeError(msg))

    def test_case_insensitive(self):
        assert is_gpu_error(RuntimeError("CUDA ERROR: device not found"))


class TestIsOomError:
    @pytest.mark.parametrize("msg", [
        "CUDA out of memory",
        "cudaErrorOutOfMemory",
        "OOM killer invoked",
        "memory allocation failed",
        "cudaMalloc failed",
        "failed to allocate 1024 bytes",
    ])
    def test_oom_errors_detected(self, msg):
        assert is_oom_error(RuntimeError(msg))

    def test_non_oom_rejected(self):
        assert not is_oom_error(RuntimeError("invalid argument"))


class TestIsCudaArchError:
    @pytest.mark.parametrize("msg", [
        "no kernel image is available for execution on the device",
        "cudaErrorNoKernel",
        "unsupported GPU architecture",
        "compute capability 3.5 not supported",
    ])
    def test_arch_errors_detected(self, msg):
        assert is_cuda_arch_error(RuntimeError(msg))

    def test_non_arch_rejected(self):
        assert not is_cuda_arch_error(RuntimeError("out of memory"))


# ---------------------------------------------------------------------------
# resolve_backend (mock check_* functions)
# ---------------------------------------------------------------------------

class TestResolveBackend:
    def test_explicit_backend_passthrough(self):
        assert resolve_backend("sklearn") == "sklearn"
        assert resolve_backend("cuml") == "cuml"
        assert resolve_backend("faiss") == "faiss"

    def test_auto_with_cuda_and_cuml(self):
        with patch("shared.utils.backend.check_cuda_available", return_value=(True, "V100")), \
             patch("shared.utils.backend.check_cuml_available", return_value=True), \
             patch("shared.utils.backend.check_faiss_available", return_value=True):
            assert resolve_backend("auto") == "cuml"

    def test_auto_without_cuda_with_faiss(self):
        with patch("shared.utils.backend.check_cuda_available", return_value=(False, "CPU only")), \
             patch("shared.utils.backend.check_cuml_available", return_value=False), \
             patch("shared.utils.backend.check_faiss_available", return_value=True):
            assert resolve_backend("auto") == "faiss"

    def test_auto_cpu_only(self):
        with patch("shared.utils.backend.check_cuda_available", return_value=(False, "CPU only")), \
             patch("shared.utils.backend.check_cuml_available", return_value=False), \
             patch("shared.utils.backend.check_faiss_available", return_value=False):
            assert resolve_backend("auto") == "sklearn"

    def test_auto_cuda_without_cuml_falls_to_faiss(self):
        with patch("shared.utils.backend.check_cuda_available", return_value=(True, "V100")), \
             patch("shared.utils.backend.check_cuml_available", return_value=False), \
             patch("shared.utils.backend.check_faiss_available", return_value=True):
            assert resolve_backend("auto") == "faiss"


# ---------------------------------------------------------------------------
# check_cuda_available (mock imports, test caching)
# ---------------------------------------------------------------------------

class TestCheckCudaAvailable:
    def test_returns_false_without_gpu(self, reset_cuda_cache):
        """On a CPU-only node, should return (False, 'CPU only')."""
        with patch.dict("sys.modules", {"torch": None, "cupy": None}):
            # Force fresh check by bypassing the cached imports
            with patch("shared.utils.backend.check_cuda_available") as mock_check:
                mock_check.return_value = (False, "CPU only")
                result = mock_check()
                assert result == (False, "CPU only")

    def test_cache_prevents_reimport(self, reset_cuda_cache):
        """Second call should return cached value."""
        import shared.utils.backend as backend_mod
        backend_mod._cuda_check_cache = (True, "V100-test")
        result = check_cuda_available()
        assert result == (True, "V100-test")
