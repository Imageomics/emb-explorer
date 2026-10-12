"""
Backend detection and resolution utilities.

Provides consistent backend selection and CUDA availability checking
across all applications.

Availability checks use importlib.find_spec() for instant package detection
without importing heavy libraries. Actual imports happen lazily when the
backend is first used.
"""

import importlib.util
import threading
from typing import Tuple, Optional
from shared.utils.logging_config import get_logger

logger = get_logger(__name__)

# --- Lightweight availability checks (find_spec, no actual import) ----------

# These are safe to call at module-load / render time — they only check
# whether the package is installed, without executing it.

HAS_CUML_PACKAGE: bool = importlib.util.find_spec("cuml") is not None
HAS_CUPY_PACKAGE: bool = importlib.util.find_spec("cupy") is not None
HAS_TORCH_PACKAGE: bool = importlib.util.find_spec("torch") is not None

# --- Cached runtime checks (perform actual import, cached after first call) -

# Cache CUDA availability to avoid repeated checks
_cuda_check_cache: Optional[Tuple[bool, str]] = None

# cuML availability: the real import is slow (tens of seconds cold) and not
# safe to run from two threads at once, so it happens under a lock and a
# success is cached. Only True is cached: a transient failure must not stick.
_cuml_check_lock = threading.Lock()
_cuml_available_cache: Optional[bool] = None


def check_cuda_available() -> Tuple[bool, str]:
    """
    Check if CUDA is available for GPU-accelerated backends.

    Returns:
        Tuple of (is_available, device_info_string)
    """
    global _cuda_check_cache

    if _cuda_check_cache is not None:
        return _cuda_check_cache

    # Try PyTorch first
    if HAS_TORCH_PACKAGE:
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                _cuda_check_cache = (True, device_name)
                logger.info(f"CUDA available via PyTorch: {device_name}")
                return _cuda_check_cache
        except ImportError:
            pass  # PyTorch not installed, try CuPy next

    # Try CuPy
    if HAS_CUPY_PACKAGE:
        try:
            import cupy as cp
            if cp.cuda.is_available():
                device = cp.cuda.Device(0)
                device_info = f"GPU {device.id}"
                _cuda_check_cache = (True, device_info)
                logger.info(f"CUDA available via CuPy: {device_info}")
                return _cuda_check_cache
        except ImportError:
            pass  # CuPy not installed, fall through to CPU-only

    _cuda_check_cache = (False, "CPU only")
    logger.info("CUDA not available, using CPU")
    return _cuda_check_cache


def _import_cuml() -> None:
    """Perform the real cuML import (factored out so tests can stub it)."""
    import cuml  # noqa: F401


def check_cuml_available() -> bool:
    """Check if cuML is available (actual import, for runtime use).

    Serialized with a lock: two Streamlit script runs resolving the backend
    at the same moment would otherwise import cuML concurrently, and the
    second importer can see a partially initialized package and get an
    ImportError, which used to be reported as "cuML not available" (#49).
    A successful import is cached so later calls return instantly.
    """
    global _cuml_available_cache

    if not HAS_CUML_PACKAGE:
        return False
    if _cuml_available_cache:
        return True

    with _cuml_check_lock:
        if _cuml_available_cache:  # another thread finished while we waited
            return True
        try:
            _import_cuml()
        except ImportError as e:
            logger.warning(f"cuML is installed but failed to import ({e}); using CPU backends")
            return False
        _cuml_available_cache = True
        return True


def resolve_backend(backend: str, operation: str = "general") -> str:
    """
    Resolve 'auto' backend to actual backend based on available hardware.

    Args:
        backend: Requested backend ("auto", "sklearn", "cuml")
        operation: Operation type for logging ("clustering", "reduction", "general")

    Returns:
        Resolved backend name. CPU paths always go through sklearn.
    """
    if backend != "auto":
        logger.debug(f"Using explicitly requested backend: {backend}")
        return backend

    cuda_available, device_info = check_cuda_available()
    # Only probe for cuML when CUDA is actually available.
    if cuda_available and check_cuml_available():
        resolved = "cuml"
        logger.info(f"Auto-resolved {operation} backend to cuML (GPU: {device_info})")
    else:
        resolved = "sklearn"
        logger.info(f"Auto-resolved {operation} backend to sklearn (CPU)")

    return resolved


def get_backend_info() -> dict:
    """
    Get comprehensive backend availability information.

    Returns:
        Dictionary with backend availability status
    """
    cuda_available, device_info = check_cuda_available()

    return {
        "cuda_available": cuda_available,
        "device_info": device_info,
        "cuml_available": check_cuml_available(),
    }


def is_gpu_error(error: Exception) -> bool:
    """
    Check if an exception is a GPU-related error.

    Args:
        error: Exception to check

    Returns:
        True if error is GPU-related
    """
    error_msg = str(error).lower()
    gpu_indicators = [
        "out of memory",
        "oom",
        "cuda",
        "gpu",
        "nvrtc",
        "libnvrtc",
        "no kernel image",
        "cudaerror",
    ]
    return any(indicator in error_msg for indicator in gpu_indicators)


def is_oom_error(error: Exception) -> bool:
    """Check if an exception is an out-of-memory error."""
    error_msg = str(error).lower()
    oom_indicators = [
        "out of memory",
        "cudaerroroutofmemory",
        "oom",
        "memory allocation failed",
        "cudamalloc failed",
        "failed to allocate",
    ]
    return any(indicator in error_msg for indicator in oom_indicators)


def is_cuda_arch_error(error: Exception) -> bool:
    """Check if an exception is a CUDA architecture incompatibility error."""
    error_msg = str(error).lower()
    arch_indicators = [
        "no kernel image",
        "cudaerrornokernel",
        "unsupported gpu",
        "compute capability",
    ]
    return any(indicator in error_msg for indicator in arch_indicators)
