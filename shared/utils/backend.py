"""
Backend detection and resolution utilities.

Provides consistent backend selection and CUDA availability checking
across all applications.

Availability checks use importlib.find_spec() for instant package detection
without importing heavy libraries. Actual imports happen lazily when the
backend is first used.
"""

import importlib.util
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

# Dataset-size range (samples) in which "auto" picks cuML over sklearn, per
# method. Chosen from measured crossovers of the app's own reduce_dim /
# run_kmeans paths (scripts/bench_backend_threshold.py); the exact numbers
# depend on the GPU, CPU count and library versions, so treat them as
# defaults to re-measure, not constants. The shape of the rule is stable:
# - PCA, KMEANS: cheap either way; cuML wins once transfer cost is amortized.
# - UMAP:   cuML runs in a subprocess, so a fixed start-up cost must be
#           amortized over a large enough dataset.
# - TSNE:   the app forces cuML's exact O(N^2) solver (#40), so cuML only
#           wins on small data and loses increasingly above the bound.
# Entries are (min_samples, max_samples); None is unbounded. An explicit
# "cuml" request is always honored. reduce_dim / run_kmeans use this same
# rule so the layers agree (#50).
CUML_AUTO_RANGE = {
    "PCA": (500, None),
    "TSNE": (None, 3000),
    "UMAP": (10000, None),
    "KMEANS": (1000, None),
}


def auto_prefers_cuml(method: str, n_samples: Optional[int]) -> bool:
    """Whether "auto" should pick cuML for ``method`` at this dataset size,
    hardware aside. Unknown size or method keeps the hardware-only rule."""
    if n_samples is None:
        return True
    lo, hi = CUML_AUTO_RANGE.get(method.upper(), (None, None))
    if lo is not None and n_samples < lo:
        return False
    if hi is not None and n_samples >= hi:
        return False
    return True


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


def check_cuml_available() -> bool:
    """Check if cuML is available (actual import, for runtime use)."""
    if not HAS_CUML_PACKAGE:
        return False
    try:
        import cuml
        return True
    except ImportError:
        return False


def resolve_backend(
    backend: str,
    operation: str = "general",
    n_samples: Optional[int] = None,
    method: Optional[str] = None,
) -> str:
    """
    Resolve 'auto' backend to actual backend based on hardware, method and data size.

    Args:
        backend: Requested backend ("auto", "sklearn", "cuml")
        operation: Operation type ("clustering", "reduction", "general")
        n_samples: Dataset size. With "auto", sizes outside the method's
            CUML_AUTO_RANGE resolve to sklearn even when a GPU is available.
            None keeps the hardware-only rule.
        method: "PCA" / "TSNE" / "UMAP" for reduction; defaults to "KMEANS"
            for clustering. Needed for the size rule.

    Returns:
        Resolved backend name. CPU paths always go through sklearn; an
        explicit "cuml" or "sklearn" request is returned unchanged.
    """
    if backend != "auto":
        logger.debug(f"Using explicitly requested backend: {backend}")
        return backend

    cuda_available, device_info = check_cuda_available()
    # Only probe for cuML when CUDA is actually available.
    if not (cuda_available and check_cuml_available()):
        logger.info(f"Auto-resolved {operation} backend to sklearn (CPU)")
        return "sklearn"

    method_key = (method or ("KMEANS" if operation == "clustering" else "")).upper()
    if n_samples is not None and method_key and not auto_prefers_cuml(method_key, n_samples):
        lo, hi = CUML_AUTO_RANGE[method_key]
        where = f"below {lo}" if lo is not None and n_samples < lo else f"at or above {hi}"
        logger.info(f"Auto-resolved {operation} backend to sklearn for {method_key}: {n_samples} "
                    f"samples is {where}, where sklearn is faster (GPU {device_info} available; "
                    f"choose 'cuml' explicitly to force it)")
        return "sklearn"

    logger.info(f"Auto-resolved {operation} backend to cuML (GPU: {device_info})")
    return "cuml"


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
