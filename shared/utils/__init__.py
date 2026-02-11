"""
Shared utilities for clustering, IO, models, and taxonomy.
"""

from shared.utils.clustering import (
    run_kmeans,
    reduce_dim,
    VRAMExceededError,
    GPUArchitectureError,
    get_gpu_memory_info,
    estimate_memory_requirement,
)
from shared.utils.io import list_image_files, copy_image
from shared.utils.models import list_available_models
from shared.utils.taxonomy_tree import (
    build_taxonomic_tree,
    format_tree_string,
    get_total_count,
    get_tree_statistics,
)

__all__ = [
    "run_kmeans",
    "reduce_dim",
    "VRAMExceededError",
    "GPUArchitectureError",
    "get_gpu_memory_info",
    "estimate_memory_requirement",
    "list_image_files",
    "copy_image",
    "list_available_models",
    "build_taxonomic_tree",
    "format_tree_string",
    "get_total_count",
    "get_tree_statistics",
]
