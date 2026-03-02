"""Shared fixtures for emb-explorer test suite."""

import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest


@pytest.fixture
def sample_embeddings():
    """Reproducible (100, 512) float32 embedding matrix."""
    rng = np.random.RandomState(42)
    return rng.randn(100, 512).astype(np.float32)


@pytest.fixture
def sample_embeddings_small():
    """Small (10, 32) float32 embedding matrix for fast edge-case tests."""
    rng = np.random.RandomState(42)
    return rng.randn(10, 32).astype(np.float32)


@pytest.fixture
def sample_paths():
    """Fake image paths matching sample_embeddings (100 items)."""
    return [f"/images/img_{i:04d}.jpg" for i in range(100)]


@pytest.fixture
def sample_uuids():
    """Fake UUIDs matching sample_embeddings (100 items)."""
    return [f"uuid-{i:04d}" for i in range(100)]


@pytest.fixture
def sample_labels():
    """Cluster labels for 100 samples across 5 clusters."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 5, size=100)


@pytest.fixture
def sample_arrow_table():
    """PyArrow table with mixed column types for filter testing."""
    return pa.table({
        "uuid": [f"id-{i}" for i in range(20)],
        "species": ["cat", "dog", "cat", "bird", "dog"] * 4,
        "family": ["felidae", "canidae", "felidae", "passeridae", "canidae"] * 4,
        "weight": [4.5, 25.0, 3.8, 0.03, 30.0] * 4,
        "notes": ["healthy", "large breed", "kitten", "sparrow", "retriever"] * 4,
        "emb": [[0.1] * 8 for _ in range(20)],
    })


@pytest.fixture
def reset_cuda_cache():
    """Reset backend CUDA cache between tests."""
    import shared.utils.backend as backend_mod
    original = backend_mod._cuda_check_cache
    backend_mod._cuda_check_cache = None
    yield
    backend_mod._cuda_check_cache = original


@pytest.fixture
def reset_logging():
    """Reset logging configuration between tests."""
    import shared.utils.logging_config as log_mod
    original = log_mod._logging_configured
    log_mod._logging_configured = False
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    root.handlers.clear()
    yield
    root.handlers.clear()
    for h in old_handlers:
        root.addHandler(h)
    log_mod._logging_configured = original
