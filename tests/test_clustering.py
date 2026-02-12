"""Tests for shared/utils/clustering.py."""

import subprocess
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from shared.utils.clustering import (
    _prepare_embeddings,
    estimate_memory_requirement,
    reduce_dim,
    run_kmeans,
    _reduce_dim_sklearn,
    _run_kmeans_sklearn,
    _run_cuml_umap_subprocess,
)


# ---------------------------------------------------------------------------
# _prepare_embeddings
# ---------------------------------------------------------------------------

class TestPrepareEmbeddings:
    def test_output_dtype_float32(self, sample_embeddings):
        result = _prepare_embeddings(sample_embeddings, "test")
        assert result.dtype == np.float32

    def test_output_l2_normalized(self, sample_embeddings):
        result = _prepare_embeddings(sample_embeddings, "test")
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_shape_preserved(self, sample_embeddings):
        result = _prepare_embeddings(sample_embeddings, "test")
        assert result.shape == sample_embeddings.shape

    def test_nan_replaced(self):
        emb = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        result = _prepare_embeddings(emb, "test")
        assert np.all(np.isfinite(result))

    def test_inf_replaced(self):
        emb = np.array([[1.0, np.inf, 3.0], [4.0, -np.inf, 6.0]], dtype=np.float32)
        result = _prepare_embeddings(emb, "test")
        assert np.all(np.isfinite(result))

    def test_zero_norm_vector_clamped(self):
        emb = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        result = _prepare_embeddings(emb, "test")
        # Zero vector stays near-zero after clamped division, no crash
        assert np.all(np.isfinite(result))

    def test_float64_input_cast(self):
        emb = np.random.RandomState(0).randn(5, 10).astype(np.float64)
        result = _prepare_embeddings(emb, "test")
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# estimate_memory_requirement
# ---------------------------------------------------------------------------

class TestEstimateMemory:
    def test_positive_for_all_methods(self):
        for method in ("PCA", "TSNE", "UMAP"):
            assert estimate_memory_requirement(1000, 512, method) > 0

    def test_tsne_greater_than_pca(self):
        pca = estimate_memory_requirement(1000, 512, "PCA")
        tsne = estimate_memory_requirement(1000, 512, "TSNE")
        assert tsne > pca

    def test_unknown_method_returns_positive(self):
        assert estimate_memory_requirement(1000, 512, "UNKNOWN") > 0


# ---------------------------------------------------------------------------
# reduce_dim — sklearn path
# ---------------------------------------------------------------------------

class TestReduceDimSklearn:
    def test_pca_output_shape(self, sample_embeddings_small):
        result = _reduce_dim_sklearn(sample_embeddings_small, "PCA", seed=42, n_workers=1)
        assert result.shape == (10, 2)

    def test_tsne_output_shape(self, sample_embeddings_small):
        result = _reduce_dim_sklearn(sample_embeddings_small, "TSNE", seed=42, n_workers=1)
        assert result.shape == (10, 2)

    def test_umap_output_shape(self, sample_embeddings_small):
        result = _reduce_dim_sklearn(sample_embeddings_small, "UMAP", seed=42, n_workers=1)
        assert result.shape == (10, 2)

    def test_deterministic_with_seed(self, sample_embeddings_small):
        r1 = _reduce_dim_sklearn(sample_embeddings_small, "PCA", seed=42, n_workers=1)
        r2 = _reduce_dim_sklearn(sample_embeddings_small, "PCA", seed=42, n_workers=1)
        np.testing.assert_array_equal(r1, r2)

    def test_invalid_method_raises(self, sample_embeddings_small):
        with pytest.raises(ValueError, match="Unsupported method"):
            _reduce_dim_sklearn(sample_embeddings_small, "INVALID", seed=42, n_workers=1)


class TestReduceDim:
    def test_sklearn_backend(self, sample_embeddings_small):
        result = reduce_dim(sample_embeddings_small, "PCA", seed=42, backend="sklearn")
        assert result.shape == (10, 2)

    def test_unknown_method_raises(self, sample_embeddings_small):
        with pytest.raises(ValueError):
            reduce_dim(sample_embeddings_small, "INVALID", seed=42, backend="sklearn")


# ---------------------------------------------------------------------------
# run_kmeans — sklearn path
# ---------------------------------------------------------------------------

class TestRunKmeansSklearn:
    def test_returns_labels_and_object(self, sample_embeddings_small):
        kmeans, labels = _run_kmeans_sklearn(
            sample_embeddings_small.astype(np.float32), n_clusters=3, seed=42
        )
        assert labels.shape == (10,)
        assert hasattr(kmeans, "cluster_centers_")

    def test_labels_in_range(self, sample_embeddings_small):
        _, labels = _run_kmeans_sklearn(
            sample_embeddings_small.astype(np.float32), n_clusters=3, seed=42
        )
        assert set(labels).issubset(set(range(3)))

    def test_deterministic_with_seed(self, sample_embeddings_small):
        _, l1 = _run_kmeans_sklearn(sample_embeddings_small.astype(np.float32), 3, seed=42)
        _, l2 = _run_kmeans_sklearn(sample_embeddings_small.astype(np.float32), 3, seed=42)
        np.testing.assert_array_equal(l1, l2)


class TestRunKmeans:
    def test_sklearn_backend(self, sample_embeddings_small):
        _, labels = run_kmeans(sample_embeddings_small, 3, seed=42, backend="sklearn")
        assert labels.shape == (10,)

    def test_auto_backend_small_dataset(self, sample_embeddings_small):
        # Small dataset (10 samples) should use sklearn even on auto
        _, labels = run_kmeans(sample_embeddings_small, 3, seed=42, backend="auto")
        assert labels.shape == (10,)


# ---------------------------------------------------------------------------
# GPU fallback (mocked)
# ---------------------------------------------------------------------------

class TestGPUFallback:
    def test_reduce_dim_cuml_fallback(self, sample_embeddings_small):
        """When cuML cp.asarray raises RuntimeError, _reduce_dim_cuml falls back to sklearn."""
        import shared.utils.clustering as clust_mod

        # Mock cupy so the cuML code path can execute, then fail
        mock_cp = MagicMock()
        mock_cp.asarray.side_effect = RuntimeError("CUDA error: no kernel image")
        mock_cp.float32 = np.float32

        original_cp = getattr(clust_mod, "cp", None)
        clust_mod.cp = mock_cp
        try:
            from shared.utils.clustering import _reduce_dim_cuml
            emb = sample_embeddings_small.astype(np.float32)
            result = _reduce_dim_cuml(emb, "PCA", seed=42, n_workers=1)
            assert result.shape == (10, 2)
        finally:
            if original_cp is not None:
                clust_mod.cp = original_cp
            else:
                delattr(clust_mod, "cp")

    def test_umap_subprocess_crash_raises(self, sample_embeddings_small):
        """Subprocess returning non-zero should raise RuntimeError."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Segmentation fault (SIGFPE)"

        with patch("shared.utils.clustering.subprocess.run", return_value=mock_result), \
             patch("shared.utils.clustering.os.path.exists", return_value=False):
            with pytest.raises(RuntimeError, match="subprocess failed"):
                _run_cuml_umap_subprocess(sample_embeddings_small.astype(np.float32), seed=42)

    def test_umap_subprocess_cleans_temp_files(self, tmp_path, sample_embeddings_small):
        """Temp files should be cleaned up even on failure."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "crash"

        with patch("shared.utils.clustering.subprocess.run", return_value=mock_result), \
             patch("shared.utils.clustering.os.path.exists", return_value=False), \
             patch("shared.utils.clustering.os.path.isdir", return_value=True), \
             patch("shared.utils.clustering.os.unlink") as mock_unlink:
            with pytest.raises(RuntimeError):
                _run_cuml_umap_subprocess(sample_embeddings_small.astype(np.float32), seed=42)
            # unlink called for both input and output paths
            assert mock_unlink.call_count == 2
