"""Tests for shared/services/clustering_service.py."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from shared.services.clustering_service import ClusteringService


# ---------------------------------------------------------------------------
# generate_clustering_summary (pure — no mocking needed)
# ---------------------------------------------------------------------------

class TestGenerateClusteringSummary:
    def _make_inputs(self, n_samples=20, n_features=32, n_clusters=3):
        rng = np.random.RandomState(42)
        embeddings = rng.randn(n_samples, n_features).astype(np.float32)
        labels = rng.randint(0, n_clusters, size=n_samples)
        df_plot = pd.DataFrame({
            "x": rng.randn(n_samples),
            "y": rng.randn(n_samples),
            "cluster": labels.astype(str),
            "image_path": [f"/img/{i}.jpg" for i in range(n_samples)],
            "idx": range(n_samples),
        })
        return embeddings, labels, df_plot

    def test_summary_columns(self):
        emb, labels, df = self._make_inputs()
        summary, _ = ClusteringService.generate_clustering_summary(emb, labels, df)
        assert set(summary.columns) == {"Cluster", "Count", "Variance"}

    def test_counts_sum_to_total(self):
        emb, labels, df = self._make_inputs(n_samples=50)
        summary, _ = ClusteringService.generate_clustering_summary(emb, labels, df)
        assert summary["Count"].sum() == 50

    def test_representatives_per_cluster(self):
        emb, labels, df = self._make_inputs(n_samples=30, n_clusters=3)
        _, reps = ClusteringService.generate_clustering_summary(emb, labels, df)
        for cluster_id, indices in reps.items():
            cluster_size = (labels == cluster_id).sum()
            assert len(indices) <= min(3, cluster_size)

    def test_single_sample_cluster(self):
        """Cluster with 1 sample should have variance 0."""
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        labels = np.array([0, 1, 2])
        df = pd.DataFrame({"x": [0, 0, 0], "y": [0, 0, 0], "cluster": ["0", "1", "2"], "idx": [0, 1, 2]})
        summary, reps = ClusteringService.generate_clustering_summary(embeddings, labels, df)
        # Each cluster has 1 sample → variance = 0
        assert all(summary["Variance"] == 0.0)
        assert all(len(v) == 1 for v in reps.values())


# ---------------------------------------------------------------------------
# run_clustering_safe — fallback chain (mocked)
# ---------------------------------------------------------------------------

class TestRunClusteringSafe:
    def _dummy_args(self):
        rng = np.random.RandomState(0)
        emb = rng.randn(20, 32).astype(np.float32)
        paths = [f"uuid-{i}" for i in range(20)]
        return emb, paths, 3, "PCA", 1, "auto", "auto", 42

    def test_success_passthrough(self):
        emb, paths, *rest = self._dummy_args()
        # Should succeed via sklearn on CPU
        df, labels = ClusteringService.run_clustering_safe(emb, paths, *rest)
        assert len(df) == 20
        assert labels.shape == (20,)

    def test_gpu_error_triggers_sklearn_fallback(self):
        emb, paths, *rest = self._dummy_args()
        call_count = {"n": 0}

        def mock_run_clustering(embeddings, valid_paths, n_clusters, method,
                                n_workers, dim_backend, cluster_backend, seed):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("CUDA error: no kernel image")
            # Second call (fallback) should use sklearn
            assert dim_backend == "sklearn"
            assert cluster_backend == "sklearn"
            return pd.DataFrame({"x": [0]*20, "y": [0]*20, "cluster": ["0"]*20,
                                 "image_path": valid_paths, "file_name": valid_paths,
                                 "idx": range(20)}), np.zeros(20, dtype=int)

        with patch.object(ClusteringService, "run_clustering", side_effect=mock_run_clustering):
            df, labels = ClusteringService.run_clustering_safe(emb, paths, *rest)
            assert call_count["n"] == 2

    def test_oom_error_reraised(self):
        emb, paths, *rest = self._dummy_args()
        with patch.object(ClusteringService, "run_clustering",
                          side_effect=RuntimeError("CUDA out of memory")):
            with pytest.raises(RuntimeError, match="out of memory"):
                ClusteringService.run_clustering_safe(emb, paths, *rest)

    def test_non_gpu_error_reraised(self):
        emb, paths, *rest = self._dummy_args()
        with patch.object(ClusteringService, "run_clustering",
                          side_effect=RuntimeError("unexpected error")):
            with pytest.raises(RuntimeError, match="unexpected error"):
                ClusteringService.run_clustering_safe(emb, paths, *rest)
