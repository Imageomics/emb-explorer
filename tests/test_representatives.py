"""Tests for shared/utils/representatives.py (find_cluster_representatives).

Covers the pure ranking logic only! Centroid ordering, global-index
correctness, the oversample cap, and label-type handling. The Streamlit
renderer is intentionally not tested here.
"""

import numpy as np

from shared.utils.representatives import find_cluster_representatives


def test_ranks_closest_to_centroid_first():
    """Members are ordered by ascending distance to the cluster centroid."""
    # Single cluster; centroid_x = (0+3+100)/3 = 34.33, so idx 1 (x=3) is
    # closest, then idx 0 (x=0), then idx 2 (x=100).
    embeddings = np.array([[0.0, 0.0], [3.0, 0.0], [100.0, 0.0]])
    labels = [0, 0, 0]                # All in one cluster

    reps = find_cluster_representatives(embeddings, labels, n_per_cluster=3)

    assert reps[0] == [1, 0, 2]


def test_indices_are_global_and_match_cluster():
    """Returned indices are global and only reference members of that cluster."""
    labels = [0, 1, 0, 1, 0]
    embeddings = np.random.RandomState(0).rand(5, 4)

    reps = find_cluster_representatives(embeddings, labels, n_per_cluster=2)

    assert set(reps.keys()) == {0, 1}
    labels_arr = np.asarray(labels)
    for cluster_id, idxs in reps.items():
        assert all(labels_arr[i] == cluster_id for i in idxs)


def test_oversample_capped_at_cluster_size():
    """Candidates = min(n_per_cluster * oversample, cluster size)."""
    # Cluster 0: 4 members, cluster 1: 10 members.
    labels = [0] * 4 + [1] * 10
    embeddings = np.random.RandomState(1).rand(14, 3)

    reps = find_cluster_representatives(
        embeddings, labels, n_per_cluster=2, oversample=3
    )  # n_candidates = 6

    assert len(reps[0]) == 4   # capped at cluster size
    assert len(reps[1]) == 6   # capped at n_candidates


def test_preserves_label_type():
    """String labels stay string keys; numpy int labels become Python ints."""
    embeddings = np.random.RandomState(2).rand(3, 2)

    str_reps = find_cluster_representatives(embeddings, ["a", "a", "b"])
    assert set(str_reps.keys()) == {"a", "b"}

    int_reps = find_cluster_representatives(embeddings, np.array([0, 0, 1]))
    assert set(int_reps.keys()) == {0, 1}
    assert all(isinstance(k, int) for k in int_reps.keys())
