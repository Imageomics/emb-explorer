"""Find representative members of clusters.

Given embeddings and cluster labels, rank each cluster's members by proximity
to the cluster centroid. Returns more candidates than strictly requested
(oversampled) so callers that render images can skip candidates whose image
fails to load and still show the desired number per cluster.
"""

from typing import Dict, List

import numpy as np

from shared.utils.logging_config import get_logger

logger = get_logger(__name__)


def find_cluster_representatives(
    embeddings: np.ndarray,
    labels,
    n_per_cluster: int = 3,
    oversample: int = 4,
) -> Dict[object, List[int]]:
    """Rank each cluster's members by closeness to the cluster centroid.

    Args:
        embeddings: (N, D) array of embeddings (row i aligns with label i).
        labels: array-like of length N with cluster labels (int or str).
        n_per_cluster: how many representatives the caller intends to show.
        oversample: multiplier for how many candidate indices to return per
            cluster (n_per_cluster * oversample), so failed image loads can be
            skipped while still surfacing n_per_cluster images.

    Returns:
        Dict mapping each cluster label to a list of global indices into
        `embeddings`, ordered closest-to-centroid first, capped at
        n_per_cluster * oversample (or the cluster size, whichever is smaller).
    """
    labels = np.asarray(labels)
    embeddings = np.asarray(embeddings)
    n_candidates = max(n_per_cluster * oversample, n_per_cluster)

    representatives: Dict[object, List[int]] = {}
    for cluster_id in np.unique(labels):
        member_idxs = np.where(labels == cluster_id)[0]
        if member_idxs.size == 0:
            continue
        cluster_embeds = embeddings[member_idxs]
        centroid = cluster_embeds.mean(axis=0)
        
        # Compute squared Euclidean distance to the centroid for each member.
        dists = np.sum((cluster_embeds - centroid) ** 2, axis=1)
        order = np.argsort(dists)[:n_candidates]
        # Keep the label's native Python type for clean dict keys / display.
        key = cluster_id.item() if hasattr(cluster_id, "item") else cluster_id
        representatives[key] = member_idxs[order].tolist()

    logger.debug(
        f"Found representatives for {len(representatives)} clusters "
        f"(up to {n_candidates} candidates each)"
    )
    return representatives
