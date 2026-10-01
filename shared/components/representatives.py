"""Shared renderer for per-cluster representative images.

Both apps surface representative images differently:
- embed_explore resolves a local image file path.
- precalculated fetches a remote image URL (which can fail).

This renderer is source-agnostic: the caller passes a `resolve_image(idx)`
callable that returns something `st.image` can display (a PIL image, a path,
or bytes) or `None` when the image is unavailable. The renderer walks each
cluster's ranked candidate indices and collects up to `n_per_cluster`
successful images, skipping any that resolve to `None` — the shared fallback.

URL-backed resolvers may also return an `ImageTooLarge` sentinel (see
`shared.utils.images`): the record's image exists but exceeded the download
cap. Real images still take priority for the cluster's slots; too-large
candidates only fill slots that would otherwise stay empty, rendered as an
informative placeholder tile with a link to the original URL.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st

from shared.utils.images import ImageTooLarge, too_large_placeholder
from shared.utils.logging_config import get_logger

logger = get_logger(__name__)


def _sorted_cluster_ids(representatives: Dict[object, List[int]]) -> List[object]:
    """Sort cluster ids numerically when possible, else as strings."""
    keys = list(representatives.keys())
    try:
        return sorted(keys, key=lambda k: int(k))
    except (ValueError, TypeError):
        return sorted(keys, key=str)


def render_representative_images(
    representatives: Dict[object, List[int]],
    resolve_image: Callable[[int], Optional[Any]],
    n_per_cluster: int = 3,
    caption_fn: Optional[Callable[[int], Optional[str]]] = None,
    columns: int = 3,
) -> None:
    """Render up to `n_per_cluster` representative images per cluster.

    Args:
        representatives: {cluster_id: [ranked candidate global indices]}, as
            returned by `find_cluster_representatives`.
        resolve_image: idx -> displayable (PIL image / path / bytes), an
            `ImageTooLarge` sentinel (over-cap URL, rendered as a placeholder
            tile), or None. None means "unavailable" and the renderer falls
            back to the next candidate.
        n_per_cluster: number of images to show per cluster.
        caption_fn: optional idx -> caption string, or None for no caption.
        columns: images per row.
    """
    for cluster_id in _sorted_cluster_ids(representatives):
        candidates = representatives[cluster_id]
        st.markdown(f"**Cluster {cluster_id}**")

        # Walk ranked candidates, collecting successful resolutions until we
        # have n_per_cluster (or run out of candidates). Over-cap candidates
        # are set aside: a later candidate with a loadable image is a better
        # slot-filler than a placeholder.
        shown: List[Tuple[Any, Optional[str]]] = []  # (displayable, caption)
        too_large: List[Tuple[ImageTooLarge, Optional[str]]] = []
        for idx in candidates:
            if len(shown) >= n_per_cluster:
                break
            try:
                img = resolve_image(idx)
            except Exception as e:  # never let one bad image break the panel
                logger.debug(f"resolve_image({idx}) raised: {e}")
                img = None
            if img is None:
                continue
            caption = caption_fn(idx) if caption_fn else None
            if isinstance(img, ImageTooLarge):
                too_large.append((img, caption))
            else:
                shown.append((img, caption))

        # Fill remaining slots with informative too-large placeholders.
        placeholders = too_large[: n_per_cluster - len(shown)]

        if not shown and not placeholders:
            st.caption("No images available for this cluster.")
            continue

        entries = shown + placeholders
        cols = st.columns(min(columns, len(entries)))
        for i, (img, caption) in enumerate(entries):
            cell = cols[i % len(cols)]
            if isinstance(img, ImageTooLarge):
                cell.image(
                    too_large_placeholder(img), caption=caption, width="stretch"
                )
                cell.caption(f"[Open original image]({img.url})")
            else:
                cell.image(img, caption=caption, width="stretch")
