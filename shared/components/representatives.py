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

import html
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components

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


# Header rendered above each tile: an optional name line plus the id/source
# line. Heights are fixed so tiles with different image sizes still align.
_ID_LINE_HEIGHT = 26
_HEADER_HEIGHT = 48


def build_tile_header_html(details: Dict[str, Any], caption: Optional[str] = None) -> str:
    """HTML for the block above a tile: the record name (``caption``) and a
    short id that copies the full id on click, plus a "source" link that
    opens the record's URL in a new tab.

    ``details`` keys: ``id`` (full text to copy; required), ``id_short``
    (display text, default the first 8 characters of ``id``; a label such as
    "File path" also works), ``url`` (optional http(s) link). Hovering the
    label shows the full value. Everything is escaped; the full value travels
    in a data attribute, never in the visible text.

    Copying uses the async Clipboard API where the page is a secure context
    (https, or http://localhost such as a port-forward) and falls back to
    the legacy execCommand path elsewhere (e.g. a node's plain-http URL);
    if both fail the full id is shown inline so it can be selected by hand.
    """
    full = str(details.get("id", ""))
    short = str(details.get("id_short") or full[:8])
    url = details.get("url")
    link = ""
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        link = (f'<span class="sep">&middot;</span>'
                f'<a href="{html.escape(url, quote=True)}" target="_blank" '
                f'rel="noopener noreferrer">source</a>')
    name = f'<div class="name">{html.escape(caption)}</div>' if caption else ""
    return f"""<style>
  body {{ margin: 0; font-family: "Source Sans Pro", "Source Sans", sans-serif;
         font-size: 14px; line-height: 22px; color: rgba(49, 51, 63, 0.6);
         overflow: hidden; white-space: nowrap; }}
  .name {{ color: rgb(49, 51, 63); overflow: hidden; text-overflow: ellipsis; }}
  a {{ color: #0068c9; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .id {{ font-family: "Source Code Pro", monospace; cursor: pointer;
         border-bottom: 1px dotted currentColor; }}
  .id:hover {{ color: rgba(49, 51, 63, 1); }}
  .sep {{ margin: 0 0.45em; }}
  @media (prefers-color-scheme: dark) {{
    body {{ color: rgba(250, 250, 250, 0.6); }}
    .name {{ color: rgb(250, 250, 250); }}
    a {{ color: #60b4ff; }}
    .id:hover {{ color: rgba(250, 250, 250, 1); }}
  }}
</style>
{name}<div><span class="id" title="Click to copy: {html.escape(full, quote=True)}" data-full="{html.escape(full, quote=True)}"
      onclick="copyId(this)">{html.escape(short)}</span>{link}</div>
<script>
function copyId(el) {{
  var text = el.dataset.full;
  var shown = el.textContent;
  function done() {{ el.textContent = "copied"; setTimeout(function () {{ el.textContent = shown; }}, 1200); }}
  function legacy() {{
    var ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    var ok = false;
    try {{ ok = document.execCommand("copy"); }} catch (e) {{ ok = false; }}
    document.body.removeChild(ta);
    if (ok) {{ done(); }} else {{ el.textContent = text; el.title = "Select and copy"; }}
  }}
  if (navigator.clipboard && window.isSecureContext) {{
    navigator.clipboard.writeText(text).then(done, legacy);
  }} else {{
    legacy();
  }}
}}
</script>"""


def _embed_html(markup: str, height: int) -> None:
    """Embed our own (fully escaped) markup. st.iframe is the current API;
    components.v1.html is the fallback below the Streamlit version that
    introduced it (the project floor is 1.50)."""
    if hasattr(st, "iframe"):
        st.iframe(markup, height=height)
    else:
        components.html(markup, height=height)


def _render_tile_header(cell: Any, caption: Optional[str], details: Optional[Dict[str, Any]]) -> None:
    if details and details.get("id"):
        with cell:
            _embed_html(build_tile_header_html(details, caption),
                        _HEADER_HEIGHT if caption else _ID_LINE_HEIGHT)


def render_representative_images(
    representatives: Dict[object, List[int]],
    resolve_image: Callable[[int], Optional[Any]],
    n_per_cluster: int = 3,
    caption_fn: Optional[Callable[[int], Optional[str]]] = None,
    columns: int = 3,
    details_fn: Optional[Callable[[int], Optional[Dict[str, Any]]]] = None,
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
        details_fn: optional idx -> {"id", "id_short", "url"}. When given, each
            tile gets a header above the image with the caption, a short id
            that copies the full id on click, and a "source" link that opens
            the URL in a new tab (see `build_tile_header_html`); the image
            itself is then rendered without a caption. None keeps the plain
            captioned-image layout.
    """
    if details_fn is not None:
        st.caption("Above each image: click the underlined label to copy its full "
                   "value (hover to preview it); \"source\" opens the image in a new tab.")

    for cluster_id in _sorted_cluster_ids(representatives):
        candidates = representatives[cluster_id]
        # One bordered card per cluster so the grid reads as separate modules.
        card = st.container(border=True)
        card.markdown(f"**Cluster {cluster_id}**")

        # Walk ranked candidates, collecting successful resolutions until we
        # have n_per_cluster (or run out of candidates). Over-cap candidates
        # are set aside: a later candidate with a loadable image is a better
        # slot-filler than a placeholder.
        shown: List[Tuple[Any, Optional[str], Optional[Dict[str, Any]]]] = []  # (displayable, caption, details)
        too_large: List[Tuple[ImageTooLarge, Optional[str], Optional[Dict[str, Any]]]] = []
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
            details = details_fn(idx) if details_fn else None
            if isinstance(img, ImageTooLarge):
                too_large.append((img, caption, details))
            else:
                shown.append((img, caption, details))

        # Fill remaining slots with informative too-large placeholders.
        placeholders = too_large[: n_per_cluster - len(shown)]

        # Nothing renderable in this cluster: every candidate (up to
        # n_per_cluster * oversample, or the whole cluster if smaller) failed
        # on all of its URL columns, and none was merely over the size cap,
        # so there is not even a placeholder to show.
        if not shown and not placeholders:
            card.caption("No images available for this cluster.")
            continue

        entries = shown + placeholders
        cols = card.columns(min(columns, len(entries)))
        for i, (img, caption, details) in enumerate(entries):
            cell = cols[i % len(cols)]
            has_header = bool(details and details.get("id"))
            if has_header:
                # Name, id and source above the image: tops stay aligned even
                # when images have different heights.
                _render_tile_header(cell, caption, details)
            tile_caption = None if has_header else caption
            if isinstance(img, ImageTooLarge):
                # The tile itself can't carry a link; point at where it is.
                hint = 'click "source" above to open' if has_header else "see the link below"
                cell.image(too_large_placeholder(img, hint=hint), caption=tile_caption, width="stretch")
                if not has_header:  # with a header, "source" already links the URL
                    cell.caption(f"[Open original image]({img.url})")
            else:
                cell.image(img, caption=tile_caption, width="stretch")
