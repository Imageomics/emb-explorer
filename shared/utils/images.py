"""Shared image-fetching utilities.

App-agnostic helpers for resolving and fetching record images from remote
URLs. Kept free of Streamlit so the helpers run safely on worker threads and
in any app (precalculated, a URL-based embed_explore, the demo Space, ...).

In-app fetch flow
-----------------
A record (parquet row) may hold image URLs in several of ``IMAGE_URL_COLUMNS``;
``resolve_record_image_urls`` lists the candidates in column order and
``get_record_image`` walks them until one actually loads (a URL can look valid
but be broken/unreachable). Two call paths consume these:

1. Cluster representatives (bulk, eager).
   ``render_cluster_representatives`` warms the cache up front by feeding each
   candidate's first URL to ``fetch_images_concurrent`` (thread pool, 8
   workers). Worker threads only run ``download_image_bytes``; the caller
   thread decodes each result with ``bytes_to_image`` as it completes and
   stores the PIL image (or ``None`` on failure) in ``_IMAGE_CACHE``. The
   renderer then resolves via ``get_record_image``; broken URLs fall back to
   the record's next URL column, and records with no loadable image fall back
   to the cluster's next candidate.

2. Click preview (single, lazy).
   ``render_data_preview`` calls ``get_record_image``, which serves cached
   images if present and otherwise does a single synchronous
   ``download_image_bytes`` -> ``bytes_to_image`` per URL and caches it.

So both paths share one fetch primitive and one cache; the only difference is
concurrent prefetch vs. on-demand single fetch.

Why a process-level cache (not ``@st.cache_data``)
--------------------------------------------------
- The bulk path fetches from worker threads, where ``st.*`` calls are unsafe;
  a plain module-level dict is thread-friendly and lets both paths share the
  same entries.
- It survives Streamlit reruns within the process, so panning/clicking does
  not refetch. A soft FIFO cap (``_IMAGE_CACHE_MAX``) bounds memory. 
  Trimming only happens at the end of a fetch call, not on every insertion.
- ``None`` is cached as a known miss, so a dead URL is fetched at most once.
- ``ImageTooLarge`` is cached the same way for URLs whose payload exceeds
  ``MAX_IMAGE_BYTES``: the fetch is abandoned early (Content-Length when
  declared, else while streaming), and renderers surface the sentinel as an
  informative placeholder with the original URL instead of a silent miss.

Each worker thread gets its own ``requests.Session`` (``requests.Session`` is
not thread-safe); every session carries the project User-Agent so data hosts
can identify / allowlist us.
"""

import concurrent.futures
import threading
import time
from io import BytesIO
from typing import Dict, Iterable, List, Optional, Union

import requests
from PIL import Image, ImageDraw, ImageFont

from shared import __version__ as _EMB_EXPLORER_VERSION
from shared.utils.logging_config import get_logger

logger = get_logger(__name__)

# Columns checked, in order, for an image URL when resolving a record's image.
IMAGE_URL_COLUMNS = ['identifier', 'image_url', 'url', 'img_url', 'image']

# Be a polite client: identify the app and link the repo so data hosts can
# contact us / allowlist us if needed.
USER_AGENT = (
    f"emb-explorer/{_EMB_EXPLORER_VERSION} "
    "(+https://github.com/Imageomics/emb-explorer)"
)

# Refuse to download image payloads over this size: a malicious or
# misconfigured URL could otherwise stall fetches and spike memory.
MAX_IMAGE_BYTES = 25 * 1024 * 1024
_STREAM_CHUNK_BYTES = 64 * 1024


class ImageTooLarge:
    """Fetch outcome for a URL whose payload exceeded the download cap.

    Cached in ``_IMAGE_CACHE`` like a known miss (so the URL is not retried),
    but distinguishable from one: renderers show an informative placeholder
    and keep ``url`` clickable so users can still open the image themselves.
    ``size_bytes`` is the declared Content-Length, or None when the server
    didn't declare one and the cap was hit mid-stream.
    """

    __slots__ = ("url", "size_bytes", "max_bytes")

    def __init__(self, url: str, size_bytes: Optional[int], max_bytes: int) -> None:
        self.url = url
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes

    def describe(self) -> str:
        """Human-readable size vs. cap, e.g. '38.5 MB (cap: 25 MB)'."""
        cap_mb = self.max_bytes / (1024 * 1024)
        if self.size_bytes is None:
            return f"over the {cap_mb:.0f} MB cap"
        return f"{self.size_bytes / (1024 * 1024):.1f} MB (cap: {cap_mb:.0f} MB)"


# requests.Session is not thread-safe, and fetch_images_concurrent calls
# download_image_bytes from a thread pool — so keep one session per thread.
_thread_local = threading.local()


def _get_session() -> requests.Session:
    """Lazily build a per-thread requests.Session carrying our User-Agent."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        _thread_local.session = session
    return session


def download_image_bytes(
    url: str, timeout: int = 5, max_bytes: Optional[int] = None
) -> Union[bytes, ImageTooLarge, None]:
    """Fetch raw image bytes via the per-thread session.

    Returns the bytes, an `ImageTooLarge` sentinel when the payload exceeds
    `max_bytes` (checked against Content-Length when declared, and enforced
    while streaming regardless — the header can be absent or lie), or None
    on any other failure. `max_bytes` defaults to the module-level
    `MAX_IMAGE_BYTES`, resolved at call time.

    Contains no Streamlit calls, so it is safe to run from worker threads.
    """
    if not isinstance(url, str) or not url.startswith(('http://', 'https://')):
        return None
    if max_bytes is None:
        max_bytes = MAX_IMAGE_BYTES
    try:
        with _get_session().get(url, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            if not resp.headers.get('content-type', '').lower().startswith('image/'):
                return None

            declared: Optional[int] = None
            try:
                declared = int(resp.headers['content-length'])
            except (KeyError, ValueError):
                pass
            if declared is not None and declared > max_bytes:
                logger.warning(
                    f"[Image] Skipping oversized payload "
                    f"({declared / (1024 * 1024):.1f} MB declared): {url[:80]}"
                )
                return ImageTooLarge(url, declared, max_bytes)

            chunks: List[bytes] = []
            received = 0
            for chunk in resp.iter_content(chunk_size=_STREAM_CHUNK_BYTES):
                received += len(chunk)
                if received > max_bytes:
                    logger.warning(
                        f"[Image] Aborting oversized download "
                        f"(>{max_bytes / (1024 * 1024):.0f} MB): {url[:80]}"
                    )
                    return ImageTooLarge(url, declared, max_bytes)
                chunks.append(chunk)
            return b"".join(chunks)
    except Exception:
        return None


def bytes_to_image(data: Optional[bytes]) -> Optional[Image.Image]:
    """Decode image bytes to a PIL image, or None on failure."""
    if not data:
        return None
    try:
        img = Image.open(BytesIO(data))
        # Force the actual decode now: Image.open is lazy, so a truncated or
        # corrupt payload would otherwise only raise later inside st.image,
        # outside this handler, and never be cached as a known miss.
        img.load()
        return img
    except Exception as e:
        logger.error(f"[Image] Failed to open: {e}")
        return None


# Process-level cache for fetched images. Survives Streamlit reruns within the
# process; value is a PIL image, an ImageTooLarge sentinel (over-cap URL,
# not retried), or None (known miss).
_IMAGE_CACHE: Dict[str, Union[Image.Image, ImageTooLarge, None]] = {}
_IMAGE_CACHE_MAX = 512


def _trim_cache() -> None:
    """Soft FIFO cap so the cache doesn't grow unbounded across sessions."""
    if len(_IMAGE_CACHE) > _IMAGE_CACHE_MAX:
        for k in list(_IMAGE_CACHE.keys())[: len(_IMAGE_CACHE) - _IMAGE_CACHE_MAX]:
            _IMAGE_CACHE.pop(k, None)


def fetch_images_concurrent(
    urls: Iterable[str], max_workers: int = 8, timeout: int = 5
) -> Dict[str, Union[Image.Image, ImageTooLarge, None]]:
    """Fetch many image URLs concurrently with a thread pool.

    Returns {url: PIL image, ImageTooLarge, or None}. Per-URL results are
    cached in a process-level dict so reruns and overlapping clusters don't
    refetch. Worker threads only do HTTP (no st.* calls, no cache writes);
    PIL decode and caching happen on the caller thread as downloads complete.
    """
    unique = [u for u in dict.fromkeys(urls) if isinstance(u, str) and u]
    missing = [u for u in unique if u not in _IMAGE_CACHE]

    if missing:
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_url = {
                ex.submit(download_image_bytes, u, timeout): u for u in missing
            }
            for fut in concurrent.futures.as_completed(future_to_url):
                u = future_to_url[fut]
                try:
                    data = fut.result()
                    if isinstance(data, ImageTooLarge):
                        _IMAGE_CACHE[u] = data
                    else:
                        _IMAGE_CACHE[u] = bytes_to_image(data)
                except Exception:
                    _IMAGE_CACHE[u] = None
        ok = sum(1 for u in missing if _IMAGE_CACHE.get(u) is not None)
        logger.info(
            f"[Image] Concurrently fetched {len(missing)} url(s) in "
            f"{time.time() - t0:.2f}s ({ok} ok)"
        )
        _trim_cache()

    return {u: _IMAGE_CACHE.get(u) for u in unique}


def get_image_from_url(
    url: str, timeout: int = 5
) -> Union[Image.Image, ImageTooLarge, None]:
    """Get a single image from a URL, using the process cache.

    Logs the request; results (including misses and over-cap sentinels) are
    cached so repeated lookups and the concurrent path share one cache.
    """
    if not url or not isinstance(url, str):
        return None
    if url in _IMAGE_CACHE:
        return _IMAGE_CACHE[url]
    if not url.startswith(('http://', 'https://')):
        logger.warning(f"[Image] Invalid URL scheme: {url[:50]}...")
        return None

    logger.info(f"[Image] Fetching: {url[:80]}...")
    start_time = time.time()
    data = download_image_bytes(url, timeout)
    image = data if isinstance(data, ImageTooLarge) else bytes_to_image(data)
    elapsed = time.time() - start_time
    if isinstance(image, Image.Image):
        logger.info(f"[Image] Loaded in {elapsed:.3f}s")
    elif image is None:
        logger.warning(f"[Image] Failed to load: {url[:50]}...")

    _IMAGE_CACHE[url] = image
    _trim_cache()
    return image


def resolve_record_image_urls(row) -> List[str]:
    """Return all candidate HTTP(S) image URLs from a record/row, in
    `IMAGE_URL_COLUMNS` order (deduplicated).

    `row` is anything supporting `col in row` membership and `row[col]`
    indexing (e.g. a pandas Series or a dict).
    """
    urls: List[str] = []
    for col in IMAGE_URL_COLUMNS:
        try:
            present = col in row.index
        except AttributeError:
            present = col in row
        if present:
            val = row[col]
            if (
                isinstance(val, str)
                and val.startswith(('http://', 'https://'))
                and val not in urls
            ):
                urls.append(val)
    return urls


def resolve_record_image_url(row) -> Optional[str]:
    """Return the first candidate HTTP(S) image URL from a record/row, else None."""
    urls = resolve_record_image_urls(row)
    return urls[0] if urls else None


def get_record_image(
    row, timeout: int = 5
) -> Union[Image.Image, ImageTooLarge, None]:
    """Fetch a record's image, falling back across candidate URL columns.

    A URL that looks valid can still be broken/unreachable; walk the record's
    candidate URLs (cached fetches) until one actually loads. An over-cap URL
    does not stop the walk — a smaller alternate column still wins — but if
    nothing loads, the first `ImageTooLarge` seen is returned (rather than
    None) so renderers can show an informative placeholder with the URL.
    """
    too_large: Optional[ImageTooLarge] = None
    for url in resolve_record_image_urls(row):
        image = get_image_from_url(url, timeout)
        if isinstance(image, ImageTooLarge):
            too_large = too_large or image
        elif image is not None:
            return image
    return too_large


def too_large_placeholder(
    info: ImageTooLarge, size: int = 280
) -> Image.Image:
    """Generate a placeholder tile for an over-cap image.

    Streamlit-free (plain PIL) so any renderer can use it; the clickable URL
    is the renderer's job since an image can't carry a link.
    """
    img = Image.new("RGB", (size, size), (233, 236, 239))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=max(12, size // 18))
    except TypeError:  # Pillow < 10.1: no size parameter
        font = ImageFont.load_default()
    lines = ["Image too large", info.describe(), "not downloaded"]
    line_heights = []
    widths = []
    for line in lines:
        x0, y0, x1, y1 = draw.textbbox((0, 0), line, font=font)
        widths.append(x1 - x0)
        line_heights.append(y1 - y0)
    spacing = max(line_heights) // 2
    total_h = sum(line_heights) + spacing * (len(lines) - 1)
    y = (size - total_h) // 2
    for line, w, h in zip(lines, widths, line_heights):
        draw.text(((size - w) // 2, y), line, fill=(108, 117, 125), font=font)
        y += h + spacing
    return img
