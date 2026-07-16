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
   workers). Each thread calls ``download_image_bytes`` -> ``bytes_to_image``
   and stores the PIL image (or ``None`` on failure) in ``_IMAGE_CACHE``. The
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

Each worker thread gets its own ``requests.Session`` (``requests.Session`` is
not thread-safe); every session carries the project User-Agent so data hosts
can identify / allowlist us.
"""

import concurrent.futures
import threading
import time
from io import BytesIO
from typing import Dict, Iterable, List, Optional

import requests
from PIL import Image

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


def download_image_bytes(url: str, timeout: int = 5) -> Optional[bytes]:
    """Fetch raw image bytes via the shared session. None on any failure.

    Contains no Streamlit calls, so it is safe to run from worker threads.
    """
    if not isinstance(url, str) or not url.startswith(('http://', 'https://')):
        return None
    try:
        with _get_session().get(url, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            if not resp.headers.get('content-type', '').lower().startswith('image/'):
                return None
            return resp.content
    except Exception:
        return None


def bytes_to_image(data: Optional[bytes]) -> Optional[Image.Image]:
    """Decode image bytes to a PIL image, or None on failure."""
    if not data:
        return None
    try:
        return Image.open(BytesIO(data))
    except Exception as e:
        logger.error(f"[Image] Failed to open: {e}")
        return None


# Process-level cache for fetched images. Survives Streamlit reruns within the
# process; value is a PIL image or None (known miss).
_IMAGE_CACHE: Dict[str, Optional[Image.Image]] = {}
_IMAGE_CACHE_MAX = 512


def _trim_cache() -> None:
    """Soft FIFO cap so the cache doesn't grow unbounded across sessions."""
    if len(_IMAGE_CACHE) > _IMAGE_CACHE_MAX:
        for k in list(_IMAGE_CACHE.keys())[: len(_IMAGE_CACHE) - _IMAGE_CACHE_MAX]:
            _IMAGE_CACHE.pop(k, None)


def fetch_images_concurrent(
    urls: Iterable[str], max_workers: int = 8, timeout: int = 5
) -> Dict[str, Optional[Image.Image]]:
    """Fetch many image URLs concurrently with a thread pool.

    Returns {url: PIL image or None}. Per-URL results are cached in a
    process-level dict so reruns and overlapping clusters don't refetch.
    Threads only do HTTP + PIL decode (no st.* calls), which is Streamlit-safe.
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
                    _IMAGE_CACHE[u] = bytes_to_image(fut.result())
                except Exception:
                    _IMAGE_CACHE[u] = None
        ok = sum(1 for u in missing if _IMAGE_CACHE.get(u) is not None)
        logger.info(
            f"[Image] Concurrently fetched {len(missing)} url(s) in "
            f"{time.time() - t0:.2f}s ({ok} ok)"
        )
        _trim_cache()

    return {u: _IMAGE_CACHE.get(u) for u in unique}


def get_image_from_url(url: str, timeout: int = 5) -> Optional[Image.Image]:
    """Get a single image from a URL, using the process cache.

    Logs the request; results (including misses) are cached so repeated
    lookups and the concurrent path share one cache.
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
    image = bytes_to_image(download_image_bytes(url, timeout))
    elapsed = time.time() - start_time
    if image is not None:
        logger.info(f"[Image] Loaded in {elapsed:.3f}s")
    else:
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


def get_record_image(row, timeout: int = 5) -> Optional[Image.Image]:
    """Fetch a record's image, falling back across candidate URL columns.

    A URL that looks valid can still be broken/unreachable; walk the record's
    candidate URLs (cached fetches) until one actually loads.
    """
    for url in resolve_record_image_urls(row):
        image = get_image_from_url(url, timeout)
        if image is not None:
            return image
    return None
