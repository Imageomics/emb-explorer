"""Thread-parallel image embedding pipeline.

Turns a list of image paths into embeddings on a single machine. Preprocessing
(decode + transform) runs on a thread pool; the model forward runs on the
calling thread that owns the device. Each batch is preprocessed while the
previous batch runs through the model (a one-batch prefetch), so CPU decoding
and the device forward overlap.

Threads — rather than worker processes — carry the preprocessing because the
work is GIL-light: PIL decode and torchvision tensor ops release the GIL, so a
thread pool scales nearly linearly. Staying in one process means no per-image
data crosses a process boundary and there is no worker-spawn cost, so small
folders are cheap and behavior does not depend on the OS.

This module is Streamlit-free and unit-testable.
"""

from __future__ import annotations

import concurrent.futures as cf
from collections import deque
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from shared.utils.logging_config import get_logger

logger = get_logger(__name__)


def _output_dim(model) -> int:
    """Best-effort embedding width, for shaping an empty result."""
    return int(getattr(getattr(model, "visual", None), "output_dim", 0) or 0)


def _preprocess_one(path: str, preprocess: Callable, color_mode: str):
    """Decode + preprocess a single image.

    Returns ``(path, tensor)`` on success or ``(path, None)`` if the file can't
    be read/decoded. Pure and device-free, so it is safe on worker threads.
    """
    try:
        with Image.open(path) as im:
            img = im.convert(color_mode)
        return path, preprocess(img)
    except Exception as e:
        logger.warning(f"[Embed] Skipping unreadable image {path}: {e}")
        return path, None


def embed_image_folder(
    image_paths: List[str],
    model,
    preprocess: Callable,
    device: torch.device,
    *,
    batch_size: int = 32,
    n_workers: int = 8,
    prefetch_batches: int = 1,
    color_mode: str = "RGB",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Embed a list of image paths, overlapping preprocessing with the forward.

    Preprocessing runs on a ``ThreadPoolExecutor``; the model forward runs on the
    calling thread (which owns ``device``). Up to ``prefetch_batches`` batches are
    preprocessed ahead of the batch currently being run through the model.

    Unreadable images are skipped (and logged), so the returned embeddings may
    have fewer rows than ``image_paths``. ``embeddings[i]`` corresponds to
    ``valid_paths[i]``.

    Args:
        image_paths: Image file paths to embed.
        model: Model exposing ``encode_image(tensor) -> tensor``.
        preprocess: Callable mapping a PIL image to a CHW tensor.
        device: Torch device the model lives on.
        batch_size: Images per forward pass.
        n_workers: Preprocessing threads.
        prefetch_batches: Batches to preprocess ahead of the forward (overlap).
        color_mode: PIL convert mode applied before preprocessing.
        progress_callback: Optional ``(fraction, message)`` progress sink.

    Returns:
        ``(embeddings [N, D] float array, valid_paths [N])``.
    """
    total = len(image_paths)
    if total == 0:
        return np.empty((0, _output_dim(model)), dtype=np.float32), []

    batches = [image_paths[i:i + batch_size] for i in range(0, total, batch_size)]
    window = max(1, prefetch_batches + 1)

    emb_chunks: List[np.ndarray] = []
    valid_paths: List[str] = []
    processed = 0
    
    # concurrent.futures.ThreadPoolExecutor(max_workers=n_workers)
    # spins up `n_workers` OS threads sitting idle, waiting for work...
    # hand it work with ex.submit(fn, *args), which returns a Future immediately, 
    # and runs fn(*args) on a worker thread when it gets scheduled by the OS...
    with cf.ThreadPoolExecutor(max_workers=n_workers) as ex:
        
        # non-blocking, starts the preprocessing of a batch on the pool
        def submit(batch: List[str]) -> List[cf.Future]:
            return [ex.submit(_preprocess_one, p, preprocess, color_mode) for p in batch]

        # Prime the pipeline so the first forward already has successors decoding.
        # pending is a queue of lists of futures, one list per batch.
        pending: deque = deque()
        next_idx = 0
        while next_idx < len(batches) and len(pending) < window:
            pending.append(submit(batches[next_idx]))
            next_idx += 1
        # pending is now a full window of batches being preprocessed
        
        with torch.no_grad():
            while pending:
                # Take the oldest in-flight batch
                futures = pending.popleft()              
                # Refill the window first: these batches preprocess on the pool
                # while we run the current batch through the model below.
                if next_idx < len(batches):
                    pending.append(submit(batches[next_idx]))
                    next_idx += 1

                # If the worker alreadt=y finished, returns immediately; 
                # otherwise blocks until the batch is ready.
                results = [f.result() for f in futures]     
                batch_paths = [p for p, t in results if t is not None]
                tensors = [t for _, t in results if t is not None]

                if tensors:
                    x = torch.stack(tensors).to(device)
                    feats = model.encode_image(x).cpu().numpy()
                    emb_chunks.append(feats)
                    valid_paths.extend(batch_paths)

                processed += len(futures)
                if progress_callback:
                    progress_callback(processed / total, f"Embedding {processed}/{total}")

    if emb_chunks:
        embeddings = np.vstack(emb_chunks)
    else:
        embeddings = np.empty((0, _output_dim(model)), dtype=np.float32)

    logger.info(
        f"[Embed] {embeddings.shape[0]}/{total} images embedded "
        f"({total - embeddings.shape[0]} skipped)"
    )
    return embeddings, valid_paths
