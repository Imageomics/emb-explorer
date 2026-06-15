"""
Embedding generation service.

Heavy libraries (torch, open_clip) are imported lazily inside methods
to avoid slowing down app startup.

Device-aware concurrency:

PyTorch has two kinds of parallelism built in, we focus on the intra-op
parallelism which is relevant to the embedding pipeline:

Intra-op is the parallelism inside a single operation. One op, say
`Normalize` on a `[3, 244, 244]` tensor, or a big matrix multiply, splits its
own work across multiple threads (via an openMP/MKL thread pool).
`torch.get_num_threads()` queries how many threads one op may use, and
`torch.set_num_threads(n)` sets it.

A single `preprocess(img)` is a chain of torch ops (resize -> to_tensor ->
normalize). With the default intra-op thread settings, each of those ops can
fan its work out across all CPU cores. So ONE preprocess call of one image
can momentarily spin up ~`cpu_count` threads to do that tiny bit of math. 

^^^ Why that's wasteful here? 

Since we already have our own parallelism layer at image level: the
`ThreadPoolExecutor` runs `workers` threads, one image per thread, and each
thread calls `preprocess(img)`. If each preprocess call fans out across all
CPU cores, then `workers` threads can easily oversubscribe the CPU with
`workers * cpu_count` threads. This causes contention and can actually slow
down the whole process. 

```
Layer 1 (ThreadPoolExecutor):  16 worker threads, each handling one image preprocess
Layer 2 (torch intra-op):      x Each preprocess call can use up to `cpu_count` threads
                               ========================================================
                               Total threads = 16 (workers) * cpu_count (intra-op) =>
                               Potentially 256 threads on a 16-core machine,
                               causing oversubscription and slowdown.
```

By setting `torch.set_num_threads(1)`, we ensure that each preprocess call
runs single-thread, no internal spliting. All parallelism comes cleanly from
one place -  the `ThreadPoolExecutor`. Instead of two nested layers that
multiply into a thread explosion, each core does one useful thing (decode a
whole image) with no scheduling thrash and no per-op thread-launch overhead. 

```
Layer 1 (ThreadPoolExecutor):  16 worker threads, each handling one image preprocess
Layer 2 (torch intra-op):      x 1 (each op runs single-threaded, instantly)
                               ========================================================
                               Total threads = 16 (workers) * 1 (intra-op) =>
                               Potentially 16 threads on a 16-core machine,
                               fully utilizing the CPU without oversubscription.
```

What Intra-op is good for? 

Intra-op parallelism is excellent for big ops. On the CPU-only path, the
forward pass of the model is the bottleneck, and it benefits from intra-op
parallelism. So we leave torch's intra-op threads alone on CPU, and cap the
worker threads to a small number (2) to avoid too much contention. On GPU,
the forward pass is fast and doesn't need CPU cores, so we maximize worker
threads for decoding and set intra-op to 1 to avoid oversubscription.

"""

import os
import numpy as np
import streamlit as st
import time
from typing import Tuple, List, Optional, Callable

from shared.utils.io import list_image_files
from shared.utils.models import list_available_models
from shared.utils.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Service for handling embedding generation workflows"""

    @staticmethod
    @st.cache_data
    def get_model_options() -> List[str]:
        """Get formatted model options for selectbox."""
        models_data = list_available_models()
        options = []

        # Add all models from list
        for model in models_data:
            name = model['name']
            pretrained = model['pretrained']

            if pretrained is None or pretrained == "":
                display_name = name
            else:
                display_name = f"{name} ({pretrained})"
            options.append(display_name)

        return options

    @staticmethod
    def parse_model_selection(selected_model: str) -> Tuple[str, Optional[str]]:
        """Parse the selected model string to extract model name and pretrained."""
        # Parse OpenCLIP format: "model_name (pretrained)" or just "model_name"
        if "(" in selected_model and selected_model.endswith(")"):
            name = selected_model.split(" (")[0]
            pretrained = selected_model.split(" (")[1].rstrip(")")
            return name, pretrained
        else:
            return selected_model, None

    @staticmethod
    @st.cache_resource(show_spinner=True)
    def load_model_unified(selected_model: str, device: str = "cuda"):
        """Unified model loading function that handles all model types."""
        import torch
        import open_clip

        model_name, pretrained = EmbeddingService.parse_model_selection(selected_model)

        logger.info(f"Loading model: {model_name} (pretrained={pretrained}) on device={device}")
        start_time = time.time()

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )

        model = torch.compile(model.to(device))

        elapsed = time.time() - start_time
        logger.info(f"Model loaded in {elapsed:.2f}s")
        return model, preprocess

    @staticmethod
    def generate_embeddings(
        image_dir: str,
        model_name: str,
        batch_size: int,
        n_workers: int,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        recursive: bool = False,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate embeddings for images in a directory.

        Preprocessing runs on a thread pool (GIL-light) overlapped with the model
        forward pass — no multiprocessing, so behavior is identical on every OS.

        Args:
            image_dir: Path to directory containing images
            model_name: Name of the model to use
            batch_size: Batch size for the forward pass
            n_workers: Max preprocessing threads (capped per device, see below)
            progress_callback: Optional callback for progress updates
            recursive: Recurse into subdirectories when listing images

        Returns:
            Tuple of (embeddings array, list of valid image paths)
        """
        import torch
        from shared.utils.image_pipeline import embed_image_folder

        logger.info(f"Starting embedding generation: dir={image_dir}, model={model_name}, "
                    f"batch_size={batch_size}, n_workers={n_workers}, recursive={recursive}")
        total_start = time.time()

        if progress_callback:
            progress_callback(0.0, "Listing images...")

        image_paths = list_image_files(image_dir, recursive=recursive)
        total = len(image_paths)
        logger.info(f"Found {total} images in {image_dir}")

        if progress_callback:
            progress_callback(0.05, f"Found {total} images. Loading model...")

        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(torch_device)
        logger.info(f"Using device: {torch_device}")
        model, preprocess = EmbeddingService.load_model_unified(model_name, torch_device)

        # Device-aware concurrency:
        cpu_count = os.cpu_count() or 1
        prev_threads = None

        if device.type == "cuda":
            # GPU: feed the GPU with parallel decode, avoid per-op oversubscription.
            # - preprocess threads: wide
            # - torch intra-op threads: forced to 1
            
            # Set the number of preprocessing threads, clamped by three ceilings:
            # 1) the user-requested n_workers 
            # 2) the number of CPU cores
            # 3) never more threads than images
            workers = max(1, min(n_workers, cpu_count, max(total, 1)))
            
            prev_threads = torch.get_num_threads()
            torch.set_num_threads(1)

        else:
            # CPU: the CPU forward is the bottleneck, needs the cores, 
            # so keep preprocess pool small and leave torch threads alone.
            workers = max(1, min(2, n_workers, max(total, 1)))

        # Map the pipeline's 0..1 progress into the 0.1..1.0 band (model load took 0..0.1).
        def _embed_progress(frac: float, msg: str):
            if progress_callback:
                progress_callback(0.1 + 0.9 * frac, msg)

        try:
            embeddings, valid_paths = embed_image_folder(
                image_paths,
                model,
                preprocess,
                device,
                batch_size=batch_size,
                n_workers=workers,
                progress_callback=_embed_progress,
            )
        finally:
            if prev_threads is not None:
                torch.set_num_threads(prev_threads)

        if progress_callback:
            progress_callback(1.0, f"Complete! Generated {embeddings.shape[0]} embeddings")

        total_elapsed = time.time() - total_start
        rate = embeddings.shape[0] / total_elapsed if total_elapsed > 0 else 0.0
        logger.info(f"Embedding generation completed: {embeddings.shape[0]} embeddings in "
                    f"{total_elapsed:.2f}s ({rate:.1f} images/sec)")

        return embeddings, valid_paths
