"""
Shared utilities for clustering, IO, models, and taxonomy.

Modules are imported lazily to avoid pulling in heavy dependencies
(sklearn, umap, faiss, cuml, torch, open_clip) at startup.
Use direct imports instead:

    from shared.utils.clustering import reduce_dim, run_kmeans
    from shared.utils.io import list_image_files
"""
