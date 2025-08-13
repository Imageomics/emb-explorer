import os
import gc

from cuml import KMeans as SingleGPU_KMeans
from cuml.decomposition import PCA as SingleGPU_PCA
from cuml.manifold.umap import UMAP as SingleGPU_UMAP

import cudf
import cupy as cp
import numpy as np
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow as pa

import pynvml

def get_array_from_df(df: cudf.DataFrame, embedding_col: str) -> cp.ndarray:
    return df[embedding_col].list.leaves.values.reshape(len(df), -1)

# Set up logging
import logging
import time
import argparse
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =================== #
# ---- Load Data ----
# =================== #
 


def main(config):
    start_time = time.time()
    logging.info("Starting data processing pipeline...")
    
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_mem_mb = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / 1e6
    logging.info(f"GPU Memory Usage: {gpu_mem_mb:.2f} MB")

    # Load your filters
    filters = config.get("filters", {})

    # Convert to PyArrow-style expression
    expr = None
    for col, values in filters.items():
        if values:  # skip empty filters
            clause = pc.field(col).isin(values)
            expr = clause if expr is None else pc.and_(expr, clause)

    # Load the dataset (Hive-style)
    dataset = ds.dataset(
        config.get("data_path"),
        format="parquet",
        partitioning="hive"
    )
    
    #table = dataset.to_table(filter=expr, columns=["uuid", "emb"], use_threads=True, batch_readahead=32)
    table = dataset.to_table(filter=expr, use_threads=True, batch_readahead=32)
    table_metadata = table.select(
        [col for col in table.column_names if col != "emb"]
    )
    
    
    
    
    chunked_array = table["emb"] 
    large_chunks = [chunk.cast(pa.large_list(pa.float32())) for chunk in chunked_array.chunks]
    large_list_array = pa.concat_arrays(large_chunks)

    num_rows = len(large_list_array)
    inner_len = len(large_list_array[0])
    flat_values = large_list_array.values

    np_view = np.frombuffer(flat_values.buffers()[1], dtype=np.float32)[:num_rows * inner_len]
    cupy_darr = cp.asarray(np_view.reshape(num_rows, inner_len))
    logging.info(f"Embedding shape: {cupy_darr.shape}")
    gpu_mem_mb = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / 1e6
    logging.info(f"GPU Memory Usage: {gpu_mem_mb:.2f} MB")
    
    # ---- Clustering ---- #
    cluster_spec = config.get("cluster_spec")
    model_kmeans = SingleGPU_KMeans(
        n_clusters=cluster_spec.get("n_clusters", 10),
        random_state=cluster_spec.get("rnd_state", 614)
    )

    model_kmeans.fit(cupy_darr)
    logging.info("KMeans clustering complete.")
    gpu_mem_mb = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / 1e6
    logging.info(f"GPU Memory Usage: {gpu_mem_mb:.2f} MB")
    
    
    # Transfer labels to CPU
    
    # labels_np = cp.asnumpy(model_kmeans.labels_)
    # del model_kmeans
    # gc.collect()  # Clear GPU memory
    # cp.get_default_memory_pool().free_all_blocks()
    # cp.get_default_pinned_memory_pool().free_all_blocks()
    # logging.info("Cleared GPU memory after KMeans.")
    # gpu_mem_mb = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / 1e6
    # logging.info(f"GPU Memory Usage: {gpu_mem_mb:.2f} MB")
    
    # ---- Dim Reduction ---- #
    dim_reduction_spec = config.get("dim_reduction_spec")
    
    if dim_reduction_spec.get('method', 'pca') == 'pca':
        model_pca = SingleGPU_PCA(
            n_components=dim_reduction_spec.get('n_components', 2)
        )
        emb_reduced = model_pca.fit_transform(cupy_darr)

        logging.info("Dimensionality reduction complete.")
        gpu_mem_mb = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / 1e6
        logging.info(f"GPU Memory Usage: {gpu_mem_mb:.2f} MB")
    elif dim_reduction_spec.get('method', 'pca') == 'umap':
        model_umap = SingleGPU_UMAP(
            n_components=dim_reduction_spec.get('n_components', 2)
        )
        emb_reduced = model_umap.fit_transform(cupy_darr)
        logging.info("Dimensionality reduction complete.")
        gpu_mem_mb = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / 1e6
        logging.info(f"GPU Memory Usage: {gpu_mem_mb:.2f} MB")

    # --- Combine ---- #
    
    del cupy_darr
    gc.collect()  # Clear GPU memory
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gpu_mem_mb = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / 1e6
    logging.info("Cleared GPU memory after dim reduction.")
    logging.info(f"GPU Memory Usage: {gpu_mem_mb:.2f} MB")

    # Combine uuid, emb_reduced, and kmeans labels into a DataFrame
    
    #uuid_series = dask_df["uuid"].compute().reset_index(drop=True)
    metadata_cudf = cudf.DataFrame.from_arrow(table_metadata).reset_index(drop=True)
    labels_series = cudf.Series(model_kmeans.labels_, name="cluster_label").reset_index(drop=True)
    labels_series = labels_series.astype("str")
    emb_reduced_cudf = cudf.DataFrame(emb_reduced, columns=["dim_1", "dim_2"]).reset_index(drop=True)
    
    logging.info(f"Convert cupy array results to cudf series")
    gpu_mem_mb = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / 1e6
    logging.info(f"GPU Memory Usage: {gpu_mem_mb:.2f} MB")
    
    combined_cudf = cudf.concat(
        [metadata_cudf, emb_reduced_cudf, labels_series],
        axis=1
    )

    logging.info(f"Constructed combined_cudf with shape: {combined_cudf.shape}")
    gpu_mem_mb = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / 1e6
    logging.info("Cleared GPU memory after PCA.")
    logging.info(f"GPU Memory Usage: {gpu_mem_mb:.2f} MB")

    os.makedirs(config.get("output_dir"), exist_ok=True)
    combined_cudf.to_parquet(
        os.path.join(config.get("output_dir"), "processed_data.parquet")
    )
    
    logging.info(
        f"Data processing complete. Results saved to {config.get('output_dir')}"
    )
    
    total_time = time.time() - start_time
    logging.info(f"Pipeline completed successfully in {total_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data processing and clustering.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    
    args = parser.parse_args()
    
    # Load configuration from file
    import json
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    main(config)
    
        
    
    
    
    
    
    
    
    
    
    
    