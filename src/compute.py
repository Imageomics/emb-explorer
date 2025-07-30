import os
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, wait

from cuml import KMeans as SingleGPU_KMeans
from cuml.decomposition import PCA as SingleGPU_PCA
#from cuml.dask.cluster import KMeans as MNMG_KMeans
import cudf
import cupy as cp

import dask
import dask.dataframe as dd

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
# ================================= #
# ---- Initialize Dask cluster ----
# ================================= #
def init_dask():
    N_WORKERS = 1
    THREADS_PER_WORKER = 4

    cluster = LocalCUDACluster(
        n_workers=N_WORKERS,
        threads_per_worker=THREADS_PER_WORKER,
        # Communication settings
        # protocol="ucx",
        # interface="ibp65s0",  # Correct InfiniBand interface for your system
        # rmm_pool_size="36GB",
        # Spilling settings for better memory management
        device_memory_limit=0.9,
        enable_cudf_spill=True,  # Improve device memory stability
        local_directory="/fs/scratch/PAS2136/netzissou"  # Use fast local storage for spilling
    )

    client = Client(cluster)

    dask.config.set({"dataframe.backend": "cudf"})
    return client, cluster

# =================== #
# ---- Load Data ----
# =================== #

def load_data(data_dir: str, filters: dict = None) -> dd.DataFrame:
    """
    Load data from parquet files in the specified directory.
    
    Args:
        data_dir: Path to the directory containing parquet files
        filters: Optional filters to apply when loading data
    
    Returns:
        Dask DataFrame containing the loaded data
    """
    
    # Parse filters if provided
    if filters:
        filters = [
            (col, "in", values)
            for col, values in filters.items()
            if values  # skip empty lists
        ]
    
    dask_df = dd.read_parquet(
        path = data_dir,
        filters = filters
    )
    return dask_df
    

# =========================== #
# ---- Close the cluster ----
# =========================== #
def close_dask(client, cluster):
    client.close()
    cluster.close()
    

def main(config):
    start_time = time.time()
    logging.info("Starting data processing pipeline...")
    
    client, cluster = init_dask()
    dask.config.set({"dataframe.backend": "cudf"})
    
    dask_df = load_data(
        config.get("data_path"),
        config.get("filters", {})
    )
    
    cupy_darr = dask_df.map_partitions(
        get_array_from_df, 
        "emb", 
        meta=cp.ndarray([1, 1])
    )
    
    cupy_darr = cupy_darr.compute()
    
    logging.info(f"Embeddings shape: {cupy_darr.shape}")

    # ---- Clustering ---- #
    cluster_spec = config.get("cluster_spec")
    model_kmeans = SingleGPU_KMeans(
        n_clusters=cluster_spec.get("n_clusters", 10),
        random_state=cluster_spec.get("rnd_state", 614)
    )

    model_kmeans.fit(cupy_darr)
    
    logging.info("KMeans clustering complete.")
    
    # ---- Dim Reduction ---- #
    dim_reduction_spec = config.get("dim_reduction_spec")
    model_pca = SingleGPU_PCA(
        n_components=dim_reduction_spec.get('n_components', 2)
    )
    emb_reduced = model_pca.fit_transform(cupy_darr)

    logging.info("Dimensionality reduction complete.")

    # --- Combine ---- #

    # Combine uuid, emb_reduced, and kmeans labels into a DataFrame
    
    uuid_series = dask_df["uuid"].compute().reset_index(drop=True)
    labels_series = cudf.Series(model_kmeans.labels_, name="cluster_label").reset_index(drop=True)
    emb_reduced_cudf = cudf.DataFrame(emb_reduced, columns=["dim_1", "dim_2"]).reset_index(drop=True)
    
    combined_cudf = cudf.concat(
        [uuid_series, emb_reduced_cudf, labels_series],
        axis=1
    )

    os.makedirs(config.get("output_dir"), exist_ok=True)
    combined_cudf.to_parquet(
        os.path.join(config.get("output_dir"), "processed_data.parquet")
    )
    
    logging.info(
        f"Data processing complete. Results saved to {config.get('output_dir')}"
    )
    
    close_dask(client, cluster)
    
    total_time = time.time() - start_time
    logging.info(f"Pipeline completed successfully in {total_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dask-based data processing and clustering.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    
    args = parser.parse_args()
    
    # Load configuration from file
    import json
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    main(config)
    
        
    
    
    
    
    
    
    
    
    
    
    