# emb-explorer Implementation Summary

## Overview
The emb-explorer is a Streamlit-based visual exploration and clustering tool for image datasets and pre-calculated image embeddings. It provides two main workflows: processing new image datasets and exploring existing pre-calculated embeddings.

## Data Storage Architecture

### Hive Partitioning
- **Storage Format**: Data is stored in Parquet files with optional Hive-style partitioning for efficient querying and organization
- **Partitioning Strategy**: When enabled, files are organized hierarchically (e.g., `dataset=bioclip/genus=Homo/species=sapiens/`) enabling efficient filtering
- **Query Optimization**: Partitioning should be optimized based on common query patterns:
  - Partition by `img_type` for image format-based filtering
  - Partition by taxonomic levels (`family`, `genus`) for biological queries
- **Embedding Format**: Embeddings are stored as float32 arrays within Parquet columns, providing efficient columnar storage

### Data Schema
- **Core Columns**: 
  - `uuid`: Unique identifier for each record
  - `emb`: **Float32** embedding vectors (typically 512-1024 dimensions)
- **Metadata Columns**: 
  - Taxonomic hierarchy: `kingdom`, `phylum`, `class`, `order`, `family`, `genus`, `species`
  - Additional metadata: `source_dataset`, `scientific_name`, `common_name`, `publisher`, `basisOfRecord`, `img_type`, `identifier`

### Image Storage and Display Strategy
- **Primary Method**: Images are rendered on-demand using `identifier` provided in metadata
  - **Advantages**: Minimal storage overhead, leverages existing image hosting
  - **Disadvantages**: URI requests can fail due to network issues, server downtime, or link rot
- **Fallback Strategy**: Store actual images as binary objects in HDF5 format
  - **Use Case**: Retrieve from HDF5 when URI requests fail or for offline usage
  - **Trade-offs**: Increased storage requirements but guaranteed availability

## Data Querying with Apache Arrow

### Query Engine
- **PyArrow Integration**: Uses PyArrow for efficient columnar data operations and zero-copy memory management
- **Dataset API**: Leverages `pyarrow.dataset` with Hive partitioning for pushdown filtering
- **Memory Efficiency**: Conversion from Arrow tables to GPU memory via CuDF/CuPy for minimal data movement

### Query Workflow
```python
# Arrow-based filtering with pushdown predicates
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")
table = dataset.to_table(filter=expr, use_threads=True, batch_readahead=32)

# Efficient embedding extraction
chunked_array = table["emb"]
large_chunks = [chunk.cast(pa.large_list(pa.float32())) for chunk in chunked_array.chunks]
cupy_array = cp.asarray(np_view.reshape(num_rows, inner_len))
```

## Backend Technologies for Clustering & Dimensionality Reduction

### Multi-Backend Architecture
The application supports multiple computational backends with automatic selection based on data size and hardware availability:

#### 1. **cuML (GPU-Accelerated)**
- **Primary Choice**: For large datasets (>5000 samples) with CUDA GPU available
- **Components**: 
  - `cuml.cluster.KMeans` for clustering
  - `cuml.decomposition.PCA`, `cuml.manifold.TSNE`, `cuml.manifold.UMAP` for dimensionality reduction
- **Memory Management**: Direct CuPy array operations, GPU memory pooling

#### 2. **FAISS (CPU/GPU)**
- **Use Case**: Large datasets (>10000 samples) when cuML unavailable
- **Features**: 
  - Optimized nearest neighbor search
  - Multi-threaded CPU execution
  - Optional GPU acceleration

#### 3. **Scikit-learn (CPU Fallback)**
- **Default Fallback**: Standard CPU-based implementations
- **Reliability**: Stable baseline when GPU backends fail
- **Components**: `sklearn.cluster.KMeans`, `sklearn.decomposition.PCA`, `sklearn.manifold.TSNE`, `umap.UMAP`

### Backend Selection Logic
```python
if backend == "auto":
    if HAS_CUML and HAS_CUDA and embeddings.shape[0] > 500:
        return _run_kmeans_cuml(embeddings, n_clusters, seed, n_workers)
    elif HAS_FAISS and embeddings.shape[0] > 500:
        return _run_kmeans_faiss(embeddings, n_clusters, seed, n_workers)
    else:
        return _run_kmeans_sklearn(embeddings, n_clusters, seed)
```


## GPU Acceleration

**Performance Gain**: GPU acceleration provides tremendous speedup for large datasets
  - cuML KMeans: 10-100x faster than CPU for large datasets
  - GPU-based dimensionality reduction: Significant improvement for t-SNE/UMAP

**GPU Availability & Resource Utilization**:
  - **Intermittent Availability**: GPUs may not always be available in shared computing environments
  - **Short Computation Bursts**: Actual computation time (clustering, dim reduction) is relatively short compared to analysis time
  - **Resource Waste**: Allocating GPU for entire app session leads to idle GPU time and poor utilization

## Critical Scaling Challenges

### 1. **GPU Memory Limitations**
- **Problem**: Large queries can result in data larger than GPU VRAM (typically 8-80GB)
- **Impact**: Out-of-memory errors when processing massive datasets

### 2. **Data Movement Bottleneck**
Another scaling bottleneck is the multi-stage data movement pipeline:

```
Disk → RAM → VRAM → RAM (results)
```

**Stages**:
1. **Disk to RAM**: Parquet file loading and Arrow table creation
2. **RAM to VRAM**: CuPy array creation and GPU transfer
3. **VRAM to RAM**: Result extraction back to CPU for further processing

### 3. **Memory Management Issues**
- **GPU Memory Fragmentation**: Repeated allocations can fragment GPU memory
- **Memory Pool Management**: CuPy memory pools need explicit cleanup
- **Process Memory Growth**: Long-running processes accumulate memory

### 4. **Visualization Scaling Challenges**
When dealing with large datasets, visualization becomes a significant bottleneck:

- **Rendering Performance**: Plotting millions of points causes browser/UI performance degradation
- **Point Labeling**: Overlapping labels become unreadable with dense point clouds
- **Interactive Response**: Hover tooltips and selection become slow with large datasets
- **Memory Consumption**: Storing all visualization data in browser memory can cause crashes


**Potential Solutions**:
- **Intelligent Sampling**: Dynamically sample representative points for display while maintaining cluster structure
- **Level-of-Detail Rendering**: Show different detail levels based on zoom level and viewport

