"""
Test script to verify FAISS and cuML clustering implementations.
"""

import sys
import numpy as np
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.clustering import run_kmeans, reduce_dim, HAS_FAISS, HAS_CUML, HAS_CUDA


def test_clustering_backends():
    """Test sklearn, FAISS, and cuML clustering backends."""
    print("🧪 Testing Clustering Backends")
    print("=" * 60)
    
    # Show available backends
    print(f"🔍 Available backends:")
    print(f"   FAISS: {'✅' if HAS_FAISS else '❌'}")
    print(f"   cuML: {'✅' if HAS_CUML else '❌'}")
    print(f"   CUDA: {'✅' if HAS_CUDA else '❌'}")
    print()
    
    # Create test data
    np.random.seed(42)
    n_samples = 10000  # Larger dataset to see performance differences
    n_features = 512   # High-dimensional like embeddings
    n_clusters = 8
    
    # Generate some clustered data
    data = []
    for i in range(n_clusters):
        cluster_center = np.random.randn(n_features) * 2
        cluster_data = np.random.randn(n_samples // n_clusters, n_features) * 0.5 + cluster_center
        data.append(cluster_data)
    
    embeddings = np.vstack(data).astype(np.float32)
    print(f"📊 Test data shape: {embeddings.shape}")
    print()
    
    # Test backends
    backends_to_test = ["sklearn"]
    if HAS_FAISS:
        backends_to_test.append("faiss")
    if HAS_CUML and HAS_CUDA:
        backends_to_test.append("cuml")
    
    results = {}
    
    for backend in backends_to_test:
        print(f"🔬 Testing {backend} backend...")
        try:
            start_time = time.time()
            
            # Test dimensionality reduction
            print(f"   Reducing dimensions with {backend}...")
            reduced = reduce_dim(embeddings, "PCA", backend=backend)
            
            # Test clustering
            print(f"   Clustering with {backend}...")
            kmeans, labels = run_kmeans(
                reduced, n_clusters, backend=backend, n_workers=4
            )
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            cluster_counts = np.bincount(labels)
            results[backend] = {
                'time': elapsed,
                'clusters': len(np.unique(labels)),
                'distribution': cluster_counts
            }
            
            print(f"✅ {backend}: {len(np.unique(labels))} clusters in {elapsed:.2f}s")
            print(f"   Cluster distribution: {cluster_counts}")
            
        except Exception as e:
            print(f"❌ {backend} failed: {e}")
            continue
        
        print()
    
    # Performance comparison
    if len(results) > 1:
        print("📈 Performance Comparison:")
        print("-" * 40)
        fastest_time = min(r['time'] for r in results.values())
        
        for backend, result in results.items():
            speedup = fastest_time / result['time']
            status = "🥇" if result['time'] == fastest_time else f"{speedup:.1f}x slower"
            print(f"   {backend:8}: {result['time']:6.2f}s {status}")
    
    # Test auto backend
    print("\n🤖 Testing auto backend...")
    try:
        start_time = time.time()
        reduced_auto = reduce_dim(embeddings, "PCA", backend="auto")
        kmeans_auto, labels_auto = run_kmeans(
            reduced_auto, n_clusters, backend="auto", n_workers=4
        )
        end_time = time.time()
        
        # Determine which backend was likely used
        if HAS_CUML and HAS_CUDA and embeddings.shape[0] > 5000:
            likely_backend = "cuML (GPU)"
        elif HAS_FAISS and embeddings.shape[0] > 10000:
            likely_backend = "FAISS (CPU)"
        else:
            likely_backend = "sklearn (CPU)"
            
        print(f"✅ Auto backend likely used: {likely_backend}")
        print(f"   Time: {end_time - start_time:.2f}s")
        print(f"   {len(np.unique(labels_auto))} clusters found")
    except Exception as e:
        print(f"❌ Auto backend failed: {e}")
    
    print("\n🎉 Testing complete!")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if HAS_CUML and HAS_CUDA:
        print("   • Use cuML for fastest GPU-accelerated clustering")
        print("   • Automatic GPU memory management")
        print("   • Best for datasets > 5,000 samples")
    elif HAS_FAISS:
        print("   • Use FAISS for fast CPU clustering")
        print("   • Good parallelization for large datasets")
        print("   • Best for datasets > 10,000 samples")
    else:
        print("   • Consider installing cuML (GPU) or FAISS (CPU) for better performance")
        print("   • cuML: conda install -c rapidsai cuml")
        print("   • FAISS: pip install faiss-cpu")
    
    return True


if __name__ == "__main__":
    success = test_clustering_backends()
    sys.exit(0 if success else 1)
