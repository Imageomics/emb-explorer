# Precalculated Embeddings: Expected Parquet Format

The precalculated embeddings app loads a `.parquet` file **or a directory of
`.parquet` files** (Hive-partitioned or flat) containing precomputed embedding
vectors alongside arbitrary metadata columns. When a directory is provided,
all parquet files within it are read and concatenated automatically.

## Column Requirements

### Must Have

| Column | Type | Description |
|--------|------|-------------|
| `uuid` | `string` | Unique identifier for each record. Used for filtering, selection, and cross-referencing between views. |
| `emb` | `list<float32>` | Precomputed embedding vector. All rows must have the same dimensionality. Used for KMeans clustering and dimensionality reduction (PCA/t-SNE/UMAP). |

The app validates these two columns on load and will reject files missing either.

### Good to Have

These columns unlock additional features but are not required.

| Column | Type | Feature Enabled |
|--------|------|-----------------|
| `identifier` or `image_url` or `url` or `img_url` or `image` | `string` (URL) | **Image preview** in the detail panel. The app tries these column names in order and displays the first valid HTTP(S) image URL found. |
| `kingdom`, `phylum`, `class`, `order`, `family`, `genus`, `species` | `string` | **Taxonomic tree** summary. Any subset works; missing levels default to "Unknown". At minimum `kingdom` must be present and non-null for a row to appear in the tree. |

### Optional (Auto-Detected)

All other columns are automatically analyzed on load:

- **Categorical** (<=100 unique values): Rendered as multi-select dropdown filters with cascading AND logic.
- **Numeric** (int/float): Rendered as range slider filters.
- **Text** (>100 unique string values): Rendered as case-insensitive substring search filters.
- **List/array columns**: Skipped (assumed to be embeddings or similar).

These columns also appear in the record detail panel when a scatter plot point is selected.

### Excluded from Filters

Columns named `uuid`, `emb`, `embedding`, `embeddings`, or `vector` are
automatically excluded from the filter UI and metadata display.

## Minimal Example

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "uuid": ["a1", "a2", "a3"],
    "emb": [np.random.randn(512).tolist() for _ in range(3)],
})
df.to_parquet("minimal.parquet")
```

## Full Example (with taxonomy and images)

```python
df = pd.DataFrame({
    "uuid": ["a1", "a2", "a3"],
    "emb": [np.random.randn(512).tolist() for _ in range(3)],
    "identifier": [
        "https://example.com/img1.jpg",
        "https://example.com/img2.jpg",
        "https://example.com/img3.jpg",
    ],
    "kingdom": ["Animalia", "Animalia", "Plantae"],
    "phylum": ["Chordata", "Chordata", "Magnoliophyta"],
    "class": ["Mammalia", "Aves", "Magnoliopsida"],
    "order": ["Carnivora", "Passeriformes", "Rosales"],
    "family": ["Felidae", "Corvidae", "Rosaceae"],
    "genus": ["Panthera", "Corvus", "Rosa"],
    "species": ["Panthera leo", "Corvus corax", "Rosa canina"],
    "source": ["iNaturalist", "iNaturalist", "GBIF"],  # auto-detected as categorical filter
})
df.to_parquet("full.parquet")
```
