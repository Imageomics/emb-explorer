# Example Data

`example_1k.parquet`: a small sample for trying out the Precalculated Embedding Exploration app.

It contains [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2) embeddings for 1,030 randomly sampled images from [TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M) (embeddings only, not the images themselves). Taxonomic information and other metadata comes from `catalog.parquet` in the TOL-200M repo.

## Schema

```
uuid: string
emb: list<element: float>
source_dataset: string
source_id: string
kingdom: string
phylum: string
class: string
order: string
family: string
genus: string
species: string
scientific_name: string
common_name: string
resolution_status: string
publisher: string
basisOfRecord: string
identifier: string
img_type: string
```
