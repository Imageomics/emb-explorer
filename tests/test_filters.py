"""Tests for filter logic in apps/precalculated/components/sidebar.py.

These functions are pure data transformations — no Streamlit dependency.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from apps.precalculated.components.sidebar import (
    apply_filters_arrow,
    get_column_info_dynamic,
    extract_embeddings_safe,
    create_cluster_dataframe,
)


# ---------------------------------------------------------------------------
# apply_filters_arrow
# ---------------------------------------------------------------------------

class TestApplyFiltersArrow:
    def test_categorical_filter(self, sample_arrow_table):
        result = apply_filters_arrow(sample_arrow_table, {"species": ["cat"]})
        species_vals = result.column("species").to_pylist()
        assert all(v == "cat" for v in species_vals)

    def test_numeric_range_filter(self, sample_arrow_table):
        result = apply_filters_arrow(sample_arrow_table, {
            "weight": {"min": 1.0, "max": 10.0}
        })
        weights = result.column("weight").to_pylist()
        assert all(1.0 <= w <= 10.0 for w in weights)

    def test_text_filter(self, sample_arrow_table):
        result = apply_filters_arrow(sample_arrow_table, {"notes": "kitten"})
        notes = result.column("notes").to_pylist()
        assert all("kitten" in n.lower() for n in notes)

    def test_text_filter_case_insensitive(self, sample_arrow_table):
        result = apply_filters_arrow(sample_arrow_table, {"notes": "HEALTHY"})
        assert len(result) > 0

    def test_multiple_filters_and_logic(self, sample_arrow_table):
        result = apply_filters_arrow(sample_arrow_table, {
            "species": ["cat"],
            "weight": {"min": 3.0, "max": 5.0},
        })
        for i in range(len(result)):
            assert result.column("species")[i].as_py() == "cat"
            assert 3.0 <= result.column("weight")[i].as_py() <= 5.0

    def test_empty_filters_returns_original(self, sample_arrow_table):
        result = apply_filters_arrow(sample_arrow_table, {})
        assert len(result) == len(sample_arrow_table)

    def test_unknown_column_skipped(self, sample_arrow_table):
        result = apply_filters_arrow(sample_arrow_table, {"nonexistent": ["x"]})
        assert len(result) == len(sample_arrow_table)

    def test_empty_list_filter_skipped(self, sample_arrow_table):
        result = apply_filters_arrow(sample_arrow_table, {"species": []})
        assert len(result) == len(sample_arrow_table)


# ---------------------------------------------------------------------------
# get_column_info_dynamic
# ---------------------------------------------------------------------------

class TestGetColumnInfoDynamic:
    def test_detects_categorical(self, sample_arrow_table):
        info = get_column_info_dynamic(sample_arrow_table)
        assert info["species"]["type"] == "categorical"

    def test_detects_numeric(self, sample_arrow_table):
        info = get_column_info_dynamic(sample_arrow_table)
        assert info["weight"]["type"] == "numeric"

    def test_skips_excluded_columns(self, sample_arrow_table):
        info = get_column_info_dynamic(sample_arrow_table)
        assert "uuid" not in info
        assert "emb" not in info

    def test_null_counting(self):
        table = pa.table({
            "col": [1, None, 3, None, 5],
        })
        info = get_column_info_dynamic(table)
        assert info["col"]["null_count"] == 2
        assert info["col"]["null_percentage"] == 40.0

    def test_high_cardinality_becomes_text(self):
        """Columns with >100 unique values should be classified as text."""
        table = pa.table({
            "many_unique": [f"val_{i}" for i in range(150)],
        })
        info = get_column_info_dynamic(table)
        assert info["many_unique"]["type"] == "text"


# ---------------------------------------------------------------------------
# extract_embeddings_safe
# ---------------------------------------------------------------------------

class TestExtractEmbeddingsSafe:
    def test_valid_extraction(self):
        emb_data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        df = pd.DataFrame({"emb": emb_data, "id": [1, 2]})
        result = extract_embeddings_safe(df)
        assert result.shape == (2, 3)
        assert result.dtype == np.float32

    def test_missing_emb_column_raises(self):
        df = pd.DataFrame({"id": [1, 2]})
        with pytest.raises(ValueError, match="emb"):
            extract_embeddings_safe(df)


# ---------------------------------------------------------------------------
# create_cluster_dataframe
# ---------------------------------------------------------------------------

class TestCreateClusterDataframe:
    def test_required_columns(self):
        df = pd.DataFrame({
            "uuid": ["a", "b", "c"],
            "emb": [[1, 2], [3, 4], [5, 6]],
            "species": ["cat", "dog", "bird"],
        })
        emb_2d = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        labels = np.array([0, 1, 0])

        result = create_cluster_dataframe(df, emb_2d, labels)
        assert "x" in result.columns
        assert "y" in result.columns
        assert "cluster" in result.columns
        assert "uuid" in result.columns
        assert "idx" in result.columns

    def test_metadata_columns_copied(self):
        df = pd.DataFrame({
            "uuid": ["a", "b"],
            "emb": [[1, 2], [3, 4]],
            "species": ["cat", "dog"],
        })
        emb_2d = np.array([[0.1, 0.2], [0.3, 0.4]])
        labels = np.array([0, 1])

        result = create_cluster_dataframe(df, emb_2d, labels)
        assert "species" in result.columns

    def test_embedding_columns_excluded(self):
        df = pd.DataFrame({
            "uuid": ["a", "b"],
            "emb": [[1, 2], [3, 4]],
            "embedding": [[1, 2], [3, 4]],
        })
        emb_2d = np.array([[0.1, 0.2], [0.3, 0.4]])
        labels = np.array([0, 1])

        result = create_cluster_dataframe(df, emb_2d, labels)
        assert "embedding" not in result.columns
