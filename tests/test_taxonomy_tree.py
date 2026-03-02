"""Tests for shared/utils/taxonomy_tree.py."""

import numpy as np
import pandas as pd
import pytest

from shared.utils.taxonomy_tree import (
    build_taxonomic_tree,
    format_tree_string,
    get_total_count,
    get_tree_statistics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_taxonomy_df(rows):
    """Create DataFrame from list of (kingdom, phylum, class, order, family, genus, species) tuples."""
    cols = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# build_taxonomic_tree
# ---------------------------------------------------------------------------

class TestBuildTaxonomicTree:
    def test_basic_nesting(self):
        df = _make_taxonomy_df([
            ("Animalia", "Chordata", "Mammalia", "Carnivora", "Felidae", "Felis", "F. catus"),
            ("Animalia", "Chordata", "Mammalia", "Carnivora", "Felidae", "Felis", "F. catus"),
            ("Animalia", "Chordata", "Aves", "Passeriformes", "Passeridae", "Passer", "P. domesticus"),
        ])
        tree = build_taxonomic_tree(df)
        assert "Animalia" in tree
        assert tree["Animalia"]["Chordata"]["Mammalia"]["Carnivora"]["Felidae"]["Felis"]["F. catus"] == 2

    def test_nan_kingdom_excluded(self):
        df = _make_taxonomy_df([
            (np.nan, "Chordata", "Mammalia", "Carnivora", "Felidae", "Felis", "F. catus"),
            ("Animalia", "Chordata", "Aves", "Passeriformes", "Passeridae", "Passer", "P. domesticus"),
        ])
        tree = build_taxonomic_tree(df)
        assert get_total_count(tree) == 1

    def test_nan_lower_level_becomes_unknown(self):
        df = _make_taxonomy_df([
            ("Animalia", "Chordata", np.nan, np.nan, np.nan, np.nan, np.nan),
        ])
        tree = build_taxonomic_tree(df)
        assert "Unknown" in tree["Animalia"]["Chordata"]

    def test_empty_dataframe(self):
        df = _make_taxonomy_df([])
        tree = build_taxonomic_tree(df)
        assert tree == {}


# ---------------------------------------------------------------------------
# get_total_count
# ---------------------------------------------------------------------------

class TestGetTotalCount:
    def test_int_leaf(self):
        assert get_total_count(5) == 5

    def test_nested_dict(self):
        tree = {"a": {"b": 3, "c": 2}, "d": 1}
        assert get_total_count(tree) == 6

    def test_empty_dict(self):
        assert get_total_count({}) == 0

    def test_non_int_non_dict(self):
        assert get_total_count("invalid") == 0


# ---------------------------------------------------------------------------
# format_tree_string
# ---------------------------------------------------------------------------

class TestFormatTreeString:
    def test_max_depth_truncation(self):
        df = _make_taxonomy_df([
            ("Animalia", "Chordata", "Mammalia", "Carnivora", "Felidae", "Felis", "F. catus"),
        ])
        tree = build_taxonomic_tree(df)
        output = format_tree_string(tree, max_depth=2)
        # Should show kingdom and phylum but not deeper
        assert "Animalia" in output
        assert "Chordata" in output
        assert "Mammalia" not in output

    def test_min_count_filtering(self):
        df = _make_taxonomy_df([
            ("Animalia", "Chordata", "Mammalia", "Carnivora", "Felidae", "Felis", "F. catus"),
            ("Animalia", "Chordata", "Mammalia", "Carnivora", "Felidae", "Felis", "F. catus"),
            ("Plantae", "Tracheophyta", "Magnoliopsida", "Rosales", "Rosaceae", "Rosa", "R. gallica"),
        ])
        tree = build_taxonomic_tree(df)
        output = format_tree_string(tree, min_count=2)
        assert "Animalia" in output
        # Plantae has count 1, should be filtered out
        assert "Plantae" not in output

    def test_tree_connector_chars(self):
        df = _make_taxonomy_df([
            ("Animalia", "Chordata", "Mammalia", "Carnivora", "Felidae", "Felis", "F. catus"),
            ("Animalia", "Chordata", "Aves", "Passeriformes", "Passeridae", "Passer", "P. domesticus"),
        ])
        tree = build_taxonomic_tree(df)
        output = format_tree_string(tree)
        # Should contain tree-drawing characters
        assert any(c in output for c in ["├──", "└──"])


# ---------------------------------------------------------------------------
# get_tree_statistics
# ---------------------------------------------------------------------------

class TestGetTreeStatistics:
    def test_counts(self):
        df = _make_taxonomy_df([
            ("Animalia", "Chordata", "Mammalia", "Carnivora", "Felidae", "Felis", "F. catus"),
            ("Animalia", "Chordata", "Aves", "Passeriformes", "Passeridae", "Passer", "P. domesticus"),
        ])
        tree = build_taxonomic_tree(df)
        stats = get_tree_statistics(tree)
        assert stats["total_records"] == 2
        assert stats["kingdoms"] == 1
        assert stats["species"] == 2

    def test_empty_tree(self):
        stats = get_tree_statistics({})
        assert stats["total_records"] == 0
        assert stats["kingdoms"] == 0
