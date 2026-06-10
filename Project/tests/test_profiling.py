"""
tests/test_profiling.py
Unit tests for the profiling and metadata extraction pipeline.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
import pandas as pd
import pytest
from pathlib import Path

from backend.profiling.extract_profile import extract_metadata, safe_get
from backend.rag.chunker import build_chunks


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 30, None, 45, 22, 60, 35, None, 28, 50],
        "salary": [50000, 75000, 90000, 120000, 45000, 95000, 80000, 70000, 55000, 110000],
        "department": ["HR", "IT", "IT", "Finance", "HR", "IT", "Finance", "HR", "IT", "Finance"],
        "score": [0.8, 0.9, 0.7, 0.95, 0.6, 0.88, 0.75, 0.82, 0.91, 0.78],
    })


@pytest.fixture
def minimal_profile_json(tmp_path):
    """Minimal profile.json that mimics ydata-profiling output."""
    profile = {
        "table": {
            "n": 10,
            "n_var": 4,
            "n_cells_missing": 2,
            "p_cells_missing": 0.05,
            "n_duplicates": 0,
            "p_duplicates": 0.0,
            "memory_size": 1234,
            "types": {"Numeric": 3, "Categorical": 1},
        },
        "variables": {
            "age": {
                "type": "Numeric",
                "n": 10,
                "n_missing": 2,
                "p_missing": 0.2,
                "n_unique": 8,
                "p_unique": 0.8,
                "mean": 37.5,
                "std": 12.3,
                "variance": 151.29,
                "min": 22,
                "max": 60,
                "50%": 33,
                "kurtosis": -0.5,
                "skewness": 0.4,
                "iqr": 17.5,
                "25%": 27.5,
                "75%": 45.0,
            },
            "salary": {
                "type": "Numeric",
                "n": 10,
                "n_missing": 0,
                "p_missing": 0.0,
                "n_unique": 10,
                "p_unique": 1.0,
                "mean": 79000,
                "std": 24000,
                "variance": 576000000,
                "min": 45000,
                "max": 120000,
                "50%": 77500,
                "kurtosis": -1.1,
                "skewness": 0.2,
                "iqr": 37500,
                "25%": 56250,
                "75%": 93750,
            },
            "department": {
                "type": "Categorical",
                "n": 10,
                "n_missing": 0,
                "p_missing": 0.0,
                "n_unique": 3,
                "p_unique": 0.3,
                "value_counts_without_nan": {"IT": 4, "HR": 3, "Finance": 3},
            },
        },
        "correlations": {
            "pearson": {
                "age": {"salary": 0.72, "score": 0.55},
                "salary": {"age": 0.72, "score": 0.48},
                "score": {"age": 0.55, "salary": 0.48},
            }
        },
    }
    json_path = tmp_path / "test_profile.json"
    json_path.write_text(json.dumps(profile))
    return json_path


# ── Tests ──────────────────────────────────────────────────────────────────

class TestSafeGet:
    def test_basic(self):
        assert safe_get({"a": {"b": 1}}, "a", "b") == 1

    def test_missing_key(self):
        assert safe_get({"a": 1}, "b", default="x") == "x"

    def test_non_dict(self):
        assert safe_get("not_a_dict", "key", default=None) is None


class TestExtractMetadata:
    def test_dataset_summary(self, minimal_profile_json):
        meta = extract_metadata(minimal_profile_json)
        ds = meta["dataset_summary"]
        assert ds["n_rows"] == 10
        assert ds["n_columns"] == 4
        assert ds["n_missing_cells"] == 2

    def test_variables(self, minimal_profile_json):
        meta = extract_metadata(minimal_profile_json)
        assert "age" in meta["variables"]
        assert meta["variables"]["age"]["n_missing"] == 2
        assert meta["variables"]["age"]["mean"] == 37.5

    def test_missing_values_sorted(self, minimal_profile_json):
        meta = extract_metadata(minimal_profile_json)
        mv_keys = list(meta["missing_values"].keys())
        # age has missing, salary and department don't
        assert "age" in mv_keys

    def test_correlations(self, minimal_profile_json):
        meta = extract_metadata(minimal_profile_json)
        pairs = meta["correlations"]["pearson_pairs"]
        assert len(pairs) > 0
        # age-salary should be above 0.5
        age_sal = next((p for p in pairs if
                        {p["col_a"], p["col_b"]} == {"age", "salary"}), None)
        assert age_sal is not None
        assert abs(age_sal["pearson"]) >= 0.5

    def test_outlier_detection(self, minimal_profile_json):
        meta = extract_metadata(minimal_profile_json)
        # age: iqr=17.5, lower=27.5-26.25=1.25, upper=45+26.25=71.25, min=22 → outlier
        assert "age" in meta["outliers"]


class TestChunker:
    def test_chunk_count(self, minimal_profile_json):
        meta = extract_metadata(minimal_profile_json)
        chunks = build_chunks(meta)
        # Should have: dataset_summary + 3 variables + correlations + mv + outliers + duplicates
        assert len(chunks) >= 7

    def test_chunk_ids_unique(self, minimal_profile_json):
        meta = extract_metadata(minimal_profile_json)
        chunks = build_chunks(meta)
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_variable_chunk_content(self, minimal_profile_json):
        meta = extract_metadata(minimal_profile_json)
        chunks = build_chunks(meta)
        age_chunk = next((c for c in chunks if c["id"] == "variable.age"), None)
        assert age_chunk is not None
        assert "age" in age_chunk["text"].lower()
        assert "37.5" in age_chunk["text"]  # mean

    def test_all_required_fields(self, minimal_profile_json):
        meta = extract_metadata(minimal_profile_json)
        chunks = build_chunks(meta)
        for c in chunks:
            assert "id" in c
            assert "text" in c
            assert "source" in c
            assert len(c["text"]) > 10