"""
tests/test_stats_tool.py
Unit tests for the statistics tool.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from backend.tools.stats_tool import (
    get_missing_values, get_column_statistics, get_correlations,
    get_outliers, compute_data_quality_score, generate_automatic_insights,
)

# ── Minimal metadata fixture ──────────────────────────────────────────────
METADATA = {
    "dataset_summary": {
        "n_rows": 100,
        "n_columns": 4,
        "n_missing_cells": 10,
        "pct_missing_cells": 0.025,
        "n_duplicate_rows": 2,
        "pct_duplicate_rows": 0.02,
        "memory_size": 5000,
        "types": {"Numeric": 3, "Categorical": 1},
    },
    "variables": {
        "age": {
            "type": "Numeric",
            "n": 100,
            "n_missing": 8,
            "p_missing": 0.08,
            "n_unique": 50,
            "p_unique": 0.5,
            "mean": 35.0,
            "median": 34.0,
            "std": 10.0,
            "min": 18.0,
            "max": 65.0,
            "skewness": 0.3,
            "kurtosis": -0.2,
            "iqr": 15.0,
        },
        "salary": {
            "type": "Numeric",
            "n": 100,
            "n_missing": 2,
            "p_missing": 0.02,
            "n_unique": 90,
            "p_unique": 0.9,
            "mean": 75000.0,
            "median": 70000.0,
            "std": 20000.0,
            "min": 30000.0,
            "max": 200000.0,
            "skewness": 1.5,
            "kurtosis": 2.1,
            "iqr": 25000.0,
        },
        "department": {
            "type": "Categorical",
            "n": 100,
            "n_missing": 0,
            "p_missing": 0.0,
            "n_unique": 4,
            "p_unique": 0.04,
        },
    },
    "correlations": {
        "pearson_pairs": [
            {"col_a": "age", "col_b": "salary", "pearson": 0.78},
            {"col_a": "age", "col_b": "experience", "pearson": 0.95},
        ]
    },
    "missing_values": {
        "age": {"n_missing": 8, "p_missing": 8.0},
        "salary": {"n_missing": 2, "p_missing": 2.0},
    },
    "outliers": {
        "salary": {
            "iqr": 25000.0,
            "lower_bound": -7500.0,
            "upper_bound": 162500.0,
            "min": 30000.0,
            "max": 200000.0,
            "has_outliers": True,
        },
        "age": {
            "iqr": 15.0,
            "lower_bound": 0.0,
            "upper_bound": 80.0,
            "min": 18.0,
            "max": 65.0,
            "has_outliers": False,
        },
    },
    "duplicates": {"n_duplicate_rows": 2, "pct_duplicate_rows": 0.02},
}


class TestGetMissingValues:
    def test_returns_dict(self):
        result = get_missing_values(METADATA)
        assert isinstance(result, dict)

    def test_sorted_by_missing_desc(self):
        result = get_missing_values(METADATA)
        values = list(result.values())
        for i in range(len(values) - 1):
            assert values[i]["n_missing"] >= values[i + 1]["n_missing"]

    def test_age_has_most_missing(self):
        result = get_missing_values(METADATA)
        keys = list(result.keys())
        assert keys[0] == "age"


class TestGetColumnStatistics:
    def test_all_numeric(self):
        result = get_column_statistics(METADATA)
        assert "age" in result
        assert "salary" in result
        assert "department" not in result  # categorical excluded

    def test_specific_column(self):
        result = get_column_statistics(METADATA, column="age")
        assert result["mean"] == 35.0
        assert result["type"] == "Numeric"

    def test_missing_column(self):
        result = get_column_statistics(METADATA, column="nonexistent")
        assert "error" in result


class TestGetCorrelations:
    def test_threshold_filter(self):
        result = get_correlations(METADATA, threshold=0.8)
        for p in result:
            assert abs(p["pearson"]) >= 0.8

    def test_default_threshold(self):
        result = get_correlations(METADATA)
        assert len(result) == 2  # both pairs are >= 0.5

    def test_structure(self):
        result = get_correlations(METADATA)
        for p in result:
            assert "col_a" in p
            assert "col_b" in p
            assert "pearson" in p


class TestGetOutliers:
    def test_only_has_outliers(self):
        result = get_outliers(METADATA)
        # Only salary has has_outliers=True
        assert "salary" in result
        assert "age" not in result

    def test_structure(self):
        result = get_outliers(METADATA)
        for col, v in result.items():
            assert "iqr" in v
            assert "has_outliers" in v
            assert v["has_outliers"] is True


class TestDataQualityScore:
    def test_returns_dict(self):
        result = compute_data_quality_score(METADATA)
        assert "total_score" in result
        assert "grade" in result
        assert "components" in result

    def test_score_range(self):
        result = compute_data_quality_score(METADATA)
        assert 0 <= result["total_score"] <= 100

    def test_grade_valid(self):
        result = compute_data_quality_score(METADATA)
        assert result["grade"] in ("A", "B", "C", "D")

    def test_perfect_dataset(self):
        perfect_meta = {
            "dataset_summary": {
                "n_rows": 1000,
                "n_columns": 5,
                "n_missing_cells": 0,
                "pct_missing_cells": 0.0,
                "n_duplicate_rows": 0,
                "pct_duplicate_rows": 0.0,
            },
            "variables": {},
            "outliers": {},
        }
        result = compute_data_quality_score(perfect_meta)
        assert result["grade"] == "A"


class TestAutomaticInsights:
    def test_returns_list(self):
        result = generate_automatic_insights(METADATA)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_insights_are_strings(self):
        result = generate_automatic_insights(METADATA)
        for insight in result:
            assert isinstance(insight, str)
            assert len(insight) > 10

    def test_contains_quality_score(self):
        result = generate_automatic_insights(METADATA)
        combined = " ".join(result)
        assert "Quality Score" in combined or "quality" in combined.lower()