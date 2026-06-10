"""
backend/profiling/extract_profile.py
Extracts structured metadata from a ydata-profiling JSON report.
"""
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def safe_get(d: dict, *keys, default=None):
    """Safely navigate nested dict."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def extract_metadata(json_path: str | Path) -> dict:
    """
    Extract structured metadata from a profile.json file.

    Returns:
        dict with keys:
            - dataset_summary
            - variables (per-column stats)
            - correlations
            - missing_values
            - duplicates
    """
    json_path = Path(json_path)
    logger.info(f"Extracting metadata from {json_path}")

    with open(json_path, "r") as f:
        profile = json.load(f)

    # ── Dataset Summary ───────────────────────────────────────────────────────
    table = profile.get("table", {})
    dataset_summary = {
        "n_rows": safe_get(table, "n", default=0),
        "n_columns": safe_get(table, "n_var", default=0),
        "n_missing_cells": safe_get(table, "n_cells_missing", default=0),
        "pct_missing_cells": safe_get(table, "p_cells_missing", default=0.0),
        "n_duplicate_rows": safe_get(table, "n_duplicates", default=0),
        "pct_duplicate_rows": safe_get(table, "p_duplicates", default=0.0),
        "memory_size": safe_get(table, "memory_size", default=0),
        "types": safe_get(table, "types", default={}),
    }

    # ── Variables (per-column) ────────────────────────────────────────────────
    raw_vars = profile.get("variables", {})
    variables = {}
    for col_name, col_data in raw_vars.items():
        variables[col_name] = {
            "type": safe_get(col_data, "type", default="Unknown"),
            "n": safe_get(col_data, "n", default=0),
            "n_missing": safe_get(col_data, "n_missing", default=0),
            "p_missing": safe_get(col_data, "p_missing", default=0.0),
            "n_unique": safe_get(col_data, "n_unique", default=0),
            "p_unique": safe_get(col_data, "p_unique", default=0.0),
            "mean": safe_get(col_data, "mean", default=None),
            "std": safe_get(col_data, "std", default=None),
            "variance": safe_get(col_data, "variance", default=None),
            "min": safe_get(col_data, "min", default=None),
            "max": safe_get(col_data, "max", default=None),
            "median": safe_get(col_data, "50%", default=safe_get(col_data, "median", default=None)),
            "kurtosis": safe_get(col_data, "kurtosis", default=None),
            "skewness": safe_get(col_data, "skewness", default=None),
            "iqr": safe_get(col_data, "iqr", default=None),
            "is_unique": safe_get(col_data, "is_unique", default=False),
            "top_values": _extract_top_values(col_data),
        }

    # ── Correlations ──────────────────────────────────────────────────────────
    correlations = _extract_correlations(profile)

    # ── Missing Values (ranked) ───────────────────────────────────────────────
    missing_values = {
        col: {
            "n_missing": v["n_missing"],
            "p_missing": v["p_missing"],
        }
        for col, v in variables.items()
        if v["n_missing"] > 0
    }
    missing_values = dict(
        sorted(missing_values.items(), key=lambda x: x[1]["n_missing"], reverse=True)
    )

    # ── Outlier Detection (IQR-based flag) ────────────────────────────────────
    outlier_cols = {}
    for col, v in variables.items():
        if v["type"] in ("Numeric", "Real", "Integer") and v.get("iqr") is not None:
            q1 = safe_get(raw_vars.get(col, {}), "25%", default=None)
            q3 = safe_get(raw_vars.get(col, {}), "75%", default=None)
            if q1 is not None and q3 is not None:
                lower = q1 - 1.5 * v["iqr"]
                upper = q3 + 1.5 * v["iqr"]
                outlier_cols[col] = {
                    "iqr": v["iqr"],
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "min": v["min"],
                    "max": v["max"],
                    "has_outliers": (v["min"] < lower or v["max"] > upper)
                    if v["min"] is not None and v["max"] is not None
                    else False,
                }

    return {
        "dataset_summary": dataset_summary,
        "variables": variables,
        "correlations": correlations,
        "missing_values": missing_values,
        "outliers": outlier_cols,
        "duplicates": {
            "n_duplicate_rows": dataset_summary["n_duplicate_rows"],
            "pct_duplicate_rows": dataset_summary["pct_duplicate_rows"],
        },
    }


def _extract_top_values(col_data: dict) -> list[dict]:
    """Extract top frequent values."""
    top_n = col_data.get("value_counts_without_nan", {})
    if not top_n:
        top_n = col_data.get("value_counts_index_sorted", {})
    if isinstance(top_n, dict):
        items = list(top_n.items())[:5]
        return [{"value": k, "count": v} for k, v in items]
    return []


def _extract_correlations(profile: dict) -> dict:
    """Extract Pearson correlation pairs above threshold 0.5."""
    pearson = safe_get(profile, "correlations", "pearson", default={})
    pairs = []
    if isinstance(pearson, dict):
        for col_a, row in pearson.items():
            if isinstance(row, dict):
                for col_b, val in row.items():
                    if col_a < col_b and isinstance(val, (int, float)) and abs(val) >= 0.5:
                        pairs.append({
                            "col_a": col_a,
                            "col_b": col_b,
                            "pearson": round(float(val), 4),
                        })
    pairs.sort(key=lambda x: abs(x["pearson"]), reverse=True)
    return {"pearson_pairs": pairs}