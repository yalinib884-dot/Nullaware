"""
backend/tools/stats_tool.py
Statistical analysis tools operating on extracted metadata.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_missing_values(metadata: dict) -> dict:
    """
    Return sorted missing value information for all columns.

    Returns:
        { col: { n_missing, p_missing, is_complete } }
    """
    mv = {}
    for col, v in metadata.get("variables", {}).items():
        mv[col] = {
            "n_missing": v.get("n_missing", 0),
            "p_missing": round(v.get("p_missing", 0.0) * 100, 2),
            "is_complete": v.get("n_missing", 0) == 0,
        }
    return dict(sorted(mv.items(), key=lambda x: x[1]["n_missing"], reverse=True))


def get_column_statistics(metadata: dict, column: str = None) -> dict:
    """
    Return descriptive statistics for one or all numeric columns.

    Args:
        metadata: Extracted metadata dict.
        column: Specific column name, or None for all.

    Returns:
        Stats dict for one column, or { col: stats } for all.
    """
    variables = metadata.get("variables", {})
    numeric_types = ("Numeric", "Real", "Integer", "Float")

    def _stats(col_name, col_data):
        return {
            "type": col_data.get("type"),
            "n": col_data.get("n"),
            "n_missing": col_data.get("n_missing"),
            "p_missing": round(col_data.get("p_missing", 0) * 100, 2),
            "n_unique": col_data.get("n_unique"),
            "mean": col_data.get("mean"),
            "median": col_data.get("median"),
            "std": col_data.get("std"),
            "min": col_data.get("min"),
            "max": col_data.get("max"),
            "skewness": col_data.get("skewness"),
            "kurtosis": col_data.get("kurtosis"),
            "iqr": col_data.get("iqr"),
        }

    if column:
        if column not in variables:
            return {"error": f"Column '{column}' not found."}
        return _stats(column, variables[column])

    return {
        col: _stats(col, data)
        for col, data in variables.items()
        if data.get("type") in numeric_types
    }


def get_correlations(metadata: dict, threshold: float = 0.5) -> list[dict]:
    """
    Return Pearson correlation pairs above a threshold.

    Args:
        metadata: Extracted metadata dict.
        threshold: Minimum absolute correlation value.

    Returns:
        List of { col_a, col_b, pearson } sorted by |pearson| descending.
    """
    pairs = metadata.get("correlations", {}).get("pearson_pairs", [])
    return [p for p in pairs if abs(p.get("pearson", 0)) >= threshold]


def get_outliers(metadata: dict) -> dict:
    """
    Return IQR-based outlier information for numeric columns.

    Returns:
        { col: { iqr, lower_bound, upper_bound, min, max, has_outliers } }
    """
    return {
        col: v
        for col, v in metadata.get("outliers", {}).items()
        if v.get("has_outliers")
    }


def compute_data_quality_score(metadata: dict) -> dict:
    """
    Compute a 0-100 data quality score based on:
      - Missing values (40 pts)
      - Duplicates (20 pts)
      - Outliers (20 pts)
      - Completeness (20 pts)
    """
    ds = metadata.get("dataset_summary", {})
    pct_missing = ds.get("pct_missing_cells", 0.0)
    pct_dup = ds.get("pct_duplicate_rows", 0.0)

    outlier_cols = [v for v in metadata.get("outliers", {}).values() if v.get("has_outliers")]
    total_cols = max(ds.get("n_columns", 1), 1)
    pct_outlier_cols = len(outlier_cols) / total_cols

    missing_score = max(0, 40 * (1 - pct_missing * 5))
    dup_score = max(0, 20 * (1 - pct_dup * 5))
    outlier_score = max(0, 20 * (1 - pct_outlier_cols))
    completeness_score = 20 * (1 - pct_missing)

    total = round(missing_score + dup_score + outlier_score + completeness_score, 1)

    return {
        "total_score": min(total, 100),
        "components": {
            "missing_values": round(missing_score, 1),
            "duplicates": round(dup_score, 1),
            "outliers": round(outlier_score, 1),
            "completeness": round(completeness_score, 1),
        },
        "grade": "A" if total >= 85 else "B" if total >= 70 else "C" if total >= 50 else "D",
    }


def generate_automatic_insights(metadata: dict) -> list[str]:
    """
    Generate a list of plain-English insight strings.
    """
    insights = []
    ds = metadata.get("dataset_summary", {})

    # Missing values
    mv = get_missing_values(metadata)
    high_missing = [(col, v) for col, v in mv.items() if v["p_missing"] > 5]
    if high_missing:
        top = high_missing[0]
        insights.append(
            f"⚠️ '{top[0]}' has the most missing values: {top[1]['n_missing']} ({top[1]['p_missing']}%)."
        )

    # Duplicates
    dup_pct = round(ds.get("pct_duplicate_rows", 0) * 100, 2)
    if dup_pct > 0:
        insights.append(
            f"🔁 {ds.get('n_duplicate_rows', 0)} duplicate rows detected ({dup_pct}% of data)."
        )

    # Correlations
    corrs = get_correlations(metadata, threshold=0.7)
    if corrs:
        top_corr = corrs[0]
        insights.append(
            f"🔗 Strong correlation between '{top_corr['col_a']}' and '{top_corr['col_b']}' (r={top_corr['pearson']})."
        )

    # Outliers
    outlier_cols = get_outliers(metadata)
    if outlier_cols:
        cols_str = ", ".join(list(outlier_cols.keys())[:3])
        insights.append(f"📊 Outliers detected in: {cols_str}.")

    # Data quality
    score = compute_data_quality_score(metadata)
    insights.append(
        f"🏆 Data Quality Score: {score['total_score']}/100 (Grade: {score['grade']})."
    )

    # Skewed columns
    for col, v in metadata.get("variables", {}).items():
        skew = v.get("skewness")
        if skew is not None and abs(skew) > 1:
            direction = "right" if skew > 0 else "left"
            insights.append(f"📈 '{col}' is heavily {direction}-skewed (skewness={round(skew, 2)}).")
            break  # just report first

    return insights