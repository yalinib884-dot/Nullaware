"""
backend/rag/chunker.py
Converts extracted metadata into semantic text chunks for embedding.
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _fmt(val: Any, decimals: int = 4) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def build_chunks(metadata: dict) -> list[dict]:
    """
    Convert metadata dict into a list of chunk dicts.
    Each chunk: { 'id': str, 'text': str, 'source': str }

    Chunk types:
        - dataset_summary
        - per-column variable
        - correlations
        - duplicates
        - outliers
    """
    chunks = []

    # ── Dataset Summary Chunk ─────────────────────────────────────────────────
    ds = metadata.get("dataset_summary", {})
    summary_text = (
        f"Dataset Summary:\n"
        f"  Total Rows: {ds.get('n_rows', 'N/A')}\n"
        f"  Total Columns: {ds.get('n_columns', 'N/A')}\n"
        f"  Missing Cells: {ds.get('n_missing_cells', 'N/A')} ({_fmt(ds.get('pct_missing_cells', 0) * 100, 2)}%)\n"
        f"  Duplicate Rows: {ds.get('n_duplicate_rows', 'N/A')} ({_fmt(ds.get('pct_duplicate_rows', 0) * 100, 2)}%)\n"
        f"  Memory Size (bytes): {ds.get('memory_size', 'N/A')}\n"
        f"  Column Types: {ds.get('types', {})}"
    )
    chunks.append({
        "id": "dataset_summary",
        "text": summary_text,
        "source": "table",
        "type": "dataset_summary",
    })

    # ── Per-column Chunks ────────────────────────────────────────────────────
    for col_name, col in metadata.get("variables", {}).items():
        top_vals = col.get("top_values", [])
        top_str = ", ".join(f"{v['value']} ({v['count']})" for v in top_vals[:3]) if top_vals else "N/A"

        col_text = (
            f"Column: {col_name}\n"
            f"  Type: {col.get('type', 'Unknown')}\n"
            f"  Total Values: {col.get('n', 'N/A')}\n"
            f"  Missing Values: {col.get('n_missing', 0)} ({_fmt(col.get('p_missing', 0) * 100, 2)}%)\n"
            f"  Unique Values: {col.get('n_unique', 'N/A')} ({_fmt(col.get('p_unique', 0) * 100, 2)}%)\n"
            f"  Mean: {_fmt(col.get('mean'))}\n"
            f"  Median: {_fmt(col.get('median'))}\n"
            f"  Std Dev: {_fmt(col.get('std'))}\n"
            f"  Min: {_fmt(col.get('min'))}\n"
            f"  Max: {_fmt(col.get('max'))}\n"
            f"  Skewness: {_fmt(col.get('skewness'))}\n"
            f"  Kurtosis: {_fmt(col.get('kurtosis'))}\n"
            f"  IQR: {_fmt(col.get('iqr'))}\n"
            f"  Top Values: {top_str}"
        )
        chunks.append({
            "id": f"variable.{col_name}",
            "text": col_text,
            "source": f"variables.{col_name}",
            "type": "variable",
            "column": col_name,
        })

    # ── Correlations Chunk ────────────────────────────────────────────────────
    corr_pairs = metadata.get("correlations", {}).get("pearson_pairs", [])
    if corr_pairs:
        lines = [f"  {p['col_a']} ↔ {p['col_b']}: {p['pearson']}" for p in corr_pairs[:20]]
        corr_text = "Correlation Analysis (Pearson, |r| ≥ 0.5):\n" + "\n".join(lines)
    else:
        corr_text = "Correlation Analysis: No strong correlations found (|r| ≥ 0.5)."
    chunks.append({
        "id": "correlations",
        "text": corr_text,
        "source": "correlations.pearson",
        "type": "correlations",
    })

    # ── Missing Values Summary Chunk ─────────────────────────────────────────
    mv = metadata.get("missing_values", {})
    if mv:
        lines = [f"  {col}: {v['n_missing']} missing ({_fmt(v['p_missing']*100, 2)}%)" for col, v in list(mv.items())[:15]]
        mv_text = "Missing Values Analysis (columns with missing data):\n" + "\n".join(lines)
    else:
        mv_text = "Missing Values Analysis: No missing values detected in any column."
    chunks.append({
        "id": "missing_values",
        "text": mv_text,
        "source": "missing_values",
        "type": "missing_values",
    })

    # ── Outliers Chunk ────────────────────────────────────────────────────────
    outliers = metadata.get("outliers", {})
    outlier_cols = {col: v for col, v in outliers.items() if v.get("has_outliers")}
    if outlier_cols:
        lines = [
            f"  {col}: IQR={_fmt(v['iqr'], 2)}, bounds=[{_fmt(v['lower_bound'], 2)}, {_fmt(v['upper_bound'], 2)}], actual=[{_fmt(v['min'], 2)}, {_fmt(v['max'], 2)}]"
            for col, v in list(outlier_cols.items())[:10]
        ]
        out_text = "Outlier Analysis (IQR method, columns with potential outliers):\n" + "\n".join(lines)
    else:
        out_text = "Outlier Analysis: No outliers detected using IQR method."
    chunks.append({
        "id": "outliers",
        "text": out_text,
        "source": "outliers",
        "type": "outliers",
    })

    # ── Duplicates Chunk ──────────────────────────────────────────────────────
    dups = metadata.get("duplicates", {})
    dup_text = (
        f"Duplicate Row Analysis:\n"
        f"  Duplicate Rows: {dups.get('n_duplicate_rows', 0)}\n"
        f"  Percentage: {_fmt(dups.get('pct_duplicate_rows', 0) * 100, 2)}%"
    )
    chunks.append({
        "id": "duplicates",
        "text": dup_text,
        "source": "duplicates",
        "type": "duplicates",
    })

    logger.info(f"Built {len(chunks)} chunks from metadata.")
    return chunks