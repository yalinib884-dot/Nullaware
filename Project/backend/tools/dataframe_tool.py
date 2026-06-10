"""
backend/tools/dataframe_tool.py
Direct DataFrame row-lookup utilities used by the row_lookup agent action.
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def find_column(df: pd.DataFrame, name: str) -> str | None:
    """Case-insensitive column name lookup. Returns actual column name or None."""
    for col in df.columns:
        if col.lower() == name.lower():
            return col
    return None


def find_rows(df: pd.DataFrame, column: str, value: str, limit: int = 20) -> list[dict]:
    """
    Find rows where `column` matches `value` (case-insensitive string match).

    Args:
        df:     Source DataFrame.
        column: Column name to filter on (case-insensitive lookup performed).
        value:  Value to match (compared as lowercase strings).
        limit:  Maximum number of rows to return.

    Returns:
        List of row dicts (empty list if no match or column not found).
    """
    # Resolve actual column name
    actual_col = find_column(df, column)
    if actual_col is None:
        logger.warning(f"find_rows: column '{column}' not found in DataFrame.")
        return []

    try:
        mask = df[actual_col].astype(str).str.strip().str.lower() == str(value).strip().lower()
        matches = df[mask].head(limit)
        if matches.empty:
            return []
        return matches.to_dict("records")
    except Exception as e:
        logger.error(f"find_rows error: {e}")
        return []