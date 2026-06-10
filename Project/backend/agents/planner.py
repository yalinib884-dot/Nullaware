"""
backend/agents/planner.py
Intent classification and action planning for the NulAware agent.
"""
import re
import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

ActionType = Literal[
    "retrieve_and_answer",
    "generate_visualization",
    "generate_report",
    "column_statistics",
    "missing_values",
    "correlations",
    "outliers",
    "duplicates",
    "data_quality",
    "row_lookup",
    "general",
]

# ── Keyword maps ──────────────────────────────────────────────────────────
VIZ_KEYWORDS = [
    "plot", "chart", "graph", "histogram", "boxplot", "box plot",
    "scatter", "heatmap", "heat map", "bar chart", "pie chart",
    "distribution", "visualize", "visualise",
]
REPORT_KEYWORDS = [
    "report", "pdf", "download", "generate report", "executive summary",
    "full report", "summary report",
]
MISSING_KEYWORDS = [
    "missing", "null", "nan", "empty", "incomplete", "n/a",
]
CORR_KEYWORDS = [
    "correlat", "relationship", "related", "association",
]
OUTLIER_KEYWORDS = [
    "outlier", "anomal", "extreme", "iqr", "unusual",
]
DUP_KEYWORDS = ["duplicate", "duplicated", "repeated row"]
QUALITY_KEYWORDS = ["quality", "score", "health", "grade", "dq score"]

CHART_TYPE_MAP = {
    "histogram": "histogram",
    "boxplot": "boxplot",
    "box plot": "boxplot",
    "scatter": "scatter",
    "heatmap": "heatmap",
    "heat map": "heatmap",
    "bar chart": "bar",
    "bar graph": "bar",
    "pie chart": "pie",
    "pie": "pie",
    "correlation": "heatmap",
}

# Words that are never treated as a lookup value
_SKIP_WORDS = {
    "show", "find", "details", "detail", "where", "with", "whose",
    "value", "values", "is", "the", "of", "equal", "equals", "a",
    "an", "me", "all", "rows", "row", "entries", "records",
}


@dataclass
class AgentPlan:
    action: ActionType
    column: str | None = None
    chart_type: str | None = None
    x_col: str | None = None
    y_col: str | None = None
    reasoning: str = ""
    extra: dict = field(default_factory=dict)


def classify_intent(question: str, available_columns: list[str] = None) -> AgentPlan:
    """
    Classify the user's question and produce a routing plan.

    Args:
        question: Resolved user question (coreferences already replaced).
        available_columns: Column names from the loaded dataset.

    Returns:
        AgentPlan with action and optional column/chart details.
    """
    q = question.lower().strip()
    cols = [c.lower() for c in (available_columns or [])]
    orig_cols = available_columns or []

    # ── Row lookup ──────────────────────────────────────────────────────
    ROW_KEYWORDS = ["show", "find", "details", "where"]
    if any(k in q for k in ROW_KEYWORDS):
        for i, col in enumerate(cols):
            if re.search(rf"\b{re.escape(col)}\b", q):
                words = re.findall(r"\b[\w\.@\-]+\b", question)
                skip = _SKIP_WORDS | {col.lower(), orig_cols[i].lower()}
                value = None
                for w in reversed(words):
                    if w.lower() not in skip:
                        value = w
                        break
                if value:
                    return AgentPlan(
                        action="row_lookup",
                        column=orig_cols[i],   # preserve original case
                        extra={"value": value},
                        reasoning=f"Lookup rows where {orig_cols[i]}={value}",
                    )

    # ── Report ──────────────────────────────────────────────────────────
    if any(kw in q for kw in REPORT_KEYWORDS):
        return AgentPlan(action="generate_report",
                         reasoning="User requested a PDF report.")

    # ── Visualization ────────────────────────────────────────────────────
    if any(kw in q for kw in VIZ_KEYWORDS):
        chart_type = _detect_chart_type(q)
        col = _detect_column(q, orig_cols)
        x_col, y_col = _detect_xy_cols(q, orig_cols)
        return AgentPlan(
            action="generate_visualization",
            column=col,
            chart_type=chart_type,
            x_col=x_col,
            y_col=y_col,
            reasoning=f"User requested a {chart_type or 'auto'} chart.",
        )

    # ── Missing Values ───────────────────────────────────────────────────
    if any(kw in q for kw in MISSING_KEYWORDS):
        col = _detect_column(q, orig_cols)
        return AgentPlan(action="missing_values", column=col,
                         reasoning="Query about missing/null values.")

    # ── Correlations ─────────────────────────────────────────────────────
    if any(kw in q for kw in CORR_KEYWORDS):
        return AgentPlan(action="correlations",
                         reasoning="Query about correlations.")

    # ── Outliers ─────────────────────────────────────────────────────────
    if any(kw in q for kw in OUTLIER_KEYWORDS):
        col = _detect_column(q, orig_cols)
        return AgentPlan(action="outliers", column=col,
                         reasoning="Query about outliers/anomalies.")

    # ── Duplicates ───────────────────────────────────────────────────────
    if any(kw in q for kw in DUP_KEYWORDS):
        return AgentPlan(action="duplicates",
                         reasoning="Query about duplicate rows.")

    # ── Data Quality ─────────────────────────────────────────────────────
    if any(kw in q for kw in QUALITY_KEYWORDS):
        return AgentPlan(action="data_quality",
                         reasoning="Query about overall data quality.")

    # ── Column Statistics ────────────────────────────────────────────────
    if any(kw in q for kw in ["statistic", "mean", "median", "std", "average",
                                "min", "max", "unique", "skew", "variance"]):
        col = _detect_column(q, orig_cols)
        return AgentPlan(action="column_statistics", column=col,
                         reasoning="Query about column statistics.")

    # ── Default: RAG retrieval + Gemini answer ───────────────────────────
    col = _detect_column(q, orig_cols)
    return AgentPlan(action="retrieve_and_answer", column=col,
                     reasoning="General question – will use RAG + LLM.")


def _detect_chart_type(q: str) -> str | None:
    for kw, ctype in CHART_TYPE_MAP.items():
        if kw in q:
            return ctype
    return None


def _detect_column(q: str, cols: list[str]) -> str | None:
    """Find the first column name mentioned in the question (preserves original case)."""
    for col in cols:
        if re.search(rf"\b{re.escape(col)}\b", q, re.IGNORECASE):
            return col
    return None


def _detect_xy_cols(q: str, cols: list[str]) -> tuple[str | None, str | None]:
    """Detect x/y columns for scatter plots."""
    matches = re.findall(
        r"(\w+)\s+(?:vs\.?|versus|against|by|and)\s+(\w+)", q, re.IGNORECASE
    )
    if matches:
        a, b = matches[0]
        x_col = next((c for c in cols if c.lower() == a.lower()), None)
        y_col = next((c for c in cols if c.lower() == b.lower()), None)
        return x_col, y_col
    return None, None