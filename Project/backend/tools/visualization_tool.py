"""
backend/tools/visualization_tool.py
Dynamic Plotly chart generation from a pandas DataFrame and metadata.
"""
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from config import CHARTS_DIR

logger = logging.getLogger(__name__)

# ── Colour palette ─────────────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Bold


def _save_fig(fig: go.Figure, name: str) -> str:
    """Persist chart as PNG and return the file path."""
    path = CHARTS_DIR / f"{name}.png"
    try:
        fig.write_image(str(path), scale=2)
    except Exception as e:
        logger.warning(f"Could not save chart PNG ({name}): {e}")
    return str(path)


# ── Individual chart functions ─────────────────────────────────────────────

def generate_histogram(df: pd.DataFrame, column: str, bins: int = 30) -> go.Figure:
    """Histogram with KDE overlay for a numeric column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    fig = px.histogram(
        df,
        x=column,
        nbins=bins,
        marginal="box",
        title=f"Distribution of {column}",
        color_discrete_sequence=[PALETTE[0]],
        template="plotly_white",
    )
    fig.update_layout(bargap=0.05)
    _save_fig(fig, f"histogram_{column}")
    return fig


def generate_boxplot(df: pd.DataFrame, column: str, group_by: str = None) -> go.Figure:
    """Box plot — optionally grouped by a categorical column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    fig = px.box(
        df,
        y=column,
        x=group_by if group_by and group_by in df.columns else None,
        title=f"Box Plot: {column}" + (f" by {group_by}" if group_by else ""),
        color=group_by if group_by and group_by in df.columns else None,
        color_discrete_sequence=PALETTE,
        template="plotly_white",
        points="outliers",
    )
    _save_fig(fig, f"boxplot_{column}")
    return fig


def generate_heatmap(df: pd.DataFrame, metadata: dict = None) -> go.Figure:
    """Pearson correlation heatmap for all numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric columns found for correlation heatmap.")

    corr = numeric_df.corr()
    fig = px.imshow(
        corr,
        title="Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
        template="plotly_white",
    )
    fig.update_layout(height=max(400, 60 * len(corr.columns)))
    _save_fig(fig, "heatmap_correlation")
    return fig


def generate_scatter_plot(
    df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None
) -> go.Figure:
    """Scatter plot between two numeric columns with optional color grouping."""
    for col in [x_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col if color_col and color_col in df.columns else None,
        title=f"Scatter Plot: {x_col} vs {y_col}",
        trendline="ols",
        color_discrete_sequence=PALETTE,
        template="plotly_white",
        opacity=0.7,
    )
    _save_fig(fig, f"scatter_{x_col}_{y_col}")
    return fig


def generate_bar_chart(
    df: pd.DataFrame, column: str, top_n: int = 15, orientation: str = "v"
) -> go.Figure:
    """Bar chart of value counts for a categorical column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    counts = df[column].value_counts().head(top_n).reset_index()
    counts.columns = [column, "count"]

    if orientation == "h":
        fig = px.bar(
            counts,
            x="count",
            y=column,
            orientation="h",
            title=f"Top {top_n} Values: {column}",
            color="count",
            color_continuous_scale="Blues",
            template="plotly_white",
        )
    else:
        fig = px.bar(
            counts,
            x=column,
            y="count",
            title=f"Top {top_n} Values: {column}",
            color="count",
            color_continuous_scale="Blues",
            template="plotly_white",
        )
    _save_fig(fig, f"bar_{column}")
    return fig


def generate_pie_chart(df: pd.DataFrame, column: str, top_n: int = 10) -> go.Figure:
    """Pie chart of value proportions for a categorical column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    counts = df[column].value_counts().head(top_n)
    fig = px.pie(
        values=counts.values,
        names=counts.index.astype(str),
        title=f"Distribution: {column}",
        color_discrete_sequence=PALETTE,
        template="plotly_white",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    _save_fig(fig, f"pie_{column}")
    return fig


def generate_missing_values_chart(metadata: dict) -> go.Figure:
    """Horizontal bar chart of missing value percentages per column."""
    mv = {
        col: v.get("p_missing", 0) * 100
        for col, v in metadata.get("variables", {}).items()
        if v.get("n_missing", 0) > 0
    }
    if not mv:
        fig = go.Figure()
        fig.add_annotation(text="No missing values!", showarrow=False, font_size=16)
        return fig

    mv_sorted = dict(sorted(mv.items(), key=lambda x: x[1], reverse=True))
    fig = px.bar(
        x=list(mv_sorted.values()),
        y=list(mv_sorted.keys()),
        orientation="h",
        title="Missing Values by Column (%)",
        labels={"x": "Missing %", "y": "Column"},
        color=list(mv_sorted.values()),
        color_continuous_scale="Reds",
        template="plotly_white",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    _save_fig(fig, "missing_values_bar")
    return fig


def generate_overview_dashboard(df: pd.DataFrame, metadata: dict) -> go.Figure:
    """
    4-panel overview dashboard:
      top-left: missing values bar
      top-right: data types pie
      bottom-left: top numeric distributions (box)
      bottom-right: correlation heatmap (if available)
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Missing Values (%)",
            "Column Types",
            "Numeric Distributions",
            "Correlation Heatmap",
        ),
        specs=[
            [{"type": "bar"}, {"type": "pie"}],
            [{"type": "box"}, {"type": "heatmap"}],
        ],
    )

    # Panel 1: Missing values
    mv = {
        col: round(v.get("p_missing", 0) * 100, 2)
        for col, v in metadata.get("variables", {}).items()
        if v.get("n_missing", 0) > 0
    }
    if mv:
        mv_s = dict(sorted(mv.items(), key=lambda x: x[1], reverse=True)[:10])
        fig.add_trace(
            go.Bar(x=list(mv_s.values()), y=list(mv_s.keys()),
                   orientation="h", marker_color="#e74c3c", showlegend=False),
            row=1, col=1,
        )

    # Panel 2: Types pie
    types = metadata.get("dataset_summary", {}).get("types", {})
    if types:
        fig.add_trace(
            go.Pie(labels=list(types.keys()), values=list(types.values()),
                   showlegend=True),
            row=1, col=2,
        )

    # Panel 3: Numeric boxplots (first 5 numeric cols)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
    for i, col in enumerate(numeric_cols):
        fig.add_trace(
            go.Box(y=df[col].dropna(), name=col,
                   marker_color=PALETTE[i % len(PALETTE)], showlegend=False),
            row=2, col=1,
        )

    # Panel 4: Correlation heatmap
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().values
        fig.add_trace(
            go.Heatmap(
                z=corr,
                x=numeric_cols,
                y=numeric_cols,
                colorscale="RdBu_r",
                zmin=-1, zmax=1,
                showscale=True,
            ),
            row=2, col=2,
        )

    fig.update_layout(
        height=700,
        title_text="Dataset Overview Dashboard",
        template="plotly_white",
    )
    _save_fig(fig, "overview_dashboard")
    return fig


# ── Smart chart dispatcher ─────────────────────────────────────────────────

def auto_chart(df: pd.DataFrame, metadata: dict, column: str = None,
               chart_type: str = "auto") -> go.Figure:
    """
    Automatically select the best chart type for a column based on its dtype.
    chart_type: 'auto' | 'histogram' | 'boxplot' | 'bar' | 'pie' | 'heatmap'
    """
    if chart_type == "heatmap" or (chart_type == "auto" and column is None):
        return generate_heatmap(df, metadata)

    if column is None:
        raise ValueError("Column name required for this chart type.")

    col_meta = metadata.get("variables", {}).get(column, {})
    col_type = col_meta.get("type", "Unknown")

    if chart_type == "auto":
        n_unique = col_meta.get("n_unique", 0)
        if col_type in ("Numeric", "Real", "Integer", "Float"):
            chart_type = "histogram"
        elif n_unique <= 20:
            chart_type = "pie"
        else:
            chart_type = "bar"

    dispatch = {
        "histogram": lambda: generate_histogram(df, column),
        "boxplot": lambda: generate_boxplot(df, column),
        "bar": lambda: generate_bar_chart(df, column),
        "pie": lambda: generate_pie_chart(df, column),
    }

    fn = dispatch.get(chart_type)
    if fn:
        return fn()
    raise ValueError(f"Unknown chart type: {chart_type}")