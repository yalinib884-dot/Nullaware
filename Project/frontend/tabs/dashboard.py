"""
frontend/pages/dashboard.py
Visualization Dashboard tab – interactive Plotly charts.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import streamlit as st
import pandas as pd

from backend.tools.visualization_tool import (
    generate_histogram, generate_boxplot, generate_heatmap,
    generate_scatter_plot, generate_bar_chart, generate_pie_chart,
    generate_missing_values_chart, generate_overview_dashboard,
)
from backend.tools.stats_tool import compute_data_quality_score


def render_dashboard_tab():
    if not st.session_state.get("dataset_name"):
        st.info("👆 Please upload a dataset first in the **Upload Dataset** tab.")
        return

    dataset_name = st.session_state["dataset_name"]
    df: pd.DataFrame = st.session_state["df"]
    metadata = st.session_state["metadata"]

    st.markdown(f"### 📊 Visualization Dashboard — **{dataset_name}**")

    # ── Overview row ──────────────────────────────────────────────────────
    quality = compute_data_quality_score(metadata)
    ds = metadata.get("dataset_summary", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{ds.get('n_rows', 0):,}")
    c2.metric("Columns", ds.get("n_columns", 0))
    c3.metric("Missing %", f"{round(ds.get('pct_missing_cells', 0)*100, 1)}%")
    c4.metric("Duplicates", ds.get("n_duplicate_rows", 0))
    c5.metric("Quality", f"{quality['total_score']}/100 ({quality['grade']})")

    st.markdown("---")

    # ── Chart type selector ───────────────────────────────────────────────
    all_cols = list(df.columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    chart_tabs = st.tabs([
        "🌐 Overview", "📈 Histogram", "📦 Boxplot",
        "🔥 Heatmap", "🔵 Scatter", "📊 Bar Chart", "🥧 Pie Chart",
        "❓ Missing Values",
    ])

    # ── Overview ──────────────────────────────────────────────────────────
    with chart_tabs[0]:
        st.markdown("##### Dataset Overview Dashboard")
        try:
            fig = generate_overview_dashboard(df, metadata)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate overview: {e}")

    # ── Histogram ─────────────────────────────────────────────────────────
    with chart_tabs[1]:
        st.markdown("##### Histogram")
        if not numeric_cols:
            st.warning("No numeric columns found.")
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                h_col = st.selectbox("Select column", numeric_cols, key="hist_col")
            with col2:
                h_bins = st.slider("Bins", 10, 100, 30, key="hist_bins")

            try:
                fig = generate_histogram(df, h_col, bins=h_bins)
                st.plotly_chart(fig, use_container_width=True)
                _show_col_stats(metadata, h_col)
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Boxplot ───────────────────────────────────────────────────────────
    with chart_tabs[2]:
        st.markdown("##### Box Plot")
        if not numeric_cols:
            st.warning("No numeric columns found.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                b_col = st.selectbox("Numeric column", numeric_cols, key="box_col")
            with col2:
                b_group_opts = ["(None)"] + categorical_cols
                b_group = st.selectbox("Group by (optional)", b_group_opts, key="box_group")
            b_group = None if b_group == "(None)" else b_group

            try:
                fig = generate_boxplot(df, b_col, group_by=b_group)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Heatmap ───────────────────────────────────────────────────────────
    with chart_tabs[3]:
        st.markdown("##### Correlation Heatmap")
        try:
            fig = generate_heatmap(df, metadata)
            st.plotly_chart(fig, use_container_width=True)

            # Top pairs table
            pairs = metadata.get("correlations", {}).get("pearson_pairs", [])
            if pairs:
                st.markdown("**Top Correlated Pairs (|r| ≥ 0.5)**")
                pairs_df = pd.DataFrame(pairs[:10])
                st.dataframe(pairs_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

    # ── Scatter ───────────────────────────────────────────────────────────
    with chart_tabs[4]:
        st.markdown("##### Scatter Plot")
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for scatter plot.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                s_x = st.selectbox("X axis", numeric_cols, key="scatter_x")
            with col2:
                s_y = st.selectbox("Y axis", numeric_cols,
                                   index=min(1, len(numeric_cols)-1), key="scatter_y")
            with col3:
                s_color_opts = ["(None)"] + categorical_cols
                s_color = st.selectbox("Color by", s_color_opts, key="scatter_color")
            s_color = None if s_color == "(None)" else s_color

            try:
                fig = generate_scatter_plot(df, s_x, s_y, color_col=s_color)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Bar Chart ─────────────────────────────────────────────────────────
    with chart_tabs[5]:
        st.markdown("##### Bar Chart")
        bar_cols = categorical_cols + [c for c in numeric_cols
                                       if df[c].nunique() <= 50]
        if not bar_cols:
            st.warning("No suitable columns found.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                br_col = st.selectbox("Column", bar_cols, key="bar_col")
            with col2:
                br_n = st.slider("Top N values", 5, 30, 15, key="bar_n")
            with col3:
                br_orient = st.radio("Orientation", ["Vertical", "Horizontal"],
                                     key="bar_orient", horizontal=True)

            try:
                fig = generate_bar_chart(df, br_col, top_n=br_n,
                                         orientation="h" if "Horizontal" in br_orient else "v")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Pie Chart ─────────────────────────────────────────────────────────
    with chart_tabs[6]:
        st.markdown("##### Pie Chart")
        pie_cols = categorical_cols + [c for c in numeric_cols if df[c].nunique() <= 20]
        if not pie_cols:
            st.warning("No suitable categorical columns found.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                pi_col = st.selectbox("Column", pie_cols, key="pie_col")
            with col2:
                pi_n = st.slider("Max slices", 3, 20, 10, key="pie_n")

            try:
                fig = generate_pie_chart(df, pi_col, top_n=pi_n)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Missing Values ────────────────────────────────────────────────────
    with chart_tabs[7]:
        st.markdown("##### Missing Values Analysis")
        try:
            fig = generate_missing_values_chart(metadata)
            st.plotly_chart(fig, use_container_width=True)

            mv = metadata.get("missing_values", {})
            if mv:
                mv_df = pd.DataFrame([
                    {"Column": col, "Missing Count": v["n_missing"],
                     "Missing %": f"{v['p_missing']*100:.2f}%"}
                    for col, v in mv.items()
                ])
                st.dataframe(mv_df, use_container_width=True)
            else:
                st.success("✅ No missing values in any column!")
        except Exception as e:
            st.error(f"Error: {e}")


def _show_col_stats(metadata: dict, column: str):
    """Show quick stats below a chart."""
    v = metadata.get("variables", {}).get(column, {})
    if not v:
        return
    with st.expander(f"📋 Stats for `{column}`", expanded=False):
        stats = {
            "Type": v.get("type", "N/A"),
            "Mean": f"{v.get('mean'):.4f}" if v.get("mean") is not None else "N/A",
            "Median": f"{v.get('median'):.4f}" if v.get("median") is not None else "N/A",
            "Std Dev": f"{v.get('std'):.4f}" if v.get("std") is not None else "N/A",
            "Min": f"{v.get('min'):.4f}" if v.get("min") is not None else "N/A",
            "Max": f"{v.get('max'):.4f}" if v.get("max") is not None else "N/A",
            "Skewness": f"{v.get('skewness'):.4f}" if v.get("skewness") is not None else "N/A",
            "Missing": f"{v.get('n_missing', 0)} ({v.get('p_missing', 0)*100:.2f}%)",
        }
        cols = st.columns(4)
        for i, (k, val) in enumerate(stats.items()):
            cols[i % 4].metric(k, val)