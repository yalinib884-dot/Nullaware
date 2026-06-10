"""
frontend/pages/report.py
Report Generator tab – PDF report with AI summary and charts.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

from backend.tools.report_tool import generate_pdf_report
from backend.tools.stats_tool import (
    compute_data_quality_score, generate_automatic_insights,
    get_missing_values, get_correlations, get_outliers,
)
from backend.tools.visualization_tool import (
    generate_heatmap, generate_missing_values_chart,
    generate_histogram, generate_overview_dashboard,
)
from backend.llm.gemini_manager import call_gemini
from prompts import REPORT_SUMMARY_PROMPT
from config import CHARTS_DIR


def render_report_tab():
    if not st.session_state.get("dataset_name"):
        st.info("👆 Please upload a dataset first in the **Upload Dataset** tab.")
        return

    dataset_name = st.session_state["dataset_name"]
    metadata = st.session_state["metadata"]
    df = st.session_state["df"]

    st.markdown(f"### 📄 Report Generator — **{dataset_name}**")
    st.markdown(
        "Generate a comprehensive PDF data quality report with AI-written insights, "
        "statistics, and embedded charts."
    )

    # ── Report preview ────────────────────────────────────────────────────
    quality = compute_data_quality_score(metadata)
    insights = generate_automatic_insights(metadata)
    ds = metadata.get("dataset_summary", {})

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📋 Report Contents")
        st.markdown("""
        The generated PDF will include:
        - ✅ Executive Summary (AI-written)
        - ✅ Dataset Information
        - ✅ Missing Values Analysis
        - ✅ Correlation Analysis
        - ✅ Outlier Detection
        - ✅ Duplicate Detection
        - ✅ Column Statistics
        - ✅ Embedded Charts
        - ✅ AI Recommendations
        """)

    with col2:
        st.markdown("#### 🏆 Data Quality Preview")
        grade_colors = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🔴"}
        grade_icon = grade_colors.get(quality["grade"], "⚪")
        st.markdown(f"""
        <div style='
            background: #1a1a2e; border: 1px solid #2a2a4a;
            border-radius: 12px; padding: 1.5rem; text-align: center;
        '>
            <div style='font-size: 3rem;'>{grade_icon}</div>
            <div style='font-size: 2rem; font-weight:700; color:#e94560;'>
                {quality['total_score']} / 100
            </div>
            <div style='color:#a0a0c0;'>Grade: <b>{quality['grade']}</b></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**Quality Breakdown:**")
        for comp, score in quality["components"].items():
            st.progress(int(score / 40 * 100) if "missing" in comp else
                        int(score / 20 * 100),
                        text=f"{comp.replace('_', ' ').title()}: {score}")

    # ── Options ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⚙️ Report Options")

    col_a, col_b = st.columns(2)
    with col_a:
        include_charts = st.checkbox("Include charts in PDF", value=True)
        include_ai_summary = st.checkbox("Include AI executive summary", value=True)
    with col_b:
        if include_ai_summary:
            st.info("🤖 AI summary will be generated using Gemini 2.5 Flash.")

    st.markdown("---")

    # ── Preview insights ──────────────────────────────────────────────────
    with st.expander("💡 Preview: Automatic Insights", expanded=True):
        for insight in insights:
            st.markdown(f"- {insight}")

    with st.expander("📊 Preview: Top Missing Columns"):
        mv = get_missing_values(metadata)
        mv_with_missing = {k: v for k, v in mv.items() if v["n_missing"] > 0}
        if mv_with_missing:
            for col, v in list(mv_with_missing.items())[:8]:
                st.markdown(f"- **`{col}`**: {v['n_missing']} missing ({v['p_missing']}%)")
        else:
            st.success("No missing values!")

    with st.expander("🔗 Preview: Top Correlations"):
        corrs = get_correlations(metadata, threshold=0.5)
        if corrs:
            for p in corrs[:5]:
                st.markdown(f"- **`{p['col_a']}`** ↔ **`{p['col_b']}`**: r = {p['pearson']}")
        else:
            st.info("No strong correlations found.")

    # ── Generate button ───────────────────────────────────────────────────
    st.markdown("---")
    if st.button("📄 Generate PDF Report", use_container_width=True, type="primary"):
        _generate_report(
            dataset_name=dataset_name,
            metadata=metadata,
            df=df,
            include_charts=include_charts,
            include_ai_summary=include_ai_summary,
        )


def _generate_report(
    dataset_name: str,
    metadata: dict,
    df,
    include_charts: bool,
    include_ai_summary: bool,
):
    progress = st.progress(0, text="Initialising…")
    status = st.empty()

    try:
        chart_paths = []

        # ── Generate charts ──────────────────────────────────────────────
        if include_charts:
            status.info("📊 Generating charts…")
            progress.progress(20, text="Generating charts…")

            try:
                fig = generate_overview_dashboard(df, metadata)
                fig.write_image(str(CHARTS_DIR / "report_overview.png"), scale=1.5)
                chart_paths.append(str(CHARTS_DIR / "report_overview.png"))
            except Exception as e:
                st.warning(f"Could not generate overview chart: {e}")

            try:
                fig = generate_missing_values_chart(metadata)
                fig.write_image(str(CHARTS_DIR / "report_missing.png"), scale=1.5)
                chart_paths.append(str(CHARTS_DIR / "report_missing.png"))
            except Exception:
                pass

            try:
                import numpy as np
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    fig = generate_histogram(df, numeric_cols[0])
                    path = str(CHARTS_DIR / f"report_hist_{numeric_cols[0]}.png")
                    fig.write_image(path, scale=1.5)
                    chart_paths.append(path)
            except Exception:
                pass

            try:
                fig = generate_heatmap(df, metadata)
                fig.write_image(str(CHARTS_DIR / "report_heatmap.png"), scale=1.5)
                chart_paths.append(str(CHARTS_DIR / "report_heatmap.png"))
            except Exception:
                pass

        # ── AI executive summary ─────────────────────────────────────────
        ai_summary = ""
        if include_ai_summary:
            status.info("🤖 Generating AI executive summary…")
            progress.progress(50, text="Writing AI summary…")

            quality = compute_data_quality_score(metadata)
            insights = generate_automatic_insights(metadata)
            corrs = get_correlations(metadata, threshold=0.5)
            ds = metadata.get("dataset_summary", {})

            top_issues = "\n".join(f"- {i}" for i in insights[:5])
            corr_text = "\n".join(
                f"- {p['col_a']} ↔ {p['col_b']}: r={p['pearson']}"
                for p in corrs[:3]
            ) or "None detected"

            prompt = REPORT_SUMMARY_PROMPT.format(
                dataset_name=dataset_name,
                n_rows=ds.get("n_rows", "?"),
                n_columns=ds.get("n_columns", "?"),
                pct_missing=round(ds.get("pct_missing_cells", 0) * 100, 2),
                n_duplicates=ds.get("n_duplicate_rows", 0),
                quality_score=quality["total_score"],
                grade=quality["grade"],
                top_issues=top_issues,
                correlations=corr_text,
            )

            try:
                ai_summary = call_gemini(prompt)
            except Exception as e:
                ai_summary = f"AI summary unavailable: {e}"

        # ── Generate PDF ─────────────────────────────────────────────────
        status.info("📄 Building PDF…")
        progress.progress(80, text="Building PDF…")

        pdf_path = generate_pdf_report(
            dataset_name=dataset_name,
            metadata=metadata,
            chart_paths=chart_paths if include_charts else [],
            ai_summary=ai_summary,
        )

        progress.progress(100, text="Done!")
        status.empty()

        st.success(f"✅ Report generated: `{os.path.basename(pdf_path)}`")

        # ── Download button ───────────────────────────────────────────────
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=os.path.basename(pdf_path),
            mime="application/pdf",
            use_container_width=True,
        )

        # ── Preview AI summary ────────────────────────────────────────────
        if ai_summary:
            with st.expander("📝 AI Executive Summary Preview", expanded=True):
                st.markdown(ai_summary)

    except Exception as e:
        progress.empty()
        status.empty()
        st.error(f"❌ Report generation failed: {e}")
        raise