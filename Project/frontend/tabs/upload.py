"""
frontend/pages/upload.py
Upload tab – CSV upload, profiling, indexing pipeline.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import uuid
import time
import pandas as pd
import streamlit as st

from config import UPLOAD_DIR, PROFILE_JSON_DIR
from backend.profiling.generate_profile import generate_profile
from backend.profiling.extract_profile import extract_metadata
from backend.rag.chunker import build_chunks
from backend.rag.chromadb_store import store_chunks
from backend.database.sqlite_manager import save_session
from backend.tools.stats_tool import (
    compute_data_quality_score, generate_automatic_insights,
)


def render_upload_tab():
    st.markdown("### 📤 Upload Your Dataset")
    st.markdown(
        "Upload a CSV file to begin profiling. "
        "NulAware AI will generate a full statistical profile, "
        "build a semantic index, and prepare the dataset for chat."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Maximum recommended size: 50 MB",
    )

    if uploaded_file is None:
        _show_placeholder()
        return

    # ── Save CSV ─────────────────────────────────────────────────────────
    csv_path = UPLOAD_DIR / uploaded_file.name
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    df = pd.read_csv(csv_path)
    dataset_name = uploaded_file.name.rsplit(".", 1)[0]

    # ── Preview ───────────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")

    st.markdown("#### 👁️ Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    col_info_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype": [str(df[c].dtype) for c in df.columns],
        "Non-Null": [df[c].notna().sum() for c in df.columns],
        "Missing": [df[c].isna().sum() for c in df.columns],
        "Unique": [df[c].nunique() for c in df.columns],
    })
    with st.expander("📋 Column Summary", expanded=False):
        st.dataframe(col_info_df, use_container_width=True)

    # ── Profile button ────────────────────────────────────────────────────
    already_loaded = (
        st.session_state.get("dataset_name") == dataset_name
        and st.session_state.get("metadata")
    )

    if already_loaded:
        st.success(f"✅ **{dataset_name}** is already indexed and ready to use!")
        _show_quick_stats()
        return

    if st.button("🚀 Profile & Index Dataset", use_container_width=True):
        _run_pipeline(df, csv_path, dataset_name)


def _run_pipeline(df: pd.DataFrame, csv_path, dataset_name: str):
    """Execute the full profiling + indexing pipeline with progress."""
    progress = st.progress(0, text="Starting pipeline...")
    status = st.empty()

    try:
        # Step 1: Generate profile
        status.info("🔬 Generating ydata-profiling report… (this may take 30-60s)")
        progress.progress(10, text="Profiling…")
        profile_result = generate_profile(csv_path, PROFILE_JSON_DIR)
        progress.progress(35, text="Profile generated.")

        # Step 2: Extract metadata
        status.info("🧠 Extracting metadata…")
        metadata = extract_metadata(profile_result["json_path"])
        progress.progress(50, text="Metadata extracted.")

        # Step 3: Build chunks
        status.info("✂️ Building semantic chunks…")
        chunks = build_chunks(metadata)
        progress.progress(65, text=f"{len(chunks)} chunks built.")

        # Step 4: Embed + store in ChromaDB
        status.info("💾 Generating embeddings and storing in ChromaDB…")
        n_stored = store_chunks(chunks, dataset_name)
        progress.progress(85, text=f"{n_stored} chunks indexed.")

        # Step 5: Persist session
        session_id = str(uuid.uuid4())
        save_session(
            session_id=session_id,
            dataset_name=dataset_name,
            csv_path=str(csv_path),
            json_path=profile_result["json_path"],
            html_path=profile_result["html_path"],
            metadata=metadata,
        )
        progress.progress(100, text="Done!")
        status.empty()

        # ── Store in session state ──────────────────────────────────────
        st.session_state.update({
            "dataset_name": dataset_name,
            "metadata": metadata,
            "df": df,
            "df_shape": df.shape,
            "csv_path": str(csv_path),
            "json_path": profile_result["json_path"],
            "html_path": profile_result["html_path"],
            "session_id": session_id,
            "conversation_id": str(uuid.uuid4()),
            "chat_history": [],
        })

        st.success(
            f"✅ **{dataset_name}** profiled and indexed successfully! "
            "Switch to the **Chat** tab to start asking questions."
        )
        _show_quick_stats()

        # HTML report download
        with open(profile_result["html_path"], "rb") as f:
            st.download_button(
                "📥 Download HTML Profile Report",
                data=f,
                file_name=f"{dataset_name}_profile.html",
                mime="text/html",
            )

    except Exception as e:
        progress.empty()
        status.empty()
        st.error(f"❌ Pipeline failed: {e}")
        raise


def _show_quick_stats():
    metadata = st.session_state.get("metadata", {})
    if not metadata:
        return

    quality = compute_data_quality_score(metadata)
    insights = generate_automatic_insights(metadata)
    ds = metadata.get("dataset_summary", {})

    st.markdown("---")
    st.markdown("#### 📊 Quick Profile Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{ds.get('n_rows', 0):,}")
    c2.metric("Columns", ds.get("n_columns", 0))
    c3.metric("Missing Cells", f"{ds.get('n_missing_cells', 0):,}")
    c4.metric("Quality Score", f"{quality['total_score']}/100 (Grade {quality['grade']})")

    st.markdown("#### 💡 Automatic Insights")
    for insight in insights:
        st.markdown(f"- {insight}")


def _show_placeholder():
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        border: 2px dashed #2a2a4a;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        margin-top: 2rem;
    '>
        <div style='font-size:3rem;'>📂</div>
        <div style='font-size:1.1rem; color:#a0a0c0; margin-top:0.5rem;'>
            Drag and drop a CSV file above to get started
        </div>
        <div style='font-size:0.8rem; color:#555; margin-top:0.5rem;'>
            NulAware AI will automatically profile your data,<br>
            build a semantic index, and enable natural-language Q&A.
        </div>
    </div>
    """, unsafe_allow_html=True)