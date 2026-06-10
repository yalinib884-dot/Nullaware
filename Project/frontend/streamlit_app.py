"""
frontend/streamlit_app.py
Main NulAware AI Streamlit application entry point.
Run: streamlit run frontend/streamlit_app.py
"""
import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="NulAware AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

from backend.database.sqlite_manager import init_db
from frontend.tabs.upload import render_upload_tab
from frontend.tabs.chat import render_chat_tab
from frontend.tabs.dashboard import render_dashboard_tab
from frontend.tabs.report import render_report_tab

# ── Init DB ───────────────────────────────────────────────────────────────
init_db()

# ── Styles ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    border-right: 1px solid #e94560;
}
[data-testid="stSidebar"] * { color: #f0f0f0 !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #e94560 !important; font-weight: 600; }

/* ── Hide multi-page nav ── */
[data-testid="stSidebarNav"] {
    display: none !important;
}            

/* ── Header ── */
.nulaware-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border: 1px solid #e94560;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.nulaware-title {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin: 0;
}
.nulaware-subtitle {
    font-size: 0.85rem;
    color: #a0a0c0;
    margin: 0;
}
.accent { color: #e94560; }

/* ── Metric cards ── */
.metric-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #e94560; }
.metric-value { font-size: 1.6rem; font-weight: 700; color: #e94560; }
.metric-label { font-size: 0.75rem; color: #888; margin-top: 2px; }

/* ── Chat bubbles ── */
.chat-user {
    background: #0f3460;
    border-radius: 12px 12px 0 12px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    max-width: 80%;
    margin-left: auto;
    color: #f0f0f0;
}
.chat-assistant {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px 12px 12px 0;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    max-width: 90%;
    color: #e0e0f0;
}
.citation-box {
    background: #0d1b2a;
    border-left: 3px solid #e94560;
    padding: 0.4rem 0.8rem;
    margin-top: 0.5rem;
    border-radius: 0 6px 6px 0;
    font-size: 0.75rem;
    color: #888;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: #1a1a2e;
    border-radius: 8px 8px 0 0;
    border: 1px solid #2a2a4a;
    color: #a0a0c0;
    font-weight: 500;
    padding: 0.5rem 1.5rem;
}
.stTabs [aria-selected="true"] {
    background: #e94560 !important;
    color: white !important;
    border-color: #e94560 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #e94560, #c0392b);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #ff6b6b, #e94560);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(233,69,96,0.4);
}

/* ── Quality badge ── */
.quality-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9rem;
}
.grade-a { background: #27ae60; color: white; }
.grade-b { background: #2980b9; color: white; }
.grade-c { background: #e67e22; color: white; }
.grade-d { background: #e74c3c; color: white; }

/* ── Code / mono ── */
code { font-family: 'JetBrains Mono', monospace !important; }

/* ── Scrollable chat ── */
.chat-container {
    max-height: 500px;
    overflow-y: auto;
    padding: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <div style='font-size:2.5rem;'>🔍</div>
        <div style='font-size:1.2rem; font-weight:700; color:#e94560;'>NulAware AI</div>
        <div style='font-size:0.7rem; color:#888;'>Intelligent Data Profiling</div>
    </div>
    <hr style='border-color:#2a2a4a; margin: 0.75rem 0;'>

    <div style='background:#0f0f1a; border:1px solid #2a2a4a; border-radius:10px;
                padding:0.9rem 1rem; margin-bottom:1rem;'>
        <div style='font-size:0.65rem; font-weight:700; color:#e94560;
                    letter-spacing:0.08em; margin-bottom:0.5rem;'>ABOUT THIS APP</div>
        <p style='font-size:0.78rem; color:#c0c0d8; line-height:1.55; margin:0 0 0.5rem;'>
            <b style='color:#f0f0f0;'>NulAware AI</b> is an intelligent data profiling
            assistant that helps you understand, explore, and report on your CSV datasets
            — powered by Gemini 2.5 Flash.
        </p>
        <p style='font-size:0.78rem; color:#c0c0d8; line-height:1.55; margin:0;'>
            Upload a dataset, chat with it using natural language, explore visual
            dashboards, and generate automated quality reports — all in one place.
        </p>
    </div>

    <div style='background:#0f0f1a; border:1px solid #2a2a4a; border-radius:10px;
                padding:0.9rem 1rem; margin-bottom:1rem;'>
        <div style='font-size:0.65rem; font-weight:700; color:#e94560;
                    letter-spacing:0.08em; margin-bottom:0.6rem;'>KEY FEATURES</div>
        <div style='font-size:0.78rem; color:#c0c0d8; line-height:1.9;'>
            📤 &nbsp;<b style='color:#f0f0f0;'>Upload</b> — CSV ingestion &amp; schema detection<br>
            💬 &nbsp;<b style='color:#f0f0f0;'>Chat</b> — Ask questions in plain English<br>
            📊 &nbsp;<b style='color:#f0f0f0;'>Dashboard</b> — Visual data exploration<br>
            📄 &nbsp;<b style='color:#f0f0f0;'>Report</b> — Automated quality reports
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Active dataset info
    

    

    

    st.markdown("<hr style='border-color:#2a2a4a;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.65rem; color:#444; text-align:center;'>
    Powered by Gemini 2.5 Flash<br>
    LangGraph · ChromaDB · ydata-profiling
    </div>
    """, unsafe_allow_html=True)


# ── Main header ───────────────────────────────────────────────────────────
st.markdown("""
<div class='nulaware-header'>
    <div style='font-size:2.5rem;'>🔍</div>
    <div>
        <p class='nulaware-title'>Nul<span class='accent'>Aware</span> AI</p>
        <p class='nulaware-subtitle'>
            AI-Powered Data Profiling Assistant · Upload · Profile · Chat · Visualize · Report
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload Dataset",
    "💬 Chat With Dataset",
    "📊 Visualization Dashboard",
    "📄 Report Generator",
])

with tab1:
    render_upload_tab()

with tab2:
    render_chat_tab()

with tab3:
    render_dashboard_tab()

with tab4:
    render_report_tab()