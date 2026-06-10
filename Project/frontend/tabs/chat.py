"""
frontend/pages/chat.py
Chat tab – natural language Q&A with the LangGraph agent.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import plotly.graph_objects as go
from backend.agents.graph import run_agent
from backend.agents.memory import ConversationMemory


EXAMPLE_QUESTIONS = [
    "Which column has the most missing values?",
    "Are there any strong correlations in the dataset?",
    "What are the outliers in numeric columns?",
    "Show me the data quality score.",
    "How many duplicate rows are there?",
    "What is the distribution of the first numeric column?",
    "Give me an executive summary of the dataset.",
]


def _get_memory() -> ConversationMemory:
    """Return or create a ConversationMemory for the current session."""
    conv_id = st.session_state.get("conversation_id", "default")
    if "memory_obj" not in st.session_state:
        st.session_state.memory_obj = ConversationMemory(conv_id)
    return st.session_state.memory_obj


def render_chat_tab():
    # ── Guard: require dataset ────────────────────────────────────────────
    st.write("INSIDE render_chat_tab")
    if not st.session_state.get("dataset_name"):
        st.info("👆 Please upload a dataset first in the **Upload Dataset** tab.")
        return

    dataset_name = st.session_state["dataset_name"]
    metadata = st.session_state["metadata"]
    df = st.session_state["df"]
    memory = _get_memory()

    st.markdown(f"### 💬 Chat with **{dataset_name}**")

    # ── Layout ────────────────────────────────────────────────────────────
    chat_col, sidebar_col = st.columns([3, 1])

    with sidebar_col:
        st.markdown("#### 💡 Try asking:")
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, key=f"eq_{q[:20]}", use_container_width=True):
                st.session_state["pending_question"] = q
                st.rerun()

        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state["chat_history"] = []
            memory.clear()
            st.rerun()

        # Conversation stats
        n_msgs = len(st.session_state.get("chat_history", []))
        st.markdown(f"""
        <div style='font-size:0.75rem; color:#666; text-align:center;'>
            {n_msgs} messages in this session
        </div>
        """, unsafe_allow_html=True)

    with chat_col:
        # ── Chat history display ─────────────────────────────────────────
        chat_history = st.session_state.get("chat_history", [])

        if not chat_history:
            st.markdown("""
            <div style='
                background:#1a1a2e; border:1px dashed #2a2a4a;
                border-radius:12px; padding:2rem; text-align:center; color:#666;
            '>
                <div style='font-size:2rem;'>💬</div>
                <div>Ask a question about your dataset to get started.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in chat_history:
                _render_message(msg)

        # ── Input ────────────────────────────────────────────────────────
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Ask a question about your data…",
                value=st.session_state.pop("pending_question", ""),
                placeholder="e.g. Which column has the most missing values?",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Send ➤", use_container_width=True)

        if submitted and user_input.strip():
            _handle_question(user_input.strip(), dataset_name, metadata, df, memory)
            st.rerun()


def _handle_question(
    question: str,
    dataset_name: str,
    metadata: dict,
    df,
    memory: ConversationMemory,
):
    """Run the agent and append result to chat history."""
    with st.spinner("🤔 Thinking…"):
        try:
            result = run_agent(
                question=question,
                dataset_name=dataset_name,
                metadata=metadata,
                df=df,
                memory=memory,
            )
        except Exception as e:
            result = {
                "answer": f"❌ Agent error: {str(e)}",
                "citations": [],
                "chart": None,
                "pdf_path": "",
                "action": "error",
            }

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "citations": result.get("citations", []),
        "chart": result.get("chart"),
        "pdf_path": result.get("pdf_path", ""),
        "action": result.get("action", ""),
    })


def _render_message(msg: dict):
    """Render a single chat message with appropriate styling."""
    role = msg["role"]

    if role == "user":
        st.markdown(f"""
        <div class='chat-user'>
            <span style='font-size:0.75rem; color:#a0c4ff;'>You</span><br>
            {msg['content']}
        </div>
        """, unsafe_allow_html=True)

    else:
        content = msg.get("content", "")
        citations = msg.get("citations", [])
        chart = msg.get("chart")
        pdf_path = msg.get("pdf_path", "")

        st.markdown(f"""
        <div class='chat-assistant'>
            <span style='font-size:0.75rem; color:#e94560;'>🔍 NulAware AI</span><br>
        """, unsafe_allow_html=True)

        st.markdown(content)

        # Chart (if any)
        if chart is not None:
            try:
                st.plotly_chart(chart, use_container_width=True, key=f"chart_{id(chart)}")
            except Exception:
                pass

        # PDF download (if any)
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📥 Download PDF Report",
                    data=f.read(),
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                    key=f"dl_{hash(pdf_path)}",
                )

        # Citations
        if citations:
            citation_text = " | ".join(citations[:3])
            st.markdown(f"""
            <div class='citation-box'>
                📌 <b>Source:</b> {citation_text}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("")
