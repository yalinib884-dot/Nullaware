"""
backend/agents/graph.py
LangGraph-based agent graph for NulAware AI.

Graph nodes:
  planner → [retriever | viz_node | report_node | stats_node] → responder
"""
import logging
import pandas as pd
from typing import TypedDict

from langgraph.graph import StateGraph, END

from .planner import classify_intent, AgentPlan
from .memory import ConversationMemory
from backend.rag.retriever import retrieve_chunks
from backend.llm.gemini_manager import call_gemini
from backend.tools.stats_tool import (
    get_missing_values, get_column_statistics, get_correlations,
    get_outliers, compute_data_quality_score,
)
from backend.tools.visualization_tool import (
    auto_chart, generate_overview_dashboard,
    generate_scatter_plot, generate_missing_values_chart,
)
from backend.tools.report_tool import generate_pdf_report
from backend.tools.dataframe_tool import find_rows
from prompts import (
    RAG_SYSTEM_PROMPT, RAG_USER_PROMPT,
    STATS_PROMPT, INSIGHTS_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


# ── State ──────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    resolved_question: str
    dataset_name: str
    metadata: dict
    df: object            # pandas DataFrame (not serialised)
    plan: object          # AgentPlan
    retrieved_chunks: list
    stats_result: object  # dict OR list[dict] depending on action
    chart: object         # plotly Figure or None
    chart_label: str
    pdf_path: str
    final_answer: str
    citations: list
    error: str


# ── Node implementations ───────────────────────────────────────────────────

def planner_node(state: AgentState) -> AgentState:
    """Resolve coreferences then classify intent."""
    memory: ConversationMemory = state.get("_memory")
    q = state["question"]

    if memory:
        q = memory.resolve_coreferences(q)

    cols = list(state.get("metadata", {}).get("variables", {}).keys())
    plan = classify_intent(q, available_columns=cols)
    logger.info(f"Plan: action={plan.action}, col={plan.column}, chart={plan.chart_type}")

    return {**state, "resolved_question": q, "plan": plan}


def retriever_node(state: AgentState) -> AgentState:
    """Fetch top-k chunks from ChromaDB."""
    chunks = retrieve_chunks(
        query=state["resolved_question"],
        dataset_name=state["dataset_name"],
        top_k=5,
    )
    return {**state, "retrieved_chunks": chunks}


def stats_node(state: AgentState) -> AgentState:
    """Run the appropriate stats tool and store results."""
    plan: AgentPlan = state["plan"]
    meta = state["metadata"]
    result = {}

    action = plan.action

    if action == "missing_values":
        result = get_missing_values(meta)
        if plan.column and plan.column in result:
            result = {plan.column: result[plan.column]}

    elif action == "column_statistics":
        result = get_column_statistics(meta, plan.column)

    elif action == "correlations":
        result = {"pearson_pairs": get_correlations(meta)}

    elif action == "outliers":
        result = get_outliers(meta)

    elif action == "duplicates":
        result = meta.get("duplicates", {})

    elif action == "data_quality":
        result = compute_data_quality_score(meta)

    elif action == "row_lookup":
        df: pd.DataFrame = state.get("df")
        if df is not None and not df.empty:
            col = plan.column or ""
            value = plan.extra.get("value", "")
            # find_rows returns list[dict] — store directly
            result = find_rows(df, col, value)
        else:
            result = []

    return {**state, "stats_result": result}


def viz_node(state: AgentState) -> AgentState:
    """Generate the requested Plotly figure."""
    plan: AgentPlan = state["plan"]
    df: pd.DataFrame = state.get("df")
    meta = state.get("metadata", {})

    if df is None or df.empty:
        return {**state, "error": "No dataset loaded for visualization."}

    try:
        chart_label = ""
        if plan.chart_type == "heatmap" or plan.column is None:
            if plan.chart_type == "heatmap":
                from backend.tools.visualization_tool import generate_heatmap
                fig = generate_heatmap(df, meta)
                chart_label = "Correlation Heatmap"
            else:
                fig = generate_overview_dashboard(df, meta)
                chart_label = "Overview Dashboard"
        elif plan.x_col and plan.y_col:
            fig = generate_scatter_plot(df, plan.x_col, plan.y_col)
            chart_label = f"Scatter: {plan.x_col} vs {plan.y_col}"
        elif plan.column == "__missing__" or (not plan.column and plan.chart_type == "missing"):
            fig = generate_missing_values_chart(meta)
            chart_label = "Missing Values"
        else:
            fig = auto_chart(df, meta, column=plan.column, chart_type=plan.chart_type or "auto")
            chart_label = f"{(plan.chart_type or 'auto').title()}: {plan.column}"

        return {**state, "chart": fig, "chart_label": chart_label}

    except Exception as e:
        logger.error(f"Visualization error: {e}")
        return {**state, "error": str(e)}


def report_node(state: AgentState) -> AgentState:
    """Generate the PDF report."""
    meta = state.get("metadata", {})
    dataset_name = state.get("dataset_name", "dataset")
    ai_summary = state.get("final_answer", "")

    try:
        pdf_path = generate_pdf_report(
            dataset_name=dataset_name,
            metadata=meta,
            ai_summary=ai_summary,
        )
        return {**state, "pdf_path": pdf_path,
                "final_answer": "PDF report generated successfully."}
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return {**state, "error": str(e)}


def responder_node(state: AgentState) -> AgentState:
    """
    Build final answer using Gemini or direct formatting.
    """
    plan: AgentPlan = state["plan"]
    question = state["resolved_question"]
    chunks = state.get("retrieved_chunks", [])
    stats = state.get("stats_result")   # may be dict or list
    pdf_path = state.get("pdf_path", "")
    error = state.get("error", "")

    if error:
        return {**state, "final_answer": f"Sorry, I encountered an error: {error}"}

    citations = []

    # ── Visualization ───────────────────────────────────────────────────
    if plan.action == "generate_visualization":
        col_info = f"column '{plan.column}'" if plan.column else "the dataset"
        answer = (
            f"Here's the **{state.get('chart_label', 'chart')}** for {col_info}. "
            "The chart is displayed above."
        )
        return {**state, "final_answer": answer, "citations": []}

    # ── Report ──────────────────────────────────────────────────────────
    if plan.action == "generate_report":
        if pdf_path:
            answer = (
                f"✅ PDF report generated: `{pdf_path}`\n"
                "You can download it from the Report Generator tab."
            )
        else:
            answer = "Report generation failed. Please check the logs."
        return {**state, "final_answer": answer, "citations": []}

    # ── Row lookup ──────────────────────────────────────────────────────
    if plan.action == "row_lookup":
        # stats is list[dict] here
        rows: list = stats if isinstance(stats, list) else []
        if not rows:
            answer = (
                f"No rows found where **`{plan.column}`** = **`{plan.extra.get('value', '?')}`**."
            )
        else:
            try:
                df_result = pd.DataFrame(rows)
                answer = (
                    f"Found **{len(rows)}** row(s) where "
                    f"**`{plan.column}`** = **`{plan.extra.get('value', '?')}`**:\n\n"
                    + df_result.to_markdown(index=False)
                )
            except Exception as e:
                answer = f"Found {len(rows)} row(s) but could not format table: {e}"
        return {
            **state,
            "final_answer": answer,
            "citations": ["Retrieved directly from dataset rows"],
        }

    # ── Stats-based answer ──────────────────────────────────────────────
    stats_actions = {
        "missing_values", "column_statistics", "correlations",
        "outliers", "duplicates", "data_quality",
    }
    if plan.action in stats_actions and stats:
        context = _format_stats_for_prompt(plan.action, stats)
        prompt = STATS_PROMPT.format(
            question=question,
            stats_context=context,
            conversation_context=state.get("_context_window", ""),
        )
        try:
            answer = call_gemini(prompt, system_prompt=INSIGHTS_SYSTEM_PROMPT)
        except Exception:
            answer = f"Based on the data:\n\n{context}"
        citations = [f"Computed from metadata: {plan.action.replace('_', ' ').title()}"]
        return {**state, "final_answer": answer, "citations": citations}

    # ── RAG answer ───────────────────────────────────────────────────────
    if chunks:
        context = "\n\n---\n\n".join(c["text"] for c in chunks)
        citations = [
            f"{c.get('source', 'metadata')} (relevance: {1 - c.get('distance', 0):.2%})"
            for c in chunks
        ]
    else:
        context = "No relevant chunks found in the vector store."

    prompt = RAG_USER_PROMPT.format(
        question=question,
        context=context,
        conversation_context=state.get("_context_window", ""),
    )

    try:
        answer = call_gemini(prompt, system_prompt=RAG_SYSTEM_PROMPT)
    except Exception as e:
        logger.error(f"Gemini call failed in responder: {e}")
        answer = f"I couldn't generate an answer due to an API error: {e}"

    return {**state, "final_answer": answer, "citations": citations}


def _format_stats_for_prompt(action: str, stats) -> str:
    """Convert stats dict to readable string for the LLM prompt."""
    import json
    lines = []
    if action == "missing_values" and isinstance(stats, dict):
        for col, v in list(stats.items())[:15]:
            lines.append(f"  {col}: {v.get('n_missing', 0)} missing ({v.get('p_missing', 0)}%)")
    elif action == "correlations" and isinstance(stats, dict):
        for p in stats.get("pearson_pairs", [])[:10]:
            lines.append(f"  {p['col_a']} ↔ {p['col_b']}: r={p['pearson']}")
    elif action == "data_quality" and isinstance(stats, dict):
        lines.append(f"  Score: {stats.get('total_score', '?')}/100 (Grade: {stats.get('grade', '?')})")
        for k, v in stats.get("components", {}).items():
            lines.append(f"  {k}: {v}")
    else:
        lines.append(json.dumps(stats, indent=2, default=str)[:800])
    return "\n".join(lines) if lines else json.dumps(stats, default=str)[:800]


# ── Route selector ─────────────────────────────────────────────────────────

def route(state: AgentState) -> str:
    action = state["plan"].action
    if action == "generate_visualization":
        return "viz"
    if action == "generate_report":
        return "report"
    if action in ("missing_values", "column_statistics", "correlations",
                  "outliers", "duplicates", "data_quality", "row_lookup"):
        return "stats"
    return "retriever"


# ── Build graph ─────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("retriever", retriever_node)
    g.add_node("stats", stats_node)
    g.add_node("viz", viz_node)
    g.add_node("report", report_node)
    g.add_node("responder", responder_node)

    g.set_entry_point("planner")
    g.add_conditional_edges("planner", route, {
        "retriever": "retriever",
        "stats": "stats",
        "viz": "viz",
        "report": "report",
    })
    g.add_edge("retriever", "responder")
    g.add_edge("stats", "responder")
    g.add_edge("viz", "responder")
    g.add_edge("report", "responder")
    g.add_edge("responder", END)

    return g.compile()


# ── Public entry point ──────────────────────────────────────────────────────

_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent(
    question: str,
    dataset_name: str,
    metadata: dict,
    df: pd.DataFrame,
    memory: ConversationMemory,
) -> dict:
    """
    Main entry point to run the agent.

    Returns:
        dict with keys: answer, citations, chart, pdf_path
    """
    graph = get_graph()

    initial_state: AgentState = {
        "question": question,
        "resolved_question": question,
        "dataset_name": dataset_name,
        "metadata": metadata,
        "df": df,
        "plan": None,
        "retrieved_chunks": [],
        "stats_result": {},
        "chart": None,
        "chart_label": "",
        "pdf_path": "",
        "final_answer": "",
        "citations": [],
        "error": "",
        "_memory": memory,
        "_context_window": memory.get_context_window(6) if memory else "",
    }

    result = graph.invoke(initial_state)

    # Persist to memory
    memory.add("user", question)
    memory.add("assistant", result.get("final_answer", ""))

    return {
        "answer": result.get("final_answer", ""),
        "citations": result.get("citations", []),
        "chart": result.get("chart"),
        "pdf_path": result.get("pdf_path", ""),
        "action": result.get("plan").action if result.get("plan") else "unknown",
    }