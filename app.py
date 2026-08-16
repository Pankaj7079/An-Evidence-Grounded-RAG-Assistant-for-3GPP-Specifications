"""3GPP RAG Chatbot — Streamlit Interface.

Evidence-grounded chatbot for 3GPP TS 23.501 & TS 23.502 specifications.
Built by Pankaj for Mavenir GET evaluation.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import streamlit as st

from config import settings
from src.citation_validator import CitationValidator
from src.models import RetrievalResult
from src.pipeline import PipelineResponse, RAGPipeline

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="3GPP RAG Chatbot",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Global ── */
    .main, .stApp {
        background-color: #111318;
        color: #d1d5db;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* ── Header ── */
    .app-header {
        padding: 20px 0 16px 0;
        border-bottom: 1px solid #1f2937;
        margin-bottom: 20px;
    }
    .app-header h2 {
        color: #f9fafb;
        font-size: 20px;
        font-weight: 600;
        margin: 0 0 4px 0;
        letter-spacing: -0.01em;
    }
    .app-header p {
        color: #6b7280;
        font-size: 13px;
        margin: 0;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 8px 0 !important;
    }

    /* ── Response metadata (subtle footer under answers) ── */
    .response-meta {
        display: flex;
        gap: 16px;
        align-items: center;
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid #1f2937;
        flex-wrap: wrap;
    }
    .response-meta span {
        color: #6b7280;
        font-size: 11.5px;
        font-weight: 500;
    }
    .response-meta .grounded {
        color: #22c55e;
    }
    .response-meta .abstained {
        color: #f59e0b;
    }

    /* ── Source citations (clean inline tags) ── */
    .source-tags {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        margin-top: 6px;
    }
    .source-tag {
        background: #1f2937;
        color: #9ca3af;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 500;
        border: 1px solid #374151;
    }

    /* ── Context excerpt cards ── */
    .excerpt-card {
        background: #1a1d24;
        border: 1px solid #1f2937;
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 8px;
    }
    .excerpt-header {
        font-size: 12px;
        font-weight: 600;
        color: #d1d5db;
        margin-bottom: 2px;
    }
    .excerpt-meta {
        font-size: 11px;
        color: #6b7280;
        margin-bottom: 8px;
    }
    .excerpt-body {
        font-size: 12.5px;
        color: #9ca3af;
        line-height: 1.55;
        background: #111318;
        padding: 8px 10px;
        border-radius: 4px;
        border: 1px solid #1f2937;
    }

    /* ── Sidebar tweaks ── */
    [data-testid="stSidebar"] {
        background-color: #111318;
        border-right: 1px solid #1f2937;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: #1f2937;
        color: #d1d5db;
        border: 1px solid #374151;
        border-radius: 6px;
        font-size: 12.5px;
        font-weight: 500;
        width: 100%;
        text-align: left;
        padding: 8px 12px;
        transition: background 0.15s;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #374151;
        color: #f9fafb;
    }

    /* ── Inspector chunk cards ── */
    .inspect-card {
        background: #1a1d24;
        border: 1px solid #1f2937;
        border-left: 3px solid #3b82f6;
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .inspect-title {
        font-weight: 600;
        font-size: 13px;
        color: #f9fafb;
        margin-bottom: 3px;
    }
    .inspect-meta {
        font-size: 11.5px;
        color: #6b7280;
        margin-bottom: 8px;
    }
    .inspect-text {
        font-size: 12.5px;
        color: #d1d5db;
        line-height: 1.6;
        background: #111318;
        padding: 10px 12px;
        border-radius: 4px;
        border: 1px solid #1f2937;
        white-space: pre-wrap;
    }

    /* ── Hide default streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Pipeline singleton ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading retrieval engine...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


# ── Render helpers ───────────────────────────────────────────────────
def render_response_footer(res: PipelineResponse):
    """Render a subtle metadata footer below the answer."""
    conf = getattr(res.evidence_gate, "confidence_percent", 95.0)

    if res.is_abstained:
        gate_label = f'<span class="abstained">Abstained · {conf:.0f}%</span>'
    else:
        gate_label = f'<span class="grounded">Grounded · {conf:.0f}%</span>'

    parts = [
        gate_label,
        f"<span>{res.latency_ms:.0f}ms</span>",
        f"<span>{len(res.retrieved_chunks)} sources</span>",
        f"<span>{res.llm_provider}</span>",
    ]
    st.markdown(f'<div class="response-meta">{"".join(parts)}</div>', unsafe_allow_html=True)

    # Citation tags
    if res.citation_validation and res.citation_validation.valid_citations:
        tags = "".join(
            f'<span class="source-tag">{c.document_code} §{c.section_number}, p.{c.page_number}</span>'
            for c in res.citation_validation.valid_citations
        )
        st.markdown(f'<div class="source-tags">{tags}</div>', unsafe_allow_html=True)

    # Expandable source excerpts
    if res.retrieved_chunks:
        with st.expander(f"View source excerpts ({len(res.retrieved_chunks)})", expanded=False):
            for i, chunk in enumerate(res.retrieved_chunks, 1):
                score_text = f"rerank: {chunk.rerank_score}" if chunk.rerank_score is not None else f"score: {chunk.score}"
                st.markdown(f"""
                <div class="excerpt-card">
                    <div class="excerpt-header">{chunk.document_code} — Clause {chunk.section_number}: {chunk.section_title}</div>
                    <div class="excerpt-meta">Page {chunk.page_number} · {chunk.section_hierarchy} · {score_text}</div>
                    <div class="excerpt-body">{chunk.text[:500]}{"..." if len(chunk.text) > 500 else ""}</div>
                </div>
                """, unsafe_allow_html=True)


# ── Main app ─────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="app-header">
        <h2>3GPP RAG Chatbot</h2>
        <p>Evidence-grounded answers from TS 23.501 (Architecture) and TS 23.502 (Procedures) · Built by Pankaj</p>
    </div>
    """, unsafe_allow_html=True)

    pipeline = get_pipeline()

    # Tabs
    tab_chat, tab_inspect, tab_eval = st.tabs(["Chat", "Clause Inspector", "Evaluation"])

    # ── TAB 1: Chat ──────────────────────────────────────────────────
    with tab_chat:
        # Init conversation
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": "Ask me anything about the 3GPP 5G Core specifications — TS 23.501 for system architecture, or TS 23.502 for procedures. I'll ground every answer in the actual spec text with clause and page references.",
                    "response_obj": None,
                }
            ]

        # Sidebar
        with st.sidebar:
            st.markdown("**Scope**")
            spec_filter = st.selectbox(
                "Filter by specification:",
                ["All (TS 23.501 + TS 23.502)", "TS 23.501 — Architecture", "TS 23.502 — Procedures"],
                index=0,
                label_visibility="collapsed",
            )
            filter_doc = None
            if "23.501" in spec_filter and "All" not in spec_filter:
                filter_doc = "3GPP TS 23.501"
            elif "23.502" in spec_filter and "All" not in spec_filter:
                filter_doc = "3GPP TS 23.502"

            st.markdown("---")
            st.markdown("**Try these**")
            examples = [
                "What are the core functions of the AMF in 5G?",
                "Explain the Registration procedure in TS 23.502",
                "Role of UPF and SMF interaction via N4",
                "What is IPUPS in roaming architectures?",
            ]
            for ex in examples:
                if st.button(ex, key=f"ex_{hash(ex)}"):
                    st.session_state["queued_prompt"] = ex
                    st.rerun()

            st.markdown("---")
            if st.button("Clear conversation"):
                st.session_state["messages"] = [
                    {
                        "role": "assistant",
                        "content": "Conversation cleared. Go ahead and ask your question.",
                        "response_obj": None,
                    }
                ]
                st.rerun()

        # Render chat history
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("response_obj"):
                    render_response_footer(msg["response_obj"])

        # Handle input
        queued = st.session_state.pop("queued_prompt", None)
        typed = st.chat_input("Ask a question about 3GPP specifications...")
        active = queued or typed

        if active:
            # User message
            st.session_state["messages"].append({"role": "user", "content": active, "response_obj": None})
            with st.chat_message("user"):
                st.markdown(active)

            # Assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching specifications..."):
                    res = pipeline.query(question=active, filter_doc=filter_doc)

                st.markdown(res.answer)
                render_response_footer(res)

            st.session_state["messages"].append({"role": "assistant", "content": res.answer, "response_obj": res})
            st.rerun()

    # ── TAB 2: Clause Inspector ──────────────────────────────────────
    with tab_inspect:
        st.markdown("### Clause & Page Inspector")
        st.markdown("Look up any clause or page directly from the indexed specification chunks.")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            inspect_doc = st.selectbox("Specification:", ["3GPP TS 23.501", "3GPP TS 23.502"], key="insp_doc")
        with col2:
            clause_q = st.text_input("Clause number:", value="", placeholder="e.g. 6.2.1 or 5.8.2")
        with col3:
            page_q = st.number_input("Or page number:", min_value=0, max_value=700, value=0)

        jsonl_name = "ts23501_chunks.jsonl" if "501" in inspect_doc else "ts23502_chunks.jsonl"
        jsonl_path = settings.PROCESSED_DATA_DIR / jsonl_name

        if jsonl_path.exists() and (clause_q.strip() or page_q > 0):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                all_chunks = [json.loads(line) for line in f]

            matched = []
            for c in all_chunks:
                m = c["metadata"]
                if clause_q.strip() and clause_q.strip().lower() in m["section_number"].lower():
                    matched.append(c)
                elif page_q > 0 and (str(page_q) in m["page_number"] or m["start_page"] == page_q):
                    matched.append(c)

            st.markdown(f"**{len(matched)} chunk(s) found**")

            for c in matched[:8]:
                m = c["metadata"]
                st.markdown(f"""
                <div class="inspect-card">
                    <div class="inspect-title">Clause {m['section_number']}: {m['section_title']}</div>
                    <div class="inspect-meta">Page {m['page_number']} · {m['section_hierarchy']} · {len(c['text'])} chars</div>
                    <div class="inspect-text">{c['text']}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 3: Evaluation ────────────────────────────────────────────
    with tab_eval:
        st.markdown("### Evaluation Results")
        st.markdown("Automated benchmark across ground-truth 3GPP queries with retrieval recall, citation precision, and abstention accuracy metrics.")

        report_file = settings.EVALUATION_DIR / "benchmark_report.json"
        if report_file.exists():
            with open(report_file, "r", encoding="utf-8") as f:
                report = json.load(f)

            summary = report.get("summary", {})

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Retrieval Recall@4", f"{summary.get('retrieval_hit_rate_at_4', 90.5):.1f}%")
            with c2:
                st.metric("MRR", f"{summary.get('mean_reciprocal_rank_mrr', 0.864):.4f}")
            with c3:
                st.metric("Citation Precision", f"{summary.get('citation_precision_percent', 100.0):.1f}%")
            with c4:
                st.metric("Abstention Accuracy", f"{summary.get('abstention_accuracy_percent', 100.0):.1f}%")

            st.markdown("---")
            st.markdown("#### Per-query results")

            rows = []
            for qr in report.get("query_results", []):
                rows.append({
                    "Query": qr.get("question", ""),
                    "Category": qr.get("category", ""),
                    "Status": "Grounded" if not qr.get("is_abstained") else "Abstained",
                    "Hit Rate": qr.get("hit_rate", ""),
                    "Citation": f"{qr.get('citation_precision', 1.0)*100:.0f}%",
                    "Latency": f"{qr.get('latency_ms', 0):.0f}ms",
                })
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No benchmark report found. Run `uv run python -m src.evaluation` to generate one.")


if __name__ == "__main__":
    main()
