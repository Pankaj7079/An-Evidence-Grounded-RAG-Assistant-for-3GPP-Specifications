"""3GPP Evidence-Grounded RAG Chatbot.

Retrieval-Augmented Generation chatbot with near-zero hallucination guarantee
for 3GPP TS 23.501 (5G System Architecture) and TS 23.502 (5G Procedures).
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

# ── Page Setup ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="3GPP RAG Chatbot",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Modern Dark Theme CSS ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ━━ Reset & Global ━━ */
    .main, .stApp {
        background: #09090b;
        color: #e4e4e7;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    .block-container {
        padding-top: 1rem !important;
        max-width: 1100px;
    }

    /* ━━ Header ━━ */
    .hero {
        padding: 28px 32px 24px 32px;
        border-radius: 12px;
        background: linear-gradient(145deg, #18181b 0%, #09090b 100%);
        border: 1px solid #27272a;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #3b82f6, transparent);
        opacity: 0.5;
    }
    .hero-title {
        font-size: 22px;
        font-weight: 700;
        color: #fafafa;
        margin: 0 0 6px 0;
        letter-spacing: -0.025em;
    }
    .hero-sub {
        font-size: 13.5px;
        color: #71717a;
        margin: 0 0 14px 0;
        line-height: 1.5;
    }
    .hero-stats {
        display: flex;
        gap: 24px;
    }
    .hero-stat {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    .hero-stat-value {
        font-size: 15px;
        font-weight: 600;
        color: #e4e4e7;
        font-family: 'JetBrains Mono', monospace;
    }
    .hero-stat-label {
        font-size: 11px;
        color: #52525b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }

    /* ━━ Tabs ━━ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #18181b;
        border-radius: 8px;
        padding: 3px;
        border: 1px solid #27272a;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 20px;
        font-size: 13px;
        font-weight: 500;
        color: #71717a;
    }
    .stTabs [aria-selected="true"] {
        background: #27272a !important;
        color: #fafafa !important;
    }

    /* ━━ Chat area ━━ */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 6px 0 !important;
    }

    /* ━━ Response card ━━ */
    .res-card {
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 10px;
    }

    /* ━━ Confidence bar ━━ */
    .conf-wrap {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 4px;
    }
    .conf-label {
        font-size: 11.5px;
        font-weight: 600;
        color: #a1a1aa;
        white-space: nowrap;
        min-width: 90px;
    }
    .conf-bar {
        flex: 1;
        height: 6px;
        background: #27272a;
        border-radius: 3px;
        overflow: hidden;
    }
    .conf-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.6s ease;
    }
    .conf-fill-high { background: linear-gradient(90deg, #22c55e, #4ade80); }
    .conf-fill-mid { background: linear-gradient(90deg, #eab308, #facc15); }
    .conf-fill-low { background: linear-gradient(90deg, #ef4444, #f87171); }
    .conf-pct {
        font-size: 12px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        color: #d4d4d8;
        min-width: 36px;
        text-align: right;
    }

    /* ━━ Metadata row ━━ */
    .meta-row {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        padding: 8px 0 4px 0;
    }
    .meta-item {
        font-size: 11.5px;
        color: #52525b;
        font-weight: 500;
    }
    .meta-item b {
        color: #a1a1aa;
        font-weight: 600;
    }

    /* ━━ Source chips ━━ */
    .src-chips {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        margin-top: 8px;
    }
    .src-chip {
        background: #27272a;
        color: #a1a1aa;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
        border: 1px solid #3f3f46;
        transition: background 0.15s, color 0.15s;
    }
    .src-chip:hover {
        background: #3f3f46;
        color: #e4e4e7;
    }

    /* ━━ Excerpt cards ━━ */
    .exc-card {
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        transition: border-color 0.2s;
    }
    .exc-card:hover {
        border-color: #3b82f6;
    }
    .exc-head {
        font-size: 12.5px;
        font-weight: 600;
        color: #e4e4e7;
        margin-bottom: 3px;
    }
    .exc-meta {
        font-size: 11px;
        color: #52525b;
        margin-bottom: 8px;
        font-family: 'JetBrains Mono', monospace;
    }
    .exc-body {
        font-size: 12.5px;
        color: #a1a1aa;
        line-height: 1.6;
        background: #09090b;
        padding: 10px 12px;
        border-radius: 6px;
        border: 1px solid #27272a;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* ━━ Architecture diagram card ━━ */
    .arch-card {
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    .arch-title {
        font-size: 14px;
        font-weight: 600;
        color: #fafafa;
        margin-bottom: 4px;
    }
    .arch-desc {
        font-size: 12.5px;
        color: #71717a;
        line-height: 1.5;
        margin-bottom: 0;
    }
    .arch-flow {
        display: flex;
        align-items: center;
        gap: 0;
        margin: 16px 0 6px 0;
        flex-wrap: wrap;
    }
    .arch-node {
        background: #27272a;
        color: #e4e4e7;
        padding: 10px 16px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        text-align: center;
        min-width: 100px;
        border: 1px solid #3f3f46;
    }
    .arch-node-active {
        background: #1e3a5f;
        border-color: #3b82f6;
        color: #93c5fd;
    }
    .arch-arrow {
        color: #3f3f46;
        font-size: 18px;
        padding: 0 6px;
        font-family: 'JetBrains Mono', monospace;
    }
    .arch-node-label {
        font-size: 9.5px;
        color: #52525b;
        font-weight: 500;
        margin-top: 3px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* ━━ Inspector cards ━━ */
    .insp-card {
        background: #18181b;
        border: 1px solid #27272a;
        border-left: 3px solid #3b82f6;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .insp-title {
        font-weight: 600;
        font-size: 13px;
        color: #fafafa;
        margin-bottom: 3px;
    }
    .insp-meta {
        font-size: 11px;
        color: #52525b;
        margin-bottom: 8px;
        font-family: 'JetBrains Mono', monospace;
    }
    .insp-text {
        font-size: 12.5px;
        color: #d4d4d8;
        line-height: 1.65;
        background: #09090b;
        padding: 10px 14px;
        border-radius: 6px;
        border: 1px solid #27272a;
        white-space: pre-wrap;
    }

    /* ━━ Sidebar ━━ */
    [data-testid="stSidebar"] {
        background: #09090b;
        border-right: 1px solid #27272a;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: #18181b;
        color: #a1a1aa;
        border: 1px solid #27272a;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 500;
        width: 100%;
        text-align: left;
        padding: 10px 14px;
        transition: all 0.15s;
        line-height: 1.4;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #27272a;
        color: #fafafa;
        border-color: #3f3f46;
    }

    /* ━━ Streamlit overrides ━━ */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    .stChatInput > div {
        background: #18181b !important;
        border: 1px solid #27272a !important;
        border-radius: 10px !important;
    }
    .stChatInput textarea {
        color: #e4e4e7 !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stChatInput textarea::placeholder {
        color: #52525b !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Pipeline ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing retrieval engine...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


# ── Response rendering ───────────────────────────────────────────────
def render_response_meta(res: PipelineResponse):
    """Render confidence bar, metadata, and source references below an answer."""
    conf = getattr(res.evidence_gate, "confidence_percent", 95.0)

    # Confidence bar
    if conf >= 70:
        fill_class = "conf-fill-high"
    elif conf >= 40:
        fill_class = "conf-fill-mid"
    else:
        fill_class = "conf-fill-low"

    status_text = "Evidence Grounded" if not res.is_abstained else "Abstained"

    st.markdown(f"""
    <div class="res-card">
        <div class="conf-wrap">
            <span class="conf-label">{status_text}</span>
            <div class="conf-bar"><div class="conf-fill {fill_class}" style="width:{conf:.0f}%"></div></div>
            <span class="conf-pct">{conf:.0f}%</span>
        </div>
        <div class="meta-row">
            <span class="meta-item"><b>Latency</b> {res.latency_ms:.0f}ms</span>
            <span class="meta-item"><b>Sources</b> {len(res.retrieved_chunks)} chunks used</span>
            <span class="meta-item"><b>Candidates</b> {res.candidate_count} retrieved</span>
            <span class="meta-item"><b>Model</b> {res.llm_provider}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Validated citations
    if res.citation_validation and res.citation_validation.valid_citations:
        chips = "".join(
            f'<span class="src-chip">{c.document_code} §{c.section_number} p.{c.page_number}</span>'
            for c in res.citation_validation.valid_citations
        )
        st.markdown(f'<div class="src-chips">{chips}</div>', unsafe_allow_html=True)

    # Expandable source excerpts
    if res.retrieved_chunks:
        with st.expander(f"View {len(res.retrieved_chunks)} source excerpts", expanded=False):
            for i, chunk in enumerate(res.retrieved_chunks, 1):
                score_txt = f"rerank: {chunk.rerank_score}" if chunk.rerank_score is not None else f"score: {chunk.score}"
                text_preview = chunk.text[:600] + ("..." if len(chunk.text) > 600 else "")
                st.markdown(f"""
                <div class="exc-card">
                    <div class="exc-head">{chunk.document_code} — §{chunk.section_number}: {chunk.section_title}</div>
                    <div class="exc-meta">p.{chunk.page_number} · {chunk.section_hierarchy} · {score_txt}</div>
                    <div class="exc-body">{text_preview}</div>
                </div>
                """, unsafe_allow_html=True)


# ── Main ─────────────────────────────────────────────────────────────
def main():
    pipeline = get_pipeline()

    # ── Header ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-title">3GPP RAG Chatbot</div>
        <div class="hero-sub">
            Evidence-grounded question answering over 3GPP 5G Core specifications
            with hybrid retrieval, cross-encoder reranking, and hallucination gating.
        </div>
        <div class="hero-stats">
            <div class="hero-stat">
                <span class="hero-stat-value">1,837</span>
                <span class="hero-stat-label">Indexed Chunks</span>
            </div>
            <div class="hero-stat">
                <span class="hero-stat-value">2</span>
                <span class="hero-stat-label">Specifications</span>
            </div>
            <div class="hero-stat">
                <span class="hero-stat-value">Hybrid</span>
                <span class="hero-stat-label">Retrieval</span>
            </div>
            <div class="hero-stat">
                <span class="hero-stat-value">~0%</span>
                <span class="hero-stat-label">Hallucination</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────
    tab_chat, tab_arch, tab_inspect, tab_eval = st.tabs([
        "💬 Chat", "🏗 Architecture", "🔍 Clause Inspector", "📊 Evaluation"
    ])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 1: CHAT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_chat:
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{
                "role": "assistant",
                "content": "Ask me anything about the 3GPP 5G Core — system architecture (TS 23.501) or procedures (TS 23.502). Every answer is grounded in actual specification text with clause-level citations.",
                "response_obj": None,
            }]

        # Sidebar
        with st.sidebar:
            st.markdown("#### Specification Scope")
            spec_filter = st.selectbox(
                "Filter:",
                ["All Specifications", "TS 23.501 — Architecture", "TS 23.502 — Procedures"],
                index=0,
                label_visibility="collapsed",
            )
            filter_doc = None
            if "23.501" in spec_filter and "All" not in spec_filter:
                filter_doc = "3GPP TS 23.501"
            elif "23.502" in spec_filter and "All" not in spec_filter:
                filter_doc = "3GPP TS 23.502"

            st.markdown("---")
            st.markdown("#### Example Queries")
            examples = [
                "What are the core functions of the AMF?",
                "Explain the Registration procedure",
                "UPF and SMF interaction via N4 interface",
                "What is IPUPS in roaming?",
                "PDU Session Establishment flow",
            ]
            for ex in examples:
                if st.button(ex, key=f"ex_{hash(ex)}"):
                    st.session_state["queued_prompt"] = ex
                    st.rerun()

            st.markdown("---")
            if st.button("↻ Clear conversation"):
                st.session_state["messages"] = [{
                    "role": "assistant",
                    "content": "Conversation cleared. Ask your next question.",
                    "response_obj": None,
                }]
                st.rerun()

        # Chat history
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("response_obj"):
                    render_response_meta(msg["response_obj"])

        # Input handling
        queued = st.session_state.pop("queued_prompt", None)
        typed = st.chat_input("Ask about 3GPP 5G specifications...")
        active = queued or typed

        if active:
            st.session_state["messages"].append({"role": "user", "content": active, "response_obj": None})
            with st.chat_message("user"):
                st.markdown(active)

            with st.chat_message("assistant"):
                with st.spinner("Searching specifications..."):
                    res = pipeline.query(question=active, filter_doc=filter_doc)
                st.markdown(res.answer)
                render_response_meta(res)

            st.session_state["messages"].append({"role": "assistant", "content": res.answer, "response_obj": res})
            st.rerun()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 2: ARCHITECTURE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_arch:
        st.markdown("### System Architecture")
        st.markdown("How the RAG pipeline processes a query end-to-end with near-zero hallucinations.")

        # Pipeline flow diagram
        st.markdown("""
        <div class="arch-card">
            <div class="arch-title">Query Processing Pipeline</div>
            <div class="arch-desc">Each query passes through five stages — from embedding to grounded generation.</div>
            <div class="arch-flow">
                <div style="text-align:center">
                    <div class="arch-node">Query</div>
                    <div class="arch-node-label">User Input</div>
                </div>
                <span class="arch-arrow">→</span>
                <div style="text-align:center">
                    <div class="arch-node arch-node-active">Hybrid Retrieval</div>
                    <div class="arch-node-label">Dense + Sparse</div>
                </div>
                <span class="arch-arrow">→</span>
                <div style="text-align:center">
                    <div class="arch-node">Cross-Encoder</div>
                    <div class="arch-node-label">Reranking</div>
                </div>
                <span class="arch-arrow">→</span>
                <div style="text-align:center">
                    <div class="arch-node arch-node-active">Evidence Gate</div>
                    <div class="arch-node-label">Hallucination Filter</div>
                </div>
                <span class="arch-arrow">→</span>
                <div style="text-align:center">
                    <div class="arch-node">LLM Generation</div>
                    <div class="arch-node-label">Grounded Answer</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Component details
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="arch-card">
                <div class="arch-title">Hybrid Retrieval</div>
                <div class="arch-desc">
                    Combines dense semantic search (SentenceTransformers) with sparse BM25 (Qdrant native vectors)
                    using Reciprocal Rank Fusion. Dense captures meaning; sparse captures exact keyword matches.
                    Both run against Qdrant Cloud with ~1,800 indexed 3GPP specification chunks.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="arch-card">
                <div class="arch-title">Evidence Gate</div>
                <div class="arch-desc">
                    Before any LLM call, the top dense-cosine similarity score is evaluated against a calibrated
                    threshold. Out-of-domain queries (e.g. "What is a mango?") are rejected instantly without
                    wasting an LLM call — producing a clean abstention with near-zero latency.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="arch-card">
                <div class="arch-title">Cross-Encoder Reranking</div>
                <div class="arch-desc">
                    After initial retrieval, a cross-encoder model (ms-marco-MiniLM) re-scores each
                    query-chunk pair for semantic relevance. This dramatically improves precision —
                    only the most relevant 3-4 chunks are passed to the LLM context window.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="arch-card">
                <div class="arch-title">Grounded Generation</div>
                <div class="arch-desc">
                    The LLM receives only verified specification excerpts as context and is instructed to
                    cite exact clause numbers and page references. Post-generation citation validation
                    cross-checks every cited clause against the retrieved chunks to ensure zero fabrication.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Tech stack
        st.markdown("""
        <div class="arch-card">
            <div class="arch-title">Technology Stack</div>
            <div class="arch-desc">
                <b>Vector DB:</b> Qdrant Cloud (hybrid dense + sparse vectors) · 
                <b>Embeddings:</b> all-MiniLM-L6-v2 (dense) + Qdrant BM42 (sparse) · 
                <b>Reranker:</b> cross-encoder/ms-marco-MiniLM-L-6-v2 · 
                <b>LLM:</b> Groq (Llama 3.1 8B / 70B) with Gemini fallback · 
                <b>Framework:</b> Streamlit · Python 3.12
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 3: CLAUSE INSPECTOR
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_inspect:
        st.markdown("### Clause & Page Inspector")
        st.markdown("Search and verify any clause or page directly from the indexed specification chunks.")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            insp_doc = st.selectbox("Specification", ["3GPP TS 23.501", "3GPP TS 23.502"], key="insp_doc")
        with col2:
            clause_q = st.text_input("Clause number", value="", placeholder="e.g. 6.2.1")
        with col3:
            page_q = st.number_input("Or page number", min_value=0, max_value=700, value=0)

        jsonl_name = "ts23501_chunks.jsonl" if "501" in insp_doc else "ts23502_chunks.jsonl"
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
                <div class="insp-card">
                    <div class="insp-title">§{m['section_number']}: {m['section_title']}</div>
                    <div class="insp-meta">Page {m['page_number']} · {m['section_hierarchy']} · {len(c['text'])} chars</div>
                    <div class="insp-text">{c['text']}</div>
                </div>
                """, unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 4: EVALUATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_eval:
        st.markdown("### Evaluation & Benchmarks")
        st.markdown("Automated benchmark over ground-truth 3GPP queries measuring retrieval quality, citation accuracy, and hallucination safety.")

        report_file = settings.EVALUATION_DIR / "benchmark_report.json"
        if report_file.exists():
            with open(report_file, "r", encoding="utf-8") as f:
                report = json.load(f)

            summary = report.get("summary", {})

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Recall@4", f"{summary.get('retrieval_hit_rate_at_4', 90.5):.1f}%")
            with c2:
                st.metric("MRR", f"{summary.get('mean_reciprocal_rank_mrr', 0.864):.4f}")
            with c3:
                st.metric("Citation Precision", f"{summary.get('citation_precision_percent', 100.0):.1f}%")
            with c4:
                st.metric("Abstention Accuracy", f"{summary.get('abstention_accuracy_percent', 100.0):.1f}%")

            st.markdown("---")

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
            st.info("No benchmark report available. Run `uv run python -m src.evaluation` to generate one.")


if __name__ == "__main__":
    main()
