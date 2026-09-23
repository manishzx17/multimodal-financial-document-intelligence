"""Interactive Visual Evidence & Citation Viewer — AuditRAG Streamlit Frontend.

Provides an interactive audit workbench to inspect answers, toggle between
visual citations [1], [2], view exact financial document page images, examine
highlighted bounding-box overlays, trace numerical reasoning, and review
comprehensive evaluation and reliability metrics (Addition 1, 2, and 3).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

# Threading constraints for PyTorch / tokenizers on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE_MONITOR"] = "1"

import torch
torch.set_num_threads(1)

import streamlit as st
from PIL import Image

from auditrag.config import PROJECT_ROOT, DEFAULT_VLM_PROVIDER
from auditrag.pipeline.engine import AuditRAGPipeline, AuditRAGResponse
from auditrag.ui.citation_viewer import (
    get_citation_details,
    get_or_create_citation_overlay,
    resolve_image_path,
)
from auditrag.ui.dashboard import render_evaluation_dashboard
from auditrag.ui.numerical_reasoning import extract_numerical_traces

# ---------------------------------------------------------------------------
# Pre-configured Sample Benchmark Queries for Instant Interactive Testing
# ---------------------------------------------------------------------------
SAMPLE_QUERIES = [
    {
        "label": "Depreciation for AMER in 2018 (Verified & Accepted, Multiple Citations)",
        "query": "What was the depreciation for AMER in 2018?",
    },
    {
        "label": "Equity in Net Earnings of Affiliates (Verified & Accepted)",
        "query": "In which years was the equity in net earnings of affiliates recorded for?",
    },
    {
        "label": "Total Assets from AMER in 2018 (Relevance Gated - Abstained)",
        "query": "What was the total assets from AMER in 2018?",
    },
    {
        "label": "Goodwill in 2018 and 2019 (Metric Mismatch - Abstained)",
        "query": "What are the respective goodwill at 2018 and 2019?",
    },
    {
        "label": "Capital Intensity Ratio Change for BCE (Calculation - Abstained)",
        "query": "What is the % change in the capital intensity ratio for BCE?",
    },
]


@st.cache_resource(show_spinner=False)
def load_pipeline() -> AuditRAGPipeline:
    """Initialize and cache the unified AuditRAG pipeline."""
    return AuditRAGPipeline()


def main():
    st.set_page_config(
        page_title="AuditRAG — Financial Audit & Reliability Workbench",
        page_icon="📑",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # -----------------------------------------------------------------------
    # Sidebar: System Configuration & Sample Queries
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.title("📑 AuditRAG Engine")
        st.markdown(
            "**Multimodal Financial RAG** with hybrid retrieval, "
            "visual bounding-box citations, factuality verification, and "
            "explainable numerical reasoning traces."
        )
        st.divider()

        st.subheader("⚙️ Pipeline Configuration")
        top_k = st.slider("Top-k Retrieval Candidates", min_value=1, max_value=10, value=5, step=1)

        provider_name = DEFAULT_VLM_PROVIDER.upper()
        st.caption(f"**Active VLM Provider:** `{provider_name}`")
        if provider_name == "MOCK":
            st.info("Running with offline deterministic `MockVLMProvider`. Set `OPENAI_API_KEY` for live GPT-4o Vision.")
        else:
            st.success("OpenAI Vision VLM active.")

        st.divider()

        st.subheader("💡 Curated Sample Queries")
        selected_sample = st.selectbox(
            "Select a benchmark sample to test:",
            options=SAMPLE_QUERIES,
            format_func=lambda s: s["label"],
            index=0,
        )

        if st.button("📥 Load Selected Query", use_container_width=True):
            st.session_state["query_input"] = selected_sample["query"]

        st.divider()
        st.markdown(
            "### 🔍 Legend\n"
            "- 🟢 **Amber / Red Box:** Document layout block containing verified evidence.\n"
            "- 🏷️ **Badge [n]:** Visual citation indicator linking answer claims to exact coordinates.\n"
            "- 🔢 **Numerical Trace:** Step-by-step formula and arithmetic execution.\n"
            "- 🛡️ **Abstention Gate:** Protects against ungrounded or wrong-metric answers."
        )

    # -----------------------------------------------------------------------
    # Top-Level Mode Selection: Interactive Workbench vs Reliability Dashboard
    # -----------------------------------------------------------------------
    tab_workbench, tab_dashboard = st.tabs([
        "🔍 Interactive Audit Workbench",
        "📊 Evaluation & Reliability Dashboard",
    ])

    with tab_workbench:
        render_interactive_workbench(top_k=top_k)

    with tab_dashboard:
        render_evaluation_dashboard()


def render_interactive_workbench(top_k: int = 5):
    """Render the interactive query, visual citation viewer, and numerical reasoning trace."""
    # -----------------------------------------------------------------------
    # Main Header
    # -----------------------------------------------------------------------
    st.title("📑 AuditRAG: Interactive Visual Evidence & Citation Viewer")
    st.markdown(
        "Inspect generated answers with **interactive bounding-box citations**, "
        "side-by-side financial page visualization, and granular evidence verification."
    )
    st.write("")

    # Query Input Box
    default_query = st.session_state.get("query_input", SAMPLE_QUERIES[0]["query"])
    query_text = st.text_input(
        "Enter financial audit question:",
        value=default_query,
        placeholder="e.g. What was the depreciation for AMER in 2018?",
        key="query_text_field",
    )

    col_btn, col_clear = st.columns([2, 8])
    with col_btn:
        run_query = st.button("🔍 Run AuditRAG Query", type="primary", use_container_width=True)

    # -----------------------------------------------------------------------
    # Pipeline Execution
    # -----------------------------------------------------------------------
    if run_query:
        if not query_text or not query_text.strip():
            st.error("Please enter a valid question.")
            return

        with st.spinner("Executing Multimodal Retrieval, VLM Generation, Visual Citation Extraction, and Verification..."):
            t0 = time.time()
            try:
                pipeline = load_pipeline()
                resp: AuditRAGResponse = pipeline.run(query_text.strip(), top_k=top_k)
                elapsed = time.time() - t0
                st.session_state["pipeline_response"] = resp
                st.session_state["pipeline_latency"] = elapsed
                st.session_state["selected_citation_idx"] = 0
            except Exception as e:
                st.error(f"Pipeline execution error: {e}")
                return

    # -----------------------------------------------------------------------
    # Response Display
    # -----------------------------------------------------------------------
    resp: Optional[AuditRAGResponse] = st.session_state.get("pipeline_response")

    if resp is not None:
        st.divider()

        # 1. Answer & Safety Decision Banner
        c_status, c_meta = st.columns([3, 1])
        with c_status:
            if resp.is_abstained:
                st.warning(
                    f"🛡️ **Hallucination Protection Active: ABSTAINED**  \n"
                    f"**Reason:** `{resp.abstention_reason}` — The system abstained from certifying this answer."
                )
            else:
                st.success("✅ **Verified & Certified Answer** — All claims are supported by retrieved financial disclosures.")

        with c_meta:
            elapsed_str = f"{st.session_state.get('pipeline_latency', 0.0):.2f} s" if "pipeline_latency" in st.session_state else "< 2.0 s"
            st.metric("Execution Latency", elapsed_str)

        st.subheader("💬 Generated Answer")
        final_answer = resp.final_answer or (resp.answer_with_citations.answer if resp.answer_with_citations else "No answer produced.")
        st.markdown(f"> {final_answer}")

        # 2. Verification Summary Metrics
        if resp.verification_report:
            vr = resp.verification_report
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Supported Claims", str(vr.supported_count))
            m2.metric("Contradicted Claims", str(vr.contradicted_count))
            m3.metric("Unsupported Claims", str(vr.not_supported_count))
            m4.metric("Question Addressed", "Yes" if vr.is_question_addressed else "No")

            # Deep-dive claims expander
            with st.expander("🔍 Inspect Granular Claim Extraction & Verification Results", expanded=False):
                for res in vr.results:
                    icon = "✅" if res.status.value == "supported" else ("❌" if res.status.value == "contradicted" else "⚠️")
                    cit_str = f" [Citation: `{res.citation_label}`]" if res.citation_label else ""
                    st.markdown(f"- {icon} **Claim:** *\"{res.claim.statement}\"* ({res.status.value.upper()}){cit_str}")
                    if res.explanation:
                        st.caption(f"  - *Verification note:* {res.explanation}")

        # -------------------------------------------------------------------
        # 3.5: Explainable Numerical Reasoning & Verification Trace (Addition 2)
        # -------------------------------------------------------------------
        if resp.verification_report:
            traces = extract_numerical_traces(resp.verification_report)

            if traces:
                st.subheader("🔢 Explainable Numerical Reasoning & Verification Trace")
                st.markdown(
                    "Deterministic, step-by-step arithmetic audit verifying source values, "
                    "financial formulas, and calculated figures against reported claims."
                )

                # Surfacing specific numerical abstention causes if triggered
                if resp.is_abstained:
                    reason_val = str(resp.abstention_reason).upper()
                    if "FAILED_ARITHMETIC_CHECK" in reason_val:
                        st.error(
                            "🛑 **Abstention Cause: FAILED_ARITHMETIC_CHECK**  \n"
                            "The answer stated a calculated figure that does not match the deterministic mathematical result."
                        )
                    elif "UNSUPPORTED_NUMERICAL_FACT" in reason_val:
                        st.warning(
                            "⚠️ **Abstention Cause: UNSUPPORTED_NUMERICAL_FACT**  \n"
                            "The numerical figures in the claim could not be grounded in the retrieved financial disclosures."
                        )

                for t_idx, trace in enumerate(traces):
                    with st.container():
                        # Header with claim & status badge
                        if trace.status == "verified":
                            badge_md = "🟢 **Status:** `✅ Arithmetic Verified`"
                        elif trace.status == "supported":
                            badge_md = "🟢 **Status:** `✅ Grounded in Evidence`"
                        elif trace.status == "failed_arithmetic":
                            badge_md = "🔴 **Status:** `❌ Arithmetic Error`"
                        elif trace.status == "unsupported":
                            badge_md = "🟡 **Status:** `⚠️ Ungrounded Numerical Value`"
                        else:
                            badge_md = f"⚪ **Status:** `{trace.status.upper()}`"

                        st.markdown(f"**Verified Claim:** *“{trace.claim_statement}”*  \n{badge_md}")

                        # 4-column inspection trace
                        col_src, col_formula, col_calc, col_rep = st.columns([1.2, 1.5, 1.0, 1.0])

                        with col_src:
                            st.markdown("**1. Source Evidence Values**")
                            if trace.source_values:
                                for sv in trace.source_values:
                                    st.code(f"{sv:,.4g}" if isinstance(sv, (int, float)) else str(sv))
                            else:
                                st.caption("No numbers extracted from text")

                        with col_formula:
                            st.markdown("**2. Mathematical Formula**")
                            st.code(trace.formula_template or "N/A", language="text")
                            if trace.formula_expression:
                                st.caption(f"Trace: `{trace.formula_expression}`")

                        with col_calc:
                            st.markdown("**3. Calculated Value**")
                            st.code(trace.calculated_value or "N/A", language="text")

                        with col_rep:
                            st.markdown("**4. Reported Value**")
                            st.code(trace.reported_value or "N/A", language="text")

                        # Explanation and citation jump button
                        col_exp, col_jump = st.columns([3, 1])
                        with col_exp:
                            if trace.explanation:
                                st.info(f"ℹ️ **Audit Trail:** {trace.explanation}")
                        with col_jump:
                            if trace.citation_label:
                                if st.button(f"🔍 Inspect Citation {trace.citation_label}", key=f"jump_cit_{t_idx}"):
                                    # Set selected citation to the referenced index if available
                                    if resp.answer_with_citations and resp.answer_with_citations.citations:
                                        for c_i, c_obj in enumerate(resp.answer_with_citations.citations):
                                            if c_obj.label == trace.citation_label:
                                                st.session_state["selected_citation_idx"] = c_i
                                                st.rerun()

                        st.write("---")
            else:
                # Answer contains only textual claims and no numerical/calculation claims
                st.caption("ℹ️ *No numerical calculation required for this response (textual audit claim).*")

        # -------------------------------------------------------------------
        # 3. Interactive Visual Evidence & Citation Viewer (Addition 1)
        # -------------------------------------------------------------------
        st.subheader("🎯 Interactive Visual Evidence & Citation Viewer")

        citations = resp.answer_with_citations.citations if resp.answer_with_citations else []

        if not citations:
            st.info("ℹ️ No visual bounding-box citations are associated with this response.")
            return

        st.markdown(
            f"This answer contains **{len(citations)} visual citation(s)** linked directly "
            "to exact coordinates in the financial disclosure source documents:"
        )

        # Multi-citation selection tabs / buttons
        selected_idx = st.session_state.get("selected_citation_idx", 0)
        if selected_idx >= len(citations):
            selected_idx = 0
            st.session_state["selected_citation_idx"] = 0

        # Build citation selection radio
        cit_options = [f"{c.label} Page {c.page_number} ({c.document_id[:8]}...)" for c in citations]
        selected_cit_label = st.radio(
            "Select Citation to Inspect:",
            options=cit_options,
            index=selected_idx,
            horizontal=True,
            key="citation_selector_radio",
        )
        selected_idx = cit_options.index(selected_cit_label)
        st.session_state["selected_citation_idx"] = selected_idx

        selected_cit = citations[selected_idx]
        details = get_citation_details(selected_cit)

        # Multi-citation layout: left inspection details, right image viewer
        col_details, col_viewer = st.columns([1, 1.4], gap="medium")

        with col_details:
            st.markdown(f"### 🏷️ Citation `{selected_cit.label}` Details")

            st.markdown(f"**Document ID:** `{details['document_id']}`")
            if details.get("document_source"):
                st.markdown(f"**Document Source:** `{details['document_source']}`")
            st.markdown(f"**Page Number:** `{details['page_number']}`")

            st.markdown("#### 📑 Supporting Evidence Text")
            st.success(f"“{details['text']}”")

            st.markdown("#### 📐 Bounding Box Geometry")
            st.code(
                f"BBox: [x0: {details['bbox'][0]}, y0: {details['bbox'][1]}, "
                f"x1: {details['bbox'][2]}, y1: {details['bbox'][3]}]\n"
                f"Dimensions: {details['width']} × {details['height']} px\n"
                f"Word Box Alignments: {details.get('word_bboxes_count', 0)}",
                language="yaml",
            )

            # View options
            st.markdown("#### 🎛️ Viewer Display Mode")
            view_mode = st.selectbox(
                "Overlay Style:",
                [
                    "Highlighted Evidence Overlay (All Citations on Page)",
                    "Focused Citation Highlight (Single Citation)",
                    "Original Document Page (Unannotated)",
                    "Side-by-Side (Original vs Annotated)",
                ],
                index=0,
            )

            # Filter citations that are on the same page
            page_citations = [
                c for c in citations
                if c.document_id == selected_cit.document_id and c.page_number == selected_cit.page_number
            ]

            # Fast switching navigation buttons
            if len(citations) > 1:
                st.write("")
                col_prev, col_next = st.columns(2)
                with col_prev:
                    if st.button("⬅️ Previous Citation", use_container_width=True, disabled=(selected_idx == 0)):
                        st.session_state["selected_citation_idx"] = selected_idx - 1
                        st.rerun()
                with col_next:
                    if st.button("Next Citation ➡️", use_container_width=True, disabled=(selected_idx == len(citations) - 1)):
                        st.session_state["selected_citation_idx"] = selected_idx + 1
                        st.rerun()

        with col_viewer:
            st.markdown("### 📄 Document Page Visual Evidence")

            source_path = resolve_image_path(selected_cit.source_image_path)

            if not source_path or not source_path.exists():
                st.warning(
                    f"⚠️ Page image not found on disk at: `{selected_cit.source_image_path}`. "
                    "Ensure TAT-DQA page images exist in `data/raw/tatdqa/test/`."
                )
            else:
                # 1. Overlay (All citations on page)
                if view_mode == "Highlighted Evidence Overlay (All Citations on Page)":
                    overlay_path, is_new = get_or_create_citation_overlay(
                        citation=selected_cit,
                        all_citations_on_page=page_citations,
                        focused_only=False,
                    )
                    img_to_show = overlay_path or source_path
                    badge_info = "Generated On-Demand" if is_new else "Pre-rendered Overlay"
                    st.image(
                        str(img_to_show),
                        caption=f"Document: {selected_cit.document_id} | Page {selected_cit.page_number} ({badge_info})",
                        use_container_width=True,
                    )

                # 2. Focused single citation highlight
                elif view_mode == "Focused Citation Highlight (Single Citation)":
                    focused_path, _ = get_or_create_citation_overlay(
                        citation=selected_cit,
                        focused_only=True,
                    )
                    img_to_show = focused_path or source_path
                    st.image(
                        str(img_to_show),
                        caption=f"Focused Highlight for {selected_cit.label} | Page {selected_cit.page_number}",
                        use_container_width=True,
                    )

                # 3. Original unannotated page
                elif view_mode == "Original Document Page (Unannotated)":
                    st.image(
                        str(source_path),
                        caption=f"Original Unannotated Page: {selected_cit.document_id} | Page {selected_cit.page_number}",
                        use_container_width=True,
                    )

                # 4. Side-by-side comparison
                elif view_mode == "Side-by-Side (Original vs Annotated)":
                    overlay_path, _ = get_or_create_citation_overlay(
                        citation=selected_cit,
                        all_citations_on_page=page_citations,
                        focused_only=False,
                    )
                    c_orig, c_anno = st.columns(2)
                    with c_orig:
                        st.markdown("**Original Page**")
                        st.image(str(source_path), use_container_width=True)
                    with c_anno:
                        st.markdown("**Annotated Bounding Box**")
                        st.image(str(overlay_path or source_path), use_container_width=True)


if __name__ == "__main__":
    main()
