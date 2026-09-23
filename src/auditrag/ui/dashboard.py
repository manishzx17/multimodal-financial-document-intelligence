"""Evaluation & Reliability Dashboard — AuditRAG Streamlit Frontend (Addition 3).

Provides a comprehensive, Master's-application grade evaluation dashboard
visualizing retrieval benchmarks, claim-level factuality, hallucination protection,
and the three-stage pipeline evolution (Baseline → Improved Grounding → Relevance-Gated).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from auditrag.config import PROJECT_ROOT

DEFAULT_TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


# ---------------------------------------------------------------------------
# Data Loading Utilities (Deterministic Artifact Re-use)
# ---------------------------------------------------------------------------

def load_json_artifact(path: Path) -> Optional[Dict[str, Any]]:
    """Safely load and parse a benchmark JSON artifact."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_all_benchmark_reports(tables_dir: Optional[Path] = None) -> Dict[str, Optional[Dict[str, Any]]]:
    """Load all 4 existing benchmark JSON reports from the reports directory."""
    base_dir = tables_dir or DEFAULT_TABLES_DIR
    return {
        "retrieval": load_json_artifact(base_dir / "retrieval_benchmark.json"),
        "baseline": load_json_artifact(base_dir / "pipeline_benchmark.json"),
        "improved": load_json_artifact(base_dir / "pipeline_benchmark_improved.json"),
        "relevance": load_json_artifact(base_dir / "pipeline_benchmark_relevance.json"),
    }


# ---------------------------------------------------------------------------
# Dashboard Rendering Components
# ---------------------------------------------------------------------------

def render_reproducibility_notice(tables_dir: Path, figures_dir: Path):
    """Display audit reproducibility and benchmark data provenance disclosure."""
    with st.expander("📌 Benchmark Methodology, Data Provenance & Reproducibility Notice", expanded=False):
        st.markdown(
            """
            **Evaluation Provenance & Scope Disclosure:**
            - **Sample Dataset:** Standard deterministic evaluation sample of **15 financial questions**
              drawn from `tatdqa_dataset_test_gold.json` (spanning complex earnings tables, 10-K disclosures,
              and percentage/difference calculations).
            - **Honest Scope Notice:** These metrics reflect the project's established 15-question benchmark sample
              rather than an extrapolation to the full TAT-DQA corpus.
            - **Deterministic Artifacts:** All metrics, confusion tables, and figures rendered on this dashboard
              are loaded directly from pre-computed, version-controlled benchmark artifacts in `reports/tables/`
              and `reports/figures/`. No LLM inferences or retrievals are re-run during dashboard rendering.
            - **Benchmarking Script:** Can be reproduced offline via `python -m auditrag.evaluation.benchmark`.
            """
        )
        st.caption(f"Artifact Paths: Tables: `{tables_dir}` | Figures: `{figures_dir}`")


def render_retrieval_section(retrieval_data: Optional[Dict[str, Any]], figures_dir: Path):
    """Render Section 1: Retrieval Performance Comparison."""
    st.subheader("1. 🔎 Multimodal Retrieval Performance")
    st.markdown(
        "Benchmarking **Dense** (sentence-transformers/all-MiniLM-L6-v2), **Sparse** (BM25 with financial tokenization), "
        "**Vision-Language** (ColPali late-interaction), and **Hybrid RRF** (Reciprocal Rank Fusion) "
        "on document and exact page retrieval."
    )

    if not retrieval_data or "results" not in retrieval_data:
        st.warning("⚠️ Retrieval benchmark data (`retrieval_benchmark.json`) not found or empty.")
        return

    results = retrieval_data["results"]

    # Build comparison table
    rows = []
    for r in results:
        rows.append({
            "Retriever Modality": r.get("retriever_name", "Unknown"),
            "Doc-Hit@1": f"{r.get('doc_hit_at_1', 0.0) * 100:.1f}%",
            "Doc-Hit@3": f"{r.get('doc_hit_at_3', 0.0) * 100:.1f}%",
            "Doc-Hit@5": f"{r.get('doc_hit_at_5', 0.0) * 100:.1f}%",
            "Doc-MRR": f"{r.get('doc_mrr', 0.0):.4f}",
            "Page-Hit@1": f"{r.get('page_hit_at_1', 0.0) * 100:.1f}%",
            "Page-Hit@3": f"{r.get('page_hit_at_3', 0.0) * 100:.1f}%",
            "Page-Hit@5": f"{r.get('page_hit_at_5', 0.0) * 100:.1f}%",
            "Page-MRR": f"{r.get('page_mrr', 0.0):.4f}",
            "Avg Latency (ms)": f"{r.get('avg_latency_ms', 0.0):.1f} ms",
        })

    df_ret = pd.DataFrame(rows)
    st.dataframe(df_ret, use_container_width=True, hide_index=True)

    # Retrieval Figure & Engineering Takeaways
    col_fig, col_insights = st.columns([1.2, 1.0])
    with col_fig:
        fig_path = figures_dir / "retrieval_comparison.png"
        if fig_path.exists():
            st.image(str(fig_path), caption="Retrieval Performance Comparison (Doc vs Page Hit@1, Hit@5, MRR)", use_container_width=True)
        else:
            st.info("Retrieval comparison figure (`retrieval_comparison.png`) not found.")

    with col_insights:
        st.markdown("#### 💡 Retrieval Architecture Analysis")
        st.markdown(
            """
            - **BM25 Dominance on Lexical Matches:** BM25 achieves highest precision (`Doc-Hit@1 = 60.0%`, `Doc-Hit@5 = 73.3%`, `Latency = 6.0 ms`)
              because financial SEC filings feature hyper-specific ticker symbols, geographic segment names (e.g. *AMER*, *APAC*), and fiscal line-items.
            - **Dense Embedding Complementarity:** Dense retrieval (`Doc-Hit@5 = 73.3%`) successfully disambiguates synonyms and multi-sentence context.
            - **ColPali Multimodal Awareness:** Directly operates over document screenshot patches to index visual layout semantics.
            - **Hybrid RRF Balance:** RRF fusion ensures robust recall across both pure textual disclosures and structured graphic tables without score calibration artifacts.
            """
        )


def render_pipeline_metrics_overview(report_data: Optional[Dict[str, Any]], title_suffix: str = ""):
    """Render metric cards for answer quality, verification, and abstention."""
    if not report_data:
        st.warning("⚠️ Benchmark data not available.")
        return

    v_sum = report_data.get("verification_summary", {})
    a_sum = report_data.get("abstention_summary", {})

    st.markdown(f"### 2. 🎯 Answer Quality & Verification Overview {title_suffix}")

    # Row 1: Generation Accuracy
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        em = report_data.get("avg_exact_match", 0.0) * 100
        st.metric("Exact Match (EM)", f"{em:.1f}%")
    with c2:
        f1 = report_data.get("avg_token_f1", 0.0) * 100
        st.metric("Token F1 Score", f"{f1:.1f}%")
    with c3:
        num = report_data.get("avg_numeric_match", 0.0) * 100
        st.metric("Numeric Match", f"{num:.1f}%")
    with c4:
        lat = report_data.get("avg_latency_ms", 0.0)
        st.metric("Mean Latency", f"{lat:.1f} ms")

    # Row 2: Verification Reliability
    st.markdown("### 3. 🛡️ Claim-Level Verification & Reliability")
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        sup = v_sum.get("supported_claims", 0)
        tot = v_sum.get("total_claims", 1)
        rate = v_sum.get("claim_support_rate", 0.0) * 100
        st.metric("Supported Claims", f"{sup} / {tot}", f"{rate:.1f}% support rate")
    with c6:
        unsup = v_sum.get("not_supported_claims", 0)
        un_rate = v_sum.get("claim_unsupported_rate", 0.0) * 100
        st.metric("Unsupported Claims", f"{unsup}", f"{un_rate:.1f}% ungrounded", delta_color="inverse")
    with c7:
        contra = v_sum.get("contradicted_claims", 0)
        st.metric("Contradicted Claims", f"{contra}", "0% contradictions")
    with c8:
        fc = v_sum.get("average_factual_consistency", 0.0) * 100
        st.metric("Mean Factual Consistency", f"{fc:.1f}%")

    # Row 3: Hallucination Protection & Abstention
    st.markdown("### 4. 🛑 Hallucination Protection & Principled Abstention")
    c9, c10, c11 = st.columns([1, 1, 2])
    with c9:
        acc = a_sum.get("accepted_count", 0)
        tot_eval = a_sum.get("total_evaluated", 15)
        st.metric("Accepted Answers", f"{acc} / {tot_eval}")
    with c10:
        abstained = a_sum.get("abstained_count", 0)
        abs_rate = a_sum.get("abstention_rate", 0.0) * 100
        st.metric("Abstention Rate", f"{abs_rate:.1f}%", f"{abstained} abstained", delta_color="inverse")
    with c11:
        st.markdown("**Abstention Breakdown by Cause:**")
        reasons = a_sum.get("reasons_breakdown", {})
        if reasons:
            for r_name, count in reasons.items():
                pct = (count / tot_eval) * 100
                st.write(f"- `{r_name}`: **{count}** query/queries ({pct:.1f}%)")
        else:
            st.write("No abstentions triggered.")


def render_pipeline_evolution_section(
    baseline_data: Optional[Dict[str, Any]],
    improved_data: Optional[Dict[str, Any]],
    relevance_data: Optional[Dict[str, Any]],
    figures_dir: Path,
):
    """Render Section 5: Baseline vs Improved vs Relevance-Gated Progression."""
    st.subheader("5. 📈 Pipeline Evolution: Baseline → Improved Grounding → Relevance Gate")
    st.markdown(
        "Demonstrating how AuditRAG evolved through engineering iterations to solve "
        "evidence grounding and question-answer relevance:"
    )

    if not (baseline_data and improved_data and relevance_data):
        st.warning("⚠️ One or more pipeline benchmark reports are missing from `reports/tables/`.")
        return

    # Extract metrics for all three stages
    def extract_summary(d: Dict[str, Any], label: str) -> Dict[str, Any]:
        v = d.get("verification_summary", {})
        a = d.get("abstention_summary", {})
        reasons = a.get("reasons_breakdown", {})
        reasons_str = ", ".join(f"{k}: {c}" for k, c in reasons.items()) if reasons else "None"
        return {
            "Pipeline Stage": label,
            "Sample Size": d.get("sample_size", 15),
            "Exact Match": f"{d.get('avg_exact_match', 0.0) * 100:.1f}%",
            "Token F1": f"{d.get('avg_token_f1', 0.0) * 100:.1f}%",
            "Numeric Match": f"{d.get('avg_numeric_match', 0.0) * 100:.1f}%",
            "Supported Claims": f"{v.get('supported_claims', 0)} / {v.get('total_claims', 0)} ({v.get('claim_support_rate', 0.0) * 100:.1f}%)",
            "Factual Consistency": f"{v.get('average_factual_consistency', 0.0) * 100:.1f}%",
            "Accepted Answers": f"{a.get('accepted_count', 0)} / {a.get('total_evaluated', 15)}",
            "Abstention Rate": f"{a.get('abstention_rate', 0.0) * 100:.1f}%",
            "Primary Abstention Causes": reasons_str,
            "Mean Latency": f"{d.get('avg_latency_ms', 0.0):.1f} ms",
        }

    evo_rows = [
        extract_summary(baseline_data, "1. Baseline Pipeline (Phase 10)"),
        extract_summary(improved_data, "2. Improved Grounding (Phase 11)"),
        extract_summary(relevance_data, "3. Relevance-Gated Final (Phase 12)"),
    ]

    df_evo = pd.DataFrame(evo_rows)
    st.dataframe(df_evo, use_container_width=True, hide_index=True)

    # Progression Analysis & Figures
    st.markdown("#### 🔬 Engineering Audit & Progression Analysis")

    col_text, col_fig = st.columns([1.1, 1.1])
    with col_text:
        st.markdown(
            """
            1. **Phase 10 (Baseline)**:
               - The initial pipeline produced generic VLM generations without tight prompt binding.
               - **13 of 16 claims (81.3%)** failed evidence verification.
               - Abstention rate was **86.7%** due to `INSUFFICIENT_EVIDENCE`.
            2. **Phase 11 (Improved Grounding)**:
               - Prompt grounding anchored the generator directly to retrieved document passages.
               - Supported claims jumped from **18.8% to 62.5%**, and factual consistency rose from **13.3% to 60.0%**.
               - Accepted answers increased from **2 to 9**.
               - *Failure Mode Discovered:* Factually grounded claims were sometimes generated for the **wrong financial metric**
                 (e.g., answering with depreciation numbers when asked for total assets).
            3. **Phase 12 (Relevance-Gated Final)**:
               - Added a deterministic **Question → Answer Relevance Gate** before certifying acceptance.
               - Caught **5 off-topic answers** (`IRRELEVANT_ANSWER: 5`), correctly refusing to certify factually true claims that don't answer the prompt.
               - Safely accepts **4 verified, relevant, audit-grade answers** while abstaining on all ungrounded or off-topic queries.
            """
        )

    with col_fig:
        # Display the progression breakdown figure
        fig_rel = figures_dir / "verification_abstention_breakdown_relevance.png"
        fig_imp = figures_dir / "verification_abstention_breakdown_improved.png"
        fig_base = figures_dir / "verification_abstention_breakdown.png"

        tab_rel, tab_imp, tab_base = st.tabs(["Relevance Gated (Final)", "Improved Grounding", "Baseline"])
        with tab_rel:
            if fig_rel.exists():
                st.image(str(fig_rel), caption="Relevance Gated: 4 Accepted, 6 Insufficient Evidence, 5 Irrelevant Answer", use_container_width=True)
            else:
                st.info("Relevance gated breakdown figure not found.")
        with tab_imp:
            if fig_imp.exists():
                st.image(str(fig_imp), caption="Improved Grounding: 9 Accepted, 6 Insufficient Evidence", use_container_width=True)
            else:
                st.info("Improved breakdown figure not found.")
        with tab_base:
            if fig_base.exists():
                st.image(str(fig_base), caption="Baseline: 2 Accepted, 13 Insufficient Evidence", use_container_width=True)
            else:
                st.info("Baseline breakdown figure not found.")


def render_sample_audit_explorer(report_data: Optional[Dict[str, Any]]):
    """Render Section 6: Auditable Benchmark Query Explorer."""
    st.subheader("6. 🔎 Auditable Benchmark Query Explorer")
    st.markdown(
        "Inspect individual queries from the deterministic benchmark sample to see "
        "gold targets, raw predictions, verification scores, and abstention decisions:"
    )

    if not report_data or "samples" not in report_data:
        st.info("No sample-level evaluation data available in benchmark report.")
        return

    samples = report_data.get("samples", [])
    if not samples:
        st.info("Benchmark sample list is empty.")
        return

    filter_mode = st.radio(
        "Filter benchmark samples:",
        ["All (15)", "Accepted Only (4)", "Abstained Only (11)"],
        horizontal=True,
    )

    if filter_mode == "Accepted Only (4)":
        filtered_samples = [s for s in samples if not s.get("is_abstained", False)]
    elif filter_mode == "Abstained Only (11)":
        filtered_samples = [s for s in samples if s.get("is_abstained", False)]
    else:
        filtered_samples = samples

    for idx, s in enumerate(filtered_samples):
        q = s.get("question", "N/A")
        is_abs = s.get("is_abstained", False)
        reason = s.get("abstention_reason") or "CERTIFIED_ACCEPTED"
        status_icon = "🛡️ ABSTAINED" if is_abs else "✅ ACCEPTED"
        fc = s.get("factual_consistency", 0.0) * 100

        with st.expander(f"{status_icon} | Q{idx+1}: {q[:75]}... [{reason}]", expanded=(idx == 0)):
            c_left, c_right = st.columns([1.5, 1.0])
            with c_left:
                st.markdown(f"**Question:** {q}")
                st.markdown(f"**Gold Target:** `{s.get('gold_answer')}`")
                st.markdown(f"**Generated Prediction:** {s.get('prediction')}")
                st.markdown(f"**Final Answer Displayed:** *{s.get('final_answer')}*")
            with c_right:
                st.markdown(f"- **Decision:** `{reason}`")
                st.markdown(f"- **Factual Consistency:** `{fc:.1f}%`")
                st.markdown(f"- **Supported Claims:** `{s.get('supported_claims', 0)}`")
                st.markdown(f"- **Unsupported Claims:** `{s.get('not_supported_claims', 0)}`")
                st.markdown(f"- **Contradicted Claims:** `{s.get('contradicted_claims', 0)}`")
                st.markdown(f"- **Exact Match:** `{s.get('exact_match', 0.0)}`")
                st.markdown(f"- **Token F1:** `{s.get('token_f1', 0.0):.3f}`")


# ---------------------------------------------------------------------------
# Main Dashboard Entrypoint
# ---------------------------------------------------------------------------

def render_evaluation_dashboard(
    tables_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
):
    """Render the full Evaluation & Reliability Dashboard inside Streamlit."""
    t_dir = tables_dir or DEFAULT_TABLES_DIR
    f_dir = figures_dir or DEFAULT_FIGURES_DIR

    st.title("📊 AuditRAG: Evaluation & Reliability Dashboard")
    st.markdown(
        "Master's-level performance and verification audit of **AuditRAG**, "
        "evaluating multimodal retrieval, claim-level factuality, and principled hallucination protection."
    )

    # 1. Reproducibility & Methodology
    render_reproducibility_notice(t_dir, f_dir)
    st.divider()

    # Load all benchmark reports
    reports = load_all_benchmark_reports(t_dir)
    retrieval_data = reports.get("retrieval")
    baseline_data = reports.get("baseline")
    improved_data = reports.get("improved")
    relevance_data = reports.get("relevance")

    # 2. Retrieval Performance Section
    render_retrieval_section(retrieval_data, f_dir)
    st.divider()

    # 3, 4, 5. Answer Quality, Verification, and Abstention for the Final System
    render_pipeline_metrics_overview(relevance_data, title_suffix="(Final Relevance-Gated System)")
    st.divider()

    # 6. Pipeline Evolution Comparison Table and Analysis
    render_pipeline_evolution_section(baseline_data, improved_data, relevance_data, f_dir)
    st.divider()

    # 7. Auditable Sample Explorer
    render_sample_audit_explorer(relevance_data)
