"""Benchmark runner for AuditRAG Phase 11 (Evaluation & Benchmarking).

Orchestrates automated evaluation of:
1. Retrieval modalities: Dense, BM25, ColPali, and Hybrid RRF.
2. End-to-end Pipeline: Answer Quality (EM, F1, Numeric), Claim Verification, and Abstention Gating.
3. Summary reports (Markdown & JSON) and visual figures (Matplotlib PNGs).
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
from pydantic import BaseModel, Field

from auditrag.data.loader import Document, Question, TATDQADataset, TATDQALoader
from auditrag.evaluation.metrics import (
    AbstentionMetricsSummary,
    VerificationMetricsSummary,
    compute_abstention_metrics,
    compute_hit_at_k,
    compute_mrr,
    compute_verification_metrics,
    exact_match_score,
    numeric_match_score,
    token_f1_score,
)
from auditrag.pipeline.engine import AuditRAGPipeline, AuditRAGResponse
from auditrag.retrieval.bm25 import BM25Retriever
from auditrag.retrieval.colpali import ColPaliRetriever
from auditrag.retrieval.dense import DenseRetriever
from auditrag.retrieval.hybrid import HybridRetriever


class RetrieverMetrics(BaseModel):
    """Evaluation summary metrics for a single retriever modality."""
    retriever_name: str
    sample_size: int
    doc_hit_at_1: float
    doc_hit_at_3: float
    doc_hit_at_5: float
    doc_mrr: float
    page_hit_at_1: float
    page_hit_at_3: float
    page_hit_at_5: float
    page_mrr: float
    avg_latency_ms: float


class RetrievalBenchmarkReport(BaseModel):
    """Aggregate benchmark results across multiple retrieval systems."""
    sample_size: int
    results: List[RetrieverMetrics] = Field(default_factory=list)


class PipelineSampleResult(BaseModel):
    """Evaluation details for a single end-to-end pipeline execution."""
    question_uid: str
    doc_uid: str
    question: str
    gold_answer: Any
    prediction: str
    final_answer: str
    is_abstained: bool
    abstention_reason: Optional[str] = None
    exact_match: float
    token_f1: float
    numeric_match: float
    supported_claims: int
    contradicted_claims: int
    not_supported_claims: int
    factual_consistency: float


class PipelineBenchmarkReport(BaseModel):
    """Aggregate end-to-end benchmark results."""
    sample_size: int
    avg_exact_match: float
    avg_token_f1: float
    avg_numeric_match: float
    verification_summary: VerificationMetricsSummary
    abstention_summary: AbstentionMetricsSummary
    avg_latency_ms: float
    samples: List[PipelineSampleResult] = Field(default_factory=list)


class BenchmarkRunner:
    """Automated benchmark executor for AuditRAG."""

    def __init__(
        self,
        dataset: Optional[TATDQADataset] = None,
        output_dir: str = "reports",
    ):
        loader = TATDQALoader()
        self.dataset = dataset or loader.load_dataset(use_gold=True, load_doc_details=True)
        self.output_dir = Path(output_dir)
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir = self.output_dir / "figures"

        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def select_sample_questions(self, sample_size: int = 25) -> List[Question]:
        """Select a deterministic, representative subset of gold test questions across documents."""
        # Group by document to ensure diversity across companies/pages
        docs_seen: Set[str] = set()
        selected: List[Question] = []

        # Round 1: Pick first question from each distinct document
        for q in self.dataset.questions:
            if q.doc_uid not in docs_seen:
                docs_seen.add(q.doc_uid)
                selected.append(q)
                if len(selected) >= sample_size:
                    break

        # Round 2: Fill remaining if sample_size > unique documents
        if len(selected) < sample_size:
            for q in self.dataset.questions:
                if q not in selected:
                    selected.append(q)
                    if len(selected) >= sample_size:
                        break

        return selected

    def _resolve_gold_pages(self, question: Question) -> Set[int]:
        """Find the gold evidence page numbers for a question."""
        doc = self.dataset.get_document(question.doc_uid)
        if doc is None:
            return set()
        spans = question.resolve_evidence_spans(doc)
        pages = {s.page_idx for s in spans if s.page_idx is not None}
        return pages

    def evaluate_retrieval(
        self,
        questions: List[Question],
        retrievers: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> RetrievalBenchmarkReport:
        """Run retrieval benchmark for Dense, BM25, ColPali, and Hybrid RRF."""
        if retrievers is None:
            retrievers = {
                "Dense": DenseRetriever(),
                "BM25": BM25Retriever(),
                "ColPali": ColPaliRetriever(),
                "Hybrid (RRF)": HybridRetriever(),
            }

        report_results: List[RetrieverMetrics] = []
        n_q = len(questions)

        for name, retriever in retrievers.items():
            doc_hits_1, doc_hits_3, doc_hits_5 = 0.0, 0.0, 0.0
            page_hits_1, page_hits_3, page_hits_5 = 0.0, 0.0, 0.0
            doc_mrrs, page_mrrs = 0.0, 0.0
            total_time_ms = 0.0

            for q in questions:
                gold_doc_id = q.doc_uid
                gold_pages = self._resolve_gold_pages(q)

                t0 = time.time()
                retrieved = retriever.retrieve(q.question, top_k=top_k)
                elapsed_ms = (time.time() - t0) * 1000.0
                total_time_ms += elapsed_ms

                # Top-1
                dh1, ph1 = compute_hit_at_k(retrieved, gold_doc_id, gold_pages, k=1)
                doc_hits_1 += dh1
                page_hits_1 += ph1

                # Top-3
                dh3, ph3 = compute_hit_at_k(retrieved, gold_doc_id, gold_pages, k=3)
                doc_hits_3 += dh3
                page_hits_3 += ph3

                # Top-5
                dh5, ph5 = compute_hit_at_k(retrieved, gold_doc_id, gold_pages, k=5)
                doc_hits_5 += dh5
                page_hits_5 += ph5

                # MRR
                dmrr, pmrr = compute_mrr(retrieved, gold_doc_id, gold_pages)
                doc_mrrs += dmrr
                page_mrrs += pmrr

            metrics = RetrieverMetrics(
                retriever_name=name,
                sample_size=n_q,
                doc_hit_at_1=round(doc_hits_1 / n_q, 4) if n_q > 0 else 0.0,
                doc_hit_at_3=round(doc_hits_3 / n_q, 4) if n_q > 0 else 0.0,
                doc_hit_at_5=round(doc_hits_5 / n_q, 4) if n_q > 0 else 0.0,
                doc_mrr=round(doc_mrrs / n_q, 4) if n_q > 0 else 0.0,
                page_hit_at_1=round(page_hits_1 / n_q, 4) if n_q > 0 else 0.0,
                page_hit_at_3=round(page_hits_3 / n_q, 4) if n_q > 0 else 0.0,
                page_hit_at_5=round(page_hits_5 / n_q, 4) if n_q > 0 else 0.0,
                page_mrr=round(page_mrrs / n_q, 4) if n_q > 0 else 0.0,
                avg_latency_ms=round(total_time_ms / n_q, 2) if n_q > 0 else 0.0,
            )
            report_results.append(metrics)

        report = RetrievalBenchmarkReport(sample_size=n_q, results=report_results)
        self._save_retrieval_reports(report)
        self._plot_retrieval_comparison(report)
        return report

    def evaluate_pipeline(
        self,
        questions: List[Question],
        pipeline: Optional[AuditRAGPipeline] = None,
        top_k: int = 5,
        suffix: str = "",
    ) -> PipelineBenchmarkReport:
        """Run end-to-end pipeline benchmark on questions."""
        p = pipeline or AuditRAGPipeline()
        n_q = len(questions)

        sample_results: List[PipelineSampleResult] = []
        all_reports = []
        all_responses = []

        total_em = 0.0
        total_f1 = 0.0
        total_num = 0.0
        total_latency_ms = 0.0

        for q in questions:
            t0 = time.time()
            resp: AuditRAGResponse = p.run(q.question, top_k=top_k)
            latency_ms = (time.time() - t0) * 1000.0
            total_latency_ms += latency_ms

            # The prediction before abstention
            vlm_answer = resp.answer_with_citations.answer
            gold = q.answer

            em = exact_match_score(vlm_answer, gold)
            f1 = token_f1_score(vlm_answer, gold)
            num = numeric_match_score(vlm_answer, gold)

            total_em += em
            total_f1 += f1
            total_num += num

            all_reports.append(resp.verification_report)
            all_responses.append(resp)

            sample_results.append(
                PipelineSampleResult(
                    question_uid=q.uid,
                    doc_uid=q.doc_uid,
                    question=q.question,
                    gold_answer=gold,
                    prediction=vlm_answer,
                    final_answer=resp.final_answer,
                    is_abstained=resp.is_abstained,
                    abstention_reason=resp.abstention_reason,
                    exact_match=em,
                    token_f1=f1,
                    numeric_match=num,
                    supported_claims=resp.verification_report.supported_count,
                    contradicted_claims=resp.verification_report.contradicted_count,
                    not_supported_claims=resp.verification_report.not_supported_count,
                    factual_consistency=resp.verification_report.factual_consistency_score,
                )
            )

        verification_summary = compute_verification_metrics(all_reports)
        abstention_summary = compute_abstention_metrics(all_responses)

        report = PipelineBenchmarkReport(
            sample_size=n_q,
            avg_exact_match=round(total_em / n_q, 4) if n_q > 0 else 0.0,
            avg_token_f1=round(total_f1 / n_q, 4) if n_q > 0 else 0.0,
            avg_numeric_match=round(total_num / n_q, 4) if n_q > 0 else 0.0,
            verification_summary=verification_summary,
            abstention_summary=abstention_summary,
            avg_latency_ms=round(total_latency_ms / n_q, 2) if n_q > 0 else 0.0,
            samples=sample_results,
        )

        self._save_pipeline_reports(report, suffix=suffix)
        self._plot_pipeline_figures(report, suffix=suffix)
        return report

    def _save_retrieval_reports(self, report: RetrievalBenchmarkReport):
        """Write retrieval markdown and JSON tables."""
        json_path = self.tables_dir / "retrieval_benchmark.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        md_path = self.tables_dir / "retrieval_benchmark.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Retrieval Modality Benchmark (Sample Size: {report.sample_size})\n\n")
            f.write("| Retriever | Doc-Hit@1 | Doc-Hit@3 | Doc-Hit@5 | Doc-MRR | Page-Hit@1 | Page-Hit@3 | Page-Hit@5 | Page-MRR | Latency (ms) |\n")
            f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
            for r in report.results:
                f.write(
                    f"| **{r.retriever_name}** | "
                    f"{r.doc_hit_at_1*100:.1f}% | {r.doc_hit_at_3*100:.1f}% | {r.doc_hit_at_5*100:.1f}% | {r.doc_mrr:.4f} | "
                    f"{r.page_hit_at_1*100:.1f}% | {r.page_hit_at_3*100:.1f}% | {r.page_hit_at_5*100:.1f}% | {r.page_mrr:.4f} | "
                    f"{r.avg_latency_ms:.1f}ms |\n"
                )

    def _save_pipeline_reports(self, report: PipelineBenchmarkReport, suffix: str = ""):
        """Write pipeline evaluation markdown and JSON tables."""
        filename = f"pipeline_benchmark{suffix}"
        json_path = self.tables_dir / f"{filename}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        md_path = self.tables_dir / f"{filename}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            title_suffix = " (Improved Pass)" if suffix else ""
            f.write(f"# End-to-End Pipeline & Protection Benchmark{title_suffix} (Sample Size: {report.sample_size})\n\n")
            f.write("## 1. Grounding & Answer Quality\n\n")
            f.write("| Metric | Score |\n")
            f.write("|:---|:---:|\n")
            f.write(f"| **Exact Match (EM)** | {report.avg_exact_match * 100:.1f}% |\n")
            f.write(f"| **Token F1** | {report.avg_token_f1 * 100:.1f}% |\n")
            f.write(f"| **Numeric Match** | {report.avg_numeric_match * 100:.1f}% |\n")
            f.write(f"| **Average Latency** | {report.avg_latency_ms:.1f} ms |\n\n")

            f.write("## 2. Claim Factuality & Verification\n\n")
            f.write("| Claim Metric | Value |\n")
            f.write("|:---|:---:|\n")
            v = report.verification_summary
            f.write(f"| Total Claims Extracted | {v.total_claims} |\n")
            f.write(f"| Supported Claims | {v.supported_claims} ({v.claim_support_rate*100:.1f}%) |\n")
            f.write(f"| Contradicted Claims | {v.contradicted_claims} ({v.claim_contradiction_rate*100:.1f}%) |\n")
            f.write(f"| Unsupported Claims | {v.not_supported_claims} ({v.claim_unsupported_rate*100:.1f}%) |\n")
            f.write(f"| Mean Factual Consistency | {v.average_factual_consistency*100:.1f}% |\n\n")

            f.write("## 3. Hallucination Protection & Abstention\n\n")
            f.write("| Protection Metric | Value |\n")
            f.write("|:---|:---:|\n")
            a = report.abstention_summary
            f.write(f"| Total Queries Evaluated | {a.total_evaluated} |\n")
            f.write(f"| Accepted Answers | {a.accepted_count} |\n")
            f.write(f"| Abstained Answers | {a.abstained_count} |\n")
            f.write(f"| **Abstention Rate** | {a.abstention_rate*100:.1f}% |\n\n")

            f.write("### Abstention Reasons Breakdown\n\n")
            if a.reasons_breakdown:
                f.write("| Reason | Count |\n|:---|:---:|\n")
                for reason, count in a.reasons_breakdown.items():
                    f.write(f"| `{reason}` | {count} |\n")
            else:
                f.write("No queries were abstained.\n")

    def _plot_retrieval_comparison(self, report: RetrievalBenchmarkReport):
        """Generate a comparison bar chart for retrieval systems."""
        try:
            names = [r.retriever_name for r in report.results]
            doc_hit5 = [r.doc_hit_at_5 * 100 for r in report.results]
            page_hit5 = [r.page_hit_at_5 * 100 for r in report.results]

            fig, ax = plt.subplots(figsize=(9, 5))
            x = range(len(names))
            width = 0.35

            ax.bar([i - width/2 for i in x], doc_hit5, width, label="Doc-Hit@5 (%)", color="#2b5c8f")
            ax.bar([i + width/2 for i in x], page_hit5, width, label="Page-Hit@5 (%)", color="#3b9c6f")

            ax.set_ylabel("Hit Rate (%)")
            ax.set_title(f"AuditRAG Retrieval Performance Comparison (N={report.sample_size})")
            ax.set_xticks(list(x))
            ax.set_xticklabels(names, rotation=10)
            ax.set_ylim(0, 105)
            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            plt.tight_layout()
            out_path = self.figures_dir / "retrieval_comparison.png"
            plt.savefig(out_path, dpi=200)
            plt.close(fig)
        except Exception:
            pass

    def _plot_pipeline_figures(self, report: PipelineBenchmarkReport, suffix: str = ""):
        """Generate summary charts for factuality and abstention behavior."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

            # Subplot 1: Claim Verification Breakdown
            v = report.verification_summary
            labels = ["Supported", "Contradicted", "Unsupported"]
            counts = [v.supported_claims, v.contradicted_claims, v.not_supported_claims]
            colors = ["#2ecc71", "#e74c3c", "#f39c12"]

            if sum(counts) > 0:
                ax1.pie(
                    counts,
                    labels=labels,
                    autopct="%1.1f%%",
                    colors=colors,
                    startangle=140,
                )
            ax1.set_title("Atomic Claim Verification Distribution")

            # Subplot 2: Protection Decision (Accepted vs Abstained)
            a = report.abstention_summary
            p_labels = ["Accepted", "Abstained"]
            p_counts = [a.accepted_count, a.abstained_count]
            p_colors = ["#3498db", "#9b59b6"]

            ax2.bar(p_labels, p_counts, color=p_colors, width=0.5)
            ax2.set_ylabel("Query Count")
            ax2.set_title("Hallucination Protection Decisions")
            for i, val in enumerate(p_counts):
                ax2.text(i, val + 0.1, str(val), ha="center", fontweight="bold")
            ax2.set_ylim(0, max(p_counts or [1]) * 1.25)
            ax2.grid(axis="y", linestyle="--", alpha=0.5)

            plt.tight_layout()
            fig_name = f"verification_abstention_breakdown{suffix}.png"
            out_path = self.figures_dir / fig_name
            plt.savefig(out_path, dpi=200)
            plt.close(fig)
        except Exception:
            pass
