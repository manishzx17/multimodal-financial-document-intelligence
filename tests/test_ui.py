"""Unit and integration tests for Addition 1: Interactive Visual Evidence & Citation Viewer."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from auditrag.citations.citation_engine import AnswerWithCitations, VisualCitation
from auditrag.config import PROJECT_ROOT
from auditrag.pipeline.engine import AuditRAGResponse
from auditrag.ui.citation_viewer import (
    get_citation_details,
    get_or_create_citation_overlay,
    resolve_image_path,
)
from auditrag.verification.abstention import AbstentionDecision, AbstentionReason
from auditrag.verification.models import Claim, VerificationReport, VerificationResult, VerificationStatus


@pytest.fixture
def sample_citation() -> VisualCitation:
    """Fixture returning a valid VisualCitation with existing sample page image."""
    return VisualCitation(
        document_id="637fab7088ea6c78a5dba55f17e833bd",
        page_number=1,
        text="AMER $ 22,531 $ 21,224 $ 19,694",
        bbox=[88, 255, 1144, 274],
        label="[1]",
        source_image_path="data/raw/tatdqa/test/637fab7088ea6c78a5dba55f17e833bd_1.png",
        overlay_image_path="data/citations/637fab7088ea6c78a5dba55f17e833bd_1_overlay.png",
    )


def test_resolve_image_path_valid_and_invalid(sample_citation: VisualCitation):
    """Verify image path resolution for existing relative, absolute, and non-existent paths."""
    # 1. Valid relative path
    resolved = resolve_image_path(sample_citation.source_image_path)
    assert resolved is not None
    assert resolved.exists()
    assert resolved.is_absolute()

    # 2. Valid absolute path
    resolved_abs = resolve_image_path(resolved)
    assert resolved_abs == resolved

    # 3. None and empty
    assert resolve_image_path(None) is None
    assert resolve_image_path("") is None

    # 4. Non-existent path
    assert resolve_image_path("data/non_existent_page_12345.png") is None


def test_get_citation_details(sample_citation: VisualCitation):
    """Verify structured metadata extraction for Streamlit inspection."""
    details = get_citation_details(sample_citation)

    assert details["label"] == "[1]"
    assert details["document_id"] == "637fab7088ea6c78a5dba55f17e833bd"
    assert details["page_number"] == 1
    assert details["text"] == "AMER $ 22,531 $ 21,224 $ 19,694"
    assert details["bbox"] == [88, 255, 1144, 274]
    assert details["width"] == 1144 - 88
    assert details["height"] == 274 - 255
    assert details["source_exists"] is True


def test_get_or_create_citation_overlay_reuse_and_dynamic(sample_citation: VisualCitation, tmp_path: Path):
    """Verify that existing overlays are reused and dynamic overlays are generated on demand."""
    # 1. Full overlay reuse (overlay_image_path exists)
    overlay_path, is_new = get_or_create_citation_overlay(
        citation=sample_citation,
        focused_only=False,
    )
    assert overlay_path is not None
    assert overlay_path.exists()
    assert is_new is False

    # 2. Focused single-citation dynamic generation
    focused_path, is_new_focused = get_or_create_citation_overlay(
        citation=sample_citation,
        focused_only=True,
        output_dir=tmp_path,
    )
    assert focused_path is not None
    assert focused_path.exists()
    assert "focused" in focused_path.name

    # 3. Missing source image graceful handling
    bad_cit = VisualCitation(
        document_id="bad_doc",
        page_number=999,
        text="Missing evidence",
        bbox=[10, 10, 50, 50],
        source_image_path="non_existent/path/image.png",
    )
    res_bad, is_new_bad = get_or_create_citation_overlay(bad_cit, output_dir=tmp_path)
    assert res_bad is None
    assert is_new_bad is False


def test_streamlit_app_initial_render():
    """Verify that the Streamlit application starts up and renders without errors."""
    at = AppTest.from_file("src/auditrag/ui/app.py", default_timeout=30)
    at.run()

    assert not at.exception
    # Verify title and essential UI elements are rendered
    assert len(at.title) >= 1
    assert "AuditRAG" in at.title[0].value


def test_streamlit_app_with_mock_citations(sample_citation: VisualCitation):
    """Verify Streamlit app renders multiple citations and allows interactive selection."""
    second_citation = VisualCitation(
        document_id="637fab7088ea6c78a5dba55f17e833bd",
        page_number=1,
        text="Capital expenditures $ 17,690",
        bbox=[88, 300, 800, 320],
        label="[2]",
        source_image_path="data/raw/tatdqa/test/637fab7088ea6c78a5dba55f17e833bd_1.png",
    )

    answer_with_cits = AnswerWithCitations(
        question="What was the depreciation and capex for AMER in 2018?",
        answer="Depreciation was $21,224 thousand and capex was $17,690 thousand.",
        citations=[sample_citation, second_citation],
        overlay_images=[sample_citation.overlay_image_path],
    )

    report = VerificationReport(
        question="What was the depreciation and capex for AMER in 2018?",
        answer=answer_with_cits.answer,
        overall_status=VerificationStatus.SUPPORTED,
        supported_count=2,
        contradicted_count=0,
        not_supported_count=0,
        factual_consistency_score=1.0,
        is_question_addressed=True,
        relevance_score=1.0,
    )

    decision = AbstentionDecision(
        is_abstained=False,
        final_answer=answer_with_cits.answer,
        abstention_reason=AbstentionReason.NONE,
        confidence_score=1.0,
        verification_report=report,
    )

    mock_response = AuditRAGResponse(
        question="What was the depreciation and capex for AMER in 2018?",
        final_answer=answer_with_cits.answer,
        is_abstained=False,
        abstention_reason=None,
        decision=decision,
        answer_with_citations=answer_with_cits,
        verification_report=report,
    )

    at = AppTest.from_file("src/auditrag/ui/app.py", default_timeout=30)
    # Inject mock response into session state
    at.session_state["pipeline_response"] = mock_response
    at.session_state["selected_citation_idx"] = 0
    at.run()

    assert not at.exception
    # Check that citation selector radio was rendered
    assert len(at.radio) >= 1
    assert any("Citation" in r.label for r in at.radio)

    # Check that bounding box code was rendered
    assert any("BBox:" in c.value for c in at.code)

    # Check that supporting evidence text was rendered
    assert any("AMER" in s.value for s in at.success)

    # Simulate switching to citation 2
    at.session_state["selected_citation_idx"] = 1
    at.run()
    assert not at.exception


def test_streamlit_app_abstained_and_no_citations():
    """Verify graceful rendering when an answer is abstained and has zero citations."""
    answer_empty_cits = AnswerWithCitations(
        question="Unanswerable question?",
        answer="I couldn't verify this answer from the available document evidence.",
        citations=[],
        overlay_images=[],
    )

    report = VerificationReport(
        question="Unanswerable question?",
        answer=answer_empty_cits.answer,
        overall_status=VerificationStatus.NOT_SUPPORTED,
        supported_count=0,
        contradicted_count=0,
        not_supported_count=1,
        factual_consistency_score=0.0,
        is_question_addressed=False,
        relevance_score=0.0,
    )

    decision = AbstentionDecision(
        is_abstained=True,
        final_answer=answer_empty_cits.answer,
        abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
        confidence_score=0.0,
        verification_report=report,
    )

    mock_abstained_resp = AuditRAGResponse(
        question="Unanswerable question?",
        final_answer=answer_empty_cits.answer,
        is_abstained=True,
        abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE.value,
        decision=decision,
        answer_with_citations=answer_empty_cits,
        verification_report=report,
    )

    at = AppTest.from_file("src/auditrag/ui/app.py", default_timeout=30)
    at.session_state["pipeline_response"] = mock_abstained_resp
    at.run()

    assert not at.exception
    # Warning for abstention should be rendered
    assert len(at.warning) >= 1
    # Info for no citations should be rendered
    assert any("No visual bounding-box citations" in info.value for info in at.info)


def test_extract_numerical_traces_verified_percentage_calculation():
    """Verify trace extraction for a verified percentage growth calculation."""
    from auditrag.ui.numerical_reasoning import extract_numerical_traces
    from auditrag.verification.models import ClaimType

    claim = Claim(
        claim_id="calc_1",
        statement="Revenues increased by 25% from 40 to 50",
        claim_type=ClaimType.CALCULATION,
        extracted_numbers=[25.0, 40.0, 50.0],
    )
    res = VerificationResult(
        claim=claim,
        status=VerificationStatus.SUPPORTED,
        confidence=1.0,
        arithmetic_check={
            "verified": True,
            "operation": "percentage_growth_check",
            "expression": "((50.0 - 40.0) / 40.0) * 100% = 25.0%",
            "computed_value": 25.0,
            "claimed_value": 25.0,
            "explanation": "Percentage calculation verified: growth from 40.0 to 50.0 is 25.0%.",
        },
        explanation="Verified in disclosure.",
        citation_label="[1]",
        matching_doc_id="doc_test_1",
        matching_page_number=1,
    )
    report = VerificationReport(
        question="What was the percentage revenue growth?",
        answer="Revenues increased by 25% from 40 to 50 [Doc: doc_test_1, Page: 1].",
        results=[res],
        overall_status=VerificationStatus.SUPPORTED,
        supported_count=1,
    )

    traces = extract_numerical_traces(report)
    assert len(traces) == 1
    t = traces[0]
    assert t.claim_id == "calc_1"
    assert t.status == "VERIFIED"
    assert t.is_verified is True
    assert t.formula_name == "Percentage Change / Growth"
    assert "((New_Value - Old_Value)" in t.formula_template
    assert t.calculated_value == "25"
    assert t.reported_value == "25"
    assert "40" in t.source_values and "50" in t.source_values


def test_extract_numerical_traces_failed_arithmetic():
    """Verify trace extraction when arithmetic check explicitly fails."""
    from auditrag.ui.numerical_reasoning import extract_numerical_traces
    from auditrag.verification.models import ClaimType

    claim = Claim(
        claim_id="calc_fail",
        statement="Operating income increased by 50.0 from 38.6 to 57.8",
        claim_type=ClaimType.CALCULATION,
        extracted_numbers=[50.0, 38.6, 57.8],
    )
    res = VerificationResult(
        claim=claim,
        status=VerificationStatus.NOT_SUPPORTED,
        confidence=0.5,
        arithmetic_check={
            "verified": False,
            "operation": "variance_check",
            "expression": "57.8 - 38.6 != 50.0",
            "computed_value": 19.2,
            "claimed_value": 50.0,
            "explanation": "Arithmetic error: difference is 19.2, not 50.0.",
        },
        explanation="Calculation mismatch.",
    )
    report = VerificationReport(
        question="How much did operating income increase?",
        answer="Operating income increased by 50.0 from 38.6 to 57.8.",
        results=[res],
        overall_status=VerificationStatus.NOT_SUPPORTED,
        not_supported_count=1,
    )

    traces = extract_numerical_traces(report)
    assert len(traces) == 1
    t = traces[0]
    assert t.status == "FAILED_ARITHMETIC_CHECK"
    assert t.is_verified is False
    assert t.calculated_value == "19.2"
    assert t.reported_value == "50"


def test_extract_numerical_traces_unsupported_numerical_claim():
    """Verify trace extraction when stated numerical amounts cannot be found in disclosures."""
    from auditrag.ui.numerical_reasoning import extract_numerical_traces
    from auditrag.verification.models import ClaimType

    claim = Claim(
        claim_id="num_unsupp",
        statement="Total debt was $999,999 thousand",
        claim_type=ClaimType.NUMERICAL,
        extracted_numbers=[999999.0],
    )
    res = VerificationResult(
        claim=claim,
        status=VerificationStatus.NOT_SUPPORTED,
        confidence=0.0,
        explanation="Stated numerical figures could not be located in retrieved evidence.",
    )
    report = VerificationReport(
        question="What was the total debt?",
        answer="Total debt was $999,999 thousand.",
        results=[res],
        overall_status=VerificationStatus.NOT_SUPPORTED,
        not_supported_count=1,
    )

    traces = extract_numerical_traces(report)
    assert len(traces) == 1
    t = traces[0]
    assert t.status == "UNSUPPORTED_NUMERICAL_FACT"
    assert t.is_verified is False
    assert "999,999" in t.source_values


def test_extract_numerical_traces_textual_claims_only():
    """Verify that answers with only textual claims return zero numerical traces."""
    from auditrag.ui.numerical_reasoning import extract_numerical_traces
    from auditrag.verification.models import ClaimType

    claim = Claim(
        claim_id="text_1",
        statement="Trade payables are paid within 30 days of recognition.",
        claim_type=ClaimType.TEXTUAL,
    )
    res = VerificationResult(
        claim=claim,
        status=VerificationStatus.SUPPORTED,
        confidence=1.0,
        explanation="Entailed by disclosure.",
    )
    report = VerificationReport(
        question="When are trade payables paid?",
        answer="Trade payables are paid within 30 days of recognition.",
        results=[res],
        overall_status=VerificationStatus.SUPPORTED,
        supported_count=1,
    )

    traces = extract_numerical_traces(report)
    assert traces == []


def test_streamlit_app_renders_numerical_reasoning_trace(sample_citation: VisualCitation):
    """Verify Streamlit app displays the Numerical Reasoning section with formula and calculated values."""
    from auditrag.verification.models import ClaimType

    calc_claim = Claim(
        claim_id="c_pct",
        statement="Growth was 20% from 50 to 60",
        claim_type=ClaimType.CALCULATION,
        extracted_numbers=[20.0, 50.0, 60.0],
    )
    res = VerificationResult(
        claim=calc_claim,
        status=VerificationStatus.SUPPORTED,
        confidence=1.0,
        arithmetic_check={
            "verified": True,
            "operation": "percentage_growth_check",
            "expression": "((60.0 - 50.0) / 50.0) * 100% = 20.0%",
            "computed_value": 20.0,
            "claimed_value": 20.0,
            "explanation": "Percentage verified.",
        },
        explanation="Verified in text.",
        citation_label="[1]",
    )
    report = VerificationReport(
        question="What was the growth?",
        answer="Growth was 20% from 50 to 60 [Doc: 637fab7088ea6c78a5dba55f17e833bd, Page: 1].",
        results=[res],
        overall_status=VerificationStatus.SUPPORTED,
        supported_count=1,
        is_question_addressed=True,
    )
    decision = AbstentionDecision(
        is_abstained=False,
        final_answer=report.answer,
        abstention_reason=AbstentionReason.NONE,
        confidence_score=1.0,
        verification_report=report,
    )
    mock_resp = AuditRAGResponse(
        question=report.question,
        final_answer=report.answer,
        is_abstained=False,
        abstention_reason=None,
        decision=decision,
        answer_with_citations=AnswerWithCitations(
            question=report.question,
            answer=report.answer,
            citations=[sample_citation],
            overlay_images=[],
        ),
        verification_report=report,
    )

    at = AppTest.from_file("src/auditrag/ui/app.py", default_timeout=30)
    at.session_state["pipeline_response"] = mock_resp
    at.run()

    assert not at.exception
    # Check that Numerical Reasoning section header is rendered
    assert any("Numerical Reasoning" in h.value for h in at.subheader)
    # Check that mathematical formula template is rendered in code blocks
    assert any("New_Value" in c.value or "Old_Value" in c.value for c in at.code)
    # Check that calculated value is rendered
    assert any("20" in c.value for c in at.code)


# ---------------------------------------------------------------------------
# Phase 12 — Addition 3: Evaluation & Reliability Dashboard Tests
# ---------------------------------------------------------------------------

def test_load_all_benchmark_reports():
    """Verify that all 4 existing benchmark JSON reports are successfully loaded."""
    from auditrag.ui.dashboard import load_all_benchmark_reports

    reports = load_all_benchmark_reports()
    assert "retrieval" in reports
    assert "baseline" in reports
    assert "improved" in reports
    assert "relevance" in reports

    ret = reports["retrieval"]
    assert ret is not None
    assert ret.get("sample_size") == 15
    assert len(ret.get("results", [])) == 4

    for key in ("baseline", "improved", "relevance"):
        rep = reports[key]
        assert rep is not None
        assert rep.get("sample_size") == 15
        assert "verification_summary" in rep
        assert "abstention_summary" in rep
        assert "samples" in rep
        assert len(rep["samples"]) == 15


def test_retrieval_benchmark_data_parsing():
    """Verify retriever metrics extraction for Dense, BM25, ColPali, and Hybrid RRF."""
    from auditrag.ui.dashboard import load_all_benchmark_reports

    reports = load_all_benchmark_reports()
    ret = reports["retrieval"]
    assert ret is not None

    names = {r["retriever_name"]: r for r in ret["results"]}
    assert "Dense" in names
    assert "BM25" in names
    assert "ColPali" in names
    assert "Hybrid (RRF)" in names

    # BM25 lexical precision
    bm25 = names["BM25"]
    assert bm25["doc_hit_at_1"] == 0.6
    assert bm25["doc_hit_at_5"] > 0.7
    assert bm25["avg_latency_ms"] < 100.0

    # Hybrid RRF latency and recall
    hybrid = names["Hybrid (RRF)"]
    assert hybrid["doc_hit_at_3"] == 0.6
    assert hybrid["doc_hit_at_5"] == 0.6


def test_pipeline_evolution_comparison_metrics():
    """Verify baseline vs improved vs relevance-gated progression metrics."""
    from auditrag.ui.dashboard import load_all_benchmark_reports

    reports = load_all_benchmark_reports()
    baseline = reports["baseline"]
    improved = reports["improved"]
    relevance = reports["relevance"]

    assert baseline is not None and improved is not None and relevance is not None

    # Baseline has low claim support and high abstention due to lack of evidence
    base_v = baseline["verification_summary"]
    base_a = baseline["abstention_summary"]
    assert base_v["supported_claims"] == 3
    assert base_v["claim_support_rate"] < 0.25
    assert base_a["accepted_count"] == 2
    assert base_a["abstained_count"] == 13

    # Improved grounding increased supported claims to 10
    imp_v = improved["verification_summary"]
    imp_a = improved["abstention_summary"]
    assert imp_v["supported_claims"] == 10
    assert imp_v["claim_support_rate"] == 0.625
    assert imp_a["accepted_count"] == 9
    assert imp_a["abstained_count"] == 6

    # Relevance-gated final catches off-topic claims, accepting exactly 4 relevant answers
    rel_v = relevance["verification_summary"]
    rel_a = relevance["abstention_summary"]
    assert rel_v["supported_claims"] == 10
    assert rel_a["accepted_count"] == 4
    assert rel_a["abstained_count"] == 11
    # Check that IRRELEVANT_ANSWER reason exists in breakdown
    assert rel_a["reasons_breakdown"].get("IRRELEVANT_ANSWER") == 5


def test_missing_report_graceful_handling(tmp_path):
    """Verify dashboard functions handle missing reports and empty directories without crashing."""
    from streamlit.testing.v1 import AppTest
    from auditrag.ui.dashboard import load_all_benchmark_reports, render_evaluation_dashboard

    # Loading from empty directory should return all None
    empty_reports = load_all_benchmark_reports(tables_dir=tmp_path)
    assert empty_reports["retrieval"] is None
    assert empty_reports["baseline"] is None
    assert empty_reports["improved"] is None
    assert empty_reports["relevance"] is None

    # Test that render_evaluation_dashboard runs without throwing unhandled exceptions
    mini_app = f"""
import streamlit as st
from pathlib import Path
from auditrag.ui.dashboard import render_evaluation_dashboard

render_evaluation_dashboard(
    tables_dir=Path("{tmp_path}"),
    figures_dir=Path("{tmp_path}"),
)
"""
    at = AppTest.from_string(mini_app)
    at.run()
    assert not at.exception
    # Warning should be rendered indicating missing benchmark data
    assert len(at.warning) >= 1


def test_streamlit_app_renders_evaluation_dashboard_tab():
    """Verify full Streamlit app renders the Evaluation & Reliability Dashboard."""
    at = AppTest.from_file("src/auditrag/ui/app.py", default_timeout=30)
    at.run()

    assert not at.exception
    # Verify subheaders include both workbench and evaluation dashboard sections
    subheaders = [h.value for h in at.subheader]
    assert any("Multimodal Retrieval Performance" in h for h in subheaders)
    assert any("Pipeline Evolution" in h for h in subheaders)
    assert any("Benchmark Query Explorer" in h for h in subheaders)

    # Verify metric cards are present
    assert len(at.metric) >= 8

    # Verify tables/dataframes are rendered
    assert len(at.dataframe) >= 2
