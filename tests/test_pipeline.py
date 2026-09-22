"""End-to-end integration tests for the unified AuditRAG Pipeline."""

import pytest
from auditrag.pipeline.engine import AuditRAGPipeline, AuditRAGResponse


def test_pipeline_initialization():
    """Verify default pipeline components initialize cleanly."""
    pipeline = AuditRAGPipeline()
    assert pipeline.retriever is not None
    assert pipeline.generator is not None
    assert pipeline.citation_engine is not None
    assert pipeline.verifier is not None
    assert pipeline.protection_engine is not None


def test_pipeline_empty_query_raises():
    """Verify empty queries raise ValueError."""
    pipeline = AuditRAGPipeline()
    with pytest.raises(ValueError, match="Question cannot be empty"):
        pipeline.run("")

    with pytest.raises(ValueError, match="Question cannot be empty"):
        pipeline.run("   ")


def test_pipeline_e2e_query_execution():
    """Verify full execution: retrieval -> generation -> citation -> verification -> decision."""
    pipeline = AuditRAGPipeline()
    response = pipeline.run("What was the depreciation for AMER in 2018?", top_k=3)

    assert isinstance(response, AuditRAGResponse)
    assert response.question == "What was the depreciation for AMER in 2018?"
    assert response.final_answer is not None
    assert len(response.final_answer) > 0
    assert response.decision is not None
    assert response.verification_report is not None
    assert response.answer_with_citations is not None
    assert len(response.retrieval_results) > 0
