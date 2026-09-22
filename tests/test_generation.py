"""Unit tests for Phase 7 VLM Answer Generation."""

import pytest

from auditrag.generation import (
    BaseVLMProvider,
    GeneratedAnswer,
    MockVLMProvider,
    OpenAIVLMProvider,
    SYSTEM_PROMPT,
    VLMGenerator,
    build_grounded_prompt,
    get_vlm_provider,
)
from auditrag.retrieval.hybrid import HybridRetrievalResult


@pytest.fixture
def sample_hybrid_results() -> list[HybridRetrievalResult]:
    """Sample hybrid retrieval results with text and visual metadata."""
    res1 = HybridRetrievalResult(
        document_id="637fab7088ea6c78a5dba55f17e833bd",
        page_number=1,
        score=0.032,
        sources=["dense", "bm25"],
        source_ranks={"dense": 3, "bm25": 2},
        source_scores={"dense": 0.599, "bm25": 13.78},
        chunk_id="chunk_637fab70_p1",
        chunk_text="Table of Contents Plexus Corp. Notes to Consolidated Financial Statements 2018 Depreciation: AMER $ 21,224 Capital expenditures: AMER $ 17,690",
        source_metadata={"bboxes": [[10, 20, 100, 200]], "page_image": "data/raw/tatdqa/test/637fab7088ea6c78a5dba55f17e833bd_1.png"},
        image_path="data/raw/tatdqa/test/637fab7088ea6c78a5dba55f17e833bd_1.png",
        visual_score=None,
    )
    res2 = HybridRetrievalResult(
        document_id="d9929fa56778e739f071d0d586171994",
        page_number=1,
        score=0.031,
        sources=["colpali"],
        source_ranks={"colpali": 1},
        source_scores={"colpali": 13.4},
        chunk_id=None,
        chunk_text=None,
        source_metadata={},
        image_path="data/raw/tatdqa/test/d9929fa56778e739f071d0d586171994_1.png",
        visual_score=13.4,
    )
    return [res1, res2]


def test_build_grounded_prompt(sample_hybrid_results):
    """Verify prompt formatting, evidence headers, and image collection."""
    question = "What was the depreciation for AMER in 2018?"
    prompt, images = build_grounded_prompt(question, sample_hybrid_results)

    # Check structure
    assert "=== RETRIEVED EVIDENCE ===" in prompt
    assert "=== USER QUESTION ===" in prompt
    assert question in prompt
    assert "[Doc: 637fab7088ea6c78a5dba55f17e833bd, Page: 1]" in prompt
    assert "Depreciation: AMER $ 21,224" in prompt
    assert "=== AUDIT-GRADE ANSWER ===" in prompt

    # Verify images gathered
    assert len(images) > 0
    assert any("637fab7088ea6c78a5dba55f17e833bd_1.png" in img for img in images)


def test_prompt_strict_grounding_rules():
    """Verify strict grounding rules are present in the system prompt."""
    assert "strictly using ONLY the provided evidence" in SYSTEM_PROMPT
    assert "Do NOT extrapolate, assume, or bring in external financial" in SYSTEM_PROMPT
    assert "[Doc: <document_id>, Page: <page_number>]" in SYSTEM_PROMPT
    assert "The provided evidence does not contain sufficient information" in SYSTEM_PROMPT


def test_mock_vlm_generator_grounded_answer(sample_hybrid_results):
    """Verify VLMGenerator with MockVLMProvider generates grounded response with citations."""
    generator = VLMGenerator(provider=MockVLMProvider())
    question = "What was the depreciation for AMER in 2018?"

    result = generator.generate_answer(question, sample_hybrid_results)

    assert isinstance(result, GeneratedAnswer)
    assert result.question == question
    assert len(result.answer) > 0
    # Must include citation in exact format [Doc: ..., Page: ...]
    assert "[Doc: 637fab7088ea6c78a5dba55f17e833bd, Page: 1]" in result.answer
    assert "$21,224" in result.answer
    assert result.evidence_count == 2
    assert len(result.sources) == 2
    assert result.model == "mock-vlm-v1"
    assert result.provider == "MockVLMProvider"


def test_mock_vlm_question_relevance(sample_hybrid_results):
    """Verify MockVLMProvider returns question-relevant metrics instead of static responses."""
    generator = VLMGenerator(provider=MockVLMProvider())

    # 1. Asking for capital expenditures returns capital expenditures ($17,690), not depreciation
    q_capex = "What was the capital expenditures for AMER in 2018?"
    res_capex = generator.generate_answer(q_capex, sample_hybrid_results)
    assert "$17,690" in res_capex.answer
    assert "[Doc: 637fab7088ea6c78a5dba55f17e833bd, Page: 1]" in res_capex.answer

    # 2. Asking for an unmentioned metric returns insufficient information
    q_unknown = "What was the solar panel efficiency in 2030?"
    res_unknown = generator.generate_answer(q_unknown, sample_hybrid_results)
    assert "does not contain sufficient information" in res_unknown.answer


def test_vlm_generator_empty_evidence():
    """Verify VLMGenerator handles empty retrieval results gracefully."""
    generator = VLMGenerator(provider=MockVLMProvider())
    result = generator.generate_answer("What were the sales in 2020?", [])

    assert "does not contain sufficient information" in result.answer
    assert result.evidence_count == 0
    assert result.sources == []
    assert result.images_provided == []


def test_empty_question_raises():
    """Verify empty question raises ValueError."""
    generator = VLMGenerator(provider=MockVLMProvider())
    with pytest.raises(ValueError, match="Question cannot be empty"):
        generator.generate_answer("", [])

    with pytest.raises(ValueError, match="Question cannot be empty"):
        generator.generate_answer("   ", [])


def test_provider_factory():
    """Verify provider factory returns expected classes and handles errors."""
    mock_p = get_vlm_provider("mock")
    assert isinstance(mock_p, MockVLMProvider)

    openai_p = get_vlm_provider("openai", api_key="dummy-test-key")
    assert isinstance(openai_p, OpenAIVLMProvider)
    assert openai_p.api_key == "dummy-test-key"

    with pytest.raises(ValueError, match="Unsupported VLM provider"):
        get_vlm_provider("unsupported-provider-xyz")


def test_openai_vlm_provider_missing_key():
    """Verify OpenAIVLMProvider raises informative error when api_key is not set."""
    provider = OpenAIVLMProvider(api_key=None)
    # Ensure env var is unset for this check
    provider.api_key = None
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY environment variable is not set"):
        provider.generate("Test prompt", [])
