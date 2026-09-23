"""Unit tests for TAT-DQA data loader and schema representation."""

from pathlib import Path
import pytest

from auditrag.data import (
    BoundingBox,
    Document,
    DocumentBlock,
    DocumentPage,
    TATDQALoader,
    WordsData,
)


@pytest.fixture
def sample_data_dir() -> Path:
    return Path("data/raw/tatdqa")


@pytest.fixture
def loader(sample_data_dir: Path) -> TATDQALoader:
    return TATDQALoader(data_dir=sample_data_dir)


def test_bounding_box_operations():
    bbox = BoundingBox(x0=50.0, y0=100.0, x1=250.0, y1=300.0)
    assert bbox.width == 200.0
    assert bbox.height == 200.0
    assert bbox.area == 40000.0

    scaled = bbox.scale(2.0, 0.5)
    assert scaled.x0 == 100.0
    assert scaled.y0 == 50.0
    assert scaled.x1 == 500.0
    assert scaled.y1 == 150.0

    norm = bbox.normalize(1000.0, 1000.0)
    assert norm.x0 == 0.05
    assert norm.y0 == 0.1
    assert norm.x1 == 0.25
    assert norm.y1 == 0.3


def test_loader_single_document(loader: TATDQALoader):
    sample_uid = "637fab7088ea6c78a5dba55f17e833bd"
    doc = loader.load_document(sample_uid, source="plexus-corp_2019.pdf", annotated_page=1)

    assert doc.uid == sample_uid
    assert doc.source == "plexus-corp_2019.pdf"
    assert doc.page_count == 1

    page = doc.get_page(1)
    assert page is not None
    assert page.page_idx == 1
    assert len(page.blocks) > 0
    assert page.image_path is not None
    assert page.image_path.exists()
    assert doc.pdf_path is not None and doc.pdf_path.exists()


def test_loader_gold_questions(loader: TATDQALoader):
    doc_meta, questions = loader.load_questions(use_gold=True)
    assert len(doc_meta) == 277
    assert len(questions) == 1663

    # Check question fields
    sample_q = questions[0]
    assert sample_q.uid is not None
    assert sample_q.doc_uid is not None
    assert sample_q.question != ""
    assert sample_q.answer_type in {"span", "arithmetic", "multi-span", "count"}


def test_evidence_span_resolution(loader: TATDQALoader):
    sample_uid = "637fab7088ea6c78a5dba55f17e833bd"
    doc = loader.load_document(sample_uid)
    _, questions = loader.load_questions(use_gold=True)

    doc_questions = [q for q in questions if q.doc_uid == sample_uid]
    assert len(doc_questions) > 0

    q0 = doc_questions[0]
    spans = q0.resolve_evidence_spans(doc)
    assert len(spans) == len(q0.block_mapping)
    for span in spans:
        assert span.page_idx == 1
        assert len(span.bboxes) > 0
        assert span.text is not None


def test_unlabelled_test_questions(loader: TATDQALoader):
    doc_meta, questions = loader.load_questions(use_gold=False)
    assert len(doc_meta) == 277
    assert len(questions) == 1663
    # In unlabelled test questions, answers are None
    assert questions[0].answer is None
