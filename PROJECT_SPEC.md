# Project Specification: AuditRAG

## 1. Project Overview & Goal
* **Project Name:** AuditRAG
* **Goal:** Build a production-grade, multimodal financial-document Retrieval-Augmented Generation (RAG) system that:
  1. Ingests and processes complex multimodal financial documents from the **TAT-DQA test dataset**.
  2. Implements a multi-stage retrieval architecture combining **Dense semantic retrieval**, sparse **BM25 retrieval**, and visual document retrieval via **ColPali**, fused and prioritized with cross-encoder **reranking**.
  3. Generates faithful answers using a state-of-the-art **Multimodal Vision-Language Model (VLM)**.
  4. Provides granular **visual evidence citations** with exact bounding-box references mapped onto document pages/tables.
  5. Performs rigorous **claim extraction** and **evidence-based verification** against retrieved source contexts.
  6. Implements **deterministic numerical verification** executing isolated Python calculations to mathematically validate numerical statements and financial metrics.
  7. Enforces strict **abstention and hallucination control** when retrieved evidence is insufficient, contradictory, or unverifiable.
  8. Provides an interactive, research-grade **Streamlit frontend** for document exploration, query execution, visual citation inspection, and verification auditing.

---

## 2. System Architecture
```
TAT-DQA Financial Documents (Test Set)
                 │
                 ▼
       Document Ingestion & Parsing (PDF / Text / Tables / Page Images)
                 │
                 ▼
       Hybrid Multi-Representation Indexing
       ├── Dense Vector Embeddings
       ├── BM25 Sparse Inverted Index
       └── ColPali Multi-Vector Visual Token Representation
                 │
                 ▼
       Multi-Engine Retrieval & Fusion
       ├── Dense Retrieval
       ├── BM25 Keyword Retrieval
       └── ColPali Visual Page Retrieval
                 │
                 ▼
       Reciprocal Rank Fusion (RRF) / Hybrid Score Normalization
                 │
                 ▼
       Cross-Encoder Reranking
                 │
                 ▼
       Multimodal VLM Context Construction & Answer Generation
                 │
                 ▼
       Generated Answer with Visual Bounding-Box Evidence Citations
                 │
                 ▼
       Post-Generation Factuality & Audit Engine
       ├── Claim Extraction (Atomic Claims)
       ├── Evidence-Based Claim Verification (Entailment / NLI)
       └── Deterministic Numerical Verification (Python Calculator)
                 │
                 ▼
       Abstention / Rejection Filter (Hallucination Control)
                 │
                 ▼
       Verified Final Response + Visual Grounding Overlay
```

---

## 3. Dataset Constraints
* **Dataset Scope:** TAT-DQA (**Test Set Only**).
* **Strict Constraint:** Do NOT download, crawl, synthesize, or modify datasets outside the specified TAT-DQA test set.
* All data exploration and benchmark evaluations must strictly adhere to this evaluation corpus.

---

## 4. Scope & Technical Boundaries
* **Included:**
  - Multimodal financial document analysis (text, tables, charts, visual layouts).
  - Dense semantic retrieval + BM25 sparse retrieval + ColPali visual patch retrieval.
  - Hybrid fusion and reranking.
  - Multimodal Vision-Language Model (VLM) generation.
  - Document image rendering and bounding box visual citations.
  - Atomic claim extraction, evidence alignment, and deterministic numerical verification in Python.
  - Selective abstention and uncertainty quantification.
  - Comprehensive empirical evaluation, ablation studies, and error analysis.
  - Streamlit user interface and thorough unit/integration test suite.
* **Strictly Out of Scope:**
  - Cloud / AWS infrastructure deployment or IaC.
  - Heavy MLOps, model training/fine-tuning cluster orchestration, or Kubernetes.
  - Sensor monitoring, predictive maintenance, or time-series vibration analysis.
  - Automatic Speech Recognition (ASR) or audio processing.
  - Traffic analysis, computer vision object tracking, or mobile applications.
  - Autonomous multi-agent swarms, authentication/authorization systems, or billing systems.

---

## 5. Locked Development Phases

| Phase | Title | Scope & Deliverables |
|:---|:---|:---|
| **Phase 0** | **Project Specification & Scaffolding** | Repository structure, project specification, configuration files (`pyproject.toml`, `requirements.txt`, `.env.example`, `.gitignore`, `README.md`, `RESEARCH.md`). |
| **Phase 1** | **Dataset Understanding** | Audit of the TAT-DQA test set: document formats, question types (arithmetic, table, text, multi-hop), gold annotations, and baseline statistics. |
| **Phase 2** | **Document Ingestion** | Extracting page images, textual blocks, tabular cells, and layout coordinates from TAT-DQA document samples. |
| **Phase 3** | **Baseline Text RAG** | Standard chunking and dense text retrieval baseline with evaluation metrics. |
| **Phase 4** | **BM25 Retrieval** | Tokenization, BM25 index creation for text/tables, and standalone keyword retrieval evaluation. |
| **Phase 5** | **ColPali Visual Retrieval** | Visual document indexation using ColPali embeddings and late-interaction patch-level scoring. |
| **Phase 6** | **Hybrid Retrieval + Reranking** | Fusion of dense, sparse, and visual scores (RRF/normalization) followed by cross-encoder reranking. |
| **Phase 7** | **Multimodal Generation** | Prompt orchestration with visual context and multimodal VLM execution for financial Q&A. |
| **Phase 8** | **Visual Evidence / Bounding-Box Citations** | Coordinate mapping, document visual grounding, and high-fidelity bounding box citation overlays. |
| **Phase 9** | **Claim Extraction** | Deconstructing generated answers into verifiable atomic claims (textual assertions & numerical assertions). |
| **Phase 10** | **Evidence-Based Verification** | Aligning claims against retrieved context to compute grounding and factual consistency scores. |
| **Phase 11** | **Deterministic Numerical Verification** | Parsing mathematical claims, generating and executing safe Python calculation checks, and comparing against stated answers. |
| **Phase 12** | **Abstention & Hallucination Control** | Automated confidence scoring, conflict detection, and selective abstention when evidence is insufficient or contradictory. |
| **Phase 13** | **End-to-End Integration** | Unifying all modules into an extensible, configurable AuditRAG pipeline. |
| **Phase 14** | **Evaluation** | Quantitative evaluation against TAT-DQA test gold answers: retrieval recall/NDCG, generation QA-F1/EM, and verification accuracy. |
| **Phase 15** | **Ablation Study** | Systematic ablation: text-only vs visual, no reranker vs reranker, verification on vs off, dense vs hybrid. |
| **Phase 16** | **Error Analysis** | Categorizing failure modes (retrieval failure, reasoning error, OCR/table misread, verification false positives). |
| **Phase 17** | **Streamlit Frontend** | Interactive web UI with document preview, citation overlays, step-by-step audit logs, and query runner. |
| **Phase 18** | **Testing** | Unit and integration test suite across ingestion, retrieval, verification, and pipeline execution. |
| **Phase 19** | **Research Documentation** | Detailed methodology report, ablation analysis documentation, and comparative benchmark findings. |
| **Phase 20** | **GitHub / Portfolio Polish** | Code cleanliness, formatting, documentation, architecture diagrams, and release readiness. |
| **Phase 21** | **Final Quality Audit** | Reproducibility verification, benchmark sanity check, and scientific integrity sign-off. |

---

## 6. Directory Structure Specification (Section 20 Scaffolding)
```
AuditRAG/
├── .env.example
├── .gitignore
├── PROJECT_SPEC.md
├── README.md
├── RESEARCH.md
├── pyproject.toml
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   └── indices/
├── docs/
│   └── architecture.md
├── notebooks/
│   └── 01_dataset_exploration.ipynb
├── reports/
│   ├── figures/
│   └── tables/
├── src/
│   └── auditrag/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── loader.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── pdf_parser.py
│       │   └── table_extractor.py
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── dense.py
│       │   ├── bm25.py
│       │   ├── colpali.py
│       │   ├── hybrid.py
│       │   └── reranker.py
│       ├── generation/
│       │   ├── __init__.py
│       │   ├── prompt.py
│       │   └── vlm.py
│       ├── citations/
│       │   ├── __init__.py
│       │   ├── visualizer.py
│       │   └── bounding_box.py
│       ├── verification/
│       │   ├── __init__.py
│       │   ├── claim_extractor.py
│       │   ├── evidence_verifier.py
│       │   ├── numerical_verifier.py
│       │   └── abstention.py
│       ├── pipeline/
│       │   ├── __init__.py
│       │   └── engine.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   └── benchmark.py
│       └── ui/
│           ├── __init__.py
│           └── app.py
└── tests/
    ├── __init__.py
    ├── test_ingestion.py
    ├── test_retrieval.py
    ├── test_verification.py
    └── test_pipeline.py
```
