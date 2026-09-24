# AuditRAG: Multimodal Financial Document RAG & Audit Engine

AuditRAG is an audit-grade, multimodal Retrieval-Augmented Generation (RAG) system engineered for complex corporate financial reports (SEC 10-Ks, annual filings, earnings tables from the TAT-DQA benchmark). It combines hybrid multimodal retrieval (Dense + BM25 + ColPali) with Vision-Language Model (VLM) generation, interactive bounding-box visual citations, deterministic Python numerical verification, a Question-Answer relevance gate, and principled hallucination protection.

---

## Key Capabilities

1. **Multimodal Hybrid Retrieval**:
   - **Dense Semantic Retrieval**: Vector embeddings using `sentence-transformers/all-MiniLM-L6-v2` over text passages and table rows with a FAISS inner-product index.
   - **Sparse Lexical Retrieval (BM25)**: Exact term and identifier matching using Okapi BM25 with financial regex tokenization (capturing fiscal years, line-item headers, dollar amounts, and ticker symbols).
   - **Visual Late-Interaction Retrieval (ColPali)**: Document screenshot patch multi-vector embeddings (`vidore/colpali-v1.2`) preserving table boundaries, multi-column typography, and footnotes.
   - **Reciprocal Rank Fusion (RRF)**: Scale-invariant candidate fusion balancing lexical precision and visual layout cues ($k=60$).

2. **Grounded Multimodal VLM Generation**:
   - Anchors generation strictly to retrieved document context.
   - Dual-provider support: Deterministic offline `MockVLMProvider` for zero-cost reproduction and CI testing, and `OpenAIVLMProvider` (`gpt-4o`) for live multimodal inference.

3. **Interactive Visual Evidence Citations**:
   - Direct spatial grounding linking generated claims `[1]`, `[2]` to exact page coordinates `[x0, y0, x1, y1]`.
   - Dynamic high-contrast translucent bounding-box composite overlays generated on-the-fly.

4. **Multi-Stage Verification & Safety Engine**:
   - **Atomic Claim Extraction**: Splits responses into discrete verifiable claims classified as `TEXTUAL`, `NUMERICAL`, or `CALCULATION`.
   - **Evidence Grounding**: NLI passage entailment verification against retrieved document blocks.
   - **Deterministic Numerical Reasoning**: Independent Python execution for financial formulas (percentage growth: `((New - Old) / |Old|) * 100`, variances: `End - Start = Diff`, sums: `Sum(Parts) = Total`), avoiding LLM calculation hallucinations.
   - **Question-Answer Relevance Gate**: Deterministic verification that extracted facts answer the user's specific requested metric, rejecting off-topic fact grounding.
   - **Principled Abstention**: Triggers safe abstention (`"I couldn't verify this answer from the available document evidence."`) when claims are ungrounded (`INSUFFICIENT_EVIDENCE`), off-topic (`IRRELEVANT_ANSWER`), or mathematically invalid (`FAILED_ARITHMETIC_CHECK`).

5. **Interactive Streamlit Interface**:
   - **Interactive Audit Workbench**: Side-by-side answer inspection, citation toggling, multi-style page viewers (full overlay, focused highlight, unannotated original, side-by-side comparison).
   - **Explainable Numerical Reasoning Trace**: Auditable chain displaying *source values → formula → calculated result → reported value → verification status*.
   - **Evaluation & Reliability Dashboard**: Built-in review dashboard presenting multimodal retrieval benchmarks, answer quality cards, factuality breakdowns, and the three-stage engineering evolution.

---

## System Architecture

AuditRAG couples hybrid multimodal retrieval with Vision-Language Model (VLM) generation, visual layout grounding, deterministic mathematical verification, and principled selective abstention. The system operates across 6 modular subsystems:

1. **Ingestion & Normalization:** Ingests TAT-DQA documents, preserving high-resolution layout blocks, words, and bounding boxes.
2. **Multimodal Hybrid Retrieval:** Fuses Dense semantic vectors (FAISS), sparse lexical tokens (Okapi BM25), and visual patch tokens (ColPali) using Reciprocal Rank Fusion (RRF).
3. **Grounded Multimodal Generation:** Constructs spatial visual prompts with retrieved passages and page context for VLM synthesis.
4. **Visual Citation Engine:** Maps generated claim citations to layout bounding boxes and renders high-contrast overlays.
5. **Verification & Safety Engine:** Decomposes answers into atomic claims, verifies passage entailment, validates calculations via deterministic Python execution, and rejects off-topic facts via a relevance gate.
6. **User Interface & Evaluation:** Interactive Streamlit audit workbench and reliability dashboard.

For technical details, subsystem interfaces, and the complete component data-flow diagram, see the Mermaid architecture specification in [docs/architecture.md](docs/architecture.md).

---

## Quickstart & Installation

### 1. Environment Setup
```bash
# Clone repository and create virtual environment
git clone https://github.com/your-username/AuditRAG.git
cd AuditRAG
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Optional Environment Variables
Copy `.env.example` to `.env` if using live OpenAI Vision endpoints:
```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY if desired.
# If omitted, AuditRAG automatically operates in deterministic offline MockVLM mode.
```

### 3. Execution & CLI Commands

- **Launch Interactive Streamlit Interface:**
  ```bash
  streamlit run src/auditrag/ui/app.py
  ```

- **Run End-to-End Pipeline Query via CLI:**
  ```bash
  python3 scripts/run_pipeline.py --query "What was the total assets from AMER in 2018?"
  ```

- **Run Evidence & Numerical Verification CLI:**
  ```bash
  python3 scripts/verify_answer.py --query "What was the total assets from AMER in 2018?"
  ```

- **Run Benchmark Evaluation:**
  ```bash
  python3 scripts/run_benchmark.py --sample-size 15
  ```

- **Run Automated Test Suite:**
  ```bash
  python3 -m pytest tests/ -v
  ```

---

## Evaluation Methodology & Benchmark Scope

- **Benchmark Corpus:** Evaluated on the standardized TAT-DQA financial document benchmark split.
- **Evaluation Sample:** Uses a deterministic, representative 15-question evaluation sample (`tatdqa_dataset_test_gold.json`) spanning multi-column earnings tables, disclosures, and arithmetic calculations.
- **Scope Disclosure:** Reported metrics represent this established deterministic 15-question benchmark sample rather than an extrapolation to the full TAT-DQA corpus.
- **Benchmark Artifacts:** Pre-computed tables and charts are preserved in `reports/tables/` and `reports/figures/`. To reproduce offline:
  ```bash
  python -m scripts.run_benchmark --sample-size 15
  ```

---

## Empirical Benchmark Summary

### Multimodal Retrieval Modality Comparison ($N=15$)
| Modality | Doc-Hit@1 | Doc-Hit@5 | Doc-MRR | Page-Hit@1 | Page-Hit@5 | Page-MRR | Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dense (all-MiniLM-L6-v2)** | 40.0% | 73.3% | 0.5189 | 40.0% | 73.3% | 0.5189 | 413.9 ms |
| **BM25 (Financial Tokens)** | 60.0% | 73.3% | 0.6667 | 60.0% | 73.3% | 0.6667 | 5.7 ms |
| **ColPali (Vision-Language)** | 13.3% | 26.7% | 0.1800 | 6.7% | 26.7% | 0.1467 | 1388.1 ms |
| **Hybrid (RRF Fusion)** | 33.3% | 60.0% | 0.4444 | 33.3% | 60.0% | 0.4444 | 1647.6 ms |

### Pipeline Evolution & Hallucination Protection
| Metric | Baseline (Phase 10) | Improved Grounding (Phase 11) | Relevance-Gated Final (Phase 12) |
|:---|:---:|:---:|:---:|
| **Claim Support Rate** | 18.8% (3/16) | 62.5% (10/16) | **62.5% (10/16)** |
| **Mean Factual Consistency** | 13.3% | 60.0% | **60.0%** |
| **Accepted Answers** | 2 / 15 | 9 / 15 | **4 / 15** |
| **Abstention Rate** | 86.7% (13/15) | 40.0% (6/15) | **73.3% (11/15)** |
| **Primary Abstention Causes** | 13 Insufficient Evidence | 6 Insufficient Evidence | **6 Insufficient Evidence, 5 Irrelevant Answer** |

---

## System Limitations

1. **OCR Layout Quality:** Text chunking relies on document OCR quality; blurred or handwritten text can reduce lexical retrieval recall.
2. **Deterministic Arithmetic Scope:** Supported calculations currently cover percentage growth/change, variances, and summations. Highly specialized non-linear financial ratios require expanding the symbolic grammar.
3. **Local Deployment:** This system is engineered as an offline-reproducible local research and evaluation prototype, not an auto-scaled production microservice.

---

## Project Structure

```
AuditRAG/
├── PROJECT_SPEC.md       # Complete 22-phase engineering specification
├── pyproject.toml        # Packaging specification
├── requirements.txt      # Python dependencies
├── .env.example          # Safe configuration template
├── data/                 # Raw/processed TAT-DQA dataset and indices (.gitignored)
├── docs/                 # In-depth documentation (architecture.md, dataset.md)
├── notebooks/            # Dataset exploration notebook (01_dataset_exploration.ipynb)
├── reports/              # Version-controlled benchmark JSON tables & figures
├── scripts/              # CLI runners (run_pipeline.py, verify_answer.py, run_benchmark.py)
├── src/auditrag/         # Core package source code
│   ├── citations/        # Bounding-box coordinate extraction & overlays
│   ├── data/             # TAT-DQA data loaders & dataset structures
│   ├── evaluation/       # Evaluation metrics & benchmark runners
│   ├── generation/       # Grounded VLM prompt assembly & providers
│   ├── ingestion/        # Document layout parsing & chunking
│   ├── pipeline/         # Unified end-to-end AuditRAG pipeline engine
│   ├── retrieval/        # Dense, BM25, ColPali, and Hybrid RRF retrievers
│   ├── ui/               # Streamlit interface, viewer, trace & dashboard
│   └── verification/     # Claim extraction, NLI, numerical & relevance gates
└── tests/                # Automated pytest suite (88 passing unit/integration tests)
```
