# TAT-DQA Dataset Technical Specification & Audit Report (Phase 1)

## 1. Executive Summary

This report provides the technical audit of the **TAT-DQA (Tabular And Textual Document Question Answering)** dataset, specifically analyzing the designated **Test Split** located in `data/raw/tatdqa/`.

TAT-DQA is designed for question answering over multimodal financial reports consisting of rich layout text, tables, and numerical data. In AuditRAG, this dataset serves as the benchmark corpus for evaluating hybrid retrieval (Dense + BM25 + ColPali), multimodal VLM reasoning, fine-grained visual citations, and deterministic post-generation numerical verification.

---

## 2. Dataset & Directory File Structure

The test dataset directory contains high-resolution financial documents across three complementary formats: JSON structural layout trees, vector/renderable PDFs, and rendered PNG page images.

```
data/raw/tatdqa/
├── tatdqa_dataset_test.json          # Unlabeled test questions (uid, order, question, doc metadata)
├── tatdqa_dataset_test_gold.json     # Ground truth benchmark (questions + answers + derivations + evidence)
└── test/                             # 866 raw document assets (277 documents)
    ├── <doc_uid>.json                # Detailed OCR & layout hierarchy (pages, blocks, words, bboxes)
    ├── <doc_uid>.pdf                 # Original PDF financial document snippet
    ├── <doc_uid>_1.png               # Rendered PNG image of Page 1 (224x224 px)
    └── <doc_uid>_2.png               # Rendered PNG image of Page 2 (for multi-page documents)
```

### Corpus Inventory & Statistics

| Asset Category | Metric | Exact Count | Description |
|:---|:---|:---:|:---|
| **Documents** | Total Unique Documents | **277** | Unique corporate financial document snippets |
| **PDF Files** | Total PDF Documents | **277** | 1 vector/rendered PDF per document |
| **JSON Files** | Total Document JSONs | **277** | 1 layout tree per document |
| **Page Images** | Total PNG Images | **312** | Rendered page images (224 × 224 pixels) |
| **Page Distribution** | 1-Page Documents | **242** (87.4%) | Exactly 1 page per document |
| **Page Distribution** | 2-Page Documents | **35** (12.6%) | Exactly 2 pages per document |
| **Total Pages** | Total Pages across corpus | **312** | 242 × 1 + 35 × 2 = 312 pages |
| **Total Questions** | Total Benchmark Questions | **1,663** | Evaluated across the 277 documents |
| **Questions / Doc** | Average Density | **6.0** | Range: 3 to 12 questions per document |

---

## 3. Document, Page, and Block Hierarchical Structure

Each document JSON (`data/raw/tatdqa/test/<doc_uid>.json`) contains a structured layout tree capturing pages, textual/tabular blocks, word-level tokens, and bounding boxes.

### Schema Hierarchy

```
Document (<doc_uid>.json)
└── pages: List[Page]
    ├── bbox: [0, 0, width, height]      # Page bounding box in high-res coordinate space
    └── blocks: List[Block]
        ├── uuid: str                    # Unique 36-char UUID (referenced by evidence annotations)
        ├── order: int                   # Reading order index within page
        ├── text: str                    # Full text content of the block / table row
        ├── bbox: [x0, y0, x1, y1]       # Block boundary in page coordinates
        └── words: Dict
            ├── word_list: List[str]     # Tokenized words in the block
            └── bbox_list: List[BBox]    # Word-level coordinates [x0, y0, x1, y1] (1:1 with word_list)
```

### Structural Metrics

* **Total Blocks across corpus:** 8,757 blocks.
* **Blocks per Page:** Average 28.1 blocks (min: 2, max: 186).
* **Words per Block:** Average 17.2 words (min: 1, max: 690).
* **Word / BBox Consistency:** 100% of all blocks have identical length between `word_list` and `bbox_list`.

---

## 4. Question and Gold-Answer Fields

The ground-truth question dataset (`tatdqa_dataset_test_gold.json`) contains 277 document entries, each containing document metadata and a list of questions:

```json
{
  "doc": {
    "uid": "637fab7088ea6c78a5dba55f17e833bd",
    "page": 1,
    "source": "plexus-corp_2019.pdf"
  },
  "questions": [
    {
      "uid": "15e4d550e10a33bb0402bebb5b791458",
      "order": 1,
      "question": "What was the total assets from AMER in 2018?",
      "answer": ["645,791"],
      "derivation": "",
      "answer_type": "span",
      "scale": "thousand",
      "req_comparison": false,
      "facts": ["645,791"],
      "block_mapping": [
        {
          "4e13ff19-60b5-44f5-9840-176eec106777": [17, 24]
        }
      ]
    }
  ]
}
```

### Question Field Definitions

| Field Name | Type | Description | Audit Observations |
|:---|:---|:---|:---|
| `uid` | `str` | Unique 32-character question identifier | Unique across all 1,663 questions. |
| `order` | `int` | Sequential question index within document | 1-indexed. |
| `question` | `str` | Natural language question | Financial domain queries covering tables, text, and notes. |
| `answer` | `List[str]`, `int`, or `float` | Ground-truth answer | Structured according to `answer_type`. |
| `derivation` | `str` | Mathematical or reasoning expression | Present in arithmetic, comparison, and counting questions. |
| `answer_type` | `str` | Taxonomy of the answer | `span`, `arithmetic`, `multi-span`, `count`. |
| `scale` | `str` | Financial magnitude scale modifier | `""`, `thousand`, `million`, `percent`. |
| `req_comparison` | `bool` | Whether answer requires ordinal/magnitude comparison | `True` or `False`. |
| `facts` | `List[str]` | Grounding text values extracted from document | String representations of source values. |
| `block_mapping` | `List[Dict[str, List[int]]]` | Map from block `uuid` to character offsets `[start, end]` | Direct links to source document layout nodes. |

### Answer Type Breakdown

| Answer Type | Count | Percentage | Answer Data Type | Example |
|:---|:---:|:---:|:---|:---|
| `span` | 714 | 42.9% | `List[str]` (1 string) | `["645,791"]` |
| `arithmetic` | 699 | 42.0% | `int` (301) or `float` (398) | `21234` or `-4.35` |
| `multi-span` | 210 | 12.6% | `List[str]` (multiple strings) | `["Oxaydo product...", "Nexafed products..."]` |
| `count` | 40 | 2.4% | `int` or `str` | `1` |
| **Total** | **1,663** | **100.0%** | | |

### Financial Scale Distribution

* **None / Unit (`""`):** 836 (50.3%)
* **`thousand`:** 319 (19.2%)
* **`percent`:** 294 (17.7%)
* **`million`:** 214 (12.9%)

### Reasoning & Comparison Distribution

* **Requires Comparison (`req_comparison: True`):** 93 questions (5.6%)
* **Arithmetic / Multi-step Derivations:** 831 questions (50.0%) contain non-empty derivation formulas (e.g. `958,744-937,510`, `(64.8-59.5)/59.5`, `2,816>2,725`).

---

## 5. Evidence Annotations: Facts, Derivations, and Block Mapping

Ground-truth evidence in TAT-DQA connects directly to the underlying document structure through three mechanisms:

1. **`facts` (`List[str]`):** Exact text fragments or numerical literals extracted from the document that serve as premises.
   - 745 questions require 1 fact.
   - 783 questions require 2 facts.
   - 135 questions require 3 or more facts (up to 9 facts for complex aggregations).

2. **`derivation` (`str`):** The symbolic computation connecting the facts to the answer.
   - Supports arithmetic operators: `+`, `-`, `*`, `/`.
   - Supports comparisons: `>`, `<`.
   - AuditRAG Phase 11 will use this gold formula structure to benchmark the post-generation deterministic Python calculator.

3. **`block_mapping` (`List[Dict[uuid, [start, end]]]`):**
   - Each element maps a block `uuid` to a character span `[start, end]` within that block's `text`.
   - **Audit Finding:** Across all 2,838 block mapping entries, 100% of referenced block UUIDs exist in the corresponding document JSON files.
   - **Offset Alignment:** In 96.9% of entries, slicing `block['text'][start:end]` exactly reproduces the annotated fact.
   - **Edge Case Detected:** In 44 entries (1.5%), the start offset is `-1` (e.g. `[-1, 88]`), indicating a boundary artifact in raw annotation. The AuditRAG loader clamps start offsets (`max(0, start)`) to ensure robust bounding box lookup.
   - **Zero Mapping Edge Case:** 23 questions have empty `block_mapping: []` (descriptive narrative questions where annotators recorded `facts` but omitted block coordinates).

---

## 6. Coordinate Spaces & Bounding-Box Localization

AuditRAG requires high-precision visual bounding-box citations overlaid on document pages. The dataset exhibits two distinct coordinate spaces:

### Coordinate Spaces

1. **High-Resolution Vector Layout Space (Document JSON & PDF Mediabox):**
   - Page dimensions: typically `[0, 0, 1240, 1564]` to `[0, 0, 1240, 1754]` (corresponding to ~150-200 DPI PDF rendering).
   - Standard PDF Mediabox: `612 × 792` points (Letter format, 72 DPI).
   - All block coordinates (`block['bbox']`) and word coordinates (`word['bbox_list']`) are formatted as `[x0, y0, x1, y1]` (`[left, top, right, bottom]`) in this high-resolution pixel space.
2. **Vision Model Thumbnail Space (`.png` files):**
   - All pre-rendered PNG page images in `data/raw/tatdqa/test/` have dimensions **224 × 224 pixels**.
   - These 224x224 images were generated for standard fixed-resolution vision models.
   - **Critical Architecture Implication for AuditRAG:** 224×224 resolution is insufficient for human inspection in Streamlit and fine-grained visual citations. In Phase 2 (Ingestion), AuditRAG will render the vector PDFs directly at high resolution (e.g. 150/200 DPI) using `pypdf`/`pdfplumber` while preserving an exact scaling transform `(scale_x, scale_y)` between high-res display coordinates, normalized `[0, 1]` coordinates, and ColPali visual patch coordinates.

---

## 7. Mapping Evidence Annotations to Visual Citations in AuditRAG

TAT-DQA's granular annotations provide an end-to-end pathway to build and evaluate visual citations:

```
Question: "What was the total assets from AMER in 2018?"
   │
   ├── Gold Evidence:
   │   ├── fact: "645,791"
   │   └── block_mapping: {"4e13ff19-60b5-44f5-9840-176eec106777": [17, 24]}
   │
   ▼
1. Block Resolution:
   Match block_uuid -> Page 1, Block 20, BBox: [88, 670, 1000, 688]
   │
   ▼
2. Word-Level Token Mapping:
   Slice block.text[17:24] -> "645,791"
   Match word in word_list -> word_index: 4
   Word BBox: [919, 670, 976, 688]
   │
   ▼
3. Normalized / Display Citation Box:
   Page BBox: [0, 0, 1240, 1605]
   Normalized: [x0: 0.741, y0: 0.417, x1: 0.787, y1: 0.428]
   │
   ▼
4. Visual Grounding Citation:
   Render highlight box over cell "645,791" on document page image.
```

By resolving `block_mapping` through `words.bbox_list`, AuditRAG can evaluate both block-level visual retrieval and word/cell-level visual citations against gold coordinates.

---

## 8. Implemented Internal Data Schema (`src/auditrag/data/loader.py`)

To represent this structure cleanly without modifying raw files, the schema implements:

1. `BoundingBox`: 4-coordinate bounding box with width, height, area, normalization, and scaling utilities.
2. `WordToken`: word string paired with its `BoundingBox`.
3. `DocumentBlock`: block containing text, bounding box, reading order, word tokens, and character-span-to-bbox resolution.
4. `DocumentPage`: page containing page bounding box, list of blocks, and lookup helpers.
5. `Document`: complete document entity holding page list, PDF path, image paths, and document metadata.
6. `EvidenceSpan`: resolved evidence reference linking a fact string, block UUID, character span, and computed bounding boxes.
7. `Question`: benchmark query entity supporting all answer types (`span`, `arithmetic`, `multi-span`, `count`), scale, comparison requirements, gold answers, derivations, and evidence span resolution.
8. `TATDQADataset`: comprehensive memory-efficient container indexing documents and questions with fast lookups.
9. `TATDQALoader`: lazy or eager dataset loader with caching and offset sanitization.
