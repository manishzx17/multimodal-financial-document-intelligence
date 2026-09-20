"""Central configuration and path constants for AuditRAG."""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Raw Data Paths
RAW_DATA_DIR = DATA_DIR / "raw" / "tatdqa"
RAW_TEST_DIR = RAW_DATA_DIR / "test"
RAW_TEST_JSON = RAW_DATA_DIR / "tatdqa_dataset_test.json"
RAW_TEST_GOLD_JSON = RAW_DATA_DIR / "tatdqa_dataset_test_gold.json"

# Processed Data Paths
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DOCUMENTS_DIR = PROCESSED_DATA_DIR / "documents"

# Indices & Cache Paths
INDICES_DIR = DATA_DIR / "indices"
DENSE_INDEX_DIR = INDICES_DIR / "dense"
BM25_INDEX_DIR = INDICES_DIR / "bm25"
COLPALI_INDEX_DIR = INDICES_DIR / "colpali"
CITATIONS_DIR = DATA_DIR / "citations"
CACHE_DIR = PROJECT_ROOT / ".cache"

# Embedding Model Configuration
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLPALI_MODEL = "vidore/colpali-v1.2"

import os
DEFAULT_VLM_PROVIDER = os.getenv(
    "VLM_PROVIDER",
    "openai" if os.getenv("OPENAI_API_KEY") else "mock",
)
DEFAULT_VLM_MODEL = os.getenv("VLM_MODEL", "gpt-4o")
VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.0"))
VLM_MAX_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "1024"))

# Hallucination Protection & Abstention Configuration
ABSTENTION_THRESHOLD = float(os.getenv("ABSTENTION_THRESHOLD", "1.0"))
DEFAULT_ABSTENTION_MESSAGE = "I couldn't verify this answer from the available document evidence."
