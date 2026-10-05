"""Central configuration. Everything here can be overridden with env vars."""

import os
from pathlib import Path

MODEL_NAME = os.getenv("DOCUQUERY_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("DOCUQUERY_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("DOCUQUERY_CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("DOCUQUERY_TOP_K", "5"))

DATA_DIR = Path(os.getenv("DOCUQUERY_DATA_DIR", "data"))
INDEX_PATH = DATA_DIR / "index" / "docuquery.faiss"
META_PATH = DATA_DIR / "index" / "metadata.json"
