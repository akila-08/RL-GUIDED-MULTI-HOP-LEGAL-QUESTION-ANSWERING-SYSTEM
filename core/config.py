import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # ── Paths ──────────────────────────────────────────────
    BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PDF_PATH    = os.getenv("PDF_PATH",    os.path.join(BASE_DIR, "data", "constitution_of_india.pdf"))
    DB_PATH     = os.getenv("DB_PATH",     os.path.join(BASE_DIR, "db",   "constitution_db"))
    LOG_PATH    = os.getenv("LOG_PATH",    os.path.join(BASE_DIR, "logs", "ingestion.log"))

    # ── ChromaDB ───────────────────────────────────────────
    COLLECTION_NAME = "constitution_of_india"

    # ── Embedding ──────────────────────────────────────────
    EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_BATCH_SZ = int(os.getenv("EMBEDDING_BATCH_SZ", "64"))

    # ── LLM (used in generation phase) ────────────────────
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    LLM_MODEL         = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")

    # ── Retrieval ──────────────────────────────────────────
    TOP_K = int(os.getenv("TOP_K", "3"))

    # ── Chunker ────────────────────────────────────────────
    MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "30"))