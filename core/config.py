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
    EMBEDDING_DIM      = int(os.getenv("EMBEDDING_DIM", "384"))   # all-MiniLM dim

    # ── LLM (used in generation phase) ────────────────────
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    LLM_MODEL         = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
    GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")

    # ── Retrieval ──────────────────────────────────────────
    TOP_K        = int(os.getenv("TOP_K", "3"))
    TOP_K_BM25   = int(os.getenv("TOP_K_BM25",   "10"))  # BM25 candidates
    TOP_K_DENSE  = int(os.getenv("TOP_K_DENSE",  "10"))  # Dense candidates
    TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "5"))   # After cross-encoder rerank

    # ── BM25 ───────────────────────────────────────────────
    BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
    BM25_B  = float(os.getenv("BM25_B",  "0.75"))

    # ── Re-ranking model ───────────────────────────────────
    RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ── Decomposer ─────────────────────────────────────────
    DECOMP_MODEL_PATH      = os.getenv("DECOMP_MODEL_PATH", os.path.join(BASE_DIR, "decomp_model"))
    DECOMP_ROUGE_THRESH    = float(os.getenv("DECOMP_ROUGE_THRESH",    "0.35"))
    DECOMP_COVERAGE_THRESH = float(os.getenv("DECOMP_COVERAGE_THRESH", "0.50"))
    DECOMP_MIN_SUBQ        = int(os.getenv("DECOMP_MIN_SUBQ", "2"))
    DECOMP_MAX_SUBQ        = int(os.getenv("DECOMP_MAX_SUBQ", "5"))

    # ── Complexity Classifier ──────────────────────────────
    CLASSIFIER_MODEL_PATH = os.getenv(
        "CLASSIFIER_MODEL_PATH",
        os.path.join(BASE_DIR, "complexity_classifier")
    )
    CLASSIFIER_BERT_NAME  = os.getenv("CLASSIFIER_BERT_NAME", "nlpaueb/legal-bert-base-uncased")
    COMPLEXITY_THRESHOLD  = float(os.getenv("COMPLEXITY_THRESHOLD", "0.5"))

    # ── Generation ────────────────────────────────────────
    GEN_TEMPERATURE_DEFAULT = float(os.getenv("GEN_TEMPERATURE_DEFAULT", "0.3"))
    GEN_TEMPERATURE_RETRY   = float(os.getenv("GEN_TEMPERATURE_RETRY",   "0.7"))
    GEN_MAX_TOKENS          = int(os.getenv("GEN_MAX_TOKENS", "512"))
    MAX_ANSWER_LEN          = int(os.getenv("MAX_ANSWER_LEN", "1000"))   # for conciseness reward

    # ── RL Agent ──────────────────────────────────────────
    RL_STATE_DIM   = int(os.getenv("RL_STATE_DIM",  "1538"))  # 4×384 + 2
    RL_ACTION_DIM  = int(os.getenv("RL_ACTION_DIM", "4"))     # macro-actions
    RL_HIDDEN_DIM  = int(os.getenv("RL_HIDDEN_DIM", "256"))
    RL_LR          = float(os.getenv("RL_LR",       "3e-4"))
    RL_GAMMA       = float(os.getenv("RL_GAMMA",    "0.99"))
    RL_GAE_LAMBDA  = float(os.getenv("RL_GAE_LAMBDA", "0.95"))
    RL_CLIP_EPS    = float(os.getenv("RL_CLIP_EPS",  "0.2"))
    RL_EPOCHS      = int(os.getenv("RL_EPOCHS",     "200"))
    RL_MAX_STEPS   = int(os.getenv("RL_MAX_STEPS",  "10"))
    RL_MODEL_PATH  = os.getenv("RL_MODEL_PATH", os.path.join(BASE_DIR, "rl_model"))

    # ── Reward weights (must sum to 1.0) ──────────────────
    RW_CORRECTNESS    = float(os.getenv("RW_CORRECTNESS",    "0.20"))
    RW_RETRIEVAL      = float(os.getenv("RW_RETRIEVAL",      "0.15"))
    RW_HALLUCINATION  = float(os.getenv("RW_HALLUCINATION",  "0.15"))  # penalty weight
    RW_ENTITY         = float(os.getenv("RW_ENTITY",         "0.10"))
    RW_ENTAILMENT     = float(os.getenv("RW_ENTAILMENT",     "0.15"))
    RW_FLUENCY        = float(os.getenv("RW_FLUENCY",        "0.10"))
    RW_CONCISENESS    = float(os.getenv("RW_CONCISENESS",    "0.05"))
    RW_QUERY_ALIGN    = float(os.getenv("RW_QUERY_ALIGN",    "0.10"))

    # ── Chunker ────────────────────────────────────────────
    MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "30"))