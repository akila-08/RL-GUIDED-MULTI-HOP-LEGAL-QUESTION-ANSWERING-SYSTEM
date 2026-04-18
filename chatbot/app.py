from __future__ import annotations
import logging
import os
import sys
import time
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in path when running with uvicorn from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from rl.agent   import PPOAgent
from rl.env     import LegalQAEnv
from rl.actions import MacroAction

log = logging.getLogger("chatbot.app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ── FastAPI setup ─────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "RL-Guided Legal QA Chatbot",
    description = (
        "Multi-hop legal question answering over the Constitution of India, "
        "guided by a PPO reinforcement learning agent."
    ),
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        example="How do Articles 14 and 16(2) together ensure fairness in public employment?",
    )


class RetrievedArticle(BaseModel):
    article_num:  str
    title:        str
    text_snippet: str
    rerank_score: float


class AskResponse(BaseModel):
    question:          str
    complexity_score:  float
    is_complex:        bool
    sub_questions:     List[str]
    retrieved_articles: List[RetrievedArticle]
    sub_answers:       List[str]
    final_answer:      str
    actions_taken:     List[str]
    rewards:           Dict[str, float]
    combined_reward:   float
    steps_taken:       int
    latency_seconds:   float


class StatusResponse(BaseModel):
    status:       str
    db_doc_count: int
    rl_agent_loaded: bool
    classifier_loaded: bool


# ── Singletons loaded at startup ──────────────────────────────────────────────

_agent:           Optional[PPOAgent]  = None
_env:             Optional[LegalQAEnv] = None
_classifier_fn    = None
_is_ready:        bool                = False


def _load_classifier():
    """Load LegalBERT complexity classifier."""
    classifier_path = Config.CLASSIFIER_MODEL_PATH
    best_model_file = os.path.join(classifier_path, "best_model.pt")

    
    def heuristic_score_fn(q: str) -> float:
        q_lower = q.lower()
        words = q_lower.replace("?", "").replace(".", "").split()
        complex_keywords = {"difference", "compare", "between", "versus", "vs", "impact", "affect", "both", "and", "together"}
        is_complex = any(kw in words for kw in complex_keywords)
        return 0.8 if is_complex else 0.2

    if not os.path.exists(best_model_file):
        log.warning("Classifier not found at %s — using Smart Heuristic instead of 0.7", classifier_path)
        return heuristic_score_fn
        
    try:
        import torch
        from transformers import AutoTokenizer
        from classifier import LegalComplexityClassifier  # type: ignore
        
        tok_path = classifier_path if os.path.exists(os.path.join(classifier_path, "tokenizer_config.json")) else Config.CLASSIFIER_BERT_NAME
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
        model     = LegalComplexityClassifier(Config.CLASSIFIER_BERT_NAME)
        ckpt      = torch.load(best_model_file, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        def score_fn(q: str) -> float:
            enc = tokenizer(q, return_tensors="pt", max_length=128,
                            truncation=True, padding="max_length")
            with torch.no_grad():
                model_score = model(enc["input_ids"], enc["attention_mask"]).item()
            
            # Heuristic boost to ensure known complex patterns score >= 0.5
            q_lower = q.lower()
            words = q_lower.replace("?", "").replace(".", "").split()
            complex_keywords = {"difference", "compare", "between", "versus", "vs", "impact", "affect", "both", "and", "together", "relationship", "relate"}
            if any(kw in words for kw in complex_keywords):
                # Shift score into the complex range while maintaining variance
                if model_score < 0.5:
                    return 0.5 + model_score
            return model_score

        log.info("Classifier loaded from %s", classifier_path)
        return score_fn
    except Exception as e:
        log.error("Failed to load classifier: %s", e)
        return heuristic_score_fn


@app.on_event("startup")
async def startup():
    global _agent, _env, _classifier_fn, _is_ready

    log.info("Loading components…")
    _classifier_fn = _load_classifier()

    _env   = LegalQAEnv(complexity_classifier=_classifier_fn)
    _agent = PPOAgent()
    _agent.load()   # loads checkpoint if available; else fresh weights

    _is_ready = True
    log.info("✅  Chatbot ready.")


# ── Simple question path ──────────────────────────────────────────────────────

def _handle_simple(question: str) -> Dict[str, Any]:
    """Single-hop: retrieve → generate, no RL agent, no decomposition."""
    import pipeline.retriever as retriever_mod
    import pipeline.generator as generator_mod
    from rl.rewards import compute_all_rewards, combined_reward

    retrieve_result = retriever_mod.retrieve([question], reformulate=False)
    generate_result = generator_mod.generate([question], retrieve_result.chunks)

    final_answer = generate_result.sub_answers[0] if generate_result.sub_answers else ""

    rewards = compute_all_rewards(
        question      = question,
        final_answer  = final_answer,
        sub_questions = [question],
        doc_texts     = retrieve_result.texts
    )

    retrieved_articles = [
        {
            "article_num":  c.get("metadata", {}).get("article_num", "?"),
            "title":        c.get("metadata", {}).get("title", ""),
            "text_snippet": c.get("text", "")[:300],
            "rerank_score": c.get("rerank_score", 0.0),
        }
        for c in retrieve_result.chunks
    ]

    return {
        "sub_questions":      [question],
        "retrieved_articles": retrieved_articles,
        "sub_answers":        generate_result.sub_answers,
        "final_answer":       final_answer,
        "actions_taken":      ["RETRIEVE", "GENERATE"],
        "rewards":            rewards,
        "combined_reward":    combined_reward(rewards),
        "steps_taken":        2,
    }


# ── RL-guided complex question path ──────────────────────────────────────────

def _handle_complex(question: str) -> Dict[str, Any]:
    """
    RL-guided multi-hop pipeline.
    Follows: DECOMPOSE → RETRIEVE → GENERATE → COMBINE
    The agent's policy weights influence action choices in the second half of training.
    """
    env   = _env
    agent = _agent

    state = env.reset(question)

    for action_id in [
        MacroAction.DECOMPOSE,
        MacroAction.RETRIEVE,
        MacroAction.GENERATE,
        MacroAction.COMBINE,
    ]:
        # Use agent's chosen action 
        action, _, _ = agent.select_action(state)
        effective_action = int(action)
        state, _, done, _ = env.step(effective_action)
        if done:
            break

    return env.get_result_snapshot()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "RL-Guided Legal QA Chatbot is running."}


@app.get("/status", response_model=StatusResponse, tags=["Health"])
async def status():
    """Check system status — DB doc count, models loaded."""
    db_count = 0
    try:
        from ingestion.embedder import get_collection
        db_count = get_collection().count()
    except Exception:
        pass

    return StatusResponse(
        status             = "ready" if _is_ready else "loading",
        db_doc_count       = db_count,
        rl_agent_loaded    = _agent is not None,
        classifier_loaded  = _classifier_fn is not None,
    )


@app.post("/ask", response_model=AskResponse, tags=["QA"])
async def ask(request: AskRequest):
    """
    Main chatbot endpoint.

    For simple questions (classifier score < 0.5): single-hop retrieve+generate.
    For complex questions (score >= 0.5): full RL-guided multi-hop pipeline.
    """
    if not _is_ready:
        raise HTTPException(status_code=503, detail="System is still loading.")

    t0       = time.perf_counter()
    question = request.question.strip()

    # ── Complexity classification ──
    try:
        complexity_score = (
            float(_classifier_fn(question))
            if _classifier_fn else 0.7
        )
    except Exception as e:
        log.warning("Classifier error: %s", e)
        complexity_score = 0.7

    is_complex = complexity_score >= Config.COMPLEXITY_THRESHOLD
    log.info(
        "Question: '%s…' | complexity=%.3f | is_complex=%s",
        question[:60], complexity_score, is_complex,
    )

    # ── Route to simple or complex handler ──
    try:
        if is_complex:
            result = _handle_complex(question)
        else:
            result = _handle_simple(question)
    except Exception as e:
        log.error("Pipeline error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    latency = time.perf_counter() - t0

    # Convert retrieved articles to Pydantic model
    retrieved_articles = [
        RetrievedArticle(
            article_num  = a.get("article_num", "?"),
            title        = a.get("title", ""),
            text_snippet = a.get("text", a.get("text_snippet", "")),
            rerank_score = a.get("rerank_score", 0.0),
        )
        for a in result.get("retrieved_articles", [])
    ]

    return AskResponse(
        question           = question,
        complexity_score   = round(complexity_score, 4),
        is_complex         = is_complex,
        sub_questions      = result.get("sub_questions", []),
        retrieved_articles = retrieved_articles,
        sub_answers        = result.get("sub_answers", []),
        final_answer       = result.get("final_answer", ""),
        actions_taken      = result.get("actions_taken", []),
        rewards            = {k: round(v, 4) for k, v in result.get("rewards", {}).items()},
        combined_reward    = round(result.get("combined_reward", 0.0), 4),
        steps_taken        = result.get("steps_taken", 0),
        latency_seconds    = round(latency, 3),
    )
