"""
pipeline/
---------
Stateless execution modules called by the RL environment.

Modules:
  decomposer  — T5 inference + evaluation + re-decompose fallback
  retriever   — BM25 + dense hybrid + cross-encoder re-ranking
  generator   — LLM sub-answer generation (Claude / Gemini)
  combiner    — concatenate or summarise sub-answers
"""
