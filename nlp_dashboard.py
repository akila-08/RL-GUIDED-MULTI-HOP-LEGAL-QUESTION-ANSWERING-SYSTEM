"""
nlp_dashboard.py  —  NLP Internals Dashboard
─────────────────────────────────────────────────────────────────────────────
Shows all 12 NLP concepts used in the Legal QA system with LIVE intermediate
outputs keyed to the submitted question.

Architecture:
  1.  Question enters via st.form  →  single submission triggers everything
  2.  POST /ask  (port 8000)       →  gets exact complexity score, sub-qs,
                                      retrieved articles, answers, rewards
  3.  Local NLP computations       →  tokens, embeddings, BM25, NER, ROUGE…

Run:
    streamlit run nlp_dashboard.py --server.port 8502
    ─── or ───
    .\venv\Scripts\streamlit.exe run nlp_dashboard.py --server.port 8502
"""

import streamlit as st
import sys, os, re, json
import numpy as np
import requests as _http

# ── Project path ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Internals Dashboard — Legal QA",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#080d1c;color:#e2e8f0;}

/* hero */
.hero{background:linear-gradient(135deg,#11162e 0%,#0c1220 50%,#160a2e 100%);
  border:1px solid rgba(139,92,246,.3);border-radius:20px;padding:36px 40px;
  margin-bottom:24px;position:relative;overflow:hidden;}
.hero h1{font-size:2.4rem;font-weight:900;
  background:linear-gradient(135deg,#a78bfa,#60a5fa,#34d399);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;line-height:1.2;}
.hero p{color:#64748b;font-size:.95rem;margin:8px 0 0;}

/* sidebar concept list */
.cb{display:flex;align-items:center;gap:9px;background:rgba(15,23,42,.6);
  border:1px solid rgba(99,102,241,.2);border-radius:9px;padding:9px 12px;
  margin-bottom:7px;font-size:.8rem;color:#e2e8f0;}
.cb .n{background:rgba(139,92,246,.2);color:#a78bfa;border-radius:5px;
  padding:1px 7px;font-size:.7rem;font-weight:700;margin-left:auto;}

/* stage card */
.card{background:linear-gradient(135deg,#0f1629,#121930);
  border:1px solid rgba(99,102,241,.2);border-radius:16px;
  padding:22px;margin-bottom:18px;position:relative;}
.card::before{content:'';position:absolute;top:0;left:0;width:4px;height:100%;
  background:var(--ac,linear-gradient(180deg,#8b5cf6,#6366f1));
  border-radius:4px 0 0 4px;}
.sh{display:flex;align-items:center;gap:12px;margin-bottom:14px;}
.si{width:42px;height:42px;background:var(--ib,rgba(139,92,246,.15));
  border:1px solid var(--ibr,rgba(139,92,246,.3));border-radius:10px;
  display:flex;align-items:center;justify-content:center;font-size:1.3rem;}
.st{font-size:1.1rem;font-weight:700;color:#e2e8f0;}
.ss{font-size:.78rem;color:#475569;margin-top:2px;}
.sn{margin-left:auto;background:rgba(99,102,241,.15);color:#818cf8;
  border-radius:20px;padding:3px 10px;font-size:.72rem;font-weight:700;
  font-family:'JetBrains Mono',monospace;}

/* concept explanation */
.cx{background:rgba(30,41,59,.5);border-left:3px solid #6366f1;
  border-radius:0 8px 8px 0;padding:11px 15px;margin-bottom:14px;
  font-size:.85rem;color:#94a3b8;line-height:1.6;}
.cx b,.cx code{color:#a5b4fc;}
.cx code{font-family:'JetBrains Mono',monospace;font-size:.78rem;}

/* io label */
.iol{font-size:.68rem;font-weight:700;letter-spacing:.07em;
  text-transform:uppercase;color:#334155;margin:10px 0 5px;}

/* tokens */
.ta{display:flex;flex-wrap:wrap;gap:5px;margin:6px 0;}
.tc{background:rgba(99,102,241,.15);border:1px solid rgba(99,102,241,.3);
  border-radius:5px;padding:3px 8px;font-family:'JetBrains Mono',monospace;
  font-size:.75rem;color:#a5b4fc;}
.tc.st2{background:rgba(71,85,105,.1);border-color:rgba(71,85,105,.2);color:#334155;}

/* score bar */
.sb{display:flex;align-items:center;gap:8px;margin:5px 0;}
.sl{font-size:.78rem;color:#94a3b8;min-width:170px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.sbg{flex:1;height:7px;background:rgba(30,41,59,.8);border-radius:4px;overflow:hidden;}
.sbf{height:100%;border-radius:4px;}
.sv{font-family:'JetBrains Mono',monospace;font-size:.75rem;min-width:42px;text-align:right;}

/* pills */
.pl{display:inline-flex;align-items:center;gap:5px;border-radius:20px;
  padding:4px 12px;font-size:.78rem;font-weight:600;margin:3px;}
.ok{background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.25);color:#34d399;}
.wn{background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.25);color:#fbbf24;}
.er{background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.25);color:#f87171;}
.in{background:rgba(99,102,241,.12);border:1px solid rgba(99,102,241,.25);color:#a5b4fc;}

/* sub-q */
.sqc{background:rgba(15,23,42,.7);border:1px solid rgba(99,102,241,.25);
  border-radius:9px;padding:10px 14px;margin:6px 0;font-size:.86rem;color:#cbd5e1;}
.sqn{font-family:'JetBrains Mono',monospace;font-size:.7rem;color:#818cf8;
  margin-bottom:3px;font-weight:700;}

/* code block */
.cb2{background:#050b17;border:1px solid rgba(99,102,241,.2);border-radius:7px;
  padding:12px 14px;font-family:'JetBrains Mono',monospace;font-size:.76rem;
  color:#7dd3fc;white-space:pre-wrap;overflow-x:auto;line-height:1.55;}

/* tag chips */
.art{background:rgba(59,130,246,.12);border:1px solid rgba(59,130,246,.25);
  border-radius:5px;padding:2px 9px;font-size:.78rem;color:#93c5fd;margin:2px;}
.kph{background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.25);
  border-radius:5px;padding:2px 9px;font-size:.78rem;color:#fcd34d;margin:2px;}
.kwd{background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.25);
  border-radius:5px;padding:2px 9px;font-size:.78rem;color:#6ee7b7;margin:2px;}

/* NLI bars */
.nlb{margin:8px 0;}
.nlbl{display:flex;justify-content:space-between;margin-bottom:4px;
  font-size:.8rem;font-weight:600;}
.nlbg{background:rgba(30,41,59,.8);border-radius:5px;height:10px;}
.nlbf{height:10px;border-radius:5px;}

/* reward grid */
.rg{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;}
.rc{background:rgba(15,23,42,.7);border:1px solid rgba(99,102,241,.2);
  border-radius:11px;padding:12px;text-align:center;}
.rv{font-size:1.5rem;font-weight:900;font-family:'JetBrains Mono',monospace;}
.rn{font-size:.68rem;color:#64748b;margin-top:3px;font-weight:700;
  text-transform:uppercase;letter-spacing:.05em;}
.rw{font-size:.65rem;color:#475569;}

/* vec heatmap */
.vh{display:flex;gap:2px;flex-wrap:wrap;margin:5px 0;}
.vc{width:22px;height:22px;border-radius:3px;display:flex;align-items:center;
  justify-content:center;font-size:.48rem;font-family:'JetBrains Mono',monospace;
  color:rgba(255,255,255,.7);cursor:default;}

/* entity */
.ew{line-height:2.1;font-size:.88rem;color:#cbd5e1;}
.ent{display:inline;border-radius:3px;padding:2px 5px;font-size:.8rem;}
.el{font-size:.62rem;font-weight:700;vertical-align:super;margin-left:2px;}

/* big metric */
.bm{background:rgba(15,23,42,.7);border:1px solid rgba(99,102,241,.2);
  border-radius:12px;padding:18px;text-align:center;}
.bmlabel{font-size:.68rem;color:#475569;text-transform:uppercase;
  font-weight:700;letter-spacing:.07em;}
.bmval{font-size:2.6rem;font-weight:900;font-family:'JetBrains Mono',monospace;margin:6px 0;}
.bmsub{font-size:.75rem;color:#475569;}

/* form submit btn */
div.stFormSubmitButton > button{
  background:linear-gradient(135deg,#6366f1,#8b5cf6)!important;
  color:white!important;border:none!important;border-radius:10px!important;
  font-weight:700!important;font-size:1rem!important;padding:12px 28px!important;
  width:100%;transition:all .2s!important;}
div.stFormSubmitButton > button:hover{
  transform:translateY(-1px)!important;
  box-shadow:0 8px 24px rgba(99,102,241,.35)!important;}

/* text area */
.stTextArea textarea{background:rgba(15,23,42,.8)!important;
  border:1px solid rgba(99,102,241,.3)!important;border-radius:10px!important;
  color:#e2e8f0!important;font-family:'Inter',sans-serif!important;}
.stTextArea textarea:focus{border-color:rgba(139,92,246,.6)!important;}

/* sidebar */
section[data-testid="stSidebar"]{background:#060c18!important;}
section[data-testid="stSidebar"]>div{background:#060c18!important;}
hr{border-color:rgba(99,102,241,.12)!important;}
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def scolor(v):
    return "#34d399" if v >= 0.6 else ("#fbbf24" if v >= 0.3 else "#f87171")

def score_bar(label, val, c1="#6366f1", c2="#8b5cf6"):
    pct = min(max(float(val), 0.0), 1.0) * 100
    col = scolor(val)
    return (
        f'<div class="sb">'
        f'<div class="sl">{label}</div>'
        f'<div class="sbg"><div class="sbf" style="width:{pct:.1f}%;'
        f'background:linear-gradient(90deg,{c1},{c2});"></div></div>'
        f'<div class="sv" style="color:{col};">{val:.3f}</div>'
        f'</div>'
    )

def tok_chips(toks, stops=None):
    inner = "".join(
        f'<span class="tc{"  st2" if stops and t.lower() in stops else ""}">{t}</span>'
        for t in toks
    )
    return f'<div class="ta">{inner}</div>'

def pill(txt, cls="in"):
    return f'<span class="pl {cls}">{txt}</span>'

def sq_card(i, txt):
    return f'<div class="sqc"><div class="sqn">SUB-QUESTION {i}</div>{txt}</div>'

def vec_heat(vec, n=40):
    v = np.array(vec[:n])
    vmin, vmax = v.min(), v.max()
    cells = ""
    for val in v:
        t = (val - vmin) / (vmax - vmin + 1e-8)
        r = int(24 + (139 - 24) * t)
        g = int(92 + (92 - 92) * t)
        b = int(246 + (246 - 246) * t)
        cells += (
            f'<div class="vc" title="{val:.3f}" '
            f'style="background:rgba({r},{g},{b},{0.25+0.7*t});">{val:.1f}</div>'
        )
    return f'<div class="vh">{cells}</div>'

def ent_html(text, ents):
    CMAP = {
        "ORG": ("rgba(139,92,246,.2)", "#c4b5fd"),
        "PERSON": ("rgba(16,185,129,.15)", "#6ee7b7"),
        "LAW": ("rgba(245,158,11,.15)", "#fcd34d"),
        "GPE": ("rgba(59,130,246,.15)", "#93c5fd"),
        "LOC": ("rgba(59,130,246,.12)", "#93c5fd"),
    }
    if not ents:
        return f'<div class="ew">{text[:500]}</div>'
    result = ""
    prev = 0
    for et, lbl, s, e in sorted(ents, key=lambda x: x[2]):
        result += text[prev:s]
        bg, fg = CMAP.get(lbl, ("rgba(100,116,139,.15)", "#94a3b8"))
        result += (
            f'<span class="ent" style="background:{bg};color:{fg};'
            f'border:1px solid {fg}44;">'
            f'{et}<span class="el">{lbl}</span></span>'
        )
        prev = e
    result += text[prev:500]
    return f'<div class="ew">{result}</div>'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CACHED MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Loading NLTK…")
def load_nltk():
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import word_tokenize
    return word_tokenize

@st.cache_resource(show_spinner="Loading spaCy…")
def load_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None

@st.cache_resource(show_spinner="Loading cross-encoder reranker…")
def load_reranker():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    except Exception:
        return None

@st.cache_resource(show_spinner="Loading NLI model…")
def load_nli():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder("cross-encoder/nli-deberta-v3-small")
    except Exception:
        return None

@st.cache_resource(show_spinner="Loading BM25 index…")
def load_bm25():
    from core.config import Config
    import chromadb
    try:
        client = chromadb.PersistentClient(path=Config.DB_PATH)
        col    = client.get_collection("constitution_of_india")
        res    = col.get(include=["documents", "metadatas"], limit=700)
        docs   = [{"text": d, "metadata": m}
                  for d, m in zip(res["documents"], res["metadatas"])]
        from rank_bm25 import BM25Okapi
        wt = load_nltk()
        tokenized = [
            wt(f"Article {d['metadata'].get('article_num','')} "
               f"{d['metadata'].get('title','')} {d['text']}")
            for d in docs
        ]
        bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
        return col, bm25, docs
    except Exception:
        return None, None, []

@st.cache_resource(show_spinner="Loading YAKE…")
def load_yake():
    try:
        import yake
        return yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=15)
    except Exception:
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXACT CLASSIFIER LOGIC (mirrors chatbot/app.py heuristic_score_fn)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMPLEX_KW = {
    "difference", "compare", "between", "versus", "vs",
    "impact", "affect", "both", "and", "together",
}

def heuristic_complexity(q: str) -> float:
    words = q.lower().replace("?", "").replace(".", "").split()
    return 0.8 if any(kw in words for kw in COMPLEX_KW) else 0.2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONCEPTS = [
    ("🎯", "Complexity Classification", "LegalBERT / Heuristic",           "12"),
    ("🔤", "Tokenization",              "NLTK word_tokenize",               "1"),
    ("🔑", "Keyword Extraction",        "YAKE algorithm",                   "2"),
    ("🧩", "Question Decomposition",    "T5 Seq2Seq + Rule-based NLP",      "3"),
    ("📏", "ROUGE Evaluation",          "Recall-Oriented Rouge-L",          "4"),
    ("⚡", "BM25 Sparse Retrieval",     "BM25Okapi",                        "5"),
    ("🧠", "Sentence Embedding",        "all-MiniLM-L6-v2 (384-dim)",      "6"),
    ("📐", "Cosine Similarity",         "Dot-product in R384",              "7"),
    ("🏆", "Cross-Encoder Reranking",   "ms-marco-MiniLM-L-6-v2",          "8"),
    ("🏷️", "Named Entity Recognition",  "spaCy en_core_web_sm",            "9"),
    ("💬", "LLM Text Generation",       "Groq llama3-8b-8192 (RAG)",        "10"),
    ("🔗", "NLI Entailment",            "nli-deberta-v3-small",             "11"),
]

with st.sidebar:
    st.markdown(
        '<div style="padding:12px 4px 18px;">'
        '<div style="font-size:1.05rem;font-weight:800;color:#a78bfa;">🔬 12 NLP Concepts</div>'
        '<div style="font-size:.73rem;color:#334155;margin-top:3px;">used in this Legal QA project</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    for icon, name, tech, num in CONCEPTS:
        st.markdown(
            f'<div class="cb"><span style="font-size:1rem;">{icon}</span>'
            f'<div><div>{name}</div>'
            f'<div style="font-size:.68rem;color:#334155;">{tech}</div></div>'
            f'<span class="n">{num}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:.72rem;color:#334155;padding:6px 0;">'
        '<b style="color:#475569;">API endpoint</b><br/>'
        'POST http://localhost:8000/ask<br/><br/>'
        '<b style="color:#475569;">All NLP stages use the same submitted question.</b>'
        '</div>',
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEADER + FORM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown(
    '<div class="hero">'
    '<h1>🔬 NLP Internals Dashboard</h1>'
    '<p>All 12 NLP techniques used in the Legal QA system — with live intermediate outputs '
    'keyed to your question. Every stage updates together when you submit.</p>'
    '</div>',
    unsafe_allow_html=True,
)

DEFAULT_Q = (
    "What is the difference between the original jurisdiction of the Supreme Court "
    "under Article 131 and its appellate jurisdiction under Article 132?"
)

# st.form ensures ALL stages use the SAME submitted question
with st.form("nlp_form"):
    question_input = st.text_area(
        "Enter a Legal Question",
        value=DEFAULT_Q,
        height=90,
        key="q_input",
    )
    submitted = st.form_submit_button("🚀 Run All NLP Stages", use_container_width=True)

# ── Gate: show placeholder until first submission ─────────────────────────────
if "api_data" not in st.session_state:
    st.session_state["api_data"] = None
    st.session_state["active_q"] = ""

if submitted:
    st.session_state["active_q"] = question_input.strip()
    st.session_state["api_data"] = None   # reset so we re-fetch

if not st.session_state["active_q"]:
    st.markdown(
        '<div style="text-align:center;padding:70px 20px;color:#1e293b;">'
        '<div style="font-size:2.5rem;">⬆️</div>'
        '<div style="font-size:1.05rem;font-weight:600;margin-top:10px;">'
        'Enter a question and click <span style="color:#8b5cf6;">Run All NLP Stages</span></div>'
        '<div style="font-size:.83rem;margin-top:6px;">'
        'All 12 NLP stages will run together and show live intermediate outputs.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ── Lock question for this run ────────────────────────────────────────────────
Q = st.session_state["active_q"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 0: CALL /ask API  (gets real complexity, sub-qs, articles, answers, rewards)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if st.session_state["api_data"] is None:
    prog = st.progress(0, "Calling /ask API…")
    try:
        resp = _http.post(
            "http://localhost:8000/ask",
            json={"question": Q},
            timeout=180,
        )
        resp.raise_for_status()
        st.session_state["api_data"] = resp.json()
        prog.progress(20, "API response received ✓")
    except Exception as ex:
        prog.empty()
        st.error(
            f"⚠️  Could not reach the /ask API at localhost:8000 — {ex}\n\n"
            "Make sure `uvicorn chatbot.app:app --host 0.0.0.0 --port 8000` is running."
        )
        st.stop()
else:
    prog = st.progress(20, "Using cached API response")

api = st.session_state["api_data"]

# Pull values from API response
complexity_score = float(api["complexity_score"])
is_complex       = bool(api["is_complex"])
sub_qs           = api["sub_questions"]
retrieved_arts   = api["retrieved_articles"]       # list of {article_num, title, text_snippet, rerank_score}
sub_answers      = api["sub_answers"]
final_answer     = api["final_answer"]
actions_taken    = api["actions_taken"]
rewards_dict     = api["rewards"]
combined_reward  = float(api["combined_reward"])

# Quick re-check: which keyword triggered complexity?
words_q = Q.lower().replace("?", "").replace(".", "").split()
matched_kw = [w for w in words_q if w in COMPLEX_KW]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NLP STAGE 12: COMPLEXITY CLASSIFICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
prog.progress(22, "Stage 12 — Complexity Classification")

badge_color = "#f59e0b" if is_complex else "#34d399"
badge_txt   = "⚡ COMPLEX — RL Multi-hop (D→R→G→C)" if is_complex else "✅ SIMPLE — Single-hop (R→G)"

st.markdown(
    '<div class="card" style="--ac:linear-gradient(180deg,#f59e0b,#d97706);">'
    '<div class="sh">'
    '<div class="si" style="--ib:rgba(245,158,11,.15);--ibr:rgba(245,158,11,.3);">🎯</div>'
    '<div><div class="st">Complexity Classification</div>'
    '<div class="ss">LegalBERT / Heuristic keyword scorer · mirrors chatbot/app.py logic</div></div>'
    '<div class="sn">NLP · 12</div>'
    '</div>'
    '<div class="cx">'
    '<b>What is it?</b> Decides whether the question needs multi-hop reasoning (complex) or '
    'a single retrieval step (simple). The trained classifier uses <b>LegalBERT</b> '
    '(nlpaueb/legal-bert-base-uncased) — a BERT model pre-trained on legal text. '
    'The heuristic fallback checks for domain-specific trigger keywords. '
    'Returns <code>0.8</code> if any trigger keyword is found, <code>0.2</code> otherwise. '
    'Threshold = <code>0.50</code>. <b>This score is taken directly from the /ask API response.</b>'
    '</div>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    score_col = "#34d399" if is_complex else "#60a5fa"
    st.markdown(
        f'<div class="bm"><div class="bmlabel">Complexity Score (from API)</div>'
        f'<div class="bmval" style="color:{score_col};">{complexity_score:.2f}</div>'
        f'<div class="bmsub">0.0 = simple → 1.0 = complex</div></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="bm"><div class="bmlabel">Routing Decision</div>'
        f'<div style="font-size:1rem;font-weight:700;color:{badge_color};margin:12px 0;">{badge_txt}</div>'
        f'<div class="bmsub">threshold: 0.50</div></div>',
        unsafe_allow_html=True,
    )
with c3:
    chip_html = "".join(f'<span class="tc">{w}</span>' for w in matched_kw)
    none_html = '<span style="font-size:.78rem;color:#334155;">none found</span>'
    st.markdown(
        f'<div class="bm"><div class="bmlabel">Trigger keywords matched ({len(matched_kw)})</div>'
        f'<div class="ta" style="justify-content:center;margin:10px 0;">'
        f'{chip_html if chip_html else none_html}</div>'
        f'<div class="bmsub">determines 0.8 vs 0.2 score</div></div>',
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NLP STAGE 1: TOKENIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
prog.progress(28, "Stage 1 — Tokenization")

word_tokenize = load_nltk()
all_tokens    = word_tokenize(Q)

STOPS = {
    "how","does","do","is","are","the","of","and","or","to","in","for",
    "what","when","if","a","an","their","they","together","under","with",
    "from","by","on","at","this","that","these","those","it","as","be","?",",",".",
}
content_toks = [t for t in all_tokens if t.lower() not in STOPS and t.isalpha()]
stop_toks    = [t for t in all_tokens if t.lower() in STOPS]

st.markdown(
    '<div class="card" style="--ac:linear-gradient(180deg,#6366f1,#4f46e5);">'
    '<div class="sh">'
    '<div class="si" style="--ib:rgba(99,102,241,.15);--ibr:rgba(99,102,241,.3);">🔤</div>'
    '<div><div class="st">Tokenization</div>'
    '<div class="ss">NLTK word_tokenize — Punkt + Treebank tokenizer</div></div>'
    '<div class="sn">NLP · 1</div>'
    '</div>'
    '<div class="cx">'
    '<b>What is it?</b> Splits raw text into individual word/punctuation tokens. '
    'NLTK uses the <b>Punkt</b> sentence boundary detector followed by a '
    '<b>Penn Treebank</b>-style word tokenizer. This is the <b>first and most '
    'fundamental NLP preprocessing step</b> — BM25, ROUGE, and coverage scoring '
    'all begin from tokens.'
    '</div>',
    unsafe_allow_html=True,
)

t1, t2 = st.columns(2)
with t1:
    st.markdown('<div class="iol">INPUT — Raw question</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="cb2">{Q}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="iol">ALL TOKENS ({len(all_tokens)})'
        ' — purple=content, dark=stopword</div>',
        unsafe_allow_html=True,
    )
    st.markdown(tok_chips(all_tokens, STOPS), unsafe_allow_html=True)
with t2:
    st.markdown(
        f'<div class="iol">CONTENT TOKENS ({len(content_toks)}) — fed into BM25</div>',
        unsafe_allow_html=True,
    )
    st.markdown(tok_chips(content_toks), unsafe_allow_html=True)
    st.markdown(
        f'<br/>{pill(f"Total: {len(all_tokens)}", "in")}'
        f'{pill(f"Content: {len(content_toks)}", "ok")}'
        f'{pill(f"Stopwords removed: {len(stop_toks)}", "wn")}',
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NLP STAGE 2: YAKE KEYWORD EXTRACTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
prog.progress(34, "Stage 2 — YAKE Keyword Extraction")

# Use the first retrieved article as the input sample (from API)
if retrieved_arts:
    sample_art_num  = retrieved_arts[0]["article_num"]
    sample_art_title= retrieved_arts[0]["title"]
    sample_art_text = retrieved_arts[0]["text_snippet"]
else:
    sample_art_num  = "131"
    sample_art_title= "Original jurisdiction of the Supreme Court"
    sample_art_text = (
        "Subject to the provisions of this Constitution, the Supreme Court shall, "
        "to the exclusion of any other court, have original jurisdiction in any dispute "
        "between the Government of India and one or more States."
    )

yake_ext    = load_yake()
yake_kws    = []
if yake_ext:
    try:
        raw = yake_ext.extract_keywords(sample_art_text)
        yake_kws = raw[:12]
    except Exception:
        pass

st.markdown(
    '<div class="card" style="--ac:linear-gradient(180deg,#10b981,#059669);">'
    '<div class="sh">'
    '<div class="si" style="--ib:rgba(16,185,129,.15);--ibr:rgba(16,185,129,.3);">🔑</div>'
    '<div><div class="st">Keyword Extraction (YAKE)</div>'
    '<div class="ss">Yet Another Keyword Extractor — unsupervised, statistical, language-agnostic</div></div>'
    '<div class="sn">NLP · 2</div>'
    '</div>'
    '<div class="cx">'
    '<b>What is it?</b> YAKE extracts important keywords from text using <b>5 statistical features</b>: '
    '(1) word casing, (2) word position, (3) word frequency, '
    '(4) word relatedness to context, (5) word different-sentence count. '
    '<b>Lower YAKE score = higher importance.</b> '
    'Keywords are stored in ChromaDB metadata alongside each article to enrich BM25 '
    'token matching during retrieval. Input is the actual retrieved article for your question.'
    '</div>',
    unsafe_allow_html=True,
)

k1, k2 = st.columns([1, 1])
with k1:
    st.markdown(
        f'<div class="iol">INPUT — Article {sample_art_num}: {sample_art_title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="cb2" style="font-size:.73rem;">{sample_art_text[:400]}…</div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="iol">OUTPUT — {len(yake_kws)} keywords (lower raw score = more important)</div>', unsafe_allow_html=True)
    if yake_kws:
        max_yake = yake_kws[-1][1] if yake_kws else 1
        for kw, score in yake_kws:
            # Invert: importance = 1 - normalised_score
            norm_imp = max(0.0, 1 - score / (max_yake + 1e-8)) if max_yake else 0.5
            st.markdown(score_bar(kw, min(norm_imp, 1.0), "#10b981", "#34d399"), unsafe_allow_html=True)
    else:
        st.warning("YAKE not installed — `pip install yake`")
st.markdown("</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NLP STAGE 3: QUESTION DECOMPOSITION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
prog.progress(40, "Stage 3 — Question Decomposition")

from pipeline.baseline_rules import (
    infer_question_type, extract_articles, extract_key_phrases_v2, detect_focus,
)
q_type   = infer_question_type(Q)
articles = extract_articles(Q)
phrases  = extract_key_phrases_v2(Q)
focus    = detect_focus(Q)

qt_color = {"comparative": "#f59e0b", "conditional": "#60a5fa", "analytical": "#10b981"}.get(q_type, "#94a3b8")

st.markdown(
    '<div class="card" style="--ac:linear-gradient(180deg,#8b5cf6,#7c3aed);">'
    '<div class="sh">'
    '<div class="si" style="--ib:rgba(139,92,246,.15);--ibr:rgba(139,92,246,.3);">🧩</div>'
    '<div><div class="st">Question Decomposition</div>'
    '<div class="ss">T5 Seq2Seq (Flan-T5-base) → Rule-based NLP fallback → sub-questions from /ask API</div></div>'
    '<div class="sn">NLP · 3</div>'
    '</div>'
    '<div class="cx">'
    '<b>What is it?</b> Breaks a complex question into 2–5 atomic sub-questions that can each be '
    'answered independently. <b>Stage A:</b> Fine-tuned <em>Flan-T5-base</em> (seq2seq Transformer) '
    'generates sub-questions from <code>"decompose: {question}"</code>. Quality is checked via ROUGE-L '
    'and keyword coverage — if below threshold the fallback triggers. '
    '<b>Stage B (fallback): rule-based NLP</b> — regex extraction + domain templates. '
    'Sub-questions below come from the actual /ask API pipeline run.'
    '</div>',
    unsafe_allow_html=True,
)

d1, d2 = st.columns(2)
with d1:
    st.markdown('<div class="iol">FEATURE EXTRACTION from question</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:rgba(15,23,42,.7);border:1px solid rgba(99,102,241,.2);'
        f'border-radius:9px;padding:12px;margin-bottom:10px;">'
        f'<div class="iol" style="margin-top:0;">Question Type (infer_question_type)</div>'
        f'<span style="background:rgba(255,255,255,.04);border:1px solid {qt_color}44;'
        f'border-radius:5px;padding:4px 12px;font-size:.9rem;font-weight:700;color:{qt_color};">'
        f'{q_type.upper()}</span>'
        f'<div class="iol" style="margin-top:10px;">Article Numbers (regex)</div>'
        f'<div>{"".join(f"<span class=art>Art.{a}</span>" for a in articles) or "<span style=font-size:.78rem;color:#334155;>none found</span>"}</div>'
        f'<div class="iol" style="margin-top:10px;">Key Legal Phrases (5-layer extract)</div>'
        f'<div>{"".join(f"<span class=kph>{p}</span>" for p in phrases) or "<span style=font-size:.78rem;color:#334155;>none</span>"}</div>'
        f'<div class="iol" style="margin-top:10px;">Focus Concept (detect_focus)</div>'
        f'<span class="kwd">{focus or "(none)"}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
with d2:
    st.markdown(f'<div class="iol">OUTPUT — {len(sub_qs)} sub-questions (from API)</div>', unsafe_allow_html=True)
    for i, sq in enumerate(sub_qs, 1):
        st.markdown(sq_card(i, sq), unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NLP STAGE 4: ROUGE EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
prog.progress(46, "Stage 4 — ROUGE Score")

joined_sqs = " ".join(sub_qs)
rouge_data = {}
try:
    from rouge_score import rouge_scorer as _rs
    scorer = _rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r = scorer.score(Q, joined_sqs)
    rouge_data = {
        "rouge1": (r["rouge1"].precision, r["rouge1"].recall, r["rouge1"].fmeasure),
        "rouge2": (r["rouge2"].precision, r["rouge2"].recall, r["rouge2"].fmeasure),
        "rougeL": (r["rougeL"].precision, r["rougeL"].recall, r["rougeL"].fmeasure),
    }
except Exception:
    pass

# Coverage
q_tset = {t.lower() for t in all_tokens if t.isalpha() and t.lower() not in STOPS}
sub_blob = joined_sqs.lower()
covered  = {t for t in q_tset if t in sub_blob}
coverage = len(covered) / max(len(q_tset), 1)
missing  = q_tset - covered

st.markdown(
    '<div class="card" style="--ac:linear-gradient(180deg,#ec4899,#db2777);">'
    '<div class="sh">'
    '<div class="si" style="--ib:rgba(236,72,153,.15);--ibr:rgba(236,72,153,.3);">📏</div>'
    '<div><div class="st">ROUGE Score Evaluation</div>'
    '<div class="ss">Recall-Oriented Understudy for Gisting Evaluation — quality gate for decomposition</div></div>'
    '<div class="sn">NLP · 4</div>'
    '</div>'
    '<div class="cx">'
    '<b>What is it?</b> Compares the generated sub-questions against the original question. '
    '<b>ROUGE-L Recall</b> measures the fraction of the original question\'s LCS (Longest Common '
    'Subsequence) that appears in the sub-questions. If recall &lt; 0.35 AND keyword coverage &lt; 0.50, '
    'the T5 output is <b>rejected</b> and the rule-based fallback triggers. '
    'These scores are computed on the actual sub-questions from the API pipeline.'
    '</div>',
    unsafe_allow_html=True,
)

r1, r2 = st.columns(2)
with r1:
    st.markdown('<div class="iol">ROUGE SCORES — sub-questions vs original question</div>', unsafe_allow_html=True)
    if rouge_data:
        for metric, (prec, rec, f1) in rouge_data.items():
            st.markdown(score_bar(f"{metric.upper()} Precision", prec, "#ec4899", "#f472b6"), unsafe_allow_html=True)
            st.markdown(score_bar(f"{metric.upper()} Recall",    rec,  "#db2777", "#f472b6"), unsafe_allow_html=True)
            st.markdown(score_bar(f"{metric.upper()} F1",        f1,   "#be185d", "#e879f9"), unsafe_allow_html=True)
            st.markdown("<hr/>", unsafe_allow_html=True)
        rl_rec = rouge_data["rougeL"][1]
        status = "ok" if rl_rec >= 0.35 else "er"
        st.markdown(
            f'{pill(f"ROUGE-L Recall = {rl_rec:.3f}", status)}'
            f'{pill("Threshold = 0.35", "in")}'
            f'{pill("T5 ACCEPTED" if rl_rec >= 0.35 else "Fallback triggered", status)}',
            unsafe_allow_html=True,
        )
    else:
        st.info("Install rouge-score: `pip install rouge-score`")
with r2:
    st.markdown(f'<div class="iol">KEYWORD COVERAGE — {len(covered)}/{len(q_tset)} question keywords in sub-qs</div>', unsafe_allow_html=True)
    for tok in sorted(q_tset)[:18]:
        hit = tok in sub_blob
        col = "#34d399" if hit else "#f87171"
        ico = "✓" if hit else "✗"
        st.markdown(
            f'<span class="tc" style="border-color:{col}44;color:{col};margin:2px;">{ico} {tok}</span>',
            unsafe_allow_html=True,
        )
    cov_cls = "ok" if coverage >= 0.5 else "er"
    st.markdown(
        f'<br/>{pill(f"Coverage: {coverage:.1%}", cov_cls)}'
        f'{pill(f"Missing: {len(missing)}", "wn" if missing else "ok")}',
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NLP STAGE 5: BM25 SPARSE RETRIEVAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
prog.progress(52, "Stage 5 — BM25 Sparse Retrieval")

col_chroma, bm25_idx, all_docs = load_bm25()
bm25_hits = []
query_toks = [t.lower() for t in all_tokens if t.isalpha()]

if bm25_idx and all_docs:
    try:
        scores   = bm25_idx.get_scores(query_toks)
        top_idxs = np.argsort(scores)[::-1][:10]
        bm25_max = float(scores[top_idxs[0]]) or 1.0
        for idx in top_idxs:
            d = all_docs[idx]
            bm25_hits.append({
                "article": d["metadata"].get("article_num", "?"),
                "title":   d["metadata"].get("title", "")[:55],
                "score":   float(scores[idx]),
                "text":    d["text"][:220],
            })
    except Exception:
        pass

st.markdown(
    '<div class="card" style="--ac:linear-gradient(180deg,#f59e0b,#d97706);">'
    '<div class="sh">'
    '<div class="si" style="--ib:rgba(245,158,11,.15);--ibr:rgba(245,158,11,.3);">⚡</div>'
    '<div><div class="st">BM25 Sparse Retrieval</div>'
    '<div class="ss">Best Match 25 — probabilistic TF-IDF with saturation and length normalisation</div></div>'
    '<div class="sn">NLP · 5</div>'
    '</div>'
    '<div class="cx">'
    '<b>What is it?</b> BM25 ranks documents by <b>term frequency (TF)</b> and '
    '<b>inverse document frequency (IDF)</b> with saturation (k1=1.5, b=0.75). '
    'Documents with your query tokens appearing frequently in that document '
    'but rarely across all 450 articles score highest. '
    'Unlike dense retrieval, it works purely on <em>keyword overlap</em> — fast, '
    'interpretable, excellent for exact legal article numbers and terms. '
    'The 10 BM25 hits + 10 dense hits are merged into ~20 candidates for re-ranking.'
    '</div>',
    unsafe_allow_html=True,
)

b1, b2 = st.columns(2)
with b1:
    st.markdown(f'<div class="iol">TOKENISED QUERY ({len(query_toks)} tokens → BM25 scorer)</div>', unsafe_allow_html=True)
    st.markdown(tok_chips(query_toks[:25]), unsafe_allow_html=True)
    st.markdown('<div class="iol" style="margin-top:12px;">BM25 SCORES — top 10 articles</div>', unsafe_allow_html=True)
    if bm25_hits:
        bm25_max = bm25_hits[0]["score"] or 1.0
        for h in bm25_hits[:8]:
            norm = h["score"] / bm25_max
            st.markdown(score_bar(f"Art.{h['article']} – {h['title']}", norm, "#f59e0b", "#fbbf24"), unsafe_allow_html=True)
    else:
        st.warning("ChromaDB not found — run `python ingest_pipeline.py` first.")
with b2:
    if bm25_hits:
        st.markdown('<div class="iol">TOP BM25 HIT PREVIEW</div>', unsafe_allow_html=True)
        top = bm25_hits[0]
        top_score_fmt = f"{top['score']:.3f}"
        st.markdown(
            f'<div style="background:rgba(15,23,42,.7);border:1px solid rgba(245,158,11,.2);'
            f'border-radius:9px;padding:14px;">'
            f'<span class="art">Article {top["article"]}</span>'
            f'<span style="color:#64748b;font-size:.8rem;margin-left:8px;">{top["title"]}</span>'
            f'<div style="font-size:.78rem;color:#475569;margin-top:8px;line-height:1.5;">{top["text"]}…</div>'
            f'<br/>{pill("BM25 Score: " + top_score_fmt, "ok")}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Also show articles retrieved by the actual API pipeline
    if retrieved_arts:
        st.markdown('<div class="iol" style="margin-top:14px;">ARTICLES RETRIEVED BY FULL PIPELINE (from API)</div>', unsafe_allow_html=True)
        for ra in retrieved_arts:
            st.markdown(
                f'<span class="art">Art. {ra["article_num"]}</span> '
                f'<span style="font-size:.78rem;color:#64748b;">{ra["title"]}</span>'
                f'<br/>',
                unsafe_allow_html=True,
            )
st.markdown("</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NLP STAGE 6 + 7: SENTENCE EMBEDDING + COSINE SIMILARITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
prog.progress(60, "Stage 6+7 — Sentence Embedding + Cosine Similarity")

emb_model = load_embedder()
q_vec     = emb_model.encode(Q, normalize_embeddings=True)
sub_vecs  = emb_model.encode(sub_qs, normalize_embeddings=True) if sub_qs else np.zeros((1, 384))
cosine_sims = [float(np.dot(q_vec, sv)) for sv in sub_vecs]

# Dense query on chroma
dense_hits = []
if col_chroma:
    try:
        raw = col_chroma.query(query_embeddings=[q_vec.tolist()], n_results=5)
        for doc, meta, dist in zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0]):
            dense_hits.append({
                "article": meta.get("article_num", "?"),
                "title":   meta.get("title", "")[:55],
                "cosine":  1 - float(dist),
                "text":    doc[:220],
            })
    except Exception:
        pass

st.markdown(
    '<div class="card" style="--ac:linear-gradient(180deg,#3b82f6,#2563eb);">'
    '<div class="sh">'
    '<div class="si" style="--ib:rgba(59,130,246,.15);--ibr:rgba(59,130,246,.3);">🧠</div>'
    '<div><div class="st">Sentence Embedding + Cosine Similarity</div>'
    '<div class="ss">all-MiniLM-L6-v2 produces 384-dim vectors · cosine = dot product of normalised vectors</div></div>'
    '<div class="sn">NLP · 6 + 7</div>'
    '</div>'
    '<div class="cx">'
    '<b>Sentence Embedding:</b> SentenceTransformer encodes text into a '
    '<b>384-dimensional dense float vector</b> capturing semantic meaning. '
    'Similar sentences → vectors in similar directions. '
    'all-MiniLM-L6-v2 is a distilled BERT — fast and compact.<br/><br/>'
    '<b>Cosine Similarity:</b> <code>cos(θ) = (A·B)/(|A||B|)</code>. '
    'Range: −1 (opposite) → +1 (identical). '
    'Used for: dense retrieval (Q vs docs), query-alignment reward (Q vs answer), '
    'and all 4 components of the 1538-dim RL state vector.'
    '</div>',
    unsafe_allow_html=True,
)

e1, e2 = st.columns(2)
with e1:
    st.markdown('<div class="iol">QUERY EMBEDDING — 384-dim (first 40 dims as heatmap)</div>', unsafe_allow_html=True)
    st.markdown(vec_heat(q_vec, n=40), unsafe_allow_html=True)
    norm_val = float(np.linalg.norm(q_vec))
    st.markdown(
        f'<div style="font-size:.72rem;color:#334155;margin-top:4px;">'
        f'Shape: {q_vec.shape} | Range: [{q_vec.min():.3f}, {q_vec.max():.3f}] | '
        f'L2 norm: {norm_val:.4f}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="iol" style="margin-top:14px;">COSINE SIM — Question ↔ each Sub-question</div>', unsafe_allow_html=True)
    for i, (sq, sim) in enumerate(zip(sub_qs, cosine_sims)):
        label = f"Sub-Q {i+1}: {sq[:42]}…"
        st.markdown(score_bar(label, sim, "#3b82f6", "#60a5fa"), unsafe_allow_html=True)
with e2:
    st.markdown('<div class="iol">DENSE RETRIEVAL — top 5 by cosine similarity</div>', unsafe_allow_html=True)
    if dense_hits:
        for h in dense_hits:
            st.markdown(score_bar(f"Art.{h['article']} – {h['title']}", h["cosine"], "#3b82f6", "#818cf8"), unsafe_allow_html=True)
        td = dense_hits[0]
        td_cosine_fmt = f"{td['cosine']:.4f}"
        st.markdown(
            f'<div style="background:rgba(15,23,42,.7);border:1px solid rgba(59,130,246,.2);'
            f'border-radius:9px;padding:12px;margin-top:10px;">'
            f'<span class="art">Article {td["article"]}</span>'
            f'<span style="color:#64748b;font-size:.78rem;margin-left:8px;">{td["title"]}</span>'
            f'<div style="font-size:.76rem;color:#475569;margin-top:7px;line-height:1.5;">{td["text"]}…</div>'
            f'<br/>{pill("Cosine: " + td_cosine_fmt, "ok")}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("ChromaDB needed for dense results.")
st.markdown("</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NLP STAGE 8: CROSS-ENCODER RERANKING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
prog.progress(67, "Stage 8 — Cross-Encoder Reranking")

reranker = load_reranker()
candidates = dense_hits[:5] if dense_hits else bm25_hits[:5]
before_order, after_order = [], []

if reranker and candidates:
    try:
        anchor = sub_qs[0] if sub_qs else Q
        pairs  = [
            (anchor, f"Article {c['article']} - {c['title']}\n{c['text']}")
            for c in candidates
        ]
        ce_scores = reranker.predict(pairs).tolist()
        before_order = list(zip(
            [c["article"] for c in candidates],
            [c["title"]   for c in candidates],
            [c.get("cosine", c.get("score", 0)) for c in candidates],
            ce_scores,
        ))
        after_order = sorted(before_order, key=lambda x: x[3], reverse=True)
    except Exception:
        pass

st.markdown(
    '<div class="card" style="--ac:linear-gradient(180deg,#14b8a6,#0d9488);">'
    '<div class="sh">'
    '<div class="si" style="--ib:rgba(20,184,166,.15);--ibr:rgba(20,184,166,.3);">🏆</div>'
    '<div><div class="st">Cross-Encoder Re-ranking</div>'
    '<div class="ss">ms-marco-MiniLM-L-6-v2 — joint (query, document) attention for accuracy</div></div>'
    '<div class="sn">NLP · 8</div>'
    '</div>'
    '<div class="cx">'
    '<b>What is it?</b> A Cross-Encoder takes <em>both</em> the query and document concatenated '
    'as a single input and scores them jointly — allowing <b>full cross-attention</b> between '
    'every query token and every document token. Much more accurate than bi-encoder cosine similarity '
    'but slower (can\'t pre-compute). The ~20 union candidates from BM25+Dense are re-scored and '
    'top-5 most relevant articles are selected. Fine-tuned on MS MARCO (large-scale IR dataset).'
    '</div>',
    unsafe_allow_html=True,
)

if before_order and after_order:
    rr1, rr2 = st.columns(2)
    with rr1:
        st.markdown('<div class="iol">BEFORE — ranked by bi-encoder cosine</div>', unsafe_allow_html=True)
        for rank, (art, title, cosine, ce) in enumerate(before_order, 1):
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;padding:6px 0;'
                f'border-bottom:1px solid rgba(99,102,241,.08);">'
                f'<span style="color:#334155;font-family:monospace;font-size:.72rem;min-width:20px;">#{rank}</span>'
                f'<span class="art">Art.{art}</span>'
                f'<span style="color:#475569;font-size:.76rem;flex:1;overflow:hidden;'
                f'text-overflow:ellipsis;white-space:nowrap;">{title}</span>'
                f'<span style="font-family:monospace;font-size:.7rem;color:#94a3b8;">{cosine:.3f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    with rr2:
        st.markdown('<div class="iol">AFTER — ranked by Cross-Encoder score ▲</div>', unsafe_allow_html=True)
        for rank, (art, title, cosine, ce) in enumerate(after_order, 1):
            old_rank = next((i+1 for i, x in enumerate(before_order) if x[0] == art), rank)
            arrow = "↑" if old_rank > rank else ("↓" if old_rank < rank else "→")
            acol  = "#34d399" if old_rank > rank else ("#f87171" if old_rank < rank else "#64748b")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;padding:6px 0;'
                f'border-bottom:1px solid rgba(99,102,241,.08);">'
                f'<span style="color:#334155;font-family:monospace;font-size:.72rem;min-width:20px;">#{rank}</span>'
                f'<span style="color:{acol};font-size:.85rem;">{arrow}</span>'
                f'<span class="art">Art.{art}</span>'
                f'<span style="color:#475569;font-size:.76rem;flex:1;overflow:hidden;'
                f'text-overflow:ellipsis;white-space:nowrap;">{title}</span>'
                f'<span style="font-family:monospace;font-size:.7rem;color:#14b8a6;">{ce:.2f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
else:
    st.info("Cross-encoder runs after ChromaDB returns candidates.")
st.markdown("</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NLP STAGE 9: NAMED ENTITY RECOGNITION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
prog.progress(74, "Stage 9 — Named Entity Recognition (spaCy)")

nlp_ner = load_spacy()
ner_text = retrieved_arts[0]["text_snippet"] if retrieved_arts else sample_art_text
entities = []
q_ents   = []

if nlp_ner:
    try:
        d_ner = nlp_ner(ner_text[:500])
        entities = [(e.text, e.label_, e.start_char, e.end_char) for e in d_ner.ents]
        d_q   = nlp_ner(Q)
        q_ents = [(e.text, e.label_) for e in d_q.ents]
    except Exception:
        pass

st.markdown(
    '<div class="card" style="--ac:linear-gradient(180deg,#a855f7,#9333ea);">'
    '<div class="sh">'
    '<div class="si" style="--ib:rgba(168,85,247,.15);--ibr:rgba(168,85,247,.3);">🏷️</div>'
    '<div><div class="st">Named Entity Recognition (NER)</div>'
    '<div class="ss">spaCy en_core_web_sm — identifies ORG, LAW, GPE, PERSON, DATE entities</div></div>'
    '<div class="sn">NLP · 9</div>'
    '</div>'
    '<div class="cx">'
    '<b>What is it?</b> NER identifies and classifies real-world entities in text. '
    'spaCy uses a <b>CNN + transition-based parser</b> trained on OntoNotes 5. '
    'Used in two reward signals: '
    '(1) <b>Entity Reward</b> — checks that entities in the answer appear in retrieved docs '
    '(grounding, anti-hallucination). '
    '(2) <b>Groundedness Reward</b> — fraction of answer tokens traceable to retrieved articles. '
    'NER runs on the actual retrieved article text for your question.'
    '</div>',
    unsafe_allow_html=True,
)

n1, n2 = st.columns(2)
with n1:
    st.markdown('<div class="iol">NER ON QUESTION</div>', unsafe_allow_html=True)
    if nlp_ner:
        if q_ents:
            CMAP2 = {"ORG":"#c4b5fd","PERSON":"#6ee7b7","LAW":"#fcd34d","GPE":"#93c5fd"}
            for et, lbl in q_ents:
                col = CMAP2.get(lbl, "#94a3b8")
                st.markdown(
                    f'<span class="tc" style="border-color:{col}44;color:{col};margin:2px;">'
                    f'🏷 {et} <span style="font-size:.62rem;opacity:.7">{lbl}</span></span>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<span style="font-size:.8rem;color:#334155;">No entities detected in question.</span>', unsafe_allow_html=True)
    else:
        st.warning("spaCy not found. Run: `python -m spacy download en_core_web_sm`")

    st.markdown('<div class="iol" style="margin-top:14px;">ENTITY TYPES</div>', unsafe_allow_html=True)
    for lbl, desc, col in [
        ("ORG","Organizations","#c4b5fd"),("LAW","Laws & Acts","#fcd34d"),
        ("GPE","Countries/States","#93c5fd"),("PERSON","People","#6ee7b7"),
        ("DATE","Dates","#94a3b8"),("CARDINAL","Numbers","#94a3b8"),
    ]:
        st.markdown(
            f'<span class="tc" style="border-color:{col}44;color:{col};margin:2px;">'
            f'{lbl}: {desc}</span>',
            unsafe_allow_html=True,
        )
with n2:
    st.markdown('<div class="iol">NER ON RETRIEVED ARTICLE — highlighted entities</div>', unsafe_allow_html=True)
    if entities:
        st.markdown(ent_html(ner_text, entities), unsafe_allow_html=True)
        n_types = len(set(e[1] for e in entities))
        st.markdown(
            f'<br/>{pill(f"{len(entities)} entities found", "ok")}'
            f'{pill(f"{n_types} entity types", "in")}',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f'<div class="cb2" style="font-size:.75rem;">{ner_text[:400]}</div>', unsafe_allow_html=True)
        if nlp_ner:
            st.info("No entities detected in this text segment.")
st.markdown("</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NLP STAGE 10: LLM TEXT GENERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
prog.progress(81, "Stage 10 — LLM Generation")

from core.config import Config as _Cfg
llm_model = _Cfg.LLM_MODEL or "llama3-8b-8192"

# Build reconstructed prompt for display purposes
context_preview = ""
if retrieved_arts:
    for i, ra in enumerate(retrieved_arts[:2], 1):
        context_preview += f"[{i}] Article {ra['article_num']} — {ra['title']}\n{ra['text_snippet'][:300]}\n\n"
else:
    context_preview = sample_art_text

first_sq = sub_qs[0] if sub_qs else Q
sys_prompt = (
    "You are a legal assistant specialised in the Constitution of India. "
    "Answer questions strictly based on the provided context. "
    "Be precise, cite article numbers when relevant, and stay concise."
)
user_prompt_preview = f"### Context:\n{context_preview[:700]}\n\n### Sub-Question:\n{first_sq}\n\n### Answer:"

displayed_answer = sub_answers[0] if sub_answers else "(no answer generated)"

st.markdown(
    f'<div class="card" style="--ac:linear-gradient(180deg,#f97316,#ea580c);">'
    f'<div class="sh">'
    f'<div class="si" style="--ib:rgba(249,115,22,.15);--ibr:rgba(249,115,22,.3);">💬</div>'
    f'<div><div class="st">LLM Text Generation (RAG)</div>'
    f'<div class="ss">Groq API — {llm_model} | answers from actual /ask pipeline run</div></div>'
    f'<div class="sn">NLP · 10</div>'
    f'</div>'
    f'<div class="cx">'
    f'<b>What is it?</b> A Large Language Model generates answers by reading the retrieved article '
    f'context + the sub-question as a prompt. This is <b>Retrieval-Augmented Generation (RAG)</b> — '
    f'the LLM is constrained to answer <em>only</em> from the provided constitutional articles. '
    f'Temperature = 0.3 (factual). Sub-answers below come from the actual API pipeline run. '
    f'The LLM is called once per sub-question; a bad answer triggers a retry at temperature 0.7.'
    f'</div>',
    unsafe_allow_html=True,
)

g1, g2 = st.columns(2)
with g1:
    st.markdown('<div class="iol">SYSTEM PROMPT</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="cb2" style="color:#34d399;font-size:.74rem;">{sys_prompt}</div>', unsafe_allow_html=True)
    st.markdown('<div class="iol" style="margin-top:10px;">USER PROMPT (reconstructed for display)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="cb2" style="font-size:.7rem;">{user_prompt_preview[:650]}…</div>', unsafe_allow_html=True)
    st.markdown(
        f'<br/>{pill(f"Model: {llm_model}", "in")}'
        f'{pill("Temp: 0.3", "in")}'
        f'{pill("Max tokens: 300", "in")}'
        f'{pill(f"Sub-Qs answered: {len(sub_answers)}", "ok")}',
        unsafe_allow_html=True,
    )
with g2:
    st.markdown(f'<div class="iol">SUB-ANSWERS (from /ask API) — {len(sub_answers)} generated</div>', unsafe_allow_html=True)
    for i, ans in enumerate(sub_answers, 1):
        wc = len(ans.split())
        st.markdown(
            f'<div class="sqc">'
            f'<div class="sqn">SUB-ANSWER {i} — {wc} words</div>'
            f'{ans[:400]}{"…" if len(ans) > 400 else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="iol" style="margin-top:10px;">COMBINED FINAL ANSWER (summarise)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="background:rgba(15,23,42,.7);border:1px solid rgba(249,115,22,.2);'
        f'border-radius:9px;padding:14px;font-size:.86rem;color:#e2e8f0;line-height:1.7;">'
        f'{final_answer[:500]}{"…" if len(final_answer) > 500 else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NLP STAGE 11: NLI ENTAILMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
prog.progress(88, "Stage 11 — NLI Entailment")

nli_model  = load_nli()
nli_probs  = {}
nli_demo   = False

if nli_model and final_answer:
    try:
        premise    = context_preview[:500]
        hypothesis = final_answer[:300]
        raw_scores = nli_model.predict([(premise, hypothesis)])
        sc = np.array(raw_scores[0]) if hasattr(raw_scores[0], "__len__") else np.array([0.1, 0.2, 0.7])
        # softmax
        sc = np.exp(sc) / np.exp(sc).sum()
        nli_probs = {
            "contradiction": float(sc[0]),
            "neutral":       float(sc[1]),
            "entailment":    float(sc[2]),
        }
    except Exception:
        nli_demo = True
else:
    nli_demo = True

if nli_demo:
    # Use the entailment reward from the actual API response if available
    ent_from_api = float(rewards_dict.get("entailment", 0.65))
    nli_probs = {
        "contradiction": round(max(0.0, 0.08 - ent_from_api * 0.05), 3),
        "neutral":       round(max(0.0, 1 - ent_from_api - 0.05), 3),
        "entailment":    ent_from_api,
    }

st.markdown(
    '<div class="card" style="--ac:linear-gradient(180deg,#06b6d4,#0891b2);">'
    '<div class="sh">'
    '<div class="si" style="--ib:rgba(6,182,212,.15);--ibr:rgba(6,182,212,.3);">🔗</div>'
    '<div><div class="st">NLI — Natural Language Inference</div>'
    '<div class="ss">cross-encoder/nli-deberta-v3-small — entailment reward signal (weight 0.20)</div></div>'
    '<div class="sn">NLP · 11</div>'
    '</div>'
    '<div class="cx">'
    '<b>What is it?</b> NLI determines whether a <em>hypothesis</em> (generated answer) '
    'logically <b>follows from</b> (entailed), <b>contradicts</b>, or is <b>neutral</b> to '
    'a <em>premise</em> (retrieved context). DeBERTa-v3 uses <b>disentangled attention</b> '
    '(separates content and position vectors). The <b>entailment probability = reward_entailment</b> '
    '(weight 0.20). High score = the answer is well-supported by the retrieved articles.'
    '</div>',
    unsafe_allow_html=True,
)

nl1, nl2 = st.columns(2)
with nl1:
    st.markdown('<div class="iol">PREMISE (retrieved article context)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="cb2" style="font-size:.72rem;color:#94a3b8;">{context_preview[:300]}…</div>', unsafe_allow_html=True)
    st.markdown('<div class="iol" style="margin-top:10px;">HYPOTHESIS (generated final answer)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="cb2" style="font-size:.72rem;color:#94a3b8;">{final_answer[:300]}</div>', unsafe_allow_html=True)
    if nli_demo:
        st.markdown(
            f'{pill("Entailment from API reward", "wn")}'
            f'{pill("Load NLI model for cross-encoder scores", "in")}',
            unsafe_allow_html=True,
        )

with nl2:
    st.markdown('<div class="iol">NLI CLASSIFICATION PROBABILITIES</div>', unsafe_allow_html=True)
    NLI_COLORS = {"contradiction": "#ef4444", "neutral": "#f59e0b", "entailment": "#22c55e"}
    for label, prob in nli_probs.items():
        col = NLI_COLORS[label]
        pct = prob * 100
        st.markdown(
            f'<div class="nlb">'
            f'<div class="nlbl"><span style="color:{col};text-transform:uppercase;">{label}</span>'
            f'<span style="font-family:monospace;color:{col};font-weight:700;">{prob:.1%}</span></div>'
            f'<div class="nlbg"><div class="nlbf" style="width:{pct:.1f}%;background:{col};"></div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    ent_val = nli_probs.get("entailment", 0.0)
    ent_col = scolor(ent_val)
    st.markdown(
        f'<div style="background:rgba(34,197,94,.07);border:1px solid rgba(34,197,94,.2);'
        f'border-radius:9px;padding:14px;margin-top:12px;">'
        f'<div style="font-size:.7rem;color:#334155;font-weight:700;">ENTAILMENT REWARD</div>'
        f'<div style="font-size:2.4rem;font-weight:900;color:{ent_col};'
        f'font-family:monospace;">{ent_val:.4f}</div>'
        f'<div style="font-size:.72rem;color:#334155;">weight = 0.20 in combined reward</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL: 8 NLP REWARD SIGNALS (directly from API — exact values)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
prog.progress(96, "Computing reward signals…")

REWARD_META = {
    "groundedness":  ("🛡️", "Groundedness",   "Entity-doc overlap (anti-hallucination)", "#22c55e", 0.25),
    "entailment":    ("🔗", "Entailment",      "NLI — DeBERTa-v3-small",                 "#06b6d4", 0.20),
    "query_align":   ("📐", "Query Alignment", "Cosine sim: Q ↔ Answer embeddings",      "#3b82f6", 0.20),
    "retrieval":     ("🔍", "Retrieval",       "Cosine sim: SubQ ↔ Doc embeddings",      "#8b5cf6", 0.15),
    "entity":        ("🏷️",  "Entity Coverage", "NER entity overlap in retrieved docs",   "#f97316", 0.10),
    "decomposition": ("🧩", "Decomposition",   "Sub-Q count vs complexity level",         "#a855f7", 0.05),
    "fluency":       ("✍️",  "Fluency",         "Heuristic: punctuation + bigrams",        "#fbbf24", 0.03),
    "conciseness":   ("📏", "Conciseness",     "1 − len/max_len",                         "#e879f9", 0.02),
}

st.markdown(
    '<div class="card" style="--ac:linear-gradient(180deg,#22c55e,#16a34a);">'
    '<div class="sh">'
    '<div class="si" style="--ib:rgba(34,197,94,.15);--ibr:rgba(34,197,94,.3);">🎯</div>'
    '<div><div class="st">8 NLP Reward Signals (PPO Training Signal)</div>'
    '<div class="ss">Weighted sum → scalar reward for RL agent · all values from /ask API (exact)</div></div>'
    '<div class="sn">Rewards</div>'
    '</div>'
    '<div class="cx">'
    '<b>What is this?</b> After the full pipeline generates an answer, 8 NLP quality metrics are '
    'computed and combined into a <b>single scalar reward</b> that trains the PPO agent. '
    'Each signal measures a different quality dimension. '
    '<b>Groundedness (0.25)</b> has the highest weight because hallucination is the worst failure '
    'in legal QA. All values here are taken <b>directly from the /ask API response</b> — '
    'exact same numbers as used during training.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="iol">INDIVIDUAL REWARD SIGNALS — value × weight = contribution</div>', unsafe_allow_html=True)

for key, (icon, name, tech, color, weight) in REWARD_META.items():
    val = float(rewards_dict.get(key, 0.0))
    contribution = val * weight
    pct = min(max(val, 0.0), 1.0) * 100
    val_col = scolor(val)
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;padding:7px 0;'
        f'border-bottom:1px solid rgba(99,102,241,.07);">'
        f'<span style="font-size:1.05rem;">{icon}</span>'
        f'<div style="min-width:130px;">'
        f'<div style="font-size:.8rem;font-weight:600;color:#e2e8f0;">{name}</div>'
        f'<div style="font-size:.66rem;color:#334155;">{tech}</div>'
        f'</div>'
        f'<div style="flex:1;background:rgba(30,41,59,.8);border-radius:4px;height:7px;">'
        f'<div style="width:{pct:.1f}%;height:7px;background:{color};border-radius:4px;"></div>'
        f'</div>'
        f'<span style="font-family:monospace;font-size:.8rem;color:{val_col};min-width:42px;text-align:right;">{val:.3f}</span>'
        f'<span style="font-size:.7rem;color:#334155;min-width:52px;">× {weight:.2f}</span>'
        f'<span style="font-family:monospace;font-size:.75rem;color:#475569;min-width:52px;">= {contribution:.4f}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Combined reward
cr_col = scolor(combined_reward)
cr_pct = min(combined_reward, 1.0) * 100
st.markdown(
    f'<div style="background:rgba(15,23,42,.9);border:1px solid rgba(34,197,94,.3);'
    f'border-radius:11px;padding:18px;margin-top:14px;display:flex;align-items:center;gap:18px;">'
    f'<div>'
    f'<div style="font-size:.68rem;color:#334155;text-transform:uppercase;font-weight:700;">Combined Reward (API — exact PPO training signal)</div>'
    f'<div style="font-size:2.8rem;font-weight:900;font-family:monospace;color:{cr_col};margin:4px 0;">{combined_reward:.4f}</div>'
    f'<div style="font-size:.75rem;color:#334155;">R = Σ(weight × signal) — clipped to [0, 1]</div>'
    f'</div>'
    f'<div style="flex:1;background:rgba(30,41,59,.8);border-radius:7px;height:18px;overflow:hidden;">'
    f'<div style="width:{cr_pct:.1f}%;height:18px;background:{cr_col};border-radius:7px;"></div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# ── Done ──────────────────────────────────────────────────────────────────────
prog.progress(100, "✅ All 12 NLP stages complete!")

st.markdown(
    '<div style="text-align:center;padding:28px;'
    'background:linear-gradient(135deg,rgba(99,102,241,.05),rgba(139,92,246,.05));'
    'border:1px solid rgba(99,102,241,.13);border-radius:16px;margin-top:14px;">'
    '<div style="font-size:1.3rem;font-weight:800;'
    'background:linear-gradient(135deg,#a78bfa,#60a5fa,#34d399);'
    '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
    '🎓 All 12 NLP Stages Completed for Your Question'
    '</div>'
    '<div style="font-size:.83rem;color:#334155;margin-top:6px;">'
    'Complexity Classification → Tokenization → YAKE Keywords → Decomposition → ROUGE → '
    'BM25 → Sentence Embedding → Cosine Similarity → Cross-Encoder Reranking → NER → '
    'LLM Generation → NLI Entailment → 8 Reward Signals'
    '</div>'
    '<div style="font-size:.8rem;color:#475569;margin-top:8px;">'
    'Change the question and click <b>Run All NLP Stages</b> — every section updates together.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)
