# ⚖️ RL-Guided Multi-Hop Legal QA System — Setup & Run Guide

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.10+ | https://python.org |
| Node.js | 18+ | https://nodejs.org |
| Git | Any | https://git-scm.com |

---

## 1. Clone the Repository

```bash
git clone https://github.com/akila-08/RL-GUIDED-MULTI-HOP-LEGAL-QUESTION-ANSWERING-SYSTEM.git
cd RL-GUIDED-MULTI-HOP-LEGAL-QUESTION-ANSWERING-SYSTEM
git checkout sujan_frontend
```

---

## 2. Python Backend Setup

### Create & activate virtual environment
```bash
# Windows
python -m venv venv1
venv1\Scripts\activate

# macOS / Linux
python3 -m venv venv1
source venv1/bin/activate
```

### Install Python dependencies
```bash
pip install -r requirements.txt
```

### Set up environment variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=llama-3.1-8b-instant
DB_PATH=./chroma_db
```
Get a free Groq API key at: https://console.groq.com

### Ingest the Constitution into the database (first time only)
```bash
python ingest_pipeline.py
```
This builds the ChromaDB vector store (~443 article chunks).

---

## 3. React Frontend Setup

```bash
cd frontend
npm install
```
This installs all Node.js dependencies (React, Vite, etc.).

---

## 4. Run the Application

You need **2 terminals open simultaneously**.

### Terminal 1 — FastAPI Backend
```bash
# From project root
venv1\Scripts\activate           # (Windows)
# source venv1/bin/activate      # (macOS/Linux)

uvicorn chatbot.app:app --host 0.0.0.0 --port 8000
```
Wait for: `✅  Chatbot ready.`

### Terminal 2 — React Frontend
```bash
cd frontend
npm run dev
```
Wait for: `VITE ready → Local: http://localhost:5173/`

### Open in browser
```
http://localhost:5173
```

---

## 5. Project Structure

```
RL-GUIDED-MULTI-HOP-LEGAL-QUESTION-ANSWERING-SYSTEM/
│
├── chatbot/
│   ├── app.py                  # FastAPI backend (main API)
│   └── streamlit_app.py        # (Legacy) Streamlit UI
│
├── frontend/                   # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx             # Root layout + state
│   │   ├── components/
│   │   │   ├── MessageBubble.jsx
│   │   │   ├── ChatInput.jsx
│   │   │   ├── AgentBrainPanel.jsx
│   │   │   ├── ArticleCard.jsx
│   │   │   └── ThinkingIndicator.jsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js          # Proxies /ask → FastAPI
│
├── pipeline/
│   ├── retriever.py            # Hybrid BM25 + Dense retrieval
│   ├── generator.py            # Groq LLM generation
│   ├── decomposer.py           # Question decomposition
│   ├── combiner.py             # Answer combining
│   └── baseline_rules.py      # Rule-based decomposition fallback
│
├── rl/
│   ├── agent.py                # PPO RL Agent
│   ├── env.py                  # Legal QA Environment
│   ├── actions.py              # Macro actions
│   └── rewards.py              # Reward computation
│
├── ingestion/
│   ├── chunker.py              # Article chunking logic
│   └── embedder.py             # Embedding + ChromaDB storage
│
├── core/
│   └── config.py               # Central configuration
│
├── requirements.txt            # Python dependencies
├── ingest_pipeline.py          # Run once to build vector DB
└── RUNNING.md                  # This file
```

---

## 6. Sample Questions to Try

### Simple (Single-Hop)
- `What is Article 1?`
- `Who appoints the Chief Justice of India?`
- `What does Article 21 say?`

### Complex (Multi-Hop HRL)
- `What is the difference between Article 19 and Article 35?`
- `How do Articles 14 and 16(2) together ensure fairness in public employment?`
- `If a person acquires foreign citizenship, what happens to their Indian citizenship?`
- `How does the duration of the Council of States differ from the House of the People?`

---

## 7. Troubleshooting

| Problem | Fix |
|---------|-----|
| `Cannot connect to backend` | Make sure Terminal 1 (uvicorn) is running |
| `npm run dev` not found | Make sure you are inside the `frontend/` folder |
| `GROQ_API_KEY` error | Add your key to the `.env` file |
| Empty answers | Re-run `python ingest_pipeline.py` to rebuild the DB |
| `node` not recognized | Install Node.js from https://nodejs.org |
