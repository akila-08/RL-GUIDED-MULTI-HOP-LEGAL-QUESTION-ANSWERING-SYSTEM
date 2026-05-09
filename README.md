# RL-Guided Multi-Hop Legal Question Answering System

## Overview

The **RL-Guided Multi-Hop Legal Question Answering System** is an AI-powered legal assistant designed to answer complex questions related to the **Constitution of India**. The system combines **Reinforcement Learning (PPO)**, **Large Language Models (LLMs)**, and **Hybrid Information Retrieval** to perform multi-hop legal reasoning.

Unlike traditional retrieval-based chatbots, this system intelligently decomposes complex legal queries into sub-questions, retrieves relevant constitutional articles, generates grounded sub-answers, and synthesizes them into a coherent final response.

---

# Key Features

* Multi-hop legal reasoning using Reinforcement Learning
* Intelligent question decomposition using Flan-T5 + rule-based fallback
* Hybrid retrieval pipeline:

  * BM25 Sparse Retrieval
  * Dense Vector Retrieval
  * Cross-Encoder Re-ranking
* ChromaDB-based vector storage
* PPO-based RL orchestration
* Legal answer generation using Llama 3 (Groq)
* Reward-based optimization for groundedness and factual consistency
* FastAPI backend and Node.js frontend integration
* Constitution of India article-level retrieval
* Interactive legal QA workflow visualization

---

# Tech Stack

## AI / Machine Learning

* PPO Reinforcement Learning
* PyTorch
* Hugging Face Transformers
* Flan-T5
* LegalBERT
* Sentence Transformers
* spaCy

## Retrieval & Vector Search

* ChromaDB
* BM25
* Cross-Encoder Re-ranking
* Dense Retrieval

## Backend & Frontend

* FastAPI
* Node.js
* Python

## LLM Integration

* Llama 3 (Groq API)
* Gemini API (Optional)
* Claude API (Optional)

## Database & Storage

* ChromaDB Persistent Vector Store
* SQLite

---

# System Architecture

```text
User Query
    │
    ▼
Frontend (Node.js)
    │
    ▼
FastAPI Backend
    │
    ▼
Complexity Classifier
    │
 ┌───────────────┴───────────────┐
 │                               │
Simple Query                 Complex Query
 │                               │
Retrieve + Generate        PPO RL Agent
                                │
                    DECOMPOSE → RETRIEVE
                    → GENERATE → COMBINE
                                │
                          Final Answer
```

---

# Workflow

## 1. Question Complexity Classification

The system first determines whether a query is simple or requires multi-hop reasoning.

## 2. Question Decomposition

Complex questions are decomposed into smaller sub-questions using:

* Fine-tuned Flan-T5
* Rule-based decomposition fallback

## 3. Hybrid Retrieval

The system retrieves relevant constitutional articles using:

* BM25 keyword retrieval
* Dense semantic retrieval
* Cross-Encoder re-ranking

## 4. Answer Generation

Sub-answers are generated using:

* Llama 3 (Groq)
* Context-grounded prompting

## 5. Answer Combination

Generated sub-answers are summarized into a final coherent legal response.

## 6. Reward Optimization

The PPO agent optimizes the workflow using reward signals such as:

* Groundedness
* Entailment
* Query Alignment
* Fluency
* Conciseness

---

# Reinforcement Learning Pipeline

The project uses a PPO-based Reinforcement Learning agent to orchestrate the legal QA pipeline.

## RL Actions

* DECOMPOSE
* RETRIEVE
* GENERATE
* COMBINE

## RL State Representation

The RL agent uses:

* Question embeddings
* Sub-question embeddings
* Retrieved document embeddings
* Generated answer embeddings
* Complexity score
* Step progress

## Reward Signals

* Groundedness Reward
* Entailment Reward
* Retrieval Reward
* Entity Consistency
* Query Alignment
* Fluency
* Conciseness

---

# Project Structure

```text
Project Root/
│
├── chatbot/
│   ├── app.py
│   └── streamlit_app.py
│
├── pipeline/
│   ├── decomposer.py
│   ├── retriever.py
│   ├── generator.py
│   └── combiner.py
│
├── rl/
│   ├── actions.py
│   ├── agent.py
│   ├── env.py
│   ├── rewards.py
│   └── state.py
│
├── ingestion/
│   ├── extractor.py
│   ├── chunker.py
│   ├── embedder.py
│   └── validator.py
│
├── db/
├── data/
├── scripts/
└── core/
```

---

# Dataset

The system uses the **Constitution of India PDF** as the primary legal knowledge source.

### Data Processing Steps

1. PDF text extraction
2. Article-level chunking
3. Keyword extraction using YAKE
4. Embedding generation using Sentence Transformers
5. Storage in ChromaDB

---

# Installation

## Clone Repository

```bash
git clone <repository-link>
cd RL-Guided-Multi-Hop-Legal-QA-System
```

## Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
LLM_MODEL=llama3-8b-8192
EMBEDDING_MODEL=all-MiniLM-L6-v2
DB_PATH=db/constitution_db
```

---

# Running the Project

## Step 1: Ingest Constitution Data

```bash
python ingest_pipeline.py
```

## Step 2: Start Backend Server

```bash
uvicorn chatbot.app:app --host 0.0.0.0 --port 8000
```

## Step 3: Start Frontend

```bash
npm install
npm start
```

---

# Example Query

### Input

```text
How do Articles 14 and 16(2) together ensure fairness in public employment?
```

### System Process

1. Decomposes the query
2. Retrieves relevant constitutional articles
3. Generates sub-answers
4. Synthesizes a final legal response

---

# Results

* Improved retrieval relevance using hybrid search
* Better factual grounding with RL-guided orchestration
* Enhanced multi-hop reasoning capability
* Reduced hallucination in legal answer generation
* Faster legal article retrieval using ChromaDB

---

