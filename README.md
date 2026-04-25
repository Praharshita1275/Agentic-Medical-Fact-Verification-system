# MedVerify — Medical Claim Verifier

**Multi-Agent Medical Fact-Checking with Glassmorphism UI**

> G Akshatha · Gaali Sai Praharshita · Srichandana — Dept of IT, CBIT Hyderabad

---

## Features

| Feature | Description |
|---------|-------------|
| **E2** | Multi-Round Rebuttal Debate (Pro & Con agents) |
| **E3** | Live PubMed Search (NCBI Entrez) + FAISS RAG |
| **E4** | Uncertainty Quantification (5 factors) |
| **UI** | Glassmorphism + animated pipeline + dark theme |
| **LLM** | `nemotron-3-nano:30b` via Ollama Cloud API |

---

## Quick Start

### Option A — Standalone HTML (No backend needed)
Just open `index.html` in a browser. Uses Claude API in demo mode.

### Option B — Full Python Backend (Recommended)

```bash
pip install -r requirements.txt
python build_index.py   # First time only (downloads PubHealth + builds FAISS index)
python server.py
```

Then open **http://localhost:8000** in your browser.

---

## Setup

1. Get an Ollama Cloud API key → [ollama.com](https://ollama.com) → Sign In → API Keys
2. Paste it in the **API Key** field in the UI
3. Type or select a medical claim and click **Verify Claim**

---

## Pipeline Steps

```
1. 🔍 Retrieve Evidence   — FAISS local index + Live PubMed search
2. 🏷️  Classify Evidence  — Support / Oppose / Neutral per passage
3. ⚔️  Round 1 Debate     — Pro agent vs Con agent opening arguments
4. 🔄 Round 2 Rebuttal   — Each agent rebuts the other's argument
5. ⚖️  Judge Verdict      — Impartial judge reads all 4 arguments
6. 📊 Uncertainty        — 5-factor uncertainty quantification
```

---

## Project Structure

```
medical-claim-verifier/
├── index.html            ← Glassmorphism UI (standalone)
├── server.py            ← FastAPI backend with full pipeline
├── build_index.py      ← Build FAISS index from medical datasets
├── requirements.txt
├── data/              ← Medical datasets (train.tsv, dev.tsv, Datensatz.csv)
└── index/             ← Built FAISS index and metadata
```

---

## API

```
POST /api/verify
{
  "claim": "Vaccines cause autism.",
  "api_key": "your_ollama_key",
  "faiss_k": 5,
  "pubmed_k": 3
}
```

Response: Full verification result JSON with verdict, confidence, debate rounds, and uncertainty breakdown.

---

## Tech Stack

- **Frontend**: Vanilla HTML/CSS/JS with glassmorphism design
- **Backend**: FastAPI + Uvicorn
- **LLM**: Ollama Cloud (nemotron-3-nano:30b) via OpenAI-compatible API
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **RAG**: FAISS + PubMed NCBI Entrez
