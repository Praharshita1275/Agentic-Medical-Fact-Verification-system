#!/usr/bin/env python3
"""
MedVerify — Medical Claim Verifier
FastAPI backend with full pipeline using local FAISS index and Ollama Cloud.

Usage:
    pip install -r requirements.txt
    python build_index.py   # First time only
    python server.py

Then open: http://localhost:8000
"""

import os
import re
import time
import json
import pickle
import threading
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

try:
    from openai import OpenAI, RateLimitError, APIStatusError
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from Bio import Entrez
    HAS_BIO = True
except ImportError:
    HAS_BIO = False

import faiss
import requests

# ═══════════════════════════════════════════════════════
# CONFIG - Nemotron 3 Nano (Cloud) - Best accuracy + speed balance
# ═══════════════════════════════════════════════════════

OLLAMA_API_KEY = "your_api_key"
OLLAMA_MODEL = "nemotron-3-nano:30b-cloud"  # Best balance - accurate + fast
OLLAMA_BASE = "https://ollama.com/v1"

MIN_CALL_GAP = 1.2
MAX_RETRIES = 4
BASE_BACKOFF = 2.0
EMBEDDING_DIM = 384

Entrez.email = "medverify@cbit.ac.in"

_CALL_LOCK = threading.Lock()
_last_call_ts = 0.0

SYS_PRO = "You are a rigorous medical debate agent. Your role is to argue IN FAVOUR of the claim using only the provided evidence. Be concise and end with a clear conclusion."
SYS_CON = "You are a rigorous medical debate agent. Your role is to argue AGAINST the claim using only the provided evidence. Be concise and end with a clear conclusion."
SYS_JDG = "You are an impartial medical fact-checking judge. Read both sides carefully and deliver a balanced, evidence-based verdict."

# Medical corpus for local FAISS index
MEDICAL_CORPUS = [
    {"claim": "Vaccines cause autism.", "label": "false", "text": "Multiple large studies have found no causal link between vaccines and autism. The original 1998 study was retracted."},
    {"claim": "Exercise reduces heart disease risk.", "label": "true", "text": "Regular physical activity lowers blood pressure, improves cholesterol, and reduces cardiovascular mortality."},
    {"claim": "Vitamin C cures the common cold.", "label": "false", "text": "There is currently no cure for the common cold. Vitamin C may reduce duration slightly but does not cure it."},
    {"claim": "COVID-19 vaccines are safe.", "label": "true", "text": "Clinical trials and post-market surveillance confirm safety and efficacy of approved COVID-19 vaccines."},
    {"claim": "Smoking causes lung cancer.", "label": "true", "text": "Tobacco smoke contains carcinogens that directly cause lung cancer in smokers."},
    {"claim": "Sugar causes hyperactivity.", "label": "false", "text": "Controlled studies show no causal link between sugar consumption and hyperactivity in children."},
    {"claim": "Coffee prevents Alzheimer's disease.", "label": "uncertain", "text": "Some studies show mild protective associations but evidence is not conclusive."},
    {"claim": "Antibiotics treat viral infections.", "label": "false", "text": "Antibiotics only work against bacterial infections; viruses are unaffected."},
    {"claim": "Meditation reduces stress.", "label": "true", "text": "Mindfulness meditation lowers cortisol levels and reduces self-reported stress."},
    {"claim": "Blood type determines personality.", "label": "false", "text": "There is no scientific evidence supporting a link between blood type and personality."},
    {"claim": "Drinking water prevents kidney stones.", "label": "true", "text": "Adequate hydration reduces the risk of kidney stone formation."},
    {"claim": "Gluten-free diet is healthier.", "label": "false", "text": "Only people with celiac disease need gluten-free diet. No benefit for others."},
    {"claim": "Organic food is more nutritious.", "label": "false", "text": "No significant nutritional difference between organic and conventional foods."},
    {"claim": "Vaccines contain harmful chemicals.", "label": "false", "text": "Vaccine ingredients are rigorously tested for safety."},
    {"claim": "Exercise improves mental health.", "label": "true", "text": "Regular exercise reduces symptoms of depression and anxiety."},
]

# ═══════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_or_create_faiss_index()
    yield

app = FastAPI(title="MedVerify API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VerifyRequest(BaseModel):
    claim: str
    api_key: Optional[str] = None
    faiss_k: int = 5
    pubmed_k: int = 3

# ═══════════════════════════════════════════════════════
# FAISS INDEX SETUP
# ═══════════════════════════════════════════════════════

faiss_index = None
corpus_records = []
st_model = None

def get_embedding_model():
    """Lazy load sentence transformer."""
    global st_model
    if st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
    return st_model

def load_or_create_faiss_index():
    """Load existing FAISS index or create from medical corpus."""
    global faiss_index, corpus_records
    
    # Try to load existing index
    index_path = "index/pubhealth.bin"
    meta_path = "index/pubhealth_meta.pkl"
    
    if os.path.exists(index_path) and os.path.exists(meta_path):
        try:
            faiss_index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                corpus_records = pickle.load(f)
            print(f"✅ Loaded FAISS index with {faiss_index.ntotal:,} vectors")
            return True
        except Exception as e:
            print(f"⚠️  Failed to load index: {e}")
    
    # Create index from medical corpus
    print("🔨 Creating FAISS index from medical corpus...")
    model = get_embedding_model()
    
    if model:
        texts = [r["text"] for r in MEDICAL_CORPUS]
        corpus_records = MEDICAL_CORPUS.copy()
        
        try:
            embeddings = model.encode(texts, convert_to_numpy=True).astype('float32')
            faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
            faiss_index.add(embeddings)
            
            # Save index
            os.makedirs("index", exist_ok=True)
            faiss.write_index(faiss_index, index_path)
            with open(meta_path, "wb") as f:
                pickle.dump(corpus_records, f)
            
            print(f"✅ Created FAISS index with {faiss_index.ntotal} vectors")
            return True
        except Exception as e:
            print(f"⚠️  Failed to create index: {e}")
    
    # Fallback: use corpus directly
    print("⚠️  Using in-memory corpus (no vector search)")
    corpus_records = MEDICAL_CORPUS.copy()
    faiss_index = None
    return False

def retrieve_local(claim: str, top_k: int = 5) -> List[Dict]:
    """Retrieve from local corpus using simple keyword matching."""
    if not corpus_records:
        return []
    
    claim_words = set(claim.lower().split())
    scored = []
    
    for r in corpus_records:
        text = r.get("text", r.get("claim", ""))
        text_words = set(text.lower().split())
        overlap = len(claim_words & text_words)
        if overlap > 0:
            scored.append((overlap, r))
    
    scored.sort(key=lambda x: -x[0])
    return [{**r, "source": "local"} for _, r in scored[:top_k]]

# ═══════════════════════════════════════════════════════
# OLLAMA CLIENT
# ═══════════════════════════════════════════════════════

def make_client(api_key: str = None):
    """Create Ollama client."""
    key = api_key or OLLAMA_API_KEY
    if not HAS_OPENAI:
        return None
    return OpenAI(base_url=OLLAMA_BASE, api_key=key)

def chat(prompt: str, api_key: str = None, max_tokens: int = 512, temperature: float = 0.3, system: str = None) -> str:
    """Send prompt to Ollama Cloud."""
    global _last_call_ts
    
    key = api_key or OLLAMA_API_KEY
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    client = make_client(key)
    
    for attempt in range(MAX_RETRIES):
        with _CALL_LOCK:
            wait = MIN_CALL_GAP - (time.time() - _last_call_ts)
            if wait > 0:
                time.sleep(wait)
            _last_call_ts = time.time()

        try:
            if client:
                resp = client.chat.completions.create(
                    model=OLLAMA_MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content.strip()
            else:
                # Fallback to requests
                r = requests.post(
                    f"{OLLAMA_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": OLLAMA_MODEL, "messages": messages,
                          "max_tokens": max_tokens, "temperature": temperature},
                    timeout=60
                )
                data = r.json()
                return data["choices"][0]["message"]["content"].strip()
                
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                backoff = BASE_BACKOFF * (2 ** attempt)
                print(f"⚠️  Rate limit — waiting {backoff:.0f}s")
                time.sleep(backoff)
            else:
                print(f"API error: {e}")
                return ""
    return ""

# ═══════════════════════════════════════════════════════
# E3: EVIDENCE RETRIEVAL
# ═══════════════════════════════════════════════════════

def search_pubmed(query: str, max_results: int = 3) -> List[Dict]:
    """Fetch live abstracts from PubMed."""
    if not HAS_BIO:
        return []
    
    results = []
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        ids = Entrez.read(handle).get("IdList", [])
        handle.close()
        
        if not ids:
            return results
        
        handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        
        for art in records.get("PubmedArticle", []):
            try:
                ml = art["MedlineCitation"]
                data = ml["Article"]
                title = str(data.get("ArticleTitle", ""))
                abs_texts = data.get("Abstract", {}).get("AbstractText", [])
                abstract = " ".join(str(a) for a in abs_texts)
                
                if len(abstract) > 50:
                    results.append({
                        "id": str(ml.get("PMID", "")),
                        "title": title,
                        "text": f"{title}. {abstract[:500]}",
                        "source": "PubMed",
                    })
            except Exception:
                continue
        time.sleep(0.35)
    except Exception as e:
        print(f"PubMed warning: {e}")
    
    return results

def retrieve_all(claim: str, faiss_k: int = 5, pubmed_k: int = 3) -> List[Dict]:
    """Retrieve evidence from local + PubMed."""
    local = retrieve_local(claim, faiss_k)
    pubmed = search_pubmed(f"{claim} clinical evidence", pubmed_k)
    
    seen, out = set(), []
    for p in local + pubmed:
        key = p.get("text", "")[:80]
        if key not in seen:
            seen.add(key)
            out.append(p)
    
    return out

# ═══════════════════════════════════════════════════════
# E2: EVIDENCE CLASSIFICATION
# ═══════════════════════════════════════════════════════

def classify_evidence(claim: str, passages: List[Dict], api_key: str = None) -> tuple:
    """Classify each passage as SUPPORT / OPPOSE / NEUTRAL."""
    if not passages:
        return [], []
    
    numbered = "\n".join(
        f'{i+1}. "{p.get("text", "")[:250]}"' 
        for i, p in enumerate(passages)
    )
    
    prompt = f"""You are a medical evidence classifier.
Claim: "{claim}"
Passages:
{numbered}
Respond ONLY with a numbered list:
<number>: SUPPORT  or  <number>: OPPOSE  or  <number>: NEUTRAL
No explanations."""
    
    raw = chat(prompt, api_key, max_tokens=len(passages)*15+30, temperature=0.1)
    
    pos_ev, neg_ev = [], []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            clean = line.replace(".", ":")
            num_s, label = clean.split(":", 1)
            idx = int(num_s.strip()) - 1
            label = label.strip().upper()
            
            if 0 <= idx < len(passages):
                p = passages[idx]
                if "SUPPORT" in label:
                    pos_ev.append({**p, "ev_label": "SUPPORT"})
                elif "OPPOSE" in label:
                    neg_ev.append({**p, "ev_label": "OPPOSE"})
        except Exception:
            continue
    
    return pos_ev, neg_ev

# ═══════════════════════════════════════════════════════
# E2: DEBATE AGENTS
# ═══════════════════════════════════════════════════════

def _ev_block(ev_list, n: int = 4) -> str:
    return "\n\n".join(
        f"[Evidence {i+1} | {e.get('source','')}]\n{e.get('text', '')}"
        for i, e in enumerate(ev_list[:n])
    )

def pro_agent_round1(claim: str, pos_ev: List[Dict], api_key: str = None) -> Dict:
    if not pos_ev:
        return {"argument": "No supporting evidence found.", "score": 0.0}
    prompt = f'Claim: "{claim}"\n\nEvidence:\n{_ev_block(pos_ev)}\n\nBuild the strongest supporting argument. Max 200 words.'
    arg = chat(prompt, api_key, max_tokens=320, system=SYS_PRO)
    return {"argument": arg or "No response.", "score": len(pos_ev)*0.35 + min(len(arg or "")/400, 1.0)}

def con_agent_round1(claim: str, neg_ev: List[Dict], api_key: str = None) -> Dict:
    if not neg_ev:
        return {"argument": "No opposing evidence found.", "score": 0.0}
    prompt = f'Claim: "{claim}"\n\nEvidence:\n{_ev_block(neg_ev)}\n\nBuild the strongest opposing argument. Max 200 words.'
    arg = chat(prompt, api_key, max_tokens=320, system=SYS_CON)
    return {"argument": arg or "No response.", "score": len(neg_ev)*0.35 + min(len(arg or "")/400, 1.0)}

def pro_rebuttal(claim: str, pro_arg: Dict, con_arg: Dict, api_key: str = None) -> Dict:
    prompt = f'Claim: "{claim}"\n\nYour argument:\n{pro_arg["argument"][:400]}\n\nOpponent:\n{con_arg["argument"][:400]}\n\nWrite a focused rebuttal. Max 150 words.'
    arg = chat(prompt, api_key, max_tokens=220, system=SYS_PRO)
    return {"argument": arg or "No response.", "score": pro_arg["score"]*0.8 + 0.2}

def con_rebuttal(claim: str, pro_arg: Dict, con_arg: Dict, api_key: str = None) -> Dict:
    prompt = f'Claim: "{claim}"\n\nYour argument:\n{con_arg["argument"][:400]}\n\nOpponent:\n{pro_arg["argument"][:400]}\n\nWrite a focused rebuttal. Max 150 words.'
    arg = chat(prompt, api_key, max_tokens=220, system=SYS_CON)
    return {"argument": arg or "No response.", "score": con_arg["score"]*0.8 + 0.2}

# ═══════════════════════════════════════════════════════════════
# JUDGE AGENT
# ═══════════════════════════════════════════════════════

def judge_agent(claim: str, pro_r1: Dict, con_r1: Dict, pro_r2: Dict, con_r2: Dict, api_key: str = None) -> Dict:
    prompt = f"""Claim: "{claim}"

=== PRO ===
{pro_r1["argument"][:300]}

=== CON ===
{con_r1["argument"][:300]}

=== PRO REBUTTAL ===
{pro_r2["argument"][:300]}

=== CON REBUTTAL ===
{con_r2["argument"][:300]}

Respond in EXACTLY this format:
VERDICT: <TRUE or FALSE or UNCERTAIN>
CONFIDENCE: <integer 0-100>
REASONING: <3-4 sentences>"""
    
    raw = chat(prompt, api_key, max_tokens=400, temperature=0.2, system=SYS_JDG)
    
    verdict, confidence, reasoning = "Uncertain", 50.0, ""
    
    for line in (raw or "").splitlines():
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            v = line.split(":", 1)[1].strip().upper()
            verdict = "True" if "TRUE" in v else ("False" if "FALSE" in v else "Uncertain")
        elif line.upper().startswith("CONFIDENCE:"):
            try:
                confidence = float(re.findall(r"[0-9.]+", line)[0])
            except:
                pass
        elif line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
    
    ps = (pro_r1["score"] + pro_r2["score"]) / 2
    cs = (con_r1["score"] + con_r2["score"]) / 2
    
    if not reasoning:
        verdict = "True" if ps > cs else ("False" if cs > ps else "Uncertain")
        confidence = round(max(ps, cs) / (ps + cs + 1e-6) * 100, 1)
        reasoning = "Score-based verdict."
    
    return {
        "claim": claim,
        "verdict": verdict,
        "confidence": confidence,
        "reason": reasoning,
        "pro_score": round(ps, 3),
        "con_score": round(cs, 3),
        "pro_r1": pro_r1["argument"],
        "con_r1": con_r1["argument"],
        "pro_rebuttal": pro_r2["argument"],
        "con_rebuttal": con_r2["argument"],
    }

# ═══════════════════════════════════════════════════════
# E4: UNCERTAINTY QUANTIFICATION
# ═══════════════════════════════════════════════════��═��═

def quantify_uncertainty(result: Dict, pos_ev: List[Dict], neg_ev: List[Dict], passages: List[Dict], api_key: str = None) -> Dict:
    report = {}
    total = 0.0
    np_, nc_ = len(pos_ev), len(neg_ev)
    n_total = len(passages)

    # 1. Evidence conflict
    if np_ > 0 and nc_ > 0:
        conflict = min(np_, nc_) / max(np_, nc_)
        d = round(conflict * 0.4, 3)
    else:
        d = 0.0
    report["evidence_conflict"] = {"score": d, "detail": f"{np_} supporting vs {nc_} opposing"}
    total += d

    # 2. Evidence scarcity
    d = round(max(0, 1 - n_total / 8) * 0.2, 3)
    report["evidence_scarcity"] = {"score": d, "detail": f"{n_total} passages"}
    total += d

    # 3. Score proximity
    ps, cs = result.get("pro_score", 0), result.get("con_score", 0)
    diff_ratio = 1 - abs(ps - cs) / (max(ps, cs) + 1e-6)
    d = round(diff_ratio * 0.2, 3)
    report["score_proximity"] = {"score": d, "detail": f"Pro={ps:.2f} Con={cs:.2f}"}
    total += d

    # 4. Source diversity
    srcs = set(p.get("source", "unknown") for p in passages)
    d = 0.0 if len(srcs) > 1 else 0.1
    report["source_diversity"] = {"score": d, "detail": f"Sources: {list(srcs)}"}
    total += d

    # 5. Claim ambiguity
    amb = 0.0
    resp = chat(
        f'Is this claim ambiguous?\nClaim: "{result.get("claim", "")}"\nReply YES or NO.',
        api_key, max_tokens=5, temperature=0.0
    ).upper()
    if "YES" in resp:
        amb = 0.15
    report["claim_ambiguity"] = {"score": amb, "detail": "Ambiguous" if amb else "Clear"}
    total += amb

    total = min(total, 1.0)
    pct = round(total * 100, 1)
    level = "LOW" if pct < 20 else ("MEDIUM" if pct < 50 else "HIGH")
    
    return {
        "uncertainty_score": pct,
        "uncertainty_level": level,
        "uncertainty_sources": report,
        "interpretation": f"Uncertainty {level} ({pct}%). {'Verdict is reliable.' if level == 'LOW' else 'Treat with caution.' if level == 'MEDIUM' else 'Verdict may be unreliable.'}"
    }

# ═══════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════

def verify_claim_full(claim: str, faiss_k: int = 5, pubmed_k: int = 3):
    """Run full verification pipeline."""
    print(f"\n[MedVerify] Verifying: {claim}")
    t0 = time.time()

    # E3: Retrieve evidence
    print("  [1/6] Retrieving evidence...")
    passages = retrieve_all(claim, faiss_k=faiss_k, pubmed_k=pubmed_k)
    pm = sum(1 for p in passages if p.get("source") == "PubMed")
    print(f"       {len(passages)} passages ({pm} from PubMed)")

    # E3: Classify evidence
    print("  [2/6] Classifying evidence...")
    pos_ev, neg_ev = classify_evidence(claim, passages)
    print(f"       {len(pos_ev)} supporting | {len(neg_ev)} opposing")

    # E2: Round 1 debate
    print("  [3/6] Round 1 debate...")
    pro_r1 = pro_agent_round1(claim, pos_ev)
    con_r1 = con_agent_round1(claim, neg_ev)

    # E2: Round 2 rebuttals
    print("  [4/6] Round 2 rebuttals...")
    pro_r2 = pro_rebuttal(claim, pro_r1, con_r1)
    con_r2 = con_rebuttal(claim, pro_r1, con_r1)

    # Judge
    print("  [5/6] Judge evaluating...")
    result = judge_agent(claim, pro_r1, con_r1, pro_r2, con_r2)
    result["retrieved_count"] = len(passages)
    result["pro_evidence_count"] = len(pos_ev)
    result["con_evidence_count"] = len(neg_ev)

    # E4: Uncertainty
    print("  [6/6] Quantifying uncertainty...")
    result.update(quantify_uncertainty(result, pos_ev, neg_ev, passages))

    elapsed = round(time.time() - t0, 1)
    print(f"  ✅ Done in {elapsed}s — Verdict: {result['verdict']} (confidence: {result['confidence']}%)")
    return result

# ═══════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════

@app.post("/api/verify")
async def api_verify(req: VerifyRequest):
    if not req.claim.strip():
        raise HTTPException(status_code=400, detail="Claim cannot be empty")
    
    try:
        result = verify_claim_full(req.claim, req.faiss_k, req.pubmed_k)
        return JSONResponse(content=result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health():
    return {
        "status": "ok", 
        "model": OLLAMA_MODEL,
        "index_loaded": faiss_index is not None,
        "corpus_size": len(corpus_records)
    }

# Serve the HTML UI
html_path = Path(__file__).parent / "index.html"

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

if __name__ == "__main__":
    import uvicorn
    print("╔══════════════════════════════════════════╗")
    print("║   MedVerify — Medical Claim Verifier     ║")
    print("║   Llama 3.3 + Local FAISS                ║")
    print("╚══════════════════════════════════════════╝")
    print(f"\n  Model  : {OLLAMA_MODEL}")
    print(f"  API Key: {OLLAMA_API_KEY[:20]}...")
    print(f"\n  ➤ Open http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
