from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client
from dotenv import load_dotenv
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel
from datetime import datetime
import os
import traceback

# ---------------------------------------------------
# ENV
# ---------------------------------------------------

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID")

if not all([SUPABASE_URL, SUPABASE_KEY, VERTEX_PROJECT_ID]):
    raise ValueError("Missing environment variables")

# ---------------------------------------------------
# APP
# ---------------------------------------------------

app = FastAPI(title="Fraud Intelligence RAG Backend")

vertexai.init(project=VERTEX_PROJECT_ID, location="us-central1")

embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
gemini_model = GenerativeModel("gemini-2.5-flash-lite")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------------------------------
# MODELS
# ---------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class RAGRequest(BaseModel):
    query: str
    top_k: int = 5

# ---------------------------------------------------
# HEALTH
# ---------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------------------------------------
# LATEST
# ---------------------------------------------------

@app.get("/latest")
def latest(limit: int = 10):

    try:
        res = supabase.table("articles_v1") \
            .select("doc_id,title,url,publish_timestamp") \
            .order("publish_timestamp", desc=True) \
            .limit(limit) \
            .execute()

        return {"count": len(res.data or []), "articles": res.data or []}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))

# ---------------------------------------------------
# SEARCH
# ---------------------------------------------------

@app.post("/search")
def search(req: SearchRequest):

    try:
        emb = embedding_model.get_embeddings([
            TextEmbeddingInput(req.query, "RETRIEVAL_QUERY")
        ])

        qvec = emb[0].values

        res = supabase.rpc(
            "match_articles_bert",
            {"query_embedding": qvec, "match_count": req.top_k}
        ).execute()

        return {
            "query": req.query,
            "results": res.data or []
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))

# ---------------------------------------------------
# 🔥 DOMAIN GUARD (CRITICAL FIX)
# ---------------------------------------------------

def is_fraud_query(query: str) -> bool:

    prompt = f"""
Classify if the query is related to fraud intelligence.

Query: {query}

Valid domains:
- financial fraud
- cybercrime
- cybersecurity threats
- scams / phishing / identity theft

Answer ONLY YES or NO.
"""

    try:
        res = gemini_model.generate_content(prompt).text.strip().lower()
        return "yes" in res
    except:
        return False

# ---------------------------------------------------
# 🔥 CONTENT FILTER
# ---------------------------------------------------

def is_relevant(doc):

    text = (
        doc.get("raw_text")
        or doc.get("snippet")
        or doc.get("title")
        or ""
    ).lower()

    keywords = [
        "fraud", "scam", "phishing", "cyber",
        "hacking", "malware", "identity",
        "attack", "theft", "ransomware",
        "impersonation", "financial crime"
    ]

    return any(k in text for k in keywords)

# ---------------------------------------------------
# RAG (SAFE INTELLIGENCE PIPELINE)
# ---------------------------------------------------

@app.post("/rag")
def rag(req: RAGRequest):

    try:
        print("\n=== RAG REQUEST ===", req.query)

        # -------------------------
        # 1. DOMAIN CHECK (BLOCK NON-FRAUD)
        # -------------------------

        if not is_fraud_query(req.query):
            return {
                "query": req.query,
                "answer": "This system only supports fraud and cybercrime intelligence queries.",
                "sources": []
            }

        # -------------------------
        # 2. EMBEDDINGS
        # -------------------------

        emb = embedding_model.get_embeddings([
            TextEmbeddingInput(req.query, "RETRIEVAL_QUERY")
        ])

        qvec = emb[0].values

        # -------------------------
        # 3. VECTOR RETRIEVAL
        # -------------------------

        retrieval = supabase.rpc(
            "match_articles_bert",
            {"query_embedding": qvec, "match_count": req.top_k}
        ).execute()

        docs = retrieval.data or []

        # -------------------------
        # 4. RELEVANCE FILTER (HARD GATE)
        # -------------------------

        docs = [d for d in docs if is_relevant(d)]

        if len(docs) == 0:
            return {
                "query": req.query,
                "answer": "No relevant fraud intelligence found for this query.",
                "sources": []
            }

        # -------------------------
        # 5. ANALYSIS JOIN
        # -------------------------

        doc_ids = [d["doc_id"] for d in docs]

        analysis = supabase.table("article_analysis") \
            .select("doc_id, cluster_id, risk_score, stage") \
            .in_("doc_id", doc_ids) \
            .execute().data or []

        analysis_map = {a["doc_id"]: a for a in analysis}

        cluster_ids = list({
            a.get("cluster_id")
            for a in analysis
            if a.get("cluster_id") is not None
        })

        themes = supabase.table("cluster_themes") \
            .select("cluster_id, theme_label") \
            .in_("cluster_id", cluster_ids) \
            .execute().data or []

        theme_map = {t["cluster_id"]: t["theme_label"] for t in themes}

        # -------------------------
        # 6. ENRICH DOCS
        # -------------------------

        for d in docs:

            meta = analysis_map.get(d["doc_id"], {})
            cluster_id = meta.get("cluster_id")

            d["cluster_id"] = cluster_id
            d["theme_label"] = theme_map.get(cluster_id)
            d["risk_score"] = meta.get("risk_score")
            d["stage"] = meta.get("stage")

        # -------------------------
        # 7. CONTEXT BUILD
        # -------------------------

        context = "\n\n---\n\n".join([
            f"""
TITLE: {d.get('title','')}
THEME: {d.get('theme_label','')}
STAGE: {d.get('stage','')}
RISK: {d.get('risk_score','')}
DATE: {d.get('publish_timestamp','')}

{d.get('raw_text','')}
"""
            for d in docs
        ])

        # -------------------------
        # 8. GEMINI PROMPT (ANTI-HALLUCINATION)
        # -------------------------

        current_date = datetime.utcnow().strftime("%Y-%m-%d")

        prompt = f"""
You are a fraud intelligence analyst.

CURRENT DATE: {current_date}

RULES:
- Do NOT list articles
- Do NOT fabricate facts
- Only use provided context
- Merge similar threats into unified insights
- Focus ONLY on fraud and cybercrime

Write structured intelligence brief.

Context:
{context}

Question:
{req.query}
"""

        res = gemini_model.generate_content(prompt)

        # -------------------------
        # RESPONSE
        # -------------------------

        return {
            "query": req.query,
            "answer": res.text,
            "retrieved_count": len(docs),

            "sources": [
                {
                    "doc_id": d.get("doc_id"),
                    "title": d.get("title"),
                    "url": d.get("url"),
                    "publish_timestamp": d.get("publish_timestamp"),
                    "cluster_id": d.get("cluster_id"),
                    "theme_label": d.get("theme_label"),
                    "risk_score": d.get("risk_score"),
                    "stage": d.get("stage"),
                    "snippet": (d.get("raw_text") or "")[:250]
                }
                for d in docs
            ]
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))

# ---------------------------------------------------
# TIMELINE
# ---------------------------------------------------

@app.get("/timeline")
def timeline(limit: int = 50):

    try:
        articles = supabase.table("articles_v1") \
            .select("doc_id,title,url,publish_timestamp") \
            .order("publish_timestamp", desc=True) \
            .limit(limit) \
            .execute().data or []

        doc_ids = [a["doc_id"] for a in articles]

        analysis = supabase.table("article_analysis") \
            .select("doc_id, cluster_id, risk_score, stage") \
            .in_("doc_id", doc_ids) \
            .execute().data or []

        analysis_map = {a["doc_id"]: a for a in analysis}

        cluster_ids = list({
            a.get("cluster_id")
            for a in analysis
            if a.get("cluster_id") is not None
        })

        themes = supabase.table("cluster_themes") \
            .select("cluster_id, theme_label") \
            .in_("cluster_id", cluster_ids) \
            .execute().data or []

        theme_map = {t["cluster_id"]: t["theme_label"] for t in themes}

        timeline = []

        for a in articles:

            meta = analysis_map.get(a["doc_id"], {})
            cluster_id = meta.get("cluster_id")

            timeline.append({
                "doc_id": a["doc_id"],
                "title": a.get("title"),
                "url": a.get("url"),
                "publish_timestamp": a.get("publish_timestamp"),
                "cluster_id": cluster_id,
                "theme_label": theme_map.get(cluster_id),
                "risk_score": meta.get("risk_score"),
                "stage": meta.get("stage")
            })

        return {
            "count": len(timeline),
            "timeline": timeline
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))