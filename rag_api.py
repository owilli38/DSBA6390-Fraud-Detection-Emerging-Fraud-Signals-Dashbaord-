from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client
from google import genai
from dotenv import load_dotenv
import os
import traceback

# ---------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL not set")

if not SUPABASE_KEY:
    raise ValueError("SUPABASE_KEY not set")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set")

# ---------------------------------------------------
# App + Clients
# ---------------------------------------------------

app = FastAPI(title="Fraud Article RAG Backend")

genai_client = genai.Client(api_key=GEMINI_API_KEY)

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

# ---------------------------------------------------
# Request Models
# ---------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class RAGRequest(BaseModel):
    query: str
    top_k: int = 5


# ---------------------------------------------------
# Health Endpoint
# ---------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "API is running"}


# ---------------------------------------------------
# Latest Articles Endpoint
# ---------------------------------------------------

@app.get("/latest")
def get_latest_articles(limit: int = 10):

    try:

        result = (
            supabase
            .table("articles_v1")
            .select("doc_id,title,url,publish_timestamp")
            .order("publish_timestamp", desc=True)
            .limit(limit)
            .execute()
        )

        return {
            "count": len(result.data),
            "articles": result.data
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------
# Search Endpoint (Vector Retrieval Only)
# ---------------------------------------------------

@app.post("/search")
def search_articles(request: SearchRequest):

    try:

        print("\n--- SEARCH REQUEST ---")
        print("Query:", request.query)

        embedding_response = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=[request.query]
        )

        query_embedding = embedding_response.embeddings[0].values

        print("Embedding dimension:", len(query_embedding))

        result = supabase.rpc(
            "match_articles",
            {
                "query_embedding": query_embedding,
                "match_count": request.top_k
            }
        ).execute()

        documents = result.data or []

        print("Retrieved documents:", len(documents))

        return {
            "query": request.query,
            "embedding_dim": len(query_embedding),
            "result_count": len(documents),
            "results": documents
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------
# RAG Endpoint (Full Pipeline)
# ---------------------------------------------------

@app.post("/rag")
def rag_endpoint(request: RAGRequest):

    try:

        print("\n========================")
        print("RAG REQUEST")
        print("Query:", request.query)
        print("========================")

        # ----------------------------------------
        # Generate query embedding
        # ----------------------------------------

        embedding_response = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=[request.query]
        )

        query_embedding = embedding_response.embeddings[0].values

        print("Embedding dimension:", len(query_embedding))

        # ----------------------------------------
        # Retrieve similar documents
        # ----------------------------------------

        retrieval = supabase.rpc(
            "match_articles",
            {
                "query_embedding": query_embedding,
                "match_count": request.top_k
            }
        ).execute()

        documents = retrieval.data or []

        print("Documents retrieved:", len(documents))

        if len(documents) == 0:
            return {
                "query": request.query,
                "answer": "No relevant fraud intelligence documents were found.",
                "retrieved_count": 0,
                "sources": []
            }

        # ----------------------------------------
        # Build LLM Context (robust)
        # ----------------------------------------

        context_chunks = []

        for doc in documents:

            text = (
                doc.get("raw_text")
                or doc.get("content")
                or doc.get("body")
                or doc.get("snippet")
                or ""
            )

            title = doc.get("title", "")

            chunk = f"""
TITLE: {title}

{text}
"""

            context_chunks.append(chunk)

        context_text = "\n\n---------------------\n\n".join(context_chunks)

        print("Context length:", len(context_text))

        # ----------------------------------------
        # Prompt for Gemini
        # ----------------------------------------

        prompt = f"""
You are a fraud intelligence analyst.

Use ONLY the provided context to answer the question.

If the answer is not supported by the context, say you do not have enough information.

Provide a clear analytical summary.

Context:
{context_text}

Question:
{request.query}

Answer:
"""

        # ----------------------------------------
        # Generate LLM Answer
        # ----------------------------------------

        llm_response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt]
        )

        answer = llm_response.text

        # ----------------------------------------
        # Return structured response
        # ----------------------------------------

        return {

            "query": request.query,

            "answer": answer,

            "retrieved_count": len(documents),

            "sources": [
                {
                    "doc_id": doc.get("doc_id"),
                    "title": doc.get("title"),
                    "url": doc.get("url"),
                    "snippet": (doc.get("snippet") or "")[:400]
                }
                for doc in documents
            ]
        }

    except Exception as e:

        print("\n!!!!!! RAG FAILURE !!!!!!")
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!\n")

        raise HTTPException(status_code=500, detail=str(e))