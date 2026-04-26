import os, time
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from google.api_core.exceptions import TooManyRequests, ServiceUnavailable

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

SUPABASE_URL    = os.getenv("SUPABASE_URL")
SUPABASE_KEY    = os.getenv("SUPABASE_KEY")
VERTEX_PROJECT  = os.getenv("VERTEX_PROJECT_ID")
BATCH_SIZE      = 10
SLEEP_BETWEEN   = 3  # seconds between batches


def embed_with_retry(model, inputs, max_retries=5):
    delay = 10
    for attempt in range(max_retries):
        try:
            return model.get_embeddings(inputs)
        except (TooManyRequests, ServiceUnavailable):
            if attempt == max_retries - 1:
                raise
            print(f"    Rate limited. Waiting {delay}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(delay)
            delay *= 2  # exponential backoff


def main():
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    vertexai.init(project=VERTEX_PROJECT, location="us-central1")
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")

    print("Fetching articles...")
    articles = sb.table("articles_v1").select("doc_id, raw_text, title").execute().data or []
    print(f"  {len(articles)} articles found")

    done = {r["doc_id"] for r in sb.table("article_embeddings_v2").select("doc_id").execute().data or []}
    articles = [a for a in articles if a["doc_id"] not in done]
    print(f"  {len(articles)} remaining to embed")

    for i in range(0, len(articles), BATCH_SIZE):
        batch = articles[i : i + BATCH_SIZE]
        print(f"  Embedding batch {i // BATCH_SIZE + 1} ({len(batch)} articles)...")

        inputs = [
            TextEmbeddingInput(
                (a.get("raw_text") or a.get("title") or "")[:2000],
                "RETRIEVAL_DOCUMENT",
            )
            for a in batch
        ]
        embeddings = embed_with_retry(model, inputs)

        rows = [
            {"doc_id": a["doc_id"], "embedding": emb.values}
            for a, emb in zip(batch, embeddings)
        ]
        sb.table("article_embeddings_v2").upsert(rows).execute()
        print(f"    Saved {len(rows)} rows")
        time.sleep(SLEEP_BETWEEN)

    print("Done.")

if __name__ == "__main__":
    main()
