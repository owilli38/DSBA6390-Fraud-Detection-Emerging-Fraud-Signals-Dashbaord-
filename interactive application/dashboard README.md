# Pickaxe Analytics — Fraud Intelligence Dashboard

A Streamlit-based fraud intelligence platform that combines semantic search, embedding-based cluster analysis, and generative AI synthesis to deliver a complete fraud intelligence operating model. Built on Supabase for data persistence and Google Vertex AI for embeddings and language model generation.

---

## Overview

Pickaxe Analytics is designed for fraud operations teams and financial crimes analysts. Rather than surfacing raw search results, the application builds a layered intelligence workflow: analysts run a search, receive a synthesised brief, then navigate through connected pages that automatically update to reflect the context of that search — from cluster pattern analysis through relationship mapping to active alerts.

---

## Application Architecture

```
Intelligence Search (RAG)
        │
        ▼  st.session_state["rag_result"]
        │
        ├──► Fraud Pattern Clusters   (retrieved clusters, KPIs, ranked list)
        ├──► Network Relationships    (constellation map centered on dominant cluster)
        └──► Alerts & Watchlists      (alerts filtered to search-relevant clusters)
```

The Executive Overview operates independently as a passive monitoring view and does not consume search context.

---

## Pages

### 1. Executive Overview
Passive monitoring mode for leadership. Displays:
- KPI row: articles monitored, active clusters, emerging themes, avg cluster risk
- Gemini-generated executive intelligence brief
- Fraud signal trend chart (weekly article ingestion)
- Cluster stage distribution (Altair bar chart)
- Evidence snapshot: top 3 signals from the highest-risk cluster
- Themes Requiring Attention: expandable cards with audience classifier

### 2. Intelligence Search
Analyst workbench and primary entry point for the connected workflow.
- **Suggested searches** generated from live cluster data via Vertex AI, validated against the article corpus before display
- **RAG pipeline**: query → Vertex AI embedding → `match_articles_v2` vector search → Gemini synthesis
- Structured brief with four sections: Threat Summary, Key Patterns, Risk Assessment, Analyst Note
- Supporting articles with cleaned excerpts, risk-scored tags, and Gemini-generated descriptions for empty articles
- Minimum risk threshold slider and document count slider
- Results stored in `st.session_state["rag_result"]` for downstream pages

### 3. Fraud Pattern Clusters
Pattern analysis layer. When a search has been run:
- Shows clusters retrieved by the search (fetched by cluster ID from `cluster_themes`)
- KPIs and ranked list reflect the search result set
- Context banner identifies the active query

When no search is active, shows the global top-10 clusters by risk score.

### 4. Network Relationships
Relationship exploration layer. Features:
- Constellation chart (Plotly) centered on the dominant cluster from the last search
- Connected clusters derived from appearance frequency in the `article_neighbors` table (not raw similarity score — those are uniformly high in this corpus)
- Clusters already in the search result set are excluded
- Two-handle connection strength slider for filtering the constellation
- Click any node to see cluster description in a detail card below the chart
- Strongest Connection card with Gemini-generated fraud profile and convergence explanation
- Fallback to highest-risk cluster when no search is active

### 5. Alerts & Watchlists
Operational monitoring layer. Features:
- Clickable KPI cards (Total / High / Medium / Low severity) that filter the alert feed
- KPI counts reflect the active filter (search context or global)
- Search context toggle: when a search is active, filters feed to clusters matching that search by both cluster ID and theme label
- Toggle resets to "Search clusters" automatically when a new search is run
- Enriched alert cards: fraud description + Gemini-generated 1–3 action recommendations
- First occurrence of each cluster shows full card; repeat alerts render as compact timeline rows

---

## Data Model (Supabase)

| Table | Purpose |
|---|---|
| `cluster_themes` | Fraud theme clusters — labels, descriptions, risk scores, stage, article counts |
| `article_analysis` | Per-article cluster membership, risk score, stage, growth/drift/acceleration metrics |
| `article_neighbors` | Pre-computed nearest-neighbor graph — top-N neighbors per article with rank and similarity score |
| `articles_v1` / `raw_articles` | Article store — title, URL, raw text, publish timestamp, source |
| `article_embeddings_v2` | BERT embedding vectors per article (used by `match_articles_v2` RPC) |
| `alerts` | Alert events — severity, cluster snapshot, trigger timestamp, read status |
| `alert_rules` | Alert rule definitions with thresholds and rule types |
| `watchlists` | Saved cluster watchlist entries |

### Key Supabase RPC
**`match_articles_v2`** — vector similarity search. Accepts `query_embedding` (float array) and `match_count` (integer). Returns articles ranked by cosine similarity to the query vector.

---

## Generative AI Layer

All AI calls use Google Vertex AI initialized with `VERTEX_PROJECT_ID`.

| Function | Model | Purpose |
|---|---|---|
| `get_models()` | `text-embedding-005` + `gemini-2.5-flash-lite` | Cached model initialization |
| `run_rag()` | Both | Full RAG pipeline — embed, retrieve, synthesise |
| `fetch_executive_brief()` | Gemini | Executive intelligence brief for Overview page |
| `generate_cluster_similarity_brief()` | Gemini | Fraud profile + convergence explanation for Network page |
| `generate_alert_recommendations()` | Gemini | Batch 1–3 action recommendations per alert cluster |
| `generate_suggested_searches()` | Both | Generate + validate suggested search queries |

### RAG Brief Prompt Structure
The intelligence brief uses a constrained 4-section prompt:
1. **Threat Summary** — 2–3 sentences on the core fraud dynamic
2. **Key Patterns** — exactly 3 bullet points of specific fraud mechanics
3. **Risk Assessment** — 1–2 sentences on exposure
4. **Analyst Note** — single most important development to monitor

A display-side safety net truncates at the Analyst Note section boundary or at 1,500 characters if Gemini exceeds the word limit.

---

## Setup

### Prerequisites
- Python 3.10+
- A Supabase project with the tables and RPC described above
- A Google Cloud project with Vertex AI API enabled
- `match_articles_v2` PostgreSQL function deployed to Supabase

### Installation

```bash
git clone <repo-url>
cd <repo-directory>
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root (one level above the dashboard folder):

```env
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_KEY=<your-supabase-anon-key>
VERTEX_PROJECT_ID=<your-gcp-project-id>
VERTEX_API_KEY=<your-vertex-api-key>
GEMINI_API_KEY=<your-gemini-api-key>
```

### Running the App

```bash
streamlit run dashboard/vizdemov4.0.5.py
```

---

## Dependencies

```
streamlit
supabase
google-cloud-aiplatform
vertexai
pandas
altair
plotly
python-dotenv
```

Install all with:
```bash
pip install streamlit supabase google-cloud-aiplatform vertexai pandas altair plotly python-dotenv
```

---

## Key Design Decisions

**Why frequency-based connection strength (not similarity score) for the constellation?**
The `article_neighbors` table stores cosine similarity scores that are uniformly ~0.99 across all stored neighbors in this corpus, regardless of rank. This is a property of the embedding model and domain — articles about financial fraud are semantically very close to each other. Raw similarity provides no variance for positioning. Frequency of cluster appearance across all neighbor slots is used instead, which genuinely varies and reflects meaningful semantic adjacency.

**Why batch Gemini calls for recommendations?**
The Alerts page generates recommendations for multiple clusters in a single Gemini call, parsing the response by `CLUSTER_N:` labels. This reduces API latency from O(n) calls to O(1) per page load, with results cached for 10 minutes.

**Why is `excluded_cluster_ids` a tuple in `fetch_constellation_data`?**
`st.cache_data` requires all arguments to be hashable. Sets and lists are not hashable in Python. Tuples are, so source cluster IDs are converted to a tuple before being passed as a cache key.

**Why does the suggestion validator use only `query_embedding` and `match_count`?**
The `match_articles_v2` RPC only accepts those two parameters. Passing additional fields (`match_threshold`, `filter_date`) causes the RPC call to fail silently, returning an empty result set regardless of actual content in the database.

---

## File Structure

```
project-root/
├── .env                          # Environment variables (not committed)
├── README.md                     # This file
└── dashboard/
    ├── vizdemov4.0.5.py          # Main application
    └── Pickaxe Analytics logo design.png
```

---

## Versioning

| Version | Notes |
|---|---|
| v3.0–v3.25 | Initial build, page-by-page polish, bug fixes, HTML rendering fixes |
| v4.0.1 | Suggested searches, cross-page search context wiring |
| v4.0.2 | Restored `generate_alert_recommendations`, fixed validation RPC params |
| v4.0.3 | Simplified suggestion generation using `article_count` filter |
| v4.0.4 | Full cross-page context: Fraud Pattern Clusters + Alerts connected to search |
| v4.0.5 | Complete validation restore, `fetch_clusters_by_ids`, alerts KPI fix, dual cluster matching |
