"""
vizdemov4.0.6.py — Pickaxe Analytics: Fraud Intelligence Dashboard
====================================================================

A Streamlit-based fraud intelligence application that combines semantic search,
generative AI synthesis, and embedding-based cluster analysis to deliver an
end-to-end fraud intelligence operating model across five connected pages.

APPLICATION FLOW
----------------
Every page except the Executive Overview is context-aware: once an analyst runs
a search on the Intelligence Search page, the result (dominant cluster, source
articles, cluster IDs) is stored in ``st.session_state["rag_result"]`` and used
to drive the content of the downstream pages automatically.

Pages
~~~~~
1. **Executive Overview** — KPI metrics, Gemini-generated executive brief,
   fraud signal trend chart, cluster stage distribution, and a live evidence
   snapshot. Passive monitoring mode — no search required.

2. **Intelligence Search** — Analyst workbench with Vertex AI-powered RAG
   pipeline. Embeds the query, retrieves semantically similar articles from
   Supabase (``match_articles_v2`` RPC), synthesises a structured intelligence
   brief via Gemini, and surfaces supporting articles with cleaned excerpts.
   Suggested searches are generated from live cluster data and validated before
   display.

3. **Fraud Pattern Clusters** — When a search has been run, shows the clusters
   retrieved by that search (top cards, KPIs, ranked list). When no search is
   active, shows the global top-10 clusters by risk score.

4. **Network Relationships** — Constellation chart of clusters semantically
   adjacent to the dominant cluster from the last search. Connection strength is
   derived from neighbor-appearance frequency across the ``article_neighbors``
   table. Excludes clusters already in the search result set.

5. **Alerts & Watchlists** — Alert feed with per-severity KPI cards that filter
   the feed when clicked. When a search is active, a search-context toggle
   filters the feed to alerts matching the search's clusters (by cluster ID and
   theme label). Enriched cards include fraud descriptions and Gemini-generated
   action recommendations.

DATA LAYER (Supabase)
---------------------
Key tables used:

* ``cluster_themes``      — Fraud theme clusters with labels, descriptions,
                            risk scores, stage labels, and article counts.
* ``article_analysis``    — Per-article metadata: cluster membership, risk score,
                            stage, growth/drift signals.
* ``article_neighbors``   — Pre-computed nearest-neighbor graph for all articles
                            (doc_id → neighbor_doc_id with rank and similarity).
* ``articles_v1``         — Raw article store: title, url, raw_text, timestamps.
* ``alerts``              — Alert events: severity, cluster snapshot, trigger time.
* ``alert_rules``         — Alert rule definitions with thresholds.
* ``watchlists``          — Saved cluster watchlist entries.

GENERATIVE AI LAYER (Vertex AI / Gemini)
-----------------------------------------
* **Embedding model** — ``text-embedding-005`` via Vertex AI, used for query
  embedding in the RAG pipeline and for suggested-search validation.
* **Generative model** — ``gemini-2.5-flash-lite`` via Vertex AI, used for:
  - Intelligence brief synthesis (RAG output)
  - Executive intelligence brief
  - Cluster similarity explanation (Network Relationships)
  - Alert action recommendations
  - Article description generation (fallback when raw_text is empty)
  - Suggested search generation

ENVIRONMENT VARIABLES (via .env)
----------------------------------
  SUPABASE_URL         Supabase project URL
  SUPABASE_KEY         Supabase publishable/anon key
  VERTEX_PROJECT_ID    GCP project ID for Vertex AI
  VERTEX_API_KEY       Vertex AI API key (optional if using ADC)
  GEMINI_API_KEY       Gemini API key (optional if using Vertex)

DEPENDENCIES
------------
  streamlit, supabase-py, google-cloud-aiplatform, vertexai,
  pandas, altair, plotly, python-dotenv

VERSION HISTORY
---------------
  v3.x  — Initial build, page-by-page polish, bug fixes
  v4.0  — Cross-page search context, suggested searches, enriched alert cards,
           constellation chart with click-to-detail, batch Gemini calls
"""

import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import math
import re
from pathlib import Path
from supabase import create_client
from dotenv import load_dotenv
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import os
import traceback
from html import escape as _esc, unescape as _unesc

# --------------------------------------------------
# ENV
# --------------------------------------------------

# Search up from the dashboard dir to find .env at project root
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path if _env_path.exists() else None)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID")

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Pickaxe Analytics",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Asset Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).parent
LOGO_PATH = BASE_DIR / "Pickaxe Analytics logo design.png"

# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    button[data-testid="stBaseButton-headerNoPadding"] {display: none !important;}

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .stApp {
        background-color: #F4F6F8;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D2B52 0%, #123765 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .page-header {
        background: linear-gradient(90deg, #0D2B52 0%, #123765 65%, #177E6C 100%);
        padding: 22px 28px;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 22px rgba(13, 43, 82, 0.12);
    }

    .page-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        letter-spacing: 0.2px;
    }

    .page-subtitle {
        font-size: 0.98rem;
        opacity: 0.92;
        margin-bottom: 0;
    }

    .info-card {
        background: white;
        border: 1px solid #E6EBF0;
        border-radius: 18px;
        padding: 20px 20px 18px 20px;
        box-shadow: 0 4px 18px rgba(16, 24, 40, 0.04);
        height: 100%;
    }

    .metric-card {
        background: white;
        border: 1px solid #E6EBF0;
        border-left: 5px solid #20B07A;
        border-radius: 16px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 4px 14px rgba(16, 24, 40, 0.04);
    }

    .metric-label {
        font-size: 0.9rem;
        color: #667085;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0D2B52;
        line-height: 1.1;
    }

    .metric-delta {
        font-size: 0.85rem;
        color: #20B07A;
        margin-top: 0.35rem;
        font-weight: 600;
    }

    .section-label {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0D2B52;
        margin-bottom: 0.65rem;
    }

    .placeholder-note {
        color: #667085;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .tag {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        background-color: #E8F7F1;
        color: #177E6C;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }

    .alert-high {
        border-left: 5px solid #D92D20;
    }

    .alert-medium {
        border-left: 5px solid #F79009;
    }

    .alert-low {
        border-left: 5px solid #12B76A;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _classify_audience(cluster: dict) -> str:
    """
    Keyword-based audience classifier for fraud cluster cards.

        Concatenates the cluster name and description into a single lowercase
        search string, then tests it against a prioritised list of keyword groups.
        The first group with a matching keyword wins and its label is returned.

        Used on the Executive Overview Themes Requiring Attention cards to render
        a "Target:" badge showing who the fraud is most likely aimed at.

        Falls back to "Broad Financial Ecosystem" when no group matches.

    """
    text = (
        (cluster.get("name") or "") + " " + (cluster.get("description") or "")
    ).lower()
    rules = [
        (["identity", "kyc", "know your customer", "impersonat", "synthetic identity", "verification"],
         "Identity Verification Systems"),
        (["investor", "investment", "crypto", "stablecoin", "digital asset", "portfolio", "trading", "defi"],
         "Investors & Crypto Users"),
        (["bank", "financial institution", "lender", "credit union", "fdic"],
         "Financial Institutions"),
        (["payment", "fintech", "digital wallet", "wire transfer", "ach", "peer-to-peer"],
         "Payment Platforms & Fintechs"),
        (["retail", "shopper", "consumer", "customer", "buyer", "e-commerce", "return fraud"],
         "Retail Consumers"),
        (["business", "enterprise", "corporate", "merchant", "vendor", "b2b", "accounts payable"],
         "Businesses & Merchants"),
        (["government", "agency", "federal", "public sector", "regulatory", "compliance"],
         "Government & Regulators"),
        (["healthcare", "medical", "patient", "hospital", "insurance claim"],
         "Healthcare Sector"),
    ]
    for keywords, label in rules:
        if any(k in text for k in keywords):
            return label
    return "Broad Financial Ecosystem"


@st.cache_data(ttl=600)
def generate_cluster_similarity_brief(
    dominant_label: str,
    dominant_desc: str,
    connected_label: str,
    connected_desc: str,
) -> dict:
    """
    Generate a fraud profile and convergence explanation for two clusters.

    Sends both cluster names + descriptions to Gemini in a structured prompt
    that asks for exactly two labelled sections:
      FRAUD_PROFILE: 2-sentence description of the connected cluster's specific
                     fraud mechanics and threat actors.
      CONVERGENCE:   2-sentence explanation of why the two clusters are
                     semantically linked — shared language, mechanics, or actors.

    Cached for 10 minutes keyed on (dominant_label, connected_label).
    Used in the Network Relationships Strongest Connection card.

    Returns:
        dict with keys "fraud_profile" and "convergence_reason".
    """
    try:
        _, gen_model = get_models()
        prompt = f"""You are a fraud intelligence analyst. Answer in plain prose, no bullet points, no markdown.

Cluster A: {dominant_label}
Description A: {dominant_desc or "No description available."}

Cluster B: {connected_label}
Description B: {connected_desc or "No description available."}

Respond with exactly two labeled sections:

FRAUD_PROFILE: In 2 sentences, describe the specific fraud mechanics and threat actors involved in Cluster B.

CONVERGENCE: In 2 sentences, explain why Cluster A and Cluster B are semantically connected — what shared language, mechanics, or threat actors link them.
"""
        raw = gen_model.generate_content(prompt).text.strip()
        fraud_profile, convergence = "", ""
        for line in raw.splitlines():
            if line.startswith("FRAUD_PROFILE:"):
                fraud_profile = line[len("FRAUD_PROFILE:"):].strip()
            elif line.startswith("CONVERGENCE:"):
                convergence = line[len("CONVERGENCE:"):].strip()
        # Fallback: split on the section headers if they appear mid-text
        if not fraud_profile and "FRAUD_PROFILE:" in raw:
            fraud_profile = raw.split("FRAUD_PROFILE:")[1].split("CONVERGENCE:")[0].strip()
        if not convergence and "CONVERGENCE:" in raw:
            convergence = raw.split("CONVERGENCE:")[1].strip()
        return {
            "fraud_profile": fraud_profile or "No profile available.",
            "convergence_reason": convergence or "No convergence analysis available.",
        }
    except Exception:
        traceback.print_exc()
        return {
            "fraud_profile": "Analysis unavailable.",
            "convergence_reason": "Analysis unavailable.",
        }

@st.cache_data(ttl=3600)
def generate_suggested_searches() -> list[str]:
    """
    Generate and validate suggested search queries from live cluster data.

        Step 1 — Queries Supabase for clusters with article_count >= 5, ordered
        by risk_score descending.  Only high-article clusters are used so that
        generated queries are guaranteed to return content.

        Step 2 — Sends the cluster list to Gemini asking for 10 candidate search
        queries in plain statement format with vocabulary matching the descriptions.

        Step 3 — Each candidate is validated via validate_search_query() which
        embeds it and calls match_articles_v2 to confirm >= 5 results exist.
        Only validated queries are surfaced; falls back to unvalidated candidates
        if none pass validation.

        Cached for 1 hour.  Returns up to 6 validated query strings.

    """
    try:
        sb = get_supabase()
        # Only use clusters with ≥5 articles — guarantees RAG will return results
        theme_rows = (
            sb.table("cluster_themes")
            .select("theme_label, theme_description, risk_score, stage, article_count")
            .gte("article_count", 5)
            .order("risk_score", desc=True)
            .limit(20)
            .execute()
            .data or []
        )
        if not theme_rows:
            return []

        themes_block = "\n".join(
            f"- {r['theme_label']} ({r.get('article_count', '?')} articles, "
            f"Risk: {round(float(r['risk_score'] or 0), 1)}, "
            f"Stage: {r.get('stage') or 'Unknown'}): "
            f"{(r.get('theme_description') or '')[:100]}"
            for r in theme_rows
        )

        _, gen_model = get_models()
        prompt = (
            "You are a fraud intelligence analyst. Based on the fraud theme clusters below, "
            "generate exactly 6 search queries for a semantic search system.\n\n"
            "RULES:\n"
            "- Use vocabulary that closely mirrors the theme descriptions (semantic match matters)\n"
            "- Be specific to fraud mechanics — avoid generic terms like 'fraud overview'\n"
            "- Plain statement format (not questions): e.g. 'AI-enabled payment fraud tactics'\n"
            "- 4-9 words each\n"
            "- Cover a variety of themes from the list\n\n"
            "Return ONLY the 6 queries, one per line, no numbering, no symbols.\n\n"
            "Themes:\n" + themes_block
        )
        raw = gen_model.generate_content(prompt).text.strip()
        queries = [
            line.strip().lstrip("•-–0123456789. ").strip()
            for line in raw.splitlines()
            if line.strip() and len(line.strip()) > 5
        ]
        candidates = [q for q in queries if 3 < len(q) < 100][:10]

        # Validate each candidate — only surface those with ≥5 matching articles
        validated = []
        for q in candidates:
            if validate_search_query(q, min_articles=5):
                validated.append(q)
            if len(validated) >= 6:
                break

        return validated if validated else candidates[:6]

    except Exception:
        traceback.print_exc()
        return []


@st.cache_data(ttl=600)
def generate_alert_recommendations(cluster_data: tuple) -> dict:
    """
    Batch-generate 1-3 action recommendations for a set of alert clusters.

        Sends all clusters in a single Gemini prompt structured with CLUSTER_N:
        labels so the response can be parsed back into a per-cluster dict without
        multiple API round-trips.  This is O(1) Gemini calls regardless of how
        many unique clusters are in the alert feed.

        Args:
            cluster_data: tuple of (cluster_name, description, severity) tuples.
                          Must be a tuple (not list) for st.cache_data hashability.

        Returns:
            {cluster_name: [rec1, rec2, ...]} dict.  Empty dict on failure.

        Cached for 10 minutes.

    """
    if not cluster_data:
        return {}
    try:
        _, gen_model = get_models()
        items_block = "\n".join(
            f"{i+1}. Cluster: {name} | Severity: {sev}\n   Description: {desc or 'No description.'}"
            for i, (name, desc, sev) in enumerate(cluster_data)
        )
        prompt = (
            "You are a fraud risk analyst. For each cluster below, provide 1-3 concise "
            "action recommendations (one per line, plain text, no bullet symbols, no markdown) "
            "that a fraud operations team should take to safeguard against the described threat.\n\n"
            "Format your response EXACTLY as:\n"
            "CLUSTER_1:\nrecommendation\nrecommendation\n\nCLUSTER_2:\nrecommendation\n\n"
            "Use the CLUSTER_N labels to match the numbered input below.\n\n"
            + items_block
        )
        raw = gen_model.generate_content(prompt).text.strip()
        # Parse response into per-cluster lists
        result = {}
        current_idx = None
        current_recs = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                if current_idx is not None and current_recs:
                    name = cluster_data[current_idx][0]
                    result[name] = current_recs[:]
                    current_recs = []
                continue
            import re as _re
            m = _re.match(r'^CLUSTER_(\d+):\s*$', line)
            if m:
                if current_idx is not None and current_recs:
                    name = cluster_data[current_idx][0]
                    result[name] = current_recs[:]
                    current_recs = []
                current_idx = int(m.group(1)) - 1
            elif current_idx is not None:
                current_recs.append(line)
        # Flush last cluster
        if current_idx is not None and current_recs:
            name = cluster_data[current_idx][0]
            result[name] = current_recs[:]
        return result
    except Exception:
        traceback.print_exc()
        return {}


def _clean_snippet(raw_text: str, title: str, max_chars: int = 220) -> str:
    """
    Extract a clean readable excerpt from raw article text.

        Processing order:
          1. Strip all HTML tags and unescape HTML entities.
          2. Collapse whitespace.
          3. Remove leading title repetition (many sources echo the headline
             as the first line of body text).
          4. Truncate at known scraper artifact strings (paywalls, newsletter
             sign-up prompts, registration confirmations, etc.).
          5. Return empty string if fewer than 30 characters remain —
             this signals run_rag to generate a description via Gemini instead.
          6. Trim to max_chars at the last word boundary and append ellipsis.

    """
    # Strip all HTML tags and unescape entities before anything else
    text = re.sub(r'<[^>]+>', ' ', raw_text)
    text = _unesc(text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Strip leading title repetition — full title match, not a 60-char prefix
    if title:
        title_clean = re.sub(r'\s+', ' ', title).strip().lower()
        if text.lower().startswith(title_clean):
            text = text[len(title_clean):].lstrip(" :-")

    # Strip common scraper artifacts
    artifacts = (
        "Highlights", "Summary:", "Subscribe to our daily newsle",
        "no additional logins required", "yes Subscribe",
        "Get Unlimited Access", "Complete the form below",
        "Please confirm your email", "Thank you for registering",
        "Sign up for free", "Already a subscriber",
    )
    for artifact in artifacts:
        idx = text.find(artifact)
        if idx != -1:
            text = text[:idx].strip()

    text = text.strip()

    # Return empty string if not enough real content — caller will use Gemini
    if len(text) < 30:
        return ""

    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text

def md_to_html(text: str) -> str:
    """
    Convert Gemini markdown output to safe HTML for card injection.

        Gemini returns a mix of markdown syntaxes that must be converted to
        HTML before being injected into st.markdown(unsafe_allow_html=True) cards.

        Handles in order:
          - Atx-style headers (##, ###, ####)
          - Gemini's "* **Label:** rest of line" bold-list pattern
          - Bare "* item" list items
          - **bold** and *italic* inline formatting
          - Dash/bullet list items
          - Consecutive <li> items wrapped in <ul>, duplicate tags collapsed
          - Blank lines converted to paragraph breaks

        NOTE: Blank lines between rendered content must be avoided in the
        surrounding HTML string — Streamlit treats blank lines as paragraph
        boundaries and can drop out of HTML rendering context.

    """
    # Headers
    text = re.sub(r'^#### (.+)$', r'<h5 style="color:#0D2B52;margin:0.6rem 0 0.3rem">\1</h5>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.+)$',  r'<h4 style="color:#0D2B52;margin:0.6rem 0 0.3rem">\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$',   r'<h3 style="color:#0D2B52;margin:0.7rem 0 0.3rem">\1</h3>', text, flags=re.MULTILINE)
    # Gemini "* **Label:** rest of line" pattern → styled list item
    text = re.sub(r'^\* \*\*(.+?)\*\*:?\s+(.*)', r'<li><b>\1:</b> \2</li>', text, flags=re.MULTILINE)
    # Remaining bare "* item" lines
    text = re.sub(r'^\* (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    # Bold then italic (order matters — bold first to avoid partial matches)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*',     r'<em>\1</em>', text)
    # Unordered list items
    text = re.sub(r'^\s*[-•]\s+(.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'(<li>.*</li>)', r'<ul>\1</ul>', text, flags=re.DOTALL)
    # Collapse multiple consecutive </ul><ul> tags created by the above
    text = re.sub(r'</ul>\s*<ul>', '', text)
    # Paragraphs — blank lines become paragraph breaks
    text = re.sub(r'\n{2,}', '</p><p>', text)
    text = f'<p>{text}</p>'
    # Clean up any leftover single newlines
    text = text.replace('\n', ' ')
    return text


# --------------------------------------------------
# Backend Init — cached resources (initialize once)
# --------------------------------------------------

@st.cache_resource
def get_supabase():
    """
    Return a cached Supabase client.

        Uses @st.cache_resource so the connection is created once per process
        lifetime and reused across all Streamlit reruns.  Credentials are read
        from SUPABASE_URL and SUPABASE_KEY loaded from the project .env file.

    """
    return create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_resource
def get_models():
    """
    Initialise and return the Vertex AI embedding and generative models.

        Decorated with @st.cache_resource so initialisation runs once per process.
        Calls vertexai.init() to authenticate against VERTEX_PROJECT_ID, then
        loads two models:
          - text-embedding-005 for query embedding in the RAG pipeline
          - gemini-2.5-flash-lite for all generative tasks

        Returns:
            (emb_model, gen_model) tuple for use throughout the app.

    """
    vertexai.init(project=VERTEX_PROJECT_ID, location="us-central1")
    emb = TextEmbeddingModel.from_pretrained("text-embedding-005")
    gen = GenerativeModel("gemini-2.5-flash-lite")
    return emb, gen


# --------------------------------------------------
# Data Fetching — cached with 5-min TTL
# --------------------------------------------------

@st.cache_data(ttl=300)
def fetch_executive_stats() -> dict:
    """
    Compute KPI statistics for the Executive Overview page.

        Makes three Supabase count queries to calculate article volume:
          - total articles across all time
          - articles in the last 7 days (this week)
          - articles from 8-14 days ago (prior week, for delta calculation)

        Queries cluster_themes for stage and risk data to derive:
          - total active clusters
          - emerging and accelerating cluster counts
          - average risk score across all clusters

        Cached for 5 minutes.  Returns a flat dict of display-ready strings.

    """
    try:
        sb = get_supabase()

        week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        two_weeks_ago = (datetime.utcnow() - timedelta(days=14)).isoformat()

        total_res = sb.table("articles_v1").select("doc_id", count="exact").execute()
        recent_res = sb.table("articles_v1").select("doc_id", count="exact").gte("publish_timestamp", week_ago).execute()
        prior_res = (
            sb.table("articles_v1")
            .select("doc_id", count="exact")
            .gte("publish_timestamp", two_weeks_ago)
            .lt("publish_timestamp", week_ago)
            .execute()
        )

        total_count = total_res.count or 0
        recent_count = recent_res.count or 0
        prior_count = prior_res.count or 0

        pct = round(((recent_count - prior_count) / max(prior_count, 1)) * 100)
        article_delta = f"↑ {pct}% vs last 7 days" if pct >= 0 else f"↓ {abs(pct)}% vs last 7 days"

        # Use cluster_themes directly — authoritative stage, risk, and count per cluster
        clusters = (
            sb.table("cluster_themes")
            .select("cluster_id, stage, risk_score")
            .execute()
            .data or []
        )

        active_clusters = len(clusters)
        emerging_count = sum(1 for c in clusters if (c.get("stage") or "").lower() == "emerging")
        accelerating_count = sum(1 for c in clusters if (c.get("stage") or "").lower() == "accelerating")
        risk_scores = [c["risk_score"] for c in clusters if c.get("risk_score") is not None]
        avg_risk = round(sum(risk_scores) / len(risk_scores), 1) if risk_scores else 0.0

        return {
            "article_count": f"{total_count:,}",
            "article_delta": article_delta,
            "active_clusters": str(active_clusters),
            "cluster_delta": f"↑ {accelerating_count} accelerating",
            "emerging_themes": str(emerging_count),
            "emerging_delta": "Active monitoring",
            "avg_risk": str(avg_risk),
            "risk_delta": "Across all clusters",
        }

    except Exception:
        traceback.print_exc()
        return {
            "article_count": "—",
            "article_delta": "Data unavailable",
            "active_clusters": "—",
            "cluster_delta": "",
            "emerging_themes": "—",
            "emerging_delta": "",
            "avg_risk": "—",
            "risk_delta": "",
        }


@st.cache_data(ttl=300)
@st.cache_data(ttl=300)
def fetch_clusters_by_ids(cluster_ids: tuple) -> list[dict]:
    """
    Fetch full cluster details for a specific set of cluster IDs.

        Used by the Fraud Pattern Clusters page to get accurate article counts,
        stage labels, risk scores, and descriptions for the clusters that were
        actually retrieved by the Intelligence Search RAG pipeline.

        Args:
            cluster_ids: tuple of integer cluster IDs (must be a tuple, not a
                         list, for st.cache_data hashability).

        Normalises the stage field — "nan", "none", "null", and empty strings
        are replaced with "Unknown".  Cached for 5 minutes.

    """
    if not cluster_ids:
        return []
    try:
        sb = get_supabase()
        rows = (
            sb.table("cluster_themes")
            .select("cluster_id, theme_label, theme_description, article_count, stage, risk_score")
            .in_("cluster_id", list(cluster_ids))
            .execute()
            .data or []
        )
        return [
            {
                "cluster_id": r["cluster_id"],
                "name": r.get("theme_label") or f"Cluster {r['cluster_id']}",
                "description": r.get("theme_description") or "",
                "articles": r.get("article_count") or "—",
                "stage": (r.get("stage") or "Unknown").capitalize()
                         if (r.get("stage") or "").lower() not in ("nan", "none", "null", "")
                         else "Unknown",
                "risk": round(float(r.get("risk_score") or 0), 1),
            }
            for r in rows
        ]
    except Exception:
        traceback.print_exc()
        return []


def fetch_top_risk_clusters(limit: int = 5) -> list[dict]:
    """
    Fetch the top N clusters ordered by risk_score descending.

        Used by the Executive Overview, Fraud Pattern Clusters (default state),
        and Network Relationships (fallback when no RAG result exists).

        Queries cluster_themes for risk_score, stage, article_count, and
        theme_evidence (JSON array of article doc_ids used for the Evidence
        Snapshot card on the Overview page).

        Resolves theme_evidence doc_ids to {title, url} pairs via articles_v1.
        Cached for 5 minutes.

    """
    try:
        sb = get_supabase()

        res = (
            sb.table("cluster_themes")
            .select("cluster_id, theme_label, theme_description, article_count, stage, risk_score")
            .order("risk_score", desc=True)
            .limit(limit)
            .execute()
            .data or []
        )

        return [
            {
                "cluster_id": r["cluster_id"],
                "name": r.get("theme_label") or f"Cluster {r['cluster_id']}",
                "description": r.get("theme_description") or "",
                "risk": str(round(r["risk_score"], 1)) if r.get("risk_score") is not None else "—",
                "articles": str(r.get("article_count") or 0),
                "stage": (r.get("stage") or "Unknown").capitalize() if (r.get("stage") or "").lower() not in ("nan", "") else "Unknown",
            }
            for r in res
        ]

    except Exception:
        traceback.print_exc()
        return []


@st.cache_data(ttl=300)
def fetch_stage_distribution() -> dict:
    """
    Count clusters in each lifecycle stage for the stage distribution chart.

        Queries all stage values from cluster_themes, excludes null/nan/empty
        entries, and returns a {stage_label: count} dict.

        Used by the Executive Overview Cluster Stage Distribution Altair chart.
        Stage labels are title-cased.  Any unrecognised stages are capitalised
        and included rather than silently dropped.  Cached for 5 minutes.

    """
    try:
        sb = get_supabase()

        res = sb.table("cluster_themes").select("stage").execute().data or []
        # Exclude null, empty, and "nan" — keep all real stage values
        excluded = {"", "nan", "none", "null"}
        stages = [
            (r.get("stage") or "").lower() for r in res
            if (r.get("stage") or "").lower() not in excluded
        ]

        label_map = {
            "emerging": "Emerging",
            "accelerating": "Accelerating",
            "expanding": "Expanding",
            "established": "Established",
            "declining": "Declining",
            "stable": "Stable",
            "trending": "Trending",
        }
        counted = Counter(stages)
        return {label_map.get(k, k.capitalize()): v for k, v in counted.items()}

    except Exception:
        traceback.print_exc()
        return {}


@st.cache_data(ttl=300)
def fetch_signal_trend(weeks: int = 8) -> pd.DataFrame:
    """
    Build a weekly article ingestion trend DataFrame for the signal trend chart.

        Fetches up to 1,000 publish_timestamp values from articles_v1 covering
        the last N weeks, parses them into datetime objects, floors to the week
        boundary (Monday), and counts articles per week.

        Returns a pandas DataFrame with a "Week" index and "Articles" column,
        indexed from the oldest week to the most recent.  Missing weeks are
        filled with zero counts.  Used by the Executive Overview line chart.
        Cached for 5 minutes.

    """
    try:
        sb = get_supabase()
        since = (datetime.utcnow() - timedelta(weeks=weeks)).isoformat()

        articles = (
            sb.table("articles_v1")
            .select("publish_timestamp")
            .gte("publish_timestamp", since)
            .order("publish_timestamp")
            .execute()
            .data or []
        )

        weekly: dict = defaultdict(int)
        for row in articles:
            ts = row.get("publish_timestamp")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                key = dt.strftime("%Y-W%W")
                weekly[key] += 1
            except Exception:
                pass

        if not weekly:
            raise ValueError("no data")

        sorted_keys = sorted(weekly.keys())[-weeks:]
        return pd.DataFrame(
            {"Signal Volume": [weekly[k] for k in sorted_keys]},
            index=[f"Wk {i + 1}" for i in range(len(sorted_keys))],
        )

    except Exception:
        # Fallback to illustrative data
        return pd.DataFrame(
            {"Signal Volume": [72, 81, 79, 88, 94, 107, 116, 132]},
            index=[f"Wk {i}" for i in range(1, 9)],
        )


@st.cache_data(ttl=300)
def fetch_timeline(limit: int = 50) -> list[dict]:
    """
    Fetch recent articles enriched with cluster and risk metadata.

        Pulls the most recent N articles from articles_v1, joins them to
        article_analysis for cluster_id/risk_score/stage, then joins to
        cluster_themes for the human-readable theme label.

        Used by the Executive Overview evidence and monitoring cards.
        Cached for 5 minutes.

    """
    try:
        sb = get_supabase()

        articles = (
            sb.table("articles_v1")
            .select("doc_id,title,url,publish_timestamp")
            .order("publish_timestamp", desc=True)
            .limit(limit)
            .execute()
            .data or []
        )

        doc_ids = [a["doc_id"] for a in articles]
        if not doc_ids:
            return []

        analysis = (
            sb.table("article_analysis")
            .select("doc_id, cluster_id, risk_score, stage")
            .in_("doc_id", doc_ids)
            .execute()
            .data or []
        )
        analysis_map = {a["doc_id"]: a for a in analysis}

        cluster_ids = list({
            a["cluster_id"] for a in analysis
            if a.get("cluster_id") is not None
        })
        themes_res = (
            sb.table("cluster_themes")
            .select("cluster_id, theme_label")
            .in_("cluster_id", cluster_ids)
            .execute()
            .data or []
        ) if cluster_ids else []
        theme_map = {t["cluster_id"]: t["theme_label"] for t in themes_res}

        result = []
        for a in articles:
            meta = analysis_map.get(a["doc_id"], {})
            cid = meta.get("cluster_id")
            result.append({
                "doc_id": a["doc_id"],
                "title": a.get("title"),
                "url": a.get("url"),
                "publish_timestamp": a.get("publish_timestamp"),
                "cluster_id": cid,
                "theme_label": theme_map.get(cid),
                "risk_score": meta.get("risk_score"),
                "stage": meta.get("stage"),
            })

        return result

    except Exception:
        traceback.print_exc()
        return []


@st.cache_data(ttl=600)
def fetch_executive_brief() -> str:
    """
    Generate the Executive Intelligence Brief via Gemini.

        Pulls the top 5 cluster themes by risk score with their descriptions
        and recent article snippets, then sends them to Gemini with a prompt
        asking for a 2-paragraph executive brief covering:
          Paragraph 1 — what is emerging across the monitored themes.
          Paragraph 2 — how the themes interconnect and what to watch.

        Cached for 10 minutes to limit Gemini API calls on page refresh.
        Returns the raw Gemini text (converted to HTML by md_to_html() at
        render time).

    """
    try:
        _, gen_model = get_models()
        sb = get_supabase()

        clusters = (
            sb.table("cluster_themes")
            .select("theme_label, theme_description, article_count, stage, risk_score")
            .order("risk_score", desc=True)
            .limit(8)
            .execute()
            .data or []
        )

        if not clusters:
            return "No cluster data available for brief generation."

        cluster_lines = "\n".join([
            f"- {c.get('theme_label', 'Unknown')}: "
            f"Risk {round(c['risk_score'], 1) if c.get('risk_score') else '?'}, "
            f"Stage {(c.get('stage') or '?').capitalize()}, "
            f"Articles {c.get('article_count') or '?'}. "
            f"{c.get('theme_description') or ''}"
            for c in clusters
        ])

        prompt = f"""You are a fraud intelligence analyst writing a concise weekly executive brief.

Based on the monitored fraud clusters below, write exactly 2 short paragraphs.
- Paragraph 1: What is accelerating or newly active, and what it signals operationally.
- Paragraph 2: What patterns connect the themes, and where attention should be focused.

Do NOT list clusters by name in a bullet list. Synthesize into analytical prose.
Do NOT fabricate facts. Only reference what the data shows.

Clusters:
{cluster_lines}

Write the brief:"""

        return gen_model.generate_content(prompt).text

    except Exception:
        traceback.print_exc()
        return "Executive brief generation failed. Check cluster data and model availability."


@st.cache_data(ttl=300)
def fetch_top_cluster_evidence() -> dict:
    """
    Fetch the highest-risk cluster with its full theme_evidence payload.

        Queries cluster_themes ordered by risk_score descending, taking the
        top result.  The theme_evidence field is a JSON array of article doc_ids
        representing the articles that most strongly anchor the cluster.

        Resolves those doc_ids to {title, url} objects via articles_v1 so the
        Evidence Snapshot card can render clickable article links.
        Cached for 5 minutes.

    """
    try:
        sb = get_supabase()

        res = (
            sb.table("cluster_themes")
            .select("cluster_id, theme_label, theme_description, theme_evidence, risk_score, stage, article_count")
            .order("risk_score", desc=True)
            .limit(1)
            .execute()
            .data or []
        )

        if not res:
            return {}

        row = res[0]
        raw = row.get("theme_evidence")

        # Resolve doc_id lists to article titles
        doc_ids = []
        if isinstance(raw, list):
            doc_ids = [e for e in raw if isinstance(e, str) and len(e) > 20]
        elif isinstance(raw, dict):
            doc_ids = [v for v in raw.values() if isinstance(v, str) and len(v) > 20]

        if doc_ids:
            articles = (
                sb.table("articles_v1")
                .select("doc_id, title, url")
                .in_("doc_id", doc_ids[:15])
                .execute()
                .data or []
            )
            title_map = {a["doc_id"]: (a.get("title") or a["doc_id"]) for a in articles}
            row["theme_evidence"] = [
                {"title": title_map.get(d, d), "url": next((a["url"] for a in articles if a["doc_id"] == d), None)}
                for d in doc_ids[:15]
            ]

        return row

    except Exception:
        traceback.print_exc()
        return {}


@st.cache_data(ttl=300)
@st.cache_data(ttl=300)
def fetch_constellation_data(
    dominant_label: str,
    excluded_cluster_ids: tuple,
) -> list[dict]:
    """
    Build connection strength data for the Network Relationships constellation.

    Seven-step pipeline:
      1. Resolve the dominant cluster's integer ID from its label.
      2. Sample up to 200 articles from that cluster.
      3. Fetch all neighbor slots at ranks 1-50 via article_neighbors.
         Rank is the distance metric — raw similarity_score is uniformly ~0.99
         in this corpus and provides no useful variance for positioning.
      4. Map neighbor doc_ids to cluster_ids via article_analysis,
         batching in chunks of 200 to stay within Supabase query limits.
      5. Count how many times each other cluster appears as a neighbor.
         Excludes the dominant cluster and any cluster in excluded_cluster_ids
         (the Intelligence Search result set — already known, not "adjacent").
      6. Resolve theme labels and descriptions for all connected clusters.
      7. Normalise counts to 0-1, sort descending, cap at 15.

    Args:
        dominant_label:       Theme label of the center cluster.
        excluded_cluster_ids: Tuple of cluster IDs to exclude (tuple for
                              st.cache_data hashability).

    Returns:
        List of dicts: {cluster_id, cluster_name, similarity, count, description}.
        Filtered at display time by the slider, not here.
    """
    try:
        sb = get_supabase()

        # 1. Resolve dominant cluster ID
        dom_rows = (
            sb.table("cluster_themes")
            .select("cluster_id, theme_label")
            .eq("theme_label", dominant_label)
            .limit(1)
            .execute()
            .data or []
        )
        if not dom_rows:
            return []
        dominant_id = dom_rows[0]["cluster_id"]

        # 2. Fetch ALL articles from dominant cluster (up to 200)
        dom_articles = (
            sb.table("article_analysis")
            .select("doc_id")
            .eq("cluster_id", dominant_id)
            .limit(200)
            .execute()
            .data or []
        )
        if not dom_articles:
            return []
        dom_doc_ids = [a["doc_id"] for a in dom_articles]

        # 3. Fetch ALL neighbor slots ranks 1–50 — wide net across all rank positions
        #    We use count of appearances as the metric, not rank position
        neighbors = (
            sb.table("article_neighbors")
            .select("doc_id, neighbor_doc_id, rank")
            .in_("doc_id", dom_doc_ids)
            .lte("rank", 50)
            .execute()
            .data or []
        )
        if not neighbors:
            return []

        # 4. Map all neighbor doc_ids → cluster_ids in one query
        neighbor_doc_ids = list({n["neighbor_doc_id"] for n in neighbors})
        # Batch in chunks of 200 to avoid query size limits
        doc_to_cluster = {}
        for i in range(0, len(neighbor_doc_ids), 200):
            batch = neighbor_doc_ids[i:i + 200]
            rows = (
                sb.table("article_analysis")
                .select("doc_id, cluster_id")
                .in_("doc_id", batch)
                .execute()
                .data or []
            )
            doc_to_cluster.update({r["doc_id"]: r["cluster_id"] for r in rows})

        # 5. Count appearances per connected cluster — frequency IS the strength signal
        excluded = set(excluded_cluster_ids)
        cluster_counts: dict = defaultdict(int)
        total_slots = 0
        for nbr in neighbors:
            cid = doc_to_cluster.get(nbr["neighbor_doc_id"])
            if cid is None or cid == dominant_id or cid in excluded:
                continue
            cluster_counts[cid] += 1
            total_slots += 1

        if not cluster_counts or total_slots == 0:
            return []

        # 6. Resolve theme labels and descriptions for connected clusters
        connected_ids = list(cluster_counts.keys())
        theme_rows = (
            sb.table("cluster_themes")
            .select("cluster_id, theme_label, theme_description")
            .in_("cluster_id", connected_ids)
            .execute()
            .data or []
        )
        theme_map = {r["cluster_id"]: r["theme_label"] for r in theme_rows}
        desc_map  = {r["cluster_id"]: r.get("theme_description") or "" for r in theme_rows}

        # 7. Normalize counts to 0–1 using max count as denominator
        max_count = max(cluster_counts.values())
        result = []
        for cid, count in cluster_counts.items():
            strength = round(count / max_count, 2)
            result.append({
                "cluster_id": cid,
                "cluster_name": theme_map.get(cid, f"Cluster {cid}"),
                "similarity": strength,
                "count": count,
                "description": desc_map.get(cid, ""),
            })

        result.sort(key=lambda x: x["similarity"], reverse=True)
        return result[:15]

    except Exception:
        traceback.print_exc()
        return []


def fetch_cluster_relationships(top_n: int = 6) -> tuple[list[dict], list[dict]]:
    """
    Compute cross-cluster relationships from the article_neighbors graph.

        Loads up to top_n clusters by risk score, then for each cluster samples
        articles and fetches their top-3 article neighbors.  A cross-cluster
        link is counted when a neighbor article belongs to a different cluster.

        Returns:
            (clusters, relationships) tuple where relationships is a list of
            {cluster_a, cluster_b, avg_similarity, connection_count} dicts
            sorted by avg_similarity descending.

        Note: this function is retained for legacy use.  The constellation
        chart uses fetch_constellation_data() instead.

    """
    try:
        sb = get_supabase()

        # 1. Top clusters by risk
        clusters = (
            sb.table("cluster_themes")
            .select("cluster_id, theme_label, risk_score, stage")
            .order("risk_score", desc=True)
            .limit(top_n)
            .execute()
            .data or []
        )
        if not clusters:
            return [], []

        cluster_ids = [c["cluster_id"] for c in clusters]
        theme_map = {c["cluster_id"]: c.get("theme_label", f"Cluster {c['cluster_id']}") for c in clusters}

        # 2. Articles belonging to these clusters
        analysis = (
            sb.table("article_analysis")
            .select("doc_id, cluster_id")
            .in_("cluster_id", cluster_ids)
            .execute()
            .data or []
        )
        doc_to_cluster = {a["doc_id"]: a["cluster_id"] for a in analysis}

        cluster_to_docs: dict = defaultdict(list)
        for a in analysis:
            cluster_to_docs[a["cluster_id"]].append(a["doc_id"])

        # 3. Sample up to 15 articles per cluster to keep the query manageable
        sample_doc_ids = []
        for cid in cluster_ids:
            sample_doc_ids.extend(cluster_to_docs[cid][:15])

        if not sample_doc_ids:
            return clusters, []

        # 4. Fetch top-3 neighbors for each sampled article
        neighbors = (
            sb.table("article_neighbors")
            .select("doc_id, neighbor_doc_id, similarity_score")
            .in_("doc_id", sample_doc_ids)
            .lte("rank", 3)
            .execute()
            .data or []
        )

        # 5. Aggregate cross-cluster connections
        connections: dict = defaultdict(lambda: {"count": 0, "similarities": []})
        for n in neighbors:
            src = doc_to_cluster.get(n["doc_id"])
            dst = doc_to_cluster.get(n["neighbor_doc_id"])
            if src is None or dst is None or src == dst:
                continue
            key = tuple(sorted([src, dst]))
            connections[key]["count"] += 1
            if n.get("similarity_score") is not None:
                connections[key]["similarities"].append(float(n["similarity_score"]))

        relationships = []
        for (cid_a, cid_b), data in connections.items():
            avg_sim = round(sum(data["similarities"]) / len(data["similarities"]), 2) if data["similarities"] else 0.0
            relationships.append({
                "cluster_a": theme_map.get(cid_a, f"Cluster {cid_a}"),
                "cluster_b": theme_map.get(cid_b, f"Cluster {cid_b}"),
                "connection_count": data["count"],
                "avg_similarity": avg_sim,
            })

        relationships.sort(key=lambda x: x["avg_similarity"], reverse=True)
        return clusters, relationships

    except Exception:
        traceback.print_exc()
        return [], []


# --------------------------------------------------
# RAG Pipeline — ported from rag_api.py
# --------------------------------------------------

def is_fraud_query(query: str) -> bool:
    """
    Domain guard — returns True only if the query is fraud-intelligence-related.

        Fires a minimal Gemini call asking for YES/NO before the full RAG pipeline
        runs.  Prevents wasting embedding + generation API calls on off-topic queries
        and blocks the system from answering questions outside its domain.

        Returns True on any exception so the pipeline is never silently blocked
        by a transient model availability issue.

    """
    _, gen_model = get_models()
    prompt = f"""Classify if the query is related to fraud intelligence.

Query: {query}

Valid domains:
- financial fraud
- cybercrime
- cybersecurity threats
- scams / phishing / identity theft

Answer ONLY YES or NO."""
    try:
        res = gen_model.generate_content(prompt).text.strip().lower()
        return "yes" in res
    except Exception:
        return False


def is_relevant(doc: dict) -> bool:
    """
    Keyword relevance filter applied to vector-retrieved documents.

        Vector similarity alone can surface tangentially related articles that
        contain no actual fraud content.  This second-pass filter checks raw_text,
        snippet, or title for at least one core fraud/cybercrime keyword before the
        document is included in the Gemini context window.

        Returns True if any keyword matches, False otherwise.

    """
    text = (
        doc.get("raw_text") or doc.get("snippet") or doc.get("title") or ""
    ).lower()
    keywords = [
        "fraud", "scam", "phishing", "cyber", "hacking", "malware",
        "identity", "attack", "theft", "ransomware", "impersonation",
        "financial crime",
    ]
    return any(k in text for k in keywords)


def validate_search_query(query: str, min_articles: int = 5) -> bool:
    """
    Validate a search query against the article corpus before surfacing it.

        Embeds the query using the Vertex AI text-embedding-005 model, then calls
        the match_articles_v2 Supabase RPC to count semantic matches.

        IMPORTANT: only query_embedding and match_count are passed to the RPC.
        Passing additional parameters (match_threshold, filter_date) causes the
        RPC to fail silently, returning zero results even when content exists.

        Returns True if at least min_articles matching articles are found.
        Returns False on any exception to avoid blocking the pipeline.

    """
    try:
        emb_model, _ = get_models()
        sb = get_supabase()
        inputs = [TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")]
        embedding = emb_model.get_embeddings(inputs)[0].values
        rows = sb.rpc(
            "match_articles_v2",
            {
                "query_embedding": embedding,
                "match_count": min_articles * 2,
            },
        ).execute().data or []
        return len(rows) >= min_articles
    except Exception:
        return False


def run_rag(query: str, top_k: int = 5, min_similarity: float = 0.50) -> dict:
    """
    Full Retrieval-Augmented Generation pipeline for fraud intelligence queries.

        Nine-step process:
          1. Domain guard  — reject non-fraud queries via is_fraud_query().
          2. Embed query   — convert query to vector via text-embedding-005.
          3. Vector search — call match_articles_v2 Supabase RPC to retrieve
                             top_k * 3 candidate articles by cosine similarity.
          4. Relevance filter — apply keyword filter via is_relevant().
          5. Analysis join  — enrich each doc with cluster_id, risk_score, and
                              stage from article_analysis and cluster_themes.
          6. Enrich docs    — attach theme label, risk, stage to each document.
          7. Build context  — format enriched docs into a structured text block.
          8. Generate brief — send context + query to Gemini with a constrained
                              4-section prompt (Threat Summary, Key Patterns,
                              Risk Assessment, Analyst Note, ≤250 words).
          9. Batch describe — for any article with an empty snippet, fire one
                              batch Gemini call to generate one-sentence summaries.

        The result dict is stored in st.session_state["rag_result"] and consumed
        by the Fraud Pattern Clusters, Network Relationships, and Alerts pages.

        Returns:
            dict with keys: query, answer, sources, retrieved_count,
            dominant_cluster, avg_similarity, avg_risk.

    """
    try:
        emb_model, gen_model = get_models()
        sb = get_supabase()

        # 1. Domain guard
        if not is_fraud_query(query):
            return {
                "query": query,
                "answer": "This system only supports fraud and cybercrime intelligence queries.",
                "sources": [],
                "retrieved_count": 0,
                "dominant_cluster": None,
                "avg_similarity": None,
                "avg_risk": None,
            }

        # 2. Embed query
        emb = emb_model.get_embeddings([TextEmbeddingInput(query, "RETRIEVAL_QUERY")])
        qvec = emb[0].values

        # 3. Vector retrieval
        retrieval = sb.rpc(
            "match_articles_v2",
            {
                "query_embedding": qvec,
                "match_count": top_k * 3,
                "min_similarity": min_similarity,
            },
        ).execute()
        docs = (retrieval.data or [])[:top_k]
        print(f"[RAG] retrieved {len(docs)} docs at min_similarity={min_similarity}")

        # 4. Relevance filter
        docs = [d for d in docs if is_relevant(d)]

        if not docs:
            return {
                "query": query,
                "answer": "No relevant fraud intelligence found for this query.",
                "sources": [],
                "retrieved_count": 0,
                "dominant_cluster": None,
                "avg_similarity": None,
                "avg_risk": None,
            }

        # 5. Analysis join
        doc_ids = [d["doc_id"] for d in docs]
        analysis = (
            sb.table("article_analysis")
            .select("doc_id, cluster_id, risk_score, stage")
            .in_("doc_id", doc_ids)
            .execute()
            .data or []
        )
        analysis_map = {a["doc_id"]: a for a in analysis}

        cluster_ids = list({
            a["cluster_id"] for a in analysis
            if a.get("cluster_id") is not None
        })
        themes_res = (
            sb.table("cluster_themes")
            .select("cluster_id, theme_label")
            .in_("cluster_id", cluster_ids)
            .execute()
            .data or []
        ) if cluster_ids else []
        theme_map = {t["cluster_id"]: t["theme_label"] for t in themes_res}

        # 6. Enrich docs
        for d in docs:
            meta = analysis_map.get(d["doc_id"], {})
            cid = meta.get("cluster_id")
            d["cluster_id"] = cid
            d["theme_label"] = theme_map.get(cid)
            d["risk_score"] = meta.get("risk_score")
            d["stage"] = meta.get("stage")

        # 7. Build context
        context = "\n\n---\n\n".join([
            f"TITLE: {d.get('title', '')}\n"
            f"THEME: {d.get('theme_label', '')}\n"
            f"STAGE: {d.get('stage', '')}\n"
            f"RISK: {d.get('risk_score', '')}\n"
            f"DATE: {d.get('publish_timestamp', '')}\n\n"
            f"{d.get('raw_text', '')}"
            for d in docs
        ])

        # 8. Generate intelligence brief
        current_date = datetime.utcnow().strftime("%Y-%m-%d")
        prompt = f"""You are a senior fraud intelligence analyst. Write a concise brief using ONLY the provided context.

CURRENT DATE: {current_date}

RULES:
- Do NOT fabricate facts or reference articles not in the context
- Do NOT list article titles or sources
- Merge similar threats into unified insights
- Focus ONLY on fraud and financial crime
- Total response must be under 250 words

OUTPUT STRUCTURE — use exactly these four sections, no others:

**Threat Summary**
2-3 sentences describing the core fraud dynamic observed across the retrieved signals.

**Key Patterns**
Exactly 3 bullet points. Each bullet is one specific fraud mechanic, tactic, or emerging behavior.

**Risk Assessment**
1-2 sentences on who is most exposed and why.

**Analyst Note**
One sentence: the single most important development to monitor going forward.

Context:
{context}

Question:
{query}
"""
        answer = gen_model.generate_content(prompt).text

        # 9. Compute summary metadata
        cluster_votes = [d.get("theme_label") for d in docs if d.get("theme_label")]
        dominant_cluster = Counter(cluster_votes).most_common(1)[0][0] if cluster_votes else None

        similarities = [d.get("similarity") for d in docs if d.get("similarity") is not None]
        avg_sim = round(sum(similarities) / len(similarities), 2) if similarities else None

        risk_scores = [d.get("risk_score") for d in docs if d.get("risk_score") is not None]
        avg_risk = round(sum(risk_scores) / len(risk_scores), 1) if risk_scores else None

        sources_built = [
            {
                "doc_id": d.get("doc_id"),
                "title": d.get("title"),
                "url": d.get("url"),
                "publish_timestamp": d.get("publish_timestamp"),
                "cluster_id": d.get("cluster_id"),
                "theme_label": d.get("theme_label"),
                "risk_score": d.get("risk_score"),
                "stage": d.get("stage"),
                "snippet": _clean_snippet(d.get("raw_text") or "", d.get("title") or ""),
                "similarity": d.get("similarity"),
                "source": d.get("source"),
            }
            for d in docs
        ]

        # Batch-generate descriptions for any articles whose snippet is empty
        no_snippet = [(i, s["title"]) for i, s in enumerate(sources_built) if not s["snippet"]]
        if no_snippet:
            titles_block = "\n".join(
                f"{j + 1}. {title}" for j, (_, title) in enumerate(no_snippet)
            )
            desc_prompt = (
                "For each article title below, write exactly one sentence (max 25 words) "
                "describing what it covers from a fraud or financial risk perspective. "
                "Number your responses to match the input. No preamble.\n\n"
                + titles_block
            )
            try:
                raw_desc = gen_model.generate_content(desc_prompt).text.strip()
                desc_lines = [
                    re.sub(r'^\d+\.\s*', '', ln).strip()
                    for ln in raw_desc.splitlines()
                    if ln.strip()
                ]
                for j, (i, _) in enumerate(no_snippet):
                    sources_built[i]["snippet"] = (
                        desc_lines[j] if j < len(desc_lines) else "No excerpt available."
                    )
            except Exception:
                for i, _ in no_snippet:
                    sources_built[i]["snippet"] = "No excerpt available."

        return {
            "query": query,
            "answer": answer,
            "sources": sources_built,
            "retrieved_count": len(docs),
            "dominant_cluster": dominant_cluster,
            "avg_similarity": avg_sim,
            "avg_risk": avg_risk,
        }

    except Exception:
        traceback.print_exc()
        return {
            "query": query,
            "answer": "An error occurred while processing your query. Please try again.",
            "sources": [],
            "retrieved_count": 0,
            "dominant_cluster": None,
            "avg_similarity": None,
            "avg_risk": None,
        }


# --------------------------------------------------
# Alerts & Watchlists — Data Functions
# --------------------------------------------------

@st.cache_data(ttl=60)
def fetch_watchlists() -> list[dict]:
    """
    Fetch all saved watchlist entries with joined cluster metadata.

        Queries the watchlists table and joins to cluster_themes to resolve
        theme_label, risk_score, and stage for display in the Alerts page.
        Cached for 1 minute (short TTL since watchlists can be modified).

    """
    try:
        sb = get_supabase()
        rows = sb.table("watchlists").select("id, cluster_id, label, notes, created_at").order("created_at", desc=True).execute().data or []
        if not rows:
            return []

        cluster_ids = list({r["cluster_id"] for r in rows if r.get("cluster_id")})
        themes = sb.table("cluster_themes").select("cluster_id, theme_label, risk_score, stage").in_("cluster_id", cluster_ids).execute().data or []
        theme_map = {t["cluster_id"]: t for t in themes}

        for r in rows:
            meta = theme_map.get(r["cluster_id"], {})
            r["theme_label"] = meta.get("theme_label", f"Cluster {r['cluster_id']}")
            r["risk_score"]  = meta.get("risk_score")
            r["stage"]       = (meta.get("stage") or "Unknown").capitalize()

        return rows
    except Exception:
        traceback.print_exc()
        return []


@st.cache_data(ttl=300)
def fetch_alert_rules() -> list[dict]:
    """
    Fetch active alert rule definitions from the database.

        Queries alert_rules filtered to is_active = True.  Returns rule names,
        types, thresholds, and descriptions for display in the Alerts page.
        Cached for 5 minutes.

    """
    try:
        sb = get_supabase()
        return sb.table("alert_rules").select("*").eq("is_active", True).order("created_at").execute().data or []
    except Exception:
        traceback.print_exc()
        return []


@st.cache_data(ttl=60)
def fetch_recent_alerts(limit: int = 50) -> list[dict]:
    """
    Fetch recent alerts enriched with cluster theme data.

        Pulls up to `limit` alerts ordered by triggered_at descending, then
        joins to cluster_themes to add theme_label, theme_description, and
        risk_score to each alert record.

        Returns all alerts sorted by risk_score descending with no deduplication —
        each alert event is its own row so the count matches the KPI cards.
        The display layer (page_alerts_watchlists) handles grouping repeated
        clusters into a full card + compact repeat rows.

        Cached for 1 minute.

    """
    try:
        sb = get_supabase()
        alerts = (
            sb.table("alerts")
            .select("*")
            .order("triggered_at", desc=True)
            .limit(limit)
            .execute()
            .data or []
        )
        if not alerts:
            return []

        cluster_ids = list({a["cluster_id"] for a in alerts if a.get("cluster_id")})
        themes = (
            sb.table("cluster_themes")
            .select("cluster_id, theme_label, theme_description, risk_score")
            .in_("cluster_id", cluster_ids)
            .execute()
            .data or []
        )
        theme_map = {t["cluster_id"]: t for t in themes}

        for a in alerts:
            t = theme_map.get(a["cluster_id"], {})
            a["theme_label"]       = t.get("theme_label") or f"Cluster {a['cluster_id']}"
            a["theme_description"] = t.get("theme_description") or ""
            a["risk_score"]        = t.get("risk_score") or a.get("cluster_risk_snapshot") or 0

        # Sort by risk score descending — no deduplication, each alert is its own event
        return sorted(alerts, key=lambda x: float(x.get("risk_score") or 0), reverse=True)

    except Exception:
        traceback.print_exc()
        return []


@st.cache_data(ttl=60)
def fetch_alert_stats() -> dict:
    """
    Compute global alert counts by severity for the KPI row.

        Queries the alerts table for all severity and is_read values and
        aggregates counts in Python.  Used as the baseline when no search
        context filter is active.  When search filter is active, the Alerts
        page recomputes counts from the filtered display_alerts list instead.

        Cached for 1 minute.

    """
    try:
        sb = get_supabase()
        alerts = sb.table("alerts").select("severity, is_read").execute().data or []
        total    = len(alerts)
        high     = sum(1 for a in alerts if a.get("severity") == "high")
        medium   = sum(1 for a in alerts if a.get("severity") == "medium")
        low      = sum(1 for a in alerts if a.get("severity") == "low")
        unread   = sum(1 for a in alerts if not a.get("is_read"))
        return {"total": total, "high": high, "medium": medium, "low": low, "unread": unread}
    except Exception:
        traceback.print_exc()
        return {"total": 0, "high": 0, "medium": 0, "low": 0, "unread": 0}


def add_to_watchlist(cluster_id: int, label: str, notes: str) -> bool:
    """
    Insert a new cluster entry into the watchlists table.

        Args:
            cluster_id: Integer cluster ID from cluster_themes.
            label:      Optional display label; stored as-is, may be None.
            notes:      Optional analyst notes; stored as-is, may be None.

        Returns True on success, False on any database error.

    """
    try:
        sb = get_supabase()
        sb.table("watchlists").insert({
            "cluster_id": cluster_id,
            "label": label or None,
            "notes": notes or None,
        }).execute()
        return True
    except Exception:
        traceback.print_exc()
        return False


def delete_watchlist(watchlist_id: str) -> bool:
    """
    Delete a watchlist entry by its UUID.

        Clears the fetch_watchlists cache after deletion so the page refreshes
        immediately without waiting for the TTL to expire.
        Returns True on success, False on any database error.

    """
    try:
        sb = get_supabase()
        sb.table("watchlists").delete().eq("id", watchlist_id).execute()
        fetch_watchlists.clear()
        return True
    except Exception:
        traceback.print_exc()
        return False


def evaluate_and_insert_alerts() -> int:
    """
    Evaluate active alert rules against current cluster data and insert new alerts.

        Runs each active rule from alert_rules against the current state of
        cluster_themes.  Supported rule types:
          - risk_threshold: fires when a cluster's risk_score exceeds the threshold.
          - stage_transition: fires when a cluster's stage matches a target value.
          - growth_spike: fires when a cluster's recent article growth is high.

        Deduplicates by checking whether an identical alert was already inserted
        for that cluster in the last 24 hours to prevent alert storms.

        Returns the count of new alerts inserted.  Called from the (now removed)
        Run Alert Evaluation button; retained for potential future use.

    """
    try:
        sb = get_supabase()
        rules = sb.table("alert_rules").select("*").eq("is_active", True).execute().data or []
        clusters = sb.table("cluster_themes").select("cluster_id, theme_label, risk_score, stage").execute().data or []
        cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()

        # Pre-fetch recent alerts to avoid duplicate inserts
        recent = sb.table("alerts").select("cluster_id, rule_id").gte("triggered_at", cutoff).execute().data or []
        already_fired = {(r["cluster_id"], r["rule_id"]) for r in recent}

        rule_map = {r["rule_type"]: r for r in rules}
        new_alerts = []

        for c in clusters:
            cid   = c["cluster_id"]
            risk  = c.get("risk_score") or 0
            stage = (c.get("stage") or "").lower()
            name  = c.get("theme_label", f"Cluster {cid}")

            # Rule: risk_escalation
            rule = rule_map.get("risk_escalation")
            if rule and risk >= (rule.get("threshold") or 3.0):
                key = (cid, rule["id"])
                if key not in already_fired:
                    severity = "high" if risk >= 3.5 else "medium"
                    new_alerts.append({
                        "cluster_id": cid,
                        "rule_id": rule["id"],
                        "severity": severity,
                        "alert_type": "risk_escalation",
                        "message": f"{name} — risk score {round(risk, 1)} exceeds threshold {round(rule['threshold'], 1)}.",
                        "cluster_risk_snapshot": risk,
                        "cluster_stage_snapshot": stage,
                    })

            # Rule: stage_transition — flag accelerating clusters
            rule = rule_map.get("stage_transition")
            if rule and stage == "accelerating":
                key = (cid, rule["id"])
                if key not in already_fired:
                    new_alerts.append({
                        "cluster_id": cid,
                        "rule_id": rule["id"],
                        "severity": "medium",
                        "alert_type": "stage_transition",
                        "message": f"{name} — cluster is in accelerating stage.",
                        "cluster_risk_snapshot": risk,
                        "cluster_stage_snapshot": stage,
                    })

        if new_alerts:
            sb.table("alerts").insert(new_alerts).execute()

        # Invalidate cached alert data
        fetch_recent_alerts.clear()
        fetch_alert_stats.clear()
        fetch_watchlists.clear()

        return len(new_alerts)

    except Exception:
        traceback.print_exc()
        return 0


# --------------------------------------------------
# Reusable UI Components (preserved from vizdemov1.py)
# --------------------------------------------------

def render_sidebar():
    """
    Render the left navigation sidebar with logo and page links.

        Displays the Pickaxe Analytics logo (PNG loaded from BASE_DIR),
        a Navigation heading, and a radio button group for page selection.
        The active page is highlighted with a red dot in the radio widget.

        Returns:
            The currently selected page name as a string.

    """
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)
        else:
            st.warning("Logo file not found in app directory.")

        st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size: 2rem; font-weight: 800; color: white; margin-bottom: 0.75rem;">
                Navigation
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected_page = st.radio(
            "",
            [
                "Executive Overview",
                "Intelligence Search",
                "Fraud Pattern Clusters",
                "Network Relationships",
                "Alerts & Watchlists",
            ],
            label_visibility="collapsed",
        )

    return selected_page


def render_page_header(title: str, description: str):
    """
    Render the full-width gradient page header banner.

        Injects a styled div using the .page-header CSS class defined in the
        global stylesheet block.  Title is displayed in 2rem bold white text;
        description is a smaller subtitle below it.

        Args:
            title:       Main page heading.
            description: One-line subtitle describing the page's purpose.

    """
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-title">{title}</div>
            <div class="page-subtitle">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, delta: str):
    """
    Render a KPI metric card with auto-scaling font size.

        Font size for the value field scales down automatically based on string
        length to prevent long cluster names or large numbers from overflowing:
          ≤12 chars → 1.8rem,  ≤22 → 1.25rem,  ≤34 → 1.0rem,  else → 0.88rem.

        Uses the .metric-card CSS class with a green left border accent.
        word-wrap: break-word is set to handle any edge cases.

    """
    val_len = len(value)
    if val_len <= 12:
        font_size = "1.8rem"
    elif val_len <= 22:
        font_size = "1.25rem"
    elif val_len <= 34:
        font_size = "1.0rem"
    else:
        font_size = "0.88rem"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="font-size:{font_size};line-height:1.25;word-wrap:break-word;">{value}</div>
            <div class="metric-delta">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(title: str, body: str, tags=None, extra_class=""):
    """
    Render a white content card with optional tag pills.

        Wraps title and HTML body in a .info-card div.  Tag strings in the
        optional `tags` list are rendered as inline .tag pill elements beneath
        the title.  Body HTML is injected with unsafe_allow_html=True so callers
        must pre-escape all user-sourced strings with _esc() before passing them.

        Args:
            title:       Card heading text.
            body:        HTML string for the card body.
            tags:        Optional list of tag label strings.
            extra_class: Optional additional CSS class on the outer div.

    """
    tag_html = ""
    if tags:
        tag_html = "".join([f'<span class="tag">{tag}</span>' for tag in tags])

    st.markdown(
        f"""
        <div class="info-card {extra_class}">
            <div class="section-label">{title}</div>
            <div>{tag_html}</div>
            <div class="placeholder-note">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ranked_list_card(title: str, items: list[dict]):
    """
    Render a list of clusters as a ranked card with divider rows.

        Each item dict must have keys: name, risk, articles, stage.
        Stage values of nan/none/null/empty are normalised to "Unknown".
        Names and stage labels are HTML-escaped with _esc() before injection.
        Used on the Fraud Pattern Clusters page for the full ranked list and
        the search-context ranked list.

    """
    rows_html = ""
    for item in items:
        raw_stage = (item.get("stage") or "").lower()
        stage_display = item["stage"] if raw_stage not in ("nan", "none", "null", "") else "Unknown"
        rows_html += (
            f'<div style="padding:0.6rem 0;border-bottom:1px solid #EEF2F6;">'
            f'<div style="font-weight:700;color:#0D2B52;font-size:0.98rem;">{_esc(item["name"])}</div>'
            f'<div style="color:#667085;font-size:0.88rem;margin-top:0.2rem;">'
            f'Risk: <b>{item["risk"]}</b> &nbsp;&nbsp;|&nbsp;&nbsp;'
            f'Articles: <b>{item["articles"]}</b> &nbsp;&nbsp;|&nbsp;&nbsp;'
            f'Stage: <b>{stage_display}</b>'
            f'</div></div>'
        )
    st.markdown(
        f'<div class="info-card"><div class="section-label">{title}</div>{rows_html}</div>',
        unsafe_allow_html=True,
    )


def render_cluster_constellation(center_name: str, center_desc: str, connections: list[dict]):
    """
    Render the Plotly cluster constellation chart for the Network Relationships page.

        Positions the dominant cluster at the center (amber node, size 20).
        Connected clusters radiate outward (green nodes, size 11) at evenly
        distributed angles.  Distance from center = 0.15 + (1 - strength) * 1.4
        so strength 1.0 sits close in and strength 0.0 sits at the outer edge.

        Each line has the connection strength score annotated at its midpoint.
        Node labels use quadrant-aware positioning (middle right/left for
        horizontal nodes, top/bottom center for vertical nodes).

        Each trace carries customdata=[name, description] so click events
        surfaced via on_select="rerun" can populate the detail card below.

        Args:
            center_name:  Label of the dominant (center) cluster.
            center_desc:  Description text for the center node detail card.
            connections:  List of dicts from fetch_constellation_data().

        Returns:
            The Streamlit chart event object (or None if no connections).

    """
    n = len(connections)
    if n == 0:
        st.info("No related clusters found in the current connection strength range.")
        return None

    fig = go.Figure()
    node_x, node_y = [], []

    for i, conn in enumerate(connections):
        angle = (2 * math.pi * i / n) - (math.pi / 2)
        dist = 0.15 + (1.0 - conn["similarity"]) * 1.4
        node_x.append(dist * math.cos(angle))
        node_y.append(dist * math.sin(angle))

    # Lines first
    for i, conn in enumerate(connections):
        fig.add_trace(go.Scatter(
            x=[0, node_x[i]], y=[0, node_y[i]],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.2)", width=1.5),
            showlegend=False,
            hoverinfo="none",
        ))
        fig.add_annotation(
            x=node_x[i] * 0.5, y=node_y[i] * 0.5,
            text=str(conn["similarity"]),
            showarrow=False,
            font=dict(size=9, color="rgba(180,210,235,0.95)"),
            bgcolor="rgba(10,30,61,0.75)",
            borderpad=2,
        )

    def _text_pos(x, y):
        # Position the node label away from center based on which quadrant the node is in
        if abs(x) >= abs(y):
            return "middle right" if x >= 0 else "middle left"
        return "top center" if y >= 0 else "bottom center"

    # Connected cluster nodes — customdata carries name + description for click card
    for i, conn in enumerate(connections):
        desc = conn.get("description") or "No description available."
        fig.add_trace(go.Scatter(
            x=[node_x[i]], y=[node_y[i]],
            mode="markers+text",
            marker=dict(size=11, color="#20B07A",
                        line=dict(color="rgba(255,255,255,0.7)", width=1.5)),
            text=[conn["cluster_name"]],
            textposition=_text_pos(node_x[i], node_y[i]),
            textfont=dict(size=10, color="rgba(255,255,255,0.9)"),
            customdata=[[conn["cluster_name"], desc]],
            showlegend=False,
            hovertemplate=(
                f"<b>{conn['cluster_name']}</b><br>"
                f"Strength: {conn['similarity']}<br>"
                f"<i>Click to view details</i><extra></extra>"
            ),
        ))

    # Center node
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers+text",
        marker=dict(size=20, color="#F79009", line=dict(color="white", width=2)),
        text=[center_name],
        textposition="bottom center",
        textfont=dict(size=11, color="white"),
        customdata=[[center_name, center_desc or "No description available."]],
        showlegend=False,
        hovertemplate=f"<b>{center_name}</b><br>Dominant Cluster<br><i>Click to view details</i><extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="#0A1E3D",
        plot_bgcolor="#0A1E3D",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.25, 1.25]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-1.25, 1.25], scaleanchor="x"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=480,
        hoverlabel=dict(bgcolor="#0D2B52", font_color="white"),
    )

    return st.plotly_chart(
        fig, use_container_width=True,
        on_select="rerun", key="constellation_chart",
    )


def render_rag_article_card(doc: dict):
    """
    Render a single supporting article card in the Intelligence Search results.

        Displays:
          - Article title as a link (URL from doc["url"])
          - Source and publication date
          - Cleaned excerpt (from _clean_snippet, or Gemini-generated fallback)
          - Tag pills: similarity score, risk score (color-coded), stage

        Risk color coding: gray < 0.5, amber 0.5-0.99, red >= 1.0.
        Stage "nan"/null values are normalised to "Uncategorized".
        All user-sourced strings are HTML-escaped before injection.

    """
    theme = doc.get("theme_label") or "Unclustered"
    risk = doc.get("risk_score")
    stage = doc.get("stage") or "Unknown"
    sim = doc.get("similarity")
    ts = doc.get("publish_timestamp", "")
    date_str = ts[:10] if ts else "Unknown"
    url = doc.get("url", "#")
    source = doc.get("source") or ""
    snippet = doc.get("snippet") or "No excerpt available."

    source_html = f"<b>Source:</b> {source} &nbsp;&nbsp;|&nbsp;&nbsp;" if source else ""
    sim_tag = f'<span class="tag">Similarity {round(sim, 2)}</span>' if sim is not None else ""

    # Clamp negatives to 0 (article-level scores can be z-score derivatives),
    # round to 2dp, and color-code for at-a-glance severity
    if risk is not None:
        risk_display = round(max(0.0, float(risk)), 2)
        if risk_display >= 1.0:
            risk_color = "#D92D20"   # red — elevated
        elif risk_display >= 0.5:
            risk_color = "#F79009"   # amber — moderate
        else:
            risk_color = "#667085"   # gray — low / minimal
        risk_tag = (
            f'<span class="tag" style="color:{risk_color};font-weight:600;">'
            f'Risk {risk_display}</span>'
        )
    else:
        risk_tag = ""

    # Normalize stage display — hide "nan" values
    raw_stage = (stage or "").lower()
    stage_display = stage.capitalize() if raw_stage not in ("nan", "none", "null", "") else "Uncategorized"
    stage_tag = f'<span class="tag">Stage {stage_display}</span>'

    # Collapse all tags into one string — empty risk_tag was creating blank lines
    # that caused Streamlit's markdown renderer to drop out of HTML context
    tags_html = "".join(filter(None, [sim_tag, risk_tag, stage_tag]))

    st.markdown(
        f'<div class="info-card" style="margin-bottom:0.85rem;">'
        f'<div style="display:flex;justify-content:space-between;align-items:start;gap:1rem;">'
        f'<div>'
        f'<div class="section-label" style="margin-bottom:0.35rem;">'
        f'<a href="{url}" target="_blank" style="color:#0D2B52;text-decoration:none;">'
        f'{_esc(doc.get("title", "Untitled"))}'
        f'</a></div>'
        f'<div class="placeholder-note" style="margin-bottom:0.5rem;">'
        f'{source_html}<b>Published:</b> {date_str}'
        f'</div></div>'
        f'<div><span class="tag">{_esc(theme)}</span></div>'
        f'</div>'
        f'<div class="placeholder-note" style="margin-bottom:0.65rem;">{_esc(snippet)}</div>'
        f'<div>{tags_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Pages
# --------------------------------------------------

def page_executive_overview():
    """
    Render the Executive Overview page — passive monitoring mode.

        Layout:
          Row 1: four KPI metric cards (articles, clusters, emerging, avg risk)
          Row 2: Executive Intelligence Brief (Gemini) | Top Risk Clusters list
          Row 3: Fraud Signal Trend chart | Stage Distribution chart | Evidence Snapshot
          Row 4: Themes Requiring Attention (expandable cards with audience badges)

        This page does NOT read from st.session_state["rag_result"].  It always
        shows the global state of the data environment regardless of any search
        the analyst may have run.

    """
    render_page_header(
        "Executive Overview",
        "A high-level view of article monitoring, theme clustering, risk acceleration, and emerging fraud pattern activity across the intelligence environment.",
    )

    with st.spinner("Loading intelligence overview..."):
        stats = fetch_executive_stats()
        top_clusters = fetch_top_risk_clusters(limit=4)
        all_clusters = fetch_top_risk_clusters(limit=20)
        stage_dist = fetch_stage_distribution()
        trend_df = fetch_signal_trend(weeks=8)
        brief = fetch_executive_brief()
        top_evidence = fetch_top_cluster_evidence()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Articles Monitored", stats["article_count"], stats["article_delta"])
    with c2:
        render_metric_card("Active Theme Clusters", stats["active_clusters"], stats["cluster_delta"])
    with c3:
        render_metric_card("Emerging Themes", stats["emerging_themes"], stats["emerging_delta"])
    with c4:
        render_metric_card("Avg Cluster Risk", stats["avg_risk"], stats["risk_delta"])

    st.markdown("")

    left, right = st.columns([1.45, 1])

    with left:
        render_info_card(
            "Executive Intelligence Brief",
            md_to_html(brief),
            tags=["Gemini Summary", "Cluster Monitoring", "Risk Review"],
        )

    with right:
        if top_clusters:
            render_ranked_list_card("Top Risk Clusters", top_clusters)
        else:
            render_info_card("Top Risk Clusters", "<p>No cluster data available.</p>", tags=["Live Data"])

    st.markdown("")

    col1, col2, col3 = st.columns([1.35, 1, 1])

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <div class="section-label">Fraud Signal Trend</div>
                <div class="placeholder-note" style="margin-bottom: 0.75rem;">
                    Weekly article ingestion volume across monitored fraud themes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.line_chart(trend_df, use_container_width=True)

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <div class="section-label">Cluster Stage Distribution</div>
                <div class="placeholder-note" style="margin-bottom: 0.75rem;">
                    Current distribution of monitored cluster groups by lifecycle stage.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if stage_dist:
            stage_order = ["Emerging", "Stable", "Trending", "Declining",
                           "Accelerating", "Expanding", "Established"]
            ordered = [(s, stage_dist[s]) for s in stage_order if s in stage_dist]
            others  = [(s, v) for s, v in stage_dist.items() if s not in stage_order]
            rows = ordered + others
            df_stage = pd.DataFrame(rows, columns=["Stage", "Clusters"])
            df_stage["_order"] = range(len(df_stage))

            base = alt.Chart(df_stage).encode(
                x=alt.X("Stage:N",
                        sort=alt.EncodingSortField(field="_order", order="ascending"),
                        axis=alt.Axis(labelAngle=-30, title="")),
                y=alt.Y("Clusters:Q", axis=alt.Axis(title="", grid=False)),
            )
            bars = base.mark_bar(color="#1565C0", cornerRadiusTopLeft=4,
                                 cornerRadiusTopRight=4)
            labels = base.mark_text(dy=-8, fontSize=12, fontWeight="bold",
                                    color="#0D2B52").encode(
                text=alt.Text("Clusters:Q")
            )
            chart = (bars + labels).properties(height=220).configure_view(
                strokeWidth=0
            ).configure_axis(domain=False)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Stage data unavailable.")

    with col3:
        ev_name  = top_evidence.get("theme_label") or (top_clusters[0]["name"] if top_clusters else "No data")
        ev_risk  = top_evidence.get("risk_score")
        ev_stage = (top_evidence.get("stage") or "—").capitalize()
        ev_count = top_evidence.get("article_count") or "—"
        raw_evidence = top_evidence.get("theme_evidence")

        # theme_evidence is now a list of {title, url} dicts resolved from doc_ids
        # Show only top 3 to keep the card clean
        top_evidence_items = raw_evidence[:3] if isinstance(raw_evidence, list) else []
        if top_evidence_items and isinstance(top_evidence_items[0], dict) and "title" in top_evidence_items[0]:
            evidence_html = "".join(
                f'<li><a href="{e["url"]}" target="_blank" style="color:#0D2B52">{_esc(e["title"])}</a></li>'
                if e.get("url") else f'<li>{_esc(e["title"])}</li>'
                for e in top_evidence_items
            )
        elif top_evidence_items:
            evidence_html = "".join(f"<li>{_esc(str(e))}</li>" for e in top_evidence_items)
        else:
            evidence_html = "<li>No evidence data available for this cluster</li>"

        total_evidence = len(raw_evidence) if isinstance(raw_evidence, list) else 0
        more_html = (
            f'<p style="color:#667085;font-size:0.82rem;margin-top:0.5rem;">'
            f'+ {total_evidence - 3} more articles in this cluster</p>'
        ) if total_evidence > 3 else ""

        render_info_card(
            "Evidence Snapshot",
            f"""
            <p><b>Theme:</b> {ev_name}</p>
            <p>
                <b>Risk:</b> {round(ev_risk, 1) if ev_risk is not None else "—"}
                &nbsp;|&nbsp; <b>Stage:</b> {ev_stage}
                &nbsp;|&nbsp; <b>Articles:</b> {ev_count}
            </p>
            <p><b>Top Signals:</b></p>
            <ul>{evidence_html}</ul>
            {more_html}
            """,
            tags=["Live Data", "theme_evidence"],
        )

    st.markdown("")

    accel = [c for c in all_clusters if c["stage"].lower() == "accelerating"]
    emerging = [c for c in all_clusters if c["stage"].lower() == "emerging"]
    priority = (accel + emerging)[:4] or all_clusters[:4]

    stage_color = {"Accelerating": "#D92D20", "Emerging": "#F79009"}

    st.markdown(
        '<div class="info-card" style="padding-bottom:0.25rem;">'
        '<div class="section-label">Themes Requiring Attention</div>'
        '<div style="margin-bottom:0.75rem">'
        '<span class="tag">Live Data</span>'
        '<span class="tag">Priority Review</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if priority:
        for i, c in enumerate(priority):
            color = stage_color.get(c["stage"], "#667085")
            expand_key = f"exec_theme_expand_{i}"
            if expand_key not in st.session_state:
                st.session_state[expand_key] = False

            is_open = st.session_state[expand_key]
            radius_bottom = "0 0 0 0" if is_open else "0 8px 8px 8px"

            st.markdown(
                f'<div style="padding:0.55rem 0.75rem;border-left:4px solid {color};'
                f'background:#FAFBFC;border-radius:0 8px {radius_bottom};margin-bottom:0;">'
                f'<div style="font-weight:700;color:#0D2B52;font-size:0.93rem;">{_esc(c["name"])}</div>'
                f'<div style="color:#667085;font-size:0.84rem;margin-top:0.15rem;">'
                f'Risk: <b>{c["risk"]}</b> &nbsp;|&nbsp; '
                f'Articles: <b>{c["articles"]}</b> &nbsp;|&nbsp; '
                f'<span style="color:{color};font-weight:600;">{c["stage"]}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            if is_open:
                desc = (c.get("description") or "No description available.")[:300]
                audience = _classify_audience(c)
                st.markdown(
                    f'<div style="padding:0.6rem 0.75rem 0.6rem 1rem;'
                    f'background:#F0F4F8;border-left:4px solid {color};'
                    f'border-radius:0 0 8px 8px;margin-bottom:0.5rem;">'
                    f'<p style="color:#344054;margin:0 0 0.4rem;font-size:0.88rem;">{_esc(desc)}</p>'
                    f'<span style="background:#E8F4FD;color:#0D2B52;padding:0.15rem 0.55rem;'
                    f'border-radius:20px;font-size:0.8rem;font-weight:600;">'
                    f'🎯 Target: {_esc(audience)}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            btn_label = "▲ Close" if is_open else "▼ Details"
            if st.button(btn_label, key=f"exec_theme_btn_{i}",
                         use_container_width=False):
                st.session_state[expand_key] = not is_open
                st.rerun()

            if not is_open:
                st.markdown('<div style="margin-bottom:0.5rem;"></div>',
                            unsafe_allow_html=True)
    else:
        st.markdown(
            '<p class="placeholder-note">No high-priority cluster data available.</p>',
            unsafe_allow_html=True,
        )


def page_intelligence_search():
    """
    Render the Intelligence Search page — RAG analyst workbench.

        The primary entry point for the connected workflow.  When a search is
        run, the result dict is stored in st.session_state["rag_result"] and
        st.session_state["last_query"] for consumption by downstream pages.

        Layout:
          - Suggested searches row (Vertex AI generated, DB validated)
          - Search controls: query input, document count slider,
            time horizon selector, minimum risk threshold slider
          - Analyze Intelligence button
          - Four result KPI cards (retrieved, dominant cluster, similarity, risk)
          - Two-column result area:
              Left:  AI Generated Intelligence brief (Gemini, md_to_html rendered)
              Right: Search Parameters card | Cluster Alignment Summary card
          - Supporting Articles section (one render_rag_article_card per source)

    """
    render_page_header(
        "Intelligence Search",
        "Query fraud schemes, actors, entities, and organizations to generate synthesized intelligence and retrieve supporting source material.",
    )

    # Suggested searches — generated from actual cluster landscape via Vertex AI
    with st.spinner("Loading suggested searches..."):
        suggestions = generate_suggested_searches()

    if suggestions:
        st.markdown(
            '<div style="margin-bottom:0.5rem;">'
            '<span style="font-size:0.82rem;font-weight:600;color:#667085;'
            'text-transform:uppercase;letter-spacing:0.05em;">Suggested Searches</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        sug_cols = st.columns(len(suggestions))
        for col, suggestion in zip(sug_cols, suggestions):
            with col:
                if st.button(
                    suggestion,
                    key=f"sug_{suggestion[:40]}",
                    use_container_width=True,
                    help=f"Search: {suggestion}",
                ):
                    st.session_state["last_query"] = suggestion
                    st.rerun()

    st.markdown("")

    col1, col2, col3, col4 = st.columns([2.2, 1, 1, 1])

    with col1:
        query = st.text_input(
            "Search Query",
            value=st.session_state.get("last_query", ""),
            placeholder="Search fraud schemes, actors, organizations, or typologies...",
        )
    with col2:
        top_k = st.slider("Documents to Retrieve", 1, 15, 5, key="intelligence_search_top_k")
    with col3:
        time_horizon = st.selectbox(
            "Time Horizon",
            ["Last 30 Days", "Last 90 Days", "Last 12 Months", "All Available"],
            index=1,
        )
    with col4:
        risk_threshold = st.slider(
            "Min Risk Threshold",
            min_value=0.0, max_value=2.0, value=0.0, step=0.1,
            key="intelligence_risk_threshold",
            help="Filter supporting articles to only those at or above this risk score.",
        )

    analyze_clicked = st.button("Analyze Intelligence", type="primary")

    if analyze_clicked and query:
        with st.spinner("Running RAG pipeline — embedding query, retrieving documents, generating brief..."):
            result = run_rag(query, top_k=top_k)
        st.session_state["rag_result"] = result
        st.session_state["last_query"] = query

    result = st.session_state.get("rag_result")

    st.markdown("")

    if result:
        sources = result.get("sources", [])
        # Apply risk threshold filter — clamp negatives to 0 before comparing
        risk_threshold = st.session_state.get("intelligence_risk_threshold", 0.0)
        if risk_threshold > 0.0:
            sources = [
                s for s in sources
                if max(0.0, float(s.get("risk_score") or 0)) >= risk_threshold
            ]
        retrieved = result.get("retrieved_count", 0)
        dominant = result.get("dominant_cluster") or "—"
        avg_sim = result.get("avg_similarity")
        avg_risk = result.get("avg_risk")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            retrieved_label = f"{len(sources)} of {retrieved}" if risk_threshold > 0.0 else str(retrieved)
            render_metric_card("Documents Retrieved", retrieved_label, "Top semantic matches")
        with c2:
            render_metric_card("Dominant Cluster", dominant, "Highest coherence")
        with c3:
            render_metric_card("Avg Similarity", str(avg_sim) if avg_sim is not None else "—", "Across retrieved set")
        with c4:
            render_metric_card("Avg Risk Signal", str(avg_risk) if avg_risk is not None else "—", "Retrieved theme match")

        st.markdown("")

        left, right = st.columns([1.75, 1])

        with left:
            brief_text = result["answer"]
            # Safety net: if Gemini ignored the word limit, truncate at the 4th section boundary
            # or at 1500 chars, whichever comes first
            section_markers = ["**Analyst Note**", "Analyst Note\n", "## Analyst Note"]
            for marker in section_markers:
                idx = brief_text.find(marker)
                if idx != -1:
                    # Find end of the Analyst Note sentence
                    end = brief_text.find("\n\n", idx + len(marker))
                    brief_text = brief_text[:end].strip() if end != -1 else brief_text[:idx + 400].strip()
                    break
            if len(brief_text) > 1500:
                brief_text = brief_text[:1500].rsplit("\n", 1)[0] + "\n\n*Brief truncated for display.*"

            render_info_card(
                "AI Generated Intelligence",
                md_to_html(brief_text),
                tags=["RAG Output", "Gemini", dominant[:30] if dominant != "—" else ""],
            )

        with right:
            render_info_card(
                "Search Parameters",
                f"""
                <ul>
                    <li><b>Query:</b> {result["query"]}</li>
                    <li><b>Documents Retrieved:</b> {retrieved}</li>
                    <li><b>Time Horizon:</b> {time_horizon}</li>
                    <li><b>Result Mode:</b> semantic retrieval + Gemini synthesis</li>
                </ul>
                """,
                tags=["Run Context"],
            )

            if sources:
                st.markdown("")
                theme_counts = Counter(s.get("theme_label") for s in sources if s.get("theme_label"))
                top_theme, top_count = theme_counts.most_common(1)[0] if theme_counts else ("—", 0)
                render_info_card(
                    "Cluster Alignment Summary",
                    f"""
                    <ul>
                        <li><b>Primary Cluster:</b> {top_theme}</li>
                        <li><b>Articles in Cluster:</b> {top_count} of {retrieved}</li>
                        <li><b>Avg Risk Score:</b> {avg_risk if avg_risk is not None else "—"}</li>
                        <li><b>Avg Similarity:</b> {avg_sim if avg_sim is not None else "—"}</li>
                    </ul>
                    """,
                    tags=["Cluster Match", "Live Data"],
                )

        st.markdown("")
        if sources:
            st.markdown("#### Supporting Articles")
            for doc in sources:
                render_rag_article_card(doc)

    else:
        # Empty state
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card("Documents Retrieved", "—", "Run a query to begin")
        with c2:
            render_metric_card("Dominant Cluster", "—", "Awaiting query")
        with c3:
            render_metric_card("Avg Similarity", "—", "Awaiting query")
        with c4:
            render_metric_card("Avg Risk Signal", "—", "Awaiting query")

        st.markdown("")
        render_info_card(
            "Analyst Search Workbench",
            "<p>Enter a query above and click <b>Analyze Intelligence</b> to retrieve documents and generate a synthesized intelligence brief.</p>",
            tags=["Ready"],
        )


def page_fraud_pattern_clusters():
    """
    Render the Fraud Pattern Clusters page — pattern analysis layer.

        Context-aware: reads st.session_state["rag_result"] to determine state.

        When a search is active (rag_result present):
          - Fetches full cluster details for the search result cluster IDs via
            fetch_clusters_by_ids() to get accurate article counts and descriptions
          - Renders a blue context banner showing the active query
          - Top cluster cards, KPI metrics, and ranked list all reflect ONLY
            the clusters retrieved by that search

        When no search is active:
          - Falls back to fetch_top_risk_clusters(limit=10) global view
          - Standard top-3 cards, global KPIs, full ranked list

    """
    render_page_header(
        "Fraud Pattern Clusters",
        "Monitor how embedding-based pattern groups form, grow, and shift over time as related fraud signals accumulate across sources.",
    )

    rag_result = st.session_state.get("rag_result")
    search_query = st.session_state.get("last_query", "")

    # Collect unique cluster IDs from search sources
    search_cluster_ids = tuple(
        dict.fromkeys(
            s["cluster_id"] for s in (rag_result.get("sources") or [])
            if s.get("cluster_id") is not None
        )
    ) if rag_result else ()

    if search_cluster_ids:
        # Fetch full details for search clusters
        with st.spinner("Loading cluster data..."):
            search_clusters = fetch_clusters_by_ids(search_cluster_ids)
            all_clusters = fetch_top_risk_clusters(limit=10)

        st.markdown(
            f'<div style="background:#F0F7FF;border-left:4px solid #1565C0;'
            f'padding:0.6rem 0.9rem;border-radius:0 8px 8px 0;margin-bottom:1rem;">'
            f'<span style="font-size:0.83rem;color:#1565C0;font-weight:600;">'
            f'Clusters retrieved for: </span>'
            f'<span style="font-size:0.83rem;color:#0D2B52;">'
            f'&ldquo;{_esc(search_query)}&rdquo;</span></div>',
            unsafe_allow_html=True,
        )

        # Top cluster cards from search
        top3 = search_clusters[:3] if search_clusters else []
        if top3:
            cols = st.columns(len(top3))
            for i, cluster in enumerate(top3):
                with cols[i]:
                    desc = cluster.get("description") or (
                        "Articles in this cluster share overlapping fraud language and "
                        "evidence signals identified through semantic embedding."
                    )
                    render_info_card(
                        cluster["name"],
                        f"<p>{_esc(desc)}</p>",
                        tags=[
                            f"Articles: {cluster['articles']}",
                            f"Risk: {cluster['risk']}",
                            f"Stage: {cluster['stage']}",
                        ],
                    )

        st.markdown("")

        # KPIs from search clusters
        avg_risk = round(
            sum(float(c["risk"]) for c in search_clusters) / len(search_clusters), 1
        ) if search_clusters else 0
        emerging = sum(1 for c in search_clusters if c["stage"].lower() == "emerging")

        m1, m2, m3 = st.columns(3)
        with m1:
            render_metric_card("Retrieved Clusters", str(len(search_clusters)), "From your search")
        with m2:
            render_metric_card("Avg Cluster Risk", str(avg_risk), "Across retrieved clusters")
        with m3:
            render_metric_card("Emerging Clusters", str(emerging), "Early-stage themes")

        st.markdown("")

        if search_clusters:
            render_ranked_list_card(
                f"Retrieved Clusters — \"{_esc(search_query)}\"",
                search_clusters,
            )

    else:
        # Default state — no search run yet
        with st.spinner("Loading cluster data..."):
            all_clusters = fetch_top_risk_clusters(limit=10)

        st.markdown(f"#### Cluster Landscape — Top {min(len(all_clusters), 10)} Themes by Risk")

        top3 = all_clusters[:3] if all_clusters else []
        cols = st.columns(max(len(top3), 1))
        for i, cluster in enumerate(top3):
            with cols[i]:
                desc = cluster.get("description") or (
                    "Articles in this cluster share overlapping language around fraud mechanics, "
                    "operational patterns, and repeated evidence signals identified through semantic embedding."
                )
                render_info_card(
                    cluster["name"],
                    f"<p>{desc}</p>",
                    tags=[
                        f"Articles: {cluster['articles']}",
                        f"Risk: {cluster['risk']}",
                        f"Stage: {cluster['stage']}",
                    ],
                )

        st.markdown("")

        avg_risk = round(sum(float(c["risk"]) for c in all_clusters) / len(all_clusters), 1) if all_clusters else 0
        emerging = sum(1 for c in all_clusters if c["stage"].lower() == "emerging")

        m1, m2, m3 = st.columns(3)
        with m1:
            render_metric_card("Tracked Clusters", str(len(all_clusters)), "From live data")
        with m2:
            render_metric_card("Avg Cluster Risk", str(avg_risk), "Across tracked groups")
        with m3:
            render_metric_card("Emerging Clusters", str(emerging), "Early-stage themes")

        st.markdown("")

        if all_clusters:
            render_ranked_list_card("All Tracked Clusters — Ranked by Risk", all_clusters)


def page_network_relationships():
    """
    Render the Network Relationships page — relationship exploration layer.

        Context-aware: reads st.session_state["rag_result"] for dominant cluster
        and excluded cluster IDs.  Falls back to the highest-risk cluster when
        no search result is present (with an info banner prompting a search).

        Layout:
          - Connection Strength Range slider (filters constellation in real time)
          - Four KPI cards: connected clusters, strongest link, avg similarity,
            high-similarity links
          - Strongest Connection card: Gemini-generated fraud profile +
            convergence explanation for the top connected pair
          - Two-column section:
              Left:  Cluster Relationship Map card + Plotly constellation chart
                     + click-to-detail card (populated when a node is selected)
              Right: How Relationships Are Computed explanatory card

        Constellation data is fetched from fetch_constellation_data() which uses
        neighbor-appearance frequency (not raw similarity score) as the connection
        strength metric.  Filtering by the slider is applied at display time,
        not in the cached fetch, to avoid re-querying Supabase on every slider move.

    """
    render_page_header(
        "Network Relationships",
        "Explore how fraud themes connect through shared article neighbors, overlapping evidence language, and proximity in embedding space.",
    )

    # Determine dominant cluster and excluded cluster IDs from Intelligence Search session
    rag_result = st.session_state.get("rag_result")
    if rag_result and rag_result.get("dominant_cluster"):
        dominant_label = rag_result["dominant_cluster"]
        excluded_ids = tuple(
            s["cluster_id"] for s in (rag_result.get("sources") or [])
            if s.get("cluster_id") is not None
        )
        using_rag = True
    else:
        fallback = fetch_top_risk_clusters(limit=1)
        dominant_label = fallback[0]["name"] if fallback else None
        excluded_ids = ()
        using_rag = False

    if not dominant_label:
        st.warning("No cluster data available to build the relationship map.")
        return

    with st.spinner("Building constellation from article neighbor data..."):
        constellation_data_all = fetch_constellation_data(dominant_label, excluded_ids)

    # Derive slider bounds from actual data so the user always starts with
    # all their data visible regardless of what scores exist in the DB
    if constellation_data_all:
        all_sims = [c["similarity"] for c in constellation_data_all]
        data_min = min(all_sims)
        data_max = max(all_sims)
    else:
        data_min, data_max = 0.0, 1.0

    sim_range = st.slider(
        "Connection Strength Range",
        min_value=0.0, max_value=1.0,
        value=(0.20, 0.99),
        step=0.01,
        key="network_sim_range",
        help="Connection strength is the normalized frequency with which each cluster "
             "appears as a neighbor of the dominant cluster's articles (1.0 = most frequent). "
             "Lower the floor to include more distant clusters.",
    )
    sim_min, sim_max = sim_range

    # Filter locally — no Supabase re-query needed
    constellation_data = [
        c for c in constellation_data_all
        if sim_min <= c["similarity"] <= sim_max
    ]

    # KPI row — derived from constellation data
    n_connected = len(constellation_data)
    sims = [c["similarity"] for c in constellation_data]
    avg_sim = round(sum(sims) / len(sims), 2) if sims else 0.0
    strongest_sim = sims[0] if sims else 0.0
    high_sim_count = sum(1 for s in sims if s >= 0.8)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Connected Clusters", str(n_connected), "Related by neighbor similarity")
    with m2:
        render_metric_card("Strongest Link", str(strongest_sim), "Highest similarity score")
    with m3:
        render_metric_card("Avg Link Similarity", str(avg_sim), "Across all connections")
    with m4:
        render_metric_card("High-Similarity Links", str(high_sim_count), "Similarity \u2265 0.80")

    # Fetch dominant cluster description for card and center node
    with st.spinner("Loading cluster details..."):
        try:
            dom_theme_rows = (
                get_supabase()
                .table("cluster_themes")
                .select("theme_description")
                .eq("theme_label", dominant_label)
                .limit(1)
                .execute()
                .data or []
            )
            dominant_desc = (dom_theme_rows[0].get("theme_description") or "") if dom_theme_rows else ""
        except Exception:
            dominant_desc = ""

    st.markdown("")

    # Strongest Connection -- full width, above the map
    if constellation_data:
        top = constellation_data[0]
        top_desc = top.get("description") or "No description available."

        with st.spinner("Generating cluster analysis..."):
            sim_brief = generate_cluster_similarity_brief(
                dominant_label, dominant_desc,
                top["cluster_name"], top_desc,
            )

        render_info_card(
            "Strongest Connection",
            f'<p style="margin-bottom:0.5rem;">'
            f'<b>{_esc(dominant_label)}</b> and <b>{_esc(top["cluster_name"])}</b> share '
            f'the strongest connection with a strength score of <b>{top["similarity"]}</b>.</p>'
            f'<p style="font-size:0.9rem;color:#344054;font-weight:600;margin:0.6rem 0 0.2rem;">'
            f'Fraud Profile — {_esc(top["cluster_name"])}</p>'
            f'<p style="font-size:0.88rem;color:#667085;margin-bottom:0.6rem;">'
            f'{_esc(sim_brief["fraud_profile"])}</p>'
            f'<p style="font-size:0.9rem;color:#344054;font-weight:600;margin:0.6rem 0 0.2rem;">'
            f'Why These Clusters Converge</p>'
            f'<p style="font-size:0.88rem;color:#667085;">'
            f'{_esc(sim_brief["convergence_reason"])}</p>',
            tags=["Strongest Bridge", "Live Data"],
        )
        st.markdown("")

    if not using_rag:
        st.info(
            "No Intelligence Search result found. Showing relationships for the highest-risk "
            "cluster. Run a search on the Intelligence Search page to focus this view on your query.",
        )
        st.markdown("")

    # Constellation + methodology side by side
    left, right = st.columns([1.8, 1])

    with left:
        st.markdown(
            f'<div class="info-card">'
            f'<div class="section-label">Cluster Relationship Map</div>'
            f'<div class="placeholder-note" style="margin-bottom:0.5rem;">'
            f'Semantic neighbors of <b>{_esc(dominant_label)}</b>. '
            f'Distance from center reflects connection strength — closer means more related. '
            f'Strength range: {sim_min}–{sim_max}.'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        chart_event = render_cluster_constellation(
            dominant_label, dominant_desc, constellation_data
        )

        # Click-to-detail card
        selected_name, selected_desc = None, None
        if chart_event and hasattr(chart_event, "selection"):
            pts = chart_event.selection.points if chart_event.selection else []
            if pts:
                cd = pts[0].get("customdata")
                if cd and len(cd) >= 2:
                    selected_name, selected_desc = cd[0], cd[1]

        if selected_name:
            st.markdown("")
            is_center = selected_name == dominant_label
            border_color = "#F79009" if is_center else "#20B07A"
            label = "Dominant Cluster" if is_center else "Selected Cluster"
            st.markdown(
                f'<div class="info-card" style="border-left:4px solid {border_color};">'
                f'<div style="font-size:0.78rem;color:#667085;text-transform:uppercase;'
                f'letter-spacing:0.05em;margin-bottom:0.3rem;">{label}</div>'
                f'<div class="section-label" style="margin-bottom:0.5rem;">'
                f'{_esc(selected_name)}</div>'
                f'<div style="font-size:0.88rem;color:#344054;">'
                f'{_esc(selected_desc)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif constellation_data:
            st.markdown(
                '<div style="color:#667085;font-size:0.85rem;text-align:center;'
                'padding:0.5rem 0;">Click any node to view cluster details</div>',
                unsafe_allow_html=True,
            )

    with right:
        render_info_card(
            "How Relationships Are Computed",
            "<p>Each connection is built by counting how frequently articles from other "
            "clusters appear as neighbors of the dominant cluster's articles across all "
            "rank positions in the neighbor graph.</p>"
            "<p>Connection strength is the normalized count — a cluster that appears as "
            "a neighbor 400 times scores higher than one that appears 50 times, regardless "
            "of rank. This makes the metric robust even when raw similarity scores are "
            "uniformly high across the corpus.</p>"
            "<p>Clusters excluded from the constellation include the dominant cluster itself "
            "and any clusters retrieved in the current Intelligence Search result set.</p>",
        )



def page_alerts_watchlists():
    """
    Render the Alerts & Watchlists page — operational monitoring layer.

        Context-aware: reads st.session_state["rag_result"] to build a search
        context filter that limits the alert feed to clusters from the last search.

        Alert matching uses both cluster_id and theme_label (normalised lowercase)
        for robustness.  The search filter toggle resets to True automatically
        when a new search is run, tracked via alerts_last_known_query session key.

        KPI card counts (Total / High / Medium / Low) reflect the active filter
        state — they show the filtered counts when search filter is on, and global
        counts when off.  KPIs are computed from display_alerts before rendering.

        Alert feed display:
          - First occurrence of each cluster: full card with description +
            Gemini-generated 1-3 action recommendations (batch-generated).
          - Repeat occurrences: compact single-line row preserving the event
            history without duplicating the description and recommendations.

    """
    render_page_header(
        "Alerts & Watchlists",
        "Review triggered alerts by severity, explore fraud descriptions, and act on recommended safeguards.",
    )

    with st.spinner("Loading alerts..."):
        alert_stats   = fetch_alert_stats()
        recent_alerts = fetch_recent_alerts(limit=200)

    # Read RAG context from Intelligence Search session
    rag_result = st.session_state.get("rag_result")
    search_query = st.session_state.get("last_query", "")

    # Build lookup sets from search sources — match by both cluster_id and theme_label
    rag_cluster_ids: set = set()
    rag_cluster_labels: set = set()
    if rag_result and rag_result.get("sources"):
        for s in rag_result["sources"]:
            if s.get("cluster_id") is not None:
                rag_cluster_ids.add(s["cluster_id"])
            if s.get("theme_label"):
                rag_cluster_labels.add(s["theme_label"].lower().strip())

    # Reset search filter toggle whenever the active query changes
    prev_query_key = "alerts_last_known_query"
    if st.session_state.get(prev_query_key) != search_query:
        st.session_state["alerts_use_search_filter"] = bool(rag_cluster_ids)
        st.session_state[prev_query_key] = search_query

    has_search_context = bool(rag_cluster_ids or rag_cluster_labels) and bool(search_query)
    using_search_filter = st.session_state.get("alerts_use_search_filter", False)

    # Pre-filter alerts for search context
    def _matches_search(a: dict) -> bool:
        # Match an alert against the search result set by cluster_id OR theme_label
        # (normalised lowercase).  Dual matching handles cases where the same cluster
        # is referenced under different IDs in alerts vs article_analysis.
        if a.get("cluster_id") in rag_cluster_ids:
            return True
        label = (a.get("theme_label") or "").lower().strip()
        return label in rag_cluster_labels

    if using_search_filter and has_search_context:
        display_alerts = [a for a in recent_alerts if _matches_search(a)]
    else:
        display_alerts = recent_alerts

    # Compute KPI counts from the display set (not global stats)
    kpi_total  = len(display_alerts)
    kpi_high   = sum(1 for a in display_alerts if a.get("severity") == "high")
    kpi_medium = sum(1 for a in display_alerts if a.get("severity") == "medium")
    kpi_low    = sum(1 for a in display_alerts if a.get("severity") == "low")
    kpi_unread = sum(1 for a in display_alerts if not a.get("is_read"))

    # --------------------------------------------------
    # KPI Row — clickable severity filters
    # --------------------------------------------------
    if "alert_severity_filter" not in st.session_state:
        st.session_state.alert_severity_filter = "all"

    active = st.session_state.alert_severity_filter
    sev_options = [
        ("all",    "Total Alerts",    str(kpi_total),  f"{kpi_unread} unread"),
        ("high",   "High Severity",   str(kpi_high),   "Require escalation"),
        ("medium", "Medium Severity", str(kpi_medium), "Under review"),
        ("low",    "Low Severity",    str(kpi_low),    "Monitor"),
    ]

    cols = st.columns(4)
    for col, (key, label, value, delta) in zip(cols, sev_options):
        with col:
            render_metric_card(label, value, delta)
            btn_label = f"✓ {'All' if key == 'all' else key.capitalize()}" if active == key else ('All' if key == 'all' else key.capitalize())
            if st.button(
                btn_label,
                key=f"sev_filter_{key}",
                use_container_width=True,
                type="primary" if active == key else "secondary",
            ):
                st.session_state.alert_severity_filter = key
                st.rerun()

    # Search context banner + toggle
    if has_search_context:
        col_banner, col_toggle = st.columns([3, 1])
        with col_banner:
            st.markdown(
                f'<div style="background:#F0F7FF;border-left:4px solid #1565C0;'
                f'padding:0.5rem 0.9rem;border-radius:0 8px 8px 0;margin-top:0.4rem;">'
                f'<span style="font-size:0.83rem;color:#1565C0;font-weight:600;">'
                f'Search context: </span>'
                f'<span style="font-size:0.83rem;color:#0D2B52;">'
                f'&ldquo;{_esc(search_query)}&rdquo;</span></div>',
                unsafe_allow_html=True,
            )
        with col_toggle:
            toggle_label = "✓ Search clusters" if using_search_filter else "All clusters"
            if st.button(toggle_label, key="alerts_search_toggle",
                         use_container_width=True,
                         type="primary" if using_search_filter else "secondary"):
                st.session_state["alerts_use_search_filter"] = not using_search_filter
                st.rerun()

    st.markdown("")

    # --------------------------------------------------
    # Apply severity filter to display set
    # --------------------------------------------------
    if active == "all":
        filtered = display_alerts
    else:
        filtered = [a for a in display_alerts if a.get("severity") == active]

    filtered = sorted(filtered, key=lambda x: float(x.get("risk_score") or 0), reverse=True)

    # --------------------------------------------------
    # Batch generate recommendations — deduplicate by cluster name
    # so Gemini isn't called with the same cluster multiple times
    # --------------------------------------------------
    seen_names: set = set()
    unique_clusters = []
    for a in filtered:
        if a["theme_label"] not in seen_names:
            seen_names.add(a["theme_label"])
            unique_clusters.append((
                a["theme_label"],
                a.get("theme_description") or "",
                a.get("severity") or "low",
            ))
    cluster_batch = tuple(unique_clusters)
    with st.spinner("Generating action recommendations..."):
        recommendations = generate_alert_recommendations(cluster_batch)

    # --------------------------------------------------
    # Alert Feed — full width, enriched cards
    # --------------------------------------------------
    severity_colors = {"high": "#D92D20", "medium": "#F79009", "low": "#12B76A"}

    if filtered:
        st.markdown(
            f'<div class="info-card"><div class="section-label">Alert Feed'
            f'<span style="color:#667085;font-size:0.82rem;font-weight:400;margin-left:0.5rem;">'
            f'— {len(filtered)} alert{"s" if len(filtered) != 1 else ""}, ranked by risk</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        seen_clusters: set = set()

        for a in filtered:
            sev   = a.get("severity", "low")
            color = severity_colors.get(sev, "#667085")
            risk  = round(float(a.get("risk_score") or 0), 1)
            ts    = (a.get("triggered_at") or "")[:16].replace("T", " ")
            label = a["theme_label"]
            is_first = label not in seen_clusters
            seen_clusters.add(label)

            if is_first:
                # Full card — description + recommendations
                desc = a.get("theme_description") or "No description available."
                recs = recommendations.get(label, [])
                rec_html = ""
                if recs:
                    rec_items = "".join(
                        f'<li style="margin-bottom:0.25rem;">{_esc(r)}</li>'
                        for r in recs[:3]
                    )
                    rec_html = (
                        f'<div style="margin-top:0.6rem;">'
                        f'<div style="font-size:0.82rem;font-weight:600;color:#344054;'
                        f'margin-bottom:0.3rem;">Recommended Actions</div>'
                        f'<ul style="margin:0;padding-left:1.2rem;color:#344054;'
                        f'font-size:0.85rem;">{rec_items}</ul></div>'
                    )
                st.markdown(
                    f'<div class="info-card" style="margin-bottom:0.75rem;'
                    f'border-left:4px solid {color};">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;margin-bottom:0.4rem;">'
                    f'<div style="font-weight:700;color:#0D2B52;font-size:0.97rem;">'
                    f'{_esc(label)}</div>'
                    f'<div style="display:flex;align-items:center;gap:0.5rem;">'
                    f'<span style="color:#667085;font-size:0.82rem;">Risk: <b>{risk}</b></span>'
                    f'<span style="background:{color}22;color:{color};padding:0.15rem 0.55rem;'
                    f'border-radius:999px;font-size:0.78rem;font-weight:700;">'
                    f'{sev.upper()}</span></div></div>'
                    f'<div style="font-size:0.87rem;color:#344054;margin-bottom:0.3rem;">'
                    f'{_esc(desc)}</div>'
                    f'<div style="color:#98A2B3;font-size:0.8rem;">{ts}</div>'
                    f'{rec_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Compact repeat card — event history without redundant detail
                st.markdown(
                    f'<div style="padding:0.5rem 0.75rem;margin-bottom:0.4rem;'
                    f'border-left:3px solid {color}44;background:#FAFBFC;'
                    f'border-radius:0 8px 8px 0;display:flex;justify-content:space-between;'
                    f'align-items:center;">'
                    f'<div style="font-size:0.86rem;color:#667085;">'
                    f'<span style="font-weight:600;color:#0D2B52;">{_esc(label)}</span>'
                    f' &nbsp;·&nbsp; repeat alert &nbsp;·&nbsp; {ts}</div>'
                    f'<span style="background:{color}22;color:{color};padding:0.1rem 0.45rem;'
                    f'border-radius:999px;font-size:0.75rem;font-weight:700;">'
                    f'{sev.upper()}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        render_info_card(
            "Alert Feed",
            f'<p>No {"alerts" if active == "all" else active + " severity alerts"} found.</p>',
        )



# --------------------------------------------------
# App Router
# --------------------------------------------------

def main():
    """App entry point — renders the sidebar and routes to the selected page function."""

    if selected_page == "Executive Overview":
        page_executive_overview()
    elif selected_page == "Intelligence Search":
        page_intelligence_search()
    elif selected_page == "Fraud Pattern Clusters":
        page_fraud_pattern_clusters()
    elif selected_page == "Network Relationships":
        page_network_relationships()
    elif selected_page == "Alerts & Watchlists":
        page_alerts_watchlists()


if __name__ == "__main__":
    main()
