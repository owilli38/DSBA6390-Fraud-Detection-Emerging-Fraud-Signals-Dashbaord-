import streamlit as st
import requests
import pandas as pd

# --------------------------------------------
# API ENDPOINTS
# --------------------------------------------

RAG_API = "http://127.0.0.1:8000/rag"
LATEST_API = "http://127.0.0.1:8000/latest"

# --------------------------------------------
# PAGE SETUP
# --------------------------------------------

st.set_page_config(
    page_title="Fraud Intelligence Platform",
    layout="wide"
)

# --------------------------------------------
# STYLING
# --------------------------------------------

st.markdown("""
<style>

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container {
    padding-top: 2rem;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #e6e6e6;
    height: 200px;
}

.card-title {
    font-size: 17px;
    font-weight: 600;
    margin-bottom: 10px;
}

.card-date {
    font-size: 13px;
    color: #666;
    margin-bottom: 10px;
}

.card-link {
    font-size: 14px;
    color: #1f77b4;
    text-decoration: none;
}

.answer-box {
    background-color: #f9fafc;
    border: 1px solid #e6e6e6;
    padding: 25px;
    border-radius: 10px;
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------
# TITLE
# --------------------------------------------

st.title("Fraud Intelligence Platform")
st.markdown("Dual-mode fraud intelligence: investigation + threat monitoring")

st.divider()

# --------------------------------------------
# TABS (KEY CHANGE)
# --------------------------------------------

tab1, tab2 = st.tabs(["🔎 Investigate", "📈 Threat Timeline"])

# =========================================================
# TAB 1 — INVESTIGATION (RAG SEARCH)
# =========================================================

with tab1:

    st.subheader("Intelligence Search")

    col1, col2, col3 = st.columns([1,3,1])

    with col2:

        query = st.text_input(
            "",
            placeholder="Search fraud schemes, actors, organizations..."
        )

        top_k = st.slider("Documents to retrieve", 1, 10, 5)

        search_button = st.button("Analyze Intelligence")

    st.divider()

    if search_button and query:

        payload = {
            "query": query,
            "top_k": top_k
        }

        with st.spinner("Analyzing fraud intelligence..."):

            response = requests.post(RAG_API, json=payload)

            if response.status_code == 200:

                data = response.json()
                sources = data["sources"]

                st.subheader("AI Intelligence Report")

                st.markdown(f"""
                <div class="answer-box">
                {data["answer"]}
                </div>
                """, unsafe_allow_html=True)

                st.divider()

                st.subheader("Supporting Articles")

                for src in sources:

                    st.markdown(f"""
                    <div class="card">
                        <div class="card-title">{src["title"]}</div>
                        <div class="card-date">
                            Theme: {src.get("theme_label","Unknown")}<br>
                            Stage: {src.get("stage","Unknown")}<br>
                            Risk Score: {src.get("risk_score","Unknown")}
                        </div>
                        <a class="card-link" href="{src["url"]}" target="_blank">
                            Read Article →
                        </a>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.error("Error contacting RAG API")

# =========================================================
# TAB 2 — THREAT TIMELINE DASHBOARD
# =========================================================

with tab2:

    st.subheader("Fraud Threat Timeline (Real Intelligence View)")

    try:
        response = requests.get("http://127.0.0.1:8000/timeline")

        if response.status_code == 200:

            articles = response.json()["timeline"]

            if not articles:
                st.warning("No timeline data available")
            else:

                df = pd.DataFrame(articles)

                df["publish_timestamp"] = pd.to_datetime(
                    df["publish_timestamp"],
                    errors="coerce"
                )

                df = df.dropna(subset=["publish_timestamp"])

                df["date"] = df["publish_timestamp"].dt.date

                # 🔥 REAL THREAT SIGNAL: risk-weighted activity
                df["risk_score"] = df["risk_score"].fillna(0)

                trend = df.groupby("date")["risk_score"].mean().reset_index()

                st.line_chart(trend.set_index("date")["risk_score"])

                st.divider()

                st.subheader("Latest Threat Intelligence Feed")

                for item in articles[:6]:

                    st.markdown(f"""
                    <div class="card">
                        <div class="card-title">{item["title"]}</div>
                        <div class="card-date">
                            Theme: {item.get("theme_label","Unknown")}<br>
                            Risk: {item.get("risk_score","Unknown")}
                        </div>
                        <a class="card-link" href="{item["url"]}" target="_blank">
                            Read Article →
                        </a>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.warning("Failed to load timeline data")

    except Exception as e:
        st.error(f"Timeline error: {e}")