import streamlit as st
import requests

# --------------------------------------------
# API Endpoints
# --------------------------------------------

RAG_API = "http://127.0.0.1:8000/rag"
LATEST_API = "http://127.0.0.1:8000/latest"

# --------------------------------------------
# Page Setup
# --------------------------------------------

st.set_page_config(
    page_title="Fraud Intelligence Platform",
    layout="wide"
)

# --------------------------------------------
# Clean Professional Styling
# --------------------------------------------

st.markdown("""
<style>

/* hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* page spacing */
.block-container {
    padding-top: 2rem;
}

/* card layout */
.card {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #e6e6e6;
    height: 180px;
    transition: all .15s ease;
}

.card:hover {
    border-color: #4A90E2;
    box-shadow: 0 3px 10px rgba(0,0,0,0.08);
}

/* card text */
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

/* AI answer box */
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
# Title
# --------------------------------------------

st.title("Fraud Intelligence Platform")

st.markdown(
    "Search fraud intelligence reports and retrieve relevant investigative articles."
)

st.divider()

# --------------------------------------------
# Search Section
# --------------------------------------------

st.subheader("Intelligence Search")

col1, col2, col3 = st.columns([1,3,1])

with col2:

    query = st.text_input(
        "",
        placeholder="Search fraud schemes, actors, organizations..."
    )

    top_k = st.slider(
        "Documents to retrieve",
        1,
        10,
        5
    )

    search_button = st.button("Analyze Intelligence")

st.divider()

# --------------------------------------------
# RAG Search Results
# --------------------------------------------

if search_button and query:

    payload = {
        "query": query,
        "top_k": top_k
    }

    with st.spinner("Analyzing fraud intelligence..."):

        response = requests.post(RAG_API, json=payload)

        if response.status_code == 200:

            data = response.json()

            st.subheader("AI Generated Intelligence")

            st.markdown(f"""
            <div class="answer-box">
            {data["answer"]}
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            st.subheader("Supporting Articles")

            for src in data["sources"]:

                st.markdown(f"""
                <div class="card">
                    <div class="card-title">{src["title"]}</div>
                    <div class="card-date">Source Article</div>
                    <a class="card-link" href="{src["url"]}" target="_blank">
                        Read Article →
                    </a>
                </div>
                """, unsafe_allow_html=True)

        else:

            st.error("Error contacting the RAG API")

# --------------------------------------------
# Default Page: Latest Articles Grid
# --------------------------------------------

else:

    st.subheader("Latest Fraud Intelligence")

    try:

        response = requests.get(LATEST_API)

        if response.status_code == 200:

            articles = response.json()["articles"]

            articles = articles[:6]

            rows = [articles[i:i+3] for i in range(0, len(articles), 3)]

            for row in rows:

                cols = st.columns(3)

                for col, article in zip(cols, row):

                    with col:

                        st.markdown(f"""
                        <div class="card">
                            <div class="card-title">{article["title"]}</div>
                            <div class="card-date">{article["publish_timestamp"]}</div>
                            <a class="card-link" href="{article["url"]}" target="_blank">
                                Read Article →
                            </a>
                        </div>
                        """, unsafe_allow_html=True)

        else:

            st.warning("Could not load articles.")

    except:

        st.warning("Backend not available.")
