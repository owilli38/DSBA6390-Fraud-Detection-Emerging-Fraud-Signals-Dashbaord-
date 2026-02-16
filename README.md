DSBA6390-Fraud-Detection-Emerging-Fraud-Signals-Dashbaord-
Final capstone project for MS in Data Science and Business Analytics 

Project Members: Owen Williamson, Anders Pierson, Baran Narravula, Declan O'Halloran, and Rituparna Bhattacharya 


**Overview**
This project is an intelligence-driven fraud R&D pipeline designed to identify emerging financial fraud patterns using publicly available data sources. The system focuses on early signal detection, trend analysis, and contextual insights to support proactive fraud research.

**Product Goals**
- Detect emerging fraud patterns from public consumer and regulatory data
- Aggregate signals across multiple authoritative sources
- Support fraud research through structured analysis and visualization
  

**Scope Boundaries**
**In Scope:**
- Public fraud and complaint data analysis
- Trend identification and exploratory analytics
- Research-oriented insights and visualization

**Out of Scope:**
- Real-time transaction-level fraud detection
- Production-grade ML deployment
- Access to proprietary or sensitive data


 **High-Level Workflow**
1. Ingest public fraud-related datasets
2. Clean and normalize data across sources
3. Perform exploratory and trend-based analysis
4. Generate insights on emerging fraud patterns


**MVP Constraints**
- Uses only publicly available and aggregated data
- AI/LLM usage limited by API capacity and cost constraints
- Focused on research feasibility rather than production scalability

**Technology Stack**
**Current:**
- Python (Pandas, NumPy)
- Jupyter Notebook
- Data visualization libraries

**Exploratory:**
- LLM APIs for research augmentation
- Supabase for data storage and visualization


**Fraud Signal Processing & Embedding Pipeline**

To support emerging fraud signal detection, we built a lightweight pipeline that converts public fraud datasets into searchable embeddings, ranked signals, and explainable insights.

**Pipeline Steps**
**Data Ingestion & Cleaning**
- Public datasets (FTC, CFPB, OCC, etc.) are loaded from CSV, JSON, and text sources. All records are converted into standardized natural language text and cleaned so everything is consistent across sources.
**Embedding Generation (Gemini)**
- Each record is converted into a semantic embedding using Google Gemini embedding models so fraud cases can be compared based on meaning instead of keywords.
**Supabase Backend Storage**
- Embeddings and cleaned records are stored in Supabase for retrieval and downstream modeling
**Similarity Search & Retrieval**
- Vector similarity search is used to find related fraud cases and surface clusters of similar activity.
**ML Ranking & Severity Classification**
- Random Forest models are used to:
- Rank fraud signals by importance
- Classify severity (Low, Medium, High)
**RAG-Based Fraud Insights (Gemini LLM)**
- Retrieved records are passed into a Gemini-based RAG workflow to generate short summaries, patterns, and evidence-backed insights.

**Supabase Backend Integration**

To support storage and querying outside of the notebook, we set up a Supabase backend that connects directly to our Python workflow.

- Cleaned fraud records and embeddings are inserted into Supabase tables
- Read and write operations were tested and validated from Python
- Stored records include document ID, text content, and embedding data
- Queries return structured data that can be used for similarity search, ranking, and RAG steps

This allows us to move beyond local notebook storage and keep all fraud records in one place that the rest of the system can access.

**How it fits into the workflow**
1. Cleaned fraud data is created in Python
2. Records and embeddings are inserted into Supabase
3. Supabase stores the data with unique IDs
4. The query layer retrieves records for:
  - similarity search
  - model ranking
  - RAG-based insights

This backend layer keeps the data organized and makes it easier to connect the modeling pipeline to a future dashboard.


**Output**
Each generated fraud signal includes:
- document ID
- similarity score
- supporting text evidence
- severity classification
- ranking score

This output feeds directly into dashboards and research analysis.


**Current Progress**
- Built ingestion pipeline for FTC and CFPB datasets
- Cleaned and standardized data for consistent processing
- Set up Supabase backend and confirmed read/write from Python
- Generated embeddings using the Gemini API
- Ran initial similarity search and ranking models on fraud records
- Integrated an early-stage RAG workflow for summarizing fraud signals

**Data Sources**

We selected a mix of authoritative government datasets and industry research sources to ensure our analysis is based on real-world, ethical, and reliable information.
- Federal Trade Commission (FTC): Consumer-reported fraud and identity theft data, including scam types, financial losses, and demographic trends. Chosen for its credibility and real-world relevance.
- Consumer Financial Protection Bureau (CFPB): Structured consumer complaint data related to financial products and fraud. Selected for clean, well-documented CSV data suitable for analysis and modeling.
- Office of the Comptroller of the Currency (OCC): Banking performance and risk indicators. Used to provide system-level context to consumer fraud trends.
- Financial Crimes Enforcement Network (FinCEN): Aggregated reports on financial crime and AML trends. Included to understand national-level fraud patterns.
- Financial Industry Regulatory Authority (FINRA): Enforcement and regulatory data related to securities markets. Adds an investment and market-fraud perspective.
- PYMNTS & Outseer: Industry research and reports used for qualitative context and interpretation of fraud and payments trends.
