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
1. Ingest public fraud-related datasets (Data Pipeline)
2. Clean and normalize data across sources (Data Pipeline)
3. Store structured data in Supabase for downstream use
4. Generate semantic embeddings and representations (Embedding Pipeline)
5. Cluster and analyze fraud signals over time
6. Generate insights on emerging fraud patterns


**MVP Constraints**
- Uses only publicly available and aggregated data
- AI/LLM usage limited by API capacity and cost constraints
- Focused on research feasibility rather than production scalability

**Technology Stack**
**Current:**
- Python (Pandas, NumPy)
- Jupyter Notebook
- Data visualization libraries
- Supabase (data storage and retrieval)
- Google Gemini API (embedding generation)
- Hugging Face Transformers (BERT embeddings)
- Scikit-learn (clustering, preprocessing)

**Exploratory:**
- LLM APIs for research augmentation
- Supabase for data storage and visualization


**Fraud Signal Processing & Embedding Pipeline**

To support emerging fraud signal detection, we built a lightweight pipeline that converts cleaned fraud datasets into semantic embeddings, clustered signals, and explainable insights.

**Pipeline Steps**
**Data Ingestion & Cleaning**
Before embeddings are generated, all raw datasets are processed through a data ingestion pipeline:

- Data is loaded from CSV, JSON, and text sources (FTC, CFPB, OCC, etc.)
- News and blog sites are scrapped on a batch, daily schedule for retrieving new articles with relevant metadata 
- Records are standardized into a unified schema
- Text fields are cleaned and normalized for consistency: lower casing and normalizing white space
- Duplicate records are removed using unique identifiers
- Missing or incomplete records are filtered out
- 
This ensures all downstream embedding and modeling steps operate on clean, structured data.

**Embedding Generation (Gemini)**
- Each record is converted into a semantic embedding using Google Gemini embedding models so fraud cases can be compared based on meaning instead of keywords.
- BERT generates contextual embeddings used for clustering and analysis
**Supabase Backend Storage**
- Embeddings and cleaned records are stored in Supabase for retrieval and downstream modeling
**Similarity Search & Retrieval**
- Vector similarity search is used to find related fraud cases and surface clusters of similar activity.
- DBSCAN/HDBSCAN identifies clusters of related fraud signals and isolates noise points that may indicate emerging or anomalous activity
**Unique Measurements for Emerging Threats**
  - Drift measures the change in the clusters centroid movement, which can indicate changes to emerging threat (i.e, phone scams become AI-powered phone scams)
  - Growth measures number of new articles within the cluster
  - Acceleration measures the difference in growth from previous week to current week to signify if topic is becoming a growing/decline threat
  - Age Days is used to classify the newness of a fraud threat
        - 0-30 Days is "Emerging"
        - 30-90 Days is "Trending"
        - Beyond 180 Days is "Established"
**Cluster Tracking & Stage Labeling**
Each processed fraud signal includes:
- document ID
- source
- cleaned text
- embedding representation
- cluster assignment
- supporting article or nearest representative article
- temporal metrics such as growth, drift, acceleration, and age_days
- stage label (for example: Emerging, Trending, Stable, or Established)
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
1. Raw fraud data is ingested and cleaned (Data Pipeline)
2. Cleaned records are stored in Supabase
3. Embeddings are generated and stored
4. Supabase serves as the central data layer
5. The system retrieves records for:
   - similarity search
   - clustering and trend detection
   - RAG-based insight generation

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

We use publicly available news and blog content from four primary sources:
- ABA (American Bankers Association)
- PYMNTS
- TechCrunch
- ACFE (Association of Certified Fraud Examiners)

These sources were selected to provide a mix of banking, payments, fraud, and technology coverage for identifying emerging fraud-related patterns.
