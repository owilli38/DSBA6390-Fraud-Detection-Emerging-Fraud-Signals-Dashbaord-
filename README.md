# DSBA6390-Fraud-Detection-Emerging-Fraud-Signals-Dashboard

Final capstone project for the MS in Data Science and Business Analytics

Project Members: Owen Williamson, Anders Pierson, Baran Narravula, Declan O'Halloran, and Rituparna Bhattacharya

---

## Overview

This project is an intelligence-driven fraud research and development pipeline designed to identify emerging financial fraud patterns using publicly available data sources. The system focuses on early signal detection, trend analysis, and contextual insights to support proactive fraud research.

---

## Product Goals

* Detect emerging fraud patterns from public consumer and regulatory data
* Aggregate signals across multiple authoritative sources
* Support fraud research through structured analysis and visualization

---

## Scope Boundaries

### In Scope

* Public fraud and complaint data analysis
* Trend identification and exploratory analytics
* Research-oriented insights and visualization

### Out of Scope

* Real-time transaction-level fraud detection
* Production-grade ML deployment
* Access to proprietary or sensitive data

---

## High-Level Workflow

1. Ingest public fraud-related datasets
2. Clean and normalize data across sources
3. Store structured data in Supabase for downstream use
4. Generate semantic embeddings and representations
5. Cluster and analyze fraud signals over time
6. Generate insights on emerging fraud patterns

---

## MVP Constraints

* Uses only publicly available and aggregated data
* AI/LLM usage limited by API capacity and cost constraints
* Focused on research feasibility rather than production scalability

---

## Technology Stack

### Current

* Python (Pandas, NumPy)
* Jupyter Notebook
* Data visualization libraries
* Supabase (data storage and retrieval)
* Google Gemini API (embedding generation and RAG)
* Hugging Face Transformers (BERT embeddings)
* Scikit-learn (clustering and preprocessing)

### Exploratory

* LLM APIs for research augmentation
* Extended Supabase integration for analytics workflows

---

## Fraud Signal Processing & Embedding Pipeline

To support emerging fraud signal detection, the system converts cleaned fraud datasets into semantic embeddings, clustered signals, and explainable insights.

### Data Ingestion & Cleaning

* Load data from CSV, JSON, and text sources (FTC, CFPB, OCC, etc.)
* Scrape news and blog sources on a batch schedule
* Standardize records into a unified schema
* Clean and normalize text (lowercasing, whitespace normalization)
* Remove duplicate records using unique identifiers
* Filter incomplete or missing records

This ensures all downstream modeling operates on consistent, structured data.

---

### Embedding Generation

* Each record is converted into a semantic embedding using Gemini models
* BERT embeddings are generated for clustering and contextual analysis

---

### Supabase Backend Storage

* Cleaned records and embeddings are stored in Supabase
* Supports structured queries, retrieval, and downstream workflows

---

### Similarity Search & Clustering

* Vector similarity search identifies related fraud cases
* DBSCAN/HDBSCAN groups similar fraud signals
* Noise points may indicate emerging or anomalous activity

---

### Unique Measurements for Emerging Threats

* Drift: Measures changes in cluster meaning over time
* Growth: Number of new articles within a cluster
* Acceleration: Change in growth rate between time periods
* Age Days:

  * 0–30 Days: Emerging
  * 30–90 Days: Trending
  * Beyond 90 Days: Established

---

### Cluster Tracking & Stage Labeling

Each fraud signal includes:

* document ID
* source
* cleaned text
* embedding representation
* cluster assignment
* supporting article
* temporal metrics (growth, drift, acceleration, age_days)
* stage label (Emerging, Trending, Stable, Established)

---

### RAG-Based Fraud Insights

* Retrieved records are passed into a Gemini-based RAG workflow
* Generates summaries, patterns, and evidence-backed insights

---

## Supabase Backend Integration

The backend supports storage, retrieval, and analysis across the full pipeline:

* Stores fraud records and embeddings
* Enables similarity search and clustering workflows
* Serves as the central data layer for dashboard integration

### Workflow Integration

1. Raw data ingestion and cleaning
2. Storage in Supabase
3. Embedding generation
4. Retrieval for similarity search, clustering, and RAG insights

---

## Output

Each generated fraud signal includes:

* document ID
* similarity score
* supporting text evidence
* severity classification
* ranking score

This output feeds directly into dashboard visualizations and research workflows.

---

## Current Progress

* Built ingestion pipeline for FTC and CFPB datasets
* Cleaned and standardized data across sources
* Implemented Supabase backend and validated read/write operations
* Generated embeddings using Gemini API
* Performed clustering and similarity analysis
* Integrated a RAG-based workflow for fraud signal summarization

---

## Key Results

* Processed over 1,400 fraud-related articles
* Identified over 100 fraud signal clusters
* Detected 40+ emerging fraud themes
* Achieved clustering silhouette score of approximately 0.54
* Enabled semantic retrieval with average similarity around 0.67

---

## Data Sources

Publicly available news and blog content from:

* ABA (American Bankers Association)
* PYMNTS
* TechCrunch
* ACFE (Association of Certified Fraud Examiners)

---

## Final Deliverables

The following documents are included in the `docs/` folder:

* Configuration & Access Guide
* Data Documentation
* Model Card
* RAG Documentation
* System Overview & Architecture Diagram
* Executive Summary
* Test Plan
* User Guide
* Runbook & Operations Guide

---

## Team Roles & Responsibilities

* Anders Pierson — Project Lead (sponsor liaison, scope, timelines, integration, testing)
* Rituparna Bhattacharya — Data Lead (data acquisition, cleaning, documentation, quality workflow)
* Owen Williamson — Model/AI Lead (ML, NLP, RAG development, integration, evaluation)
* Declan O’Halloran — Product/Engineering Lead (dashboard, visualization, application development, integration)
* Baran Narravula — Documentation & Communication Lead (README, updates, demo narrative, final deliverables)

---

## Limitations

* Batch-based ingestion rather than real-time updates
* Limited coverage of external data sources
* No supervised evaluation dataset for validation
* Heuristic-based stage classification
* Prototype-level system not designed for production deployment

---

## Future Work

* Implement real-time or scheduled automated ingestion
* Expand data source coverage
* Improve NLP enrichment (NER, tagging, classification)
* Develop formal evaluation benchmarks
* Add authentication and multi-user support
* Integrate with enterprise fraud monitoring systems

---
