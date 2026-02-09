DSBA6390-Fraud-Detection-Emerging-Fraud-Signals-Dashbaord-
Final capstone project for MS in Data Science and Business Analytics 

Project Members: Owen Williamson, Anders Pierson, Baran Narravula, Declan O'Halloran, and Rituparna Bhattacharya 


Overview
This project is an intelligence-driven fraud R&D pipeline designed to identify emerging financial fraud patterns using publicly available data sources. The system focuses on early signal detection, trend analysis, and contextual insights to support proactive fraud research.

Product Goals
- Detect emerging fraud patterns from public consumer and regulatory data
- Aggregate signals across multiple authoritative sources
- Support fraud research through structured analysis and visualization
  

Scope Boundaries
**In Scope**
- Public fraud and complaint data analysis
- Trend identification and exploratory analytics
- Research-oriented insights and visualization

**Out of Scope**
- Real-time transaction-level fraud detection
- Production-grade ML deployment
- Access to proprietary or sensitive data


 High-Level Workflow
1. Ingest public fraud-related datasets
2. Clean and normalize data across sources
3. Perform exploratory and trend-based analysis
4. Generate insights on emerging fraud patterns


MVP Constraints
- Uses only publicly available and aggregated data
- AI/LLM usage limited by API capacity and cost constraints
- Focused on research feasibility rather than production scalability

Technology Stack
**Current**
- Python (Pandas, NumPy)
- Jupyter Notebook
- Data visualization libraries

**Exploratory**
- LLM APIs for research augmentation
- Supabase for data storage and visualization


Data Sources

We selected a mix of authoritative government datasets and industry research sources to ensure our analysis is based on real-world, ethical, and reliable information.
- Federal Trade Commission (FTC): Consumer-reported fraud and identity theft data, including scam types, financial losses, and demographic trends. Chosen for its credibility and real-world relevance.
- Consumer Financial Protection Bureau (CFPB): Structured consumer complaint data related to financial products and fraud. Selected for clean, well-documented CSV data suitable for analysis and modeling.
- Office of the Comptroller of the Currency (OCC): Banking performance and risk indicators. Used to provide system-level context to consumer fraud trends.
- Financial Crimes Enforcement Network (FinCEN): Aggregated reports on financial crime and AML trends. Included to understand national-level fraud patterns.
- Financial Industry Regulatory Authority (FINRA): Enforcement and regulatory data related to securities markets. Adds an investment and market-fraud perspective.
- PYMNTS & Outseer: Industry research and reports used for qualitative context and interpretation of fraud and payments trends.
