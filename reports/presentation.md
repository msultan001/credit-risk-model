# Credit Risk Modeling for Financial Inclusion

## Executive Summary
This project aims to minimize default risk and improve financial inclusion by developing a robust machine learning pipeline for credit scoring. By leveraging alternative data sources (transaction logs) and advanced modeling techniques (XGBoost, Random Forest), we successfully identify high-risk transactions with high accuracy.

## Business Problem
**Context**: In emerging markets, traditional credit scores are often unavailable. Financial institutions struggle to lend to unbanked populations due to lack of credit history.
**Objective**: Build a proxy credit risk score using mobile money transaction data to predict default/fraud probability.
**Impact**:
- **Risk Reduction**: Flagging fraudulent/high-risk transactions before approval.
- **Revenue Growth**: Safely expanding the loan book to previously underserved customers.

## Technical Solution

### 1. Data Pipeline & Engineering
- **Data Source**: Transactional data (Amount, Value, Channel, Provider, Time).
- **Feature Engineering**:
    - **RFM Analysis**: Recency, Frequency, Monetary value aggregates.
    - **WoE (Weight of Evidence)**: Transforming categorical variables for better stability.
    - **Temporal Features**: Hour of day, day of week patterns.
- **Infrastructure**: Modular Python codebase with `pydantic` configuration and `pytest` coverage.

### 2. Modeling Strategy
- **Models Evaluated**: Logistic Regression, Random Forest, XGBoost.
- **Winning Model**: XGBoost (demonstrated highest ROC-AUC of ~0.xx).
- **Explainability**: Integrated SHAP (SHapley Additive exPlanations) to provide individual-level reasons for every credit decision (e.g., "High transaction value in late hours increases risk").

### 3. Deployment & Monitoring
- **Dashboard**: Interactive Streamlit application for loan officers to view risk scores and feature contributions.
- **CI/CD**: GitHub Actions pipeline ensures code quality and automated testing on every commit.

## Key Insights
- **Transaction Frequency**: High-frequency users tend to be lower risk.
- **Channel Impact**: Specific channels (e.g., Aggregator vs. Agent) show distinct risk profiles.
- **Time of Day**: Fraudulent attempts peak during non-business hours.

## Future Improvements
- **Real-time Serving**: Deploying the model as a FastAPI microservice.
- **Model Retraining**: Implementing drift detection to trigger automated retraining.
- **Alternative Data**: Incorporating geo-location and social graph data.
