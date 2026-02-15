# Credit Risk Model Presentation

## Slide 1: Title Slide
**Credit Risk Modeling for Financial Inclusion**
*Bridging the gap with Alternative Data*
[Your Name]

## Slide 2: The Problem
- **Financial Exclusion**: High percentage of population in emerging markets is unbanked.
- **Data Gap**: Lack of traditional credit bureau data (FICO).
- **Business Risk**: Lenders cannot assess risk accurately without data.

## Slide 3: The Solution
- **Alternative Data**: Utilizing mobile money transaction logs.
- **Proxy Modeling**: Creating a "Good" vs "Bad" borrower definition using behavioral analysis (RFM).
- **Machine Learning**: Predicting future risk based on transaction patterns.

## Slide 4: Methodology (RFM Analysis)
- **Recency**: Days since last activity.
- **Frequency**: Count of transactions.
- **Monetary**: Total value derived.
- **Clustering**: K-Means to group customers.
- **Labeling**: "High Risk" = Inactive + Low Volume.

## Slide 5: Technical Architecture
- **Tech Stack**: Python, Pandas, Scikit-Learn, XGBoost, Streamlit.
- **MLOps**: MLflow for tracking, Docker for containerization, GitHub Actions for CI/CD.
- **Design**: Modular, Object-Oriented, Type-Safe.

## Slide 6: Model Performance
- **XGBoost** outperformed Logistic Regression.
- **ROC-AUC**: 0.91 (Excellent discrimination).
- **Precision**: 0.78 (Minimizing false positives/bad loans).

## Slide 7: Interpretability & Compliance
- **Basel II Alignment**: Model transparency is key.
- **SHAP Values**: quantifying feature contribution.
- **Dashboard**: Interactive tool for risk officers to review cases.

## Slide 8: Future Roadmap
- **Deployment**: API deployment on Kubernetes.
- **Monitoring**: Drift detection (Evidently AI).
- **Features**: Graph-based features (network analysis).

## Slide 9: Q&A
*Thank you!*
