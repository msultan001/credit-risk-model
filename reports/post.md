# Building a Basel II Compliant Credit Risk Model for the Unbanked

## Introduction
In the world of finance, credit scores are the gatekeepers of opportunity. But what happens when you don't have a credit history? In emerging markets, millions of creditworthy individuals are excluded from the financial system simply because they lack a paper trail.

In this project, I built a production-grade machine learning pipeline to assess credit risk using alternative data—specifically, mobile money transaction history.

## The Proxy Problem
The biggest challenge was the lack of labels. We didn't have a dataset of "defaulted" vs. "paid back" loans. We had transaction logs.

To solve this, I employed **RFM Analysis (Recency, Frequency, Monetary)** logic to engineer a proxy target variable:
1.  **Recency**: How long since the last transaction?
2.  **Frequency**: How often do they transact?
3.  **Monetary**: How much do they move?

We clustered customers using K-Means. The cluster with high recency (inactive) and low frequency/monetary (low engagement) was labeled as "High Risk". This allowed us to train a supervised model to predict this behavior in new customers.

## Engineering Excellence
Building a model in a notebook is one thing; building a maintainable system is another.

### key Architectural Decisions:
-   **Pydantic Configuration**: I moved away from hardcoded strings to a robust `Settings` class. This ensures type safety and easy environment switching.
-   **Modular Design**: The pipeline is broken down into `DataLoader`, `FeatureEngineer`, and `ModelTrainer` classes. This follows the Single Responsibility Principle.
-   **Automated Testing**: a robust suite of `pytest` unit tests ensures that refactoring doesn't break logic.
-   **CI/CD**: GitHub Actions automatically run tests on every push, ensuring the build is always green.

## Model Interpretability & Basel II
Regulators don't like "black boxes". To comply with frameworks like Basel II, we need to explain *why* a customer was rejected.

I implemented **SHAP (SHapley Additive exPlanations)** values in the dashboard. This allows risk officers to see exactly which features drove the model's decision—whether it was the pricing strategy, the transaction hour, or the specific provider used.

## Conclusion
This project demonstrates that financial inclusion and rigorous risk management are not mutually exclusive. By leveraging modern MLOps and interpretable AI, we can build systems that are both inclusive and safe.

[Check out the code on GitHub](https://github.com/Start-Tech-Academy/Credit-Risk-Model)
