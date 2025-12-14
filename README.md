# Credit Risk Model

## Project Overview
This project implements a comprehensive credit risk modeling system for fraud detection and risk assessment in financial services. The analysis is based on transaction data from a financial services provider in Africa.

---

## Credit Scoring Business Understanding

### Basel II Regulatory Framework and Model Interpretability

**Basel II Influence on Model Selection**

The Basel II Accord, specifically the Internal Ratings-Based (IRB) approach, mandates that financial institutions must demonstrate robust, transparent, and interpretable credit risk models to regulators. This regulatory requirement significantly impacts our model selection process:

1. **Transparency Requirements**: Basel II requires that risk models be explainable to regulators, auditors, and stakeholders. This means we must be able to clearly articulate how our model makes predictions and which factors drive credit decisions.

2. **Model Validation**: Regulators must be able to validate model logic and ensure it aligns with sound credit risk principles. Complex "black box" models make this validation challenging.

3. **Risk Weight Calculations**: Basel II uses model outputs to calculate capital requirements. Interpretable models allow institutions to understand and justify their capital allocation decisions.

4. **Documentation Standards**: The framework requires comprehensive documentation of model development, validation, and ongoing monitoring - easier with interpretable approaches.

### Proxy Variable: TotalTransactionValue and Business Risks

**Why We Need a Proxy Variable**

Traditional credit scoring relies on credit history (FICO scores, payment history, credit utilization). However, in markets with limited credit bureau data or for customers without traditional banking relationships, we need alternative indicators of creditworthiness.

**Chosen Proxy: TotalTransactionValue**

We use `TotalTransactionValue` (aggregated transaction activity per customer) as a proxy for creditworthiness because:
- Higher transaction volumes suggest active financial engagement
- Consistent transaction patterns indicate financial stability
- Transaction diversity reflects broader financial ecosystem participation

**Business Risks Associated with This Proxy**

1. **Survivor Bias**: High transaction volumes might represent desperate borrowing rather than creditworthiness
2. **Circular Logic**: Excluding individuals without transaction history perpetuates financial exclusion
3. **Gaming Risk**: Sophisticated fraudsters might artificially inflate transaction volumes
4. **Economic Shocks**: Transaction patterns during crises may not reflect true risk profiles
5. **Data Quality**: Transaction data may be incomplete or manipulated
6. **Regulatory Scrutiny**: Using non-traditional proxies requires additional regulatory justification

### Trade-offs: Interpretable vs Complex Models

| Aspect | Interpretable Models (Logistic Regression, Decision Trees) | Complex Models (XGBoost, Neural Networks) |
|--------|-----------------------------------------------------------|------------------------------------------|
| **Regulatory Compliance** | ✅ Easier to explain to regulators | ❌ Difficult to justify decisions |
| **Model Performance** | ⚠️ May sacrifice some accuracy | ✅ Typically higher predictive power |
| **Feature Importance** | ✅ Clear coefficient interpretation | ⚠️ SHAP values add complexity |
| **Debugging** | ✅ Easy to identify model issues | ❌ Hard to diagnose problems |
| **Trust** | ✅ Stakeholders understand logic | ❌ "Black box" concerns |
| **Adversarial Robustness** | ✅ Harder to game | ❌ Vulnerable to adversarial attacks |
| **Implementation** | ✅ Simple deployment | ⚠️ Requires specialized infrastructure |
| **Monitoring** | ✅ Easy to track drift | ❌ Complex drift patterns |

**Our Recommendation**

We adopt a **hybrid approach**:
1. Start with interpretable **Logistic Regression** as baseline (regulatory-friendly)
2. Develop **XGBoost** for production (superior performance)
3. Use **SHAP values** to explain complex model decisions
4. Maintain **model governance framework** for ongoing validation

This balances Basel II compliance with competitive model performance.

---

## Project Structure

```
credit-risk-model/
├── data/
│   ├── raw/              # Original data files (gitignored)
│   └── processed/        # Cleaned datasets (gitignored)
├── notebooks/
│   └── eda.ipynb         # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py  # Feature engineering & preprocessing
│   ├── train.py            # Model training pipeline
│   └── predict.py          # Inference functions
├── api/
│   ├── main.py             # FastAPI application
│   └── pydantic_models.py  # API data validation
├── tests/
│   └── test_data_processing.py  # Unit tests
├── .github/workflows/
│   └── ci.yml              # CI/CD pipeline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup Instructions

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd credit-risk-model
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run EDA Notebook**
   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```

### Docker Deployment

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`

---

## Usage

### Training Models

```bash
python src/train.py
```

### Running API

```bash
uvicorn api.main:app --reload
```

### Running Tests

```bash
pytest tests/ --cov=src
```

---

## Key Insights from EDA

Detailed exploratory data analysis can be found in `notebooks/eda.ipynb`. Key findings include:
- Transaction patterns vary significantly across product categories
- Fraud rate is approximately 0.03% of all transactions
- High-value transactions show elevated fraud risk
- Channel and provider combinations influence risk profiles

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD |

---

## API Endpoints

- `GET /` - Health check
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /model/info` - Model metadata

---

## Contributing

1. Create feature branch
2. Make changes with tests
3. Submit pull request
4. Ensure CI/CD passes

---

## License

MIT License
