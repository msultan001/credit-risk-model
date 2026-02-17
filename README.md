# Credit Risk Modeling for Financial Inclusion 🚀

![CI/CD Pipeline](https://github.com/msultan001/credit-risk-model/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A production-grade machine learning pipeline for credit risk assessment in emerging markets. This project demonstrates end-to-end MLOps practices, from data engineering to model deployment and interactive dashboards.

## 📌 Business Context

### The Challenge
Financial institutions in emerging markets often lack traditional credit bureau data (FICO scores), leading to financial exclusion for millions of unbanked individuals. To bridge this gap, we use alternative data—specifically mobile money transaction history—to assess creditworthiness.

### Regulatory Compliance (Basel II)
This solution is designed with **Basel II** compliance in mind:
- **Interpretability**: We prioritize explainable models (Logistic Regression, SHAP values) to satisfy regulatory requirements for transparency.
- **Risk Management**: The model helps in calculating risk-weighted assets by providing accurate Probability of Default (PD) estimates.

### The Proxy Variable strategy
We derive a proxy for credit risk based on **RFM (Recency, Frequency, Monetary)** analysis:
- **High Risk**: Inactive customers (High Recency) with low usage (Low Frequency/Monetary).
- **Low Risk**: Active, consistent users.
This approach allows us to label potential defaulters even without historical loan data.

---

## 🛠️ Project Structure

```bash
credit-risk-model/
├── .github/workflows/   # CI/CD Pipeline (GitHub Actions)
├── data/                # Data storage (gitignored)
├── models/              # Serialized models and artifacts
├── notebooks/           # Jupyter notebooks for experimentation
├── reports/             # Generated reports and presentations
├── src/                 # Source code
│   ├── config.py        # Centralized configuration
│   ├── dashboard.py     # Streamlit interactive dashboard
│   ├── data_processing.py # Feature engineering pipeline
│   ├── predict.py       # Inference engine
│   ├── rfm_analysis.py  # Label engineering
│   └── train.py         # Model training & MLflow tracking
├── tests/               # Unit tests
├── Dockerfile           # Containerization
├── requirements.txt     # Dependencies
└── README.md            # You are here
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker (optional)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/credit-risk-model.git
   cd credit-risk-model
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Windows users might need to install C++ Build Tools for some libraries.*

4. **Run the Dashboard**
   ```bash
   streamlit run src/dashboard.py
   ```

---

## 💻 Usage

### 1. Training the Model
Run the full training pipeline, which includes data loading, feature engineering, RFM analysis, oversampling (SMOTE), and model evaluation.
```bash
python src/train.py
```
*Artifacts will be saved to `models/` and metrics logged to MLflow.*

### 2. Making Predictions
Use the CLI to make predictions on new data:
```bash
python src/predict.py
```

### 3. Interactive Dashboard
Explore the data and explain model decisions:
```bash
streamlit run src/dashboard.py
```

---

## 📊 Key Features

- **Engineering Excellence**:
  - Modular, object-oriented code with type hinting.
  - Centralized configuration using `pydantic`.
  - Comprehensive unit tests with `pytest`.
  
- **MLOps Integration**:
  - Experiment tracking with **MLflow**.
  - CI/CD pipeline for automated testing.
  - Docker support for reproducible environments.

- **Advanced Modeling**:
  - **RFM Analysis** for unsupervised labeling.
  - **SMOTE** for handling class imbalance.
  - **WoE (Weight of Evidence)** transformation for categorical features.
  - **XGBoost** for high-performance classification.
  - **SHAP** values for model interpretability.

---

## 📈 Results

| Model | ROC-AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| Logistic Regression | 0.82 | 0.65 | 0.78 | 0.71 |
| Random Forest | 0.88 | 0.72 | 0.75 | 0.73 |
| **XGBoost** | **0.91** | **0.78** | **0.80** | **0.79** |

*Note: Results based on initial validation set. Run `src/train.py` for latest metrics.*

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
