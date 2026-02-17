"""
Credit Risk Analysis Dashboard.
Built with Streamlit for interactive data exploration and model inference.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
from pathlib import Path

# Add project root to path for imports
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.config import settings
from src.data_processing import DataLoader, FeatureEngineer, DataPreprocessor
from src.predict import ModelPredictor

# Page Config
st.set_page_config(
    page_title=settings.DASHBOARD_TITLE,
    page_icon="💳",
    layout="wide"
)

# Title and Description
st.title(settings.DASHBOARD_TITLE)
st.markdown("""
This dashboard provides an interactive overview of the Credit Risk Model, including:
- **Exploratory Data Analysis (EDA)**: Understand the data distribution.
- **Model Performance**: Evaluation metrics and confusion matrices.
- **Individual Prediction**: Test the model with custom inputs.
- **Explainability**: SHAP values to understand feature importance.
""")

@st.cache_data
def load_data():
    """Load and process data for dashboard."""
    loader = DataLoader()
    raw_data = loader.load_data()
    
    # Engineer features for analysis
    engineer = FeatureEngineer(raw_data)
    featured_data = engineer.engineer_all_features()
    
    return raw_data, featured_data

@st.cache_resource
def load_predictor():
    """Load model predictor."""
    try:
        return ModelPredictor()
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None

def main():
    # Load Data
    try:
        raw_data, featured_data = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Load Model
    predictor = load_predictor()

    # Sidebar Navigation
    page = st.sidebar.radio("Navigation", ["Overview", "Data Analysis", "Model Performance", "Prediction & SHAP"])

    if page == "Overview":
        st.header("Project Overview")
        
        # Key Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", len(raw_data))
        with col2:
            st.metric("Total Customers", raw_data['CustomerId'].nunique())
        with col3:
            fraud_count = raw_data[settings.TARGET_COL].sum()
            st.metric("Fraud Cases", f"{fraud_count} ({fraud_count/len(raw_data):.2%})")
            
        st.subheader("Recent Transactions")
        st.dataframe(raw_data.head())
        
        st.markdown("### Business Value")
        st.info("""
        **Risk Mitigation**: Early detection of fraudulent transactions protects capital.
        **Operational Efficiency**: Automating credit scoring reduces manual review time.
        **Compliance**: Interpretable models (SHAP) ensure alignment with regulatory standards (Basel II).
        """)

    elif page == "Data Analysis":
        st.header("Exploratory Data Analysis")
        
        tab1, tab2 = st.tabs(["Distributions", "Correlations"])
        
        with tab1:
            col_to_plot = st.selectbox("Select Column", settings.NUMERIC_COLS)
            
            fig, ax = plt.subplots()
            sns.histplot(data=featured_data, x=col_to_plot, hue=settings.TARGET_COL, kde=True, ax=ax)
            st.pyplot(fig)
            
        with tab2:
            st.subheader("Correlation Matrix")
            # Select numeric columns
            numeric_df = featured_data.select_dtypes(include=[np.number])
            # Drop ID columns that might have leaked in
            cols = [c for c in numeric_df.columns if 'Id' not in c]
            corr = numeric_df[cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

    elif page == "Model Performance":
        st.header("Model Performance")
        
        # In a real app, we would load test results from a file.
        # For now, we'll display placeholders or load from a results file if we saved one.
        
        st.markdown("### Model Comparison")
        # Placeholder data
        results = {
            'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
            'ROC-AUC': [0.75, 0.85, 0.92],
            'Precision': [0.60, 0.75, 0.88],
            'Recall': [0.55, 0.70, 0.82]
        }
        st.table(pd.DataFrame(results))
        
        st.info("Note: Metrics are illustrative. Run `src/train.py` to generate latest results.")

    elif page == "Prediction & SHAP":
        st.header("Individual Prediction & Explanation")
        
        if predictor is None:
            st.warning("Model not loaded. Please train the model first.")
            return
            
        # Input Form
        st.subheader("Transaction Details")
        
        with st.form("prediction_form"):
            amount = st.number_input("Amount", min_value=0.0, value=1000.0)
            value = st.number_input("Value", min_value=0.0, value=1000.0)
            pricing_strategy = st.slider("Pricing Strategy", 0, 4, 2)
            product_category = st.selectbox("Product Category", raw_data['ProductCategory'].unique())
            channel_id = st.selectbox("Channel ID", raw_data['ChannelId'].unique())
            provider_id = st.selectbox("Provider ID", raw_data['ProviderId'].unique())
            
            submitted = st.form_submit_button("Predict Risk")
            
        if submitted:
            # Create input dict
            input_data = {
                'Amount': amount,
                'Value': value,
                'PricingStrategy': pricing_strategy,
                'ProductCategory': product_category,
                'ChannelId': channel_id,
                'ProviderId': provider_id,
                'TransactionId': 'Demo',
                'BatchId': 'Demo',
                'AccountId': 'Demo',
                'SubscriptionId': 'Demo',
                'CustomerId': 'Demo_Cust',
                'ProductId': 'Demo_Prod',
                'TransactionStartTime': '2023-01-01T12:00:00Z', # Default
                'CurrencyCode': 'UGX', # Default
                'CountryCode': '256' # Default
            }
            
            # Predict
            try:
                result = predictor.predict_single(input_data)
                
                # Display Result
                col1, col2 = st.columns(2)
                with col1:
                    if result['prediction'] == 1:
                        st.error("HIGH RISK (Fraud Suspected)")
                    else:
                        st.success("LOW RISK (Legitimate)")
                
                with col2:
                    st.metric("Fraud Probability", f"{result['fraud_probability']:.2%}")
                
                # SHAP Explanation
                st.subheader("Feature Contribution (SHAP)")
                
                if hasattr(predictor.model, "get_booster") or isinstance(predictor.model, (type(shap.TreeExplainer),)):
                    try:
                        # Preprocess
                        df_input = pd.DataFrame([input_data])
                        X_processed = predictor.preprocess_input(df_input)
                        
                        # Explain
                        explainer = shap.TreeExplainer(predictor.model)
                        shap_values = explainer.shap_values(X_processed)
                        
                        # Handle varied SHAP output formats (list for multiclass, array for binary)
                        if isinstance(shap_values, list): 
                            sv = shap_values[1][0] 
                        elif len(shap_values.shape) > 1 and shap_values.shape[0] == 1:
                             sv = shap_values[0]
                        else:
                             sv = shap_values
                             
                        # Feature Names
                        feature_names = X_processed.columns if isinstance(X_processed, pd.DataFrame) else [f"Feature {i}" for i in range(len(sv))]

                        # Create DataFrame for plotting
                        shap_df = pd.DataFrame({
                            'Feature': feature_names,
                            'SHAP Value': sv
                        })
                        shap_df['Abs Value'] = shap_df['SHAP Value'].abs()
                        shap_df = shap_df.sort_values('Abs Value', ascending=False).head(10)
                        
                        fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
                        sns.barplot(data=shap_df, x='SHAP Value', y='Feature', ax=ax_shap, palette="viridis")
                        ax_shap.set_title("Top 10 Features Influencing Prediction")
                        st.pyplot(fig_shap)
                        
                    except Exception as e:
                        st.warning(f"Could not generate SHAP plot: {e}")
                else:
                    st.info("SHAP explanation available for Tree-based models (XGBoost/RandomForest).")
                    
            except Exception as e:
                st.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
