
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.config import settings
from src.train import ModelTrainer
from src.predict import ModelPredictor

@pytest.fixture
def synthetic_data(tmp_path):
    """Create synthetic data for integration testing."""
    # Create enough data for train/test split and cross-validation
    n_samples = 100
    data = {
        'TransactionId': [f'T{i}' for i in range(n_samples)],
        'BatchId': [f'B{i}' for i in range(n_samples)],
        'AccountId': [f'A{i}' for i in range(n_samples)],
        'SubscriptionId': [f'S{i}' for i in range(n_samples)],
        'CustomerId': [f'C{i%10}' for i in range(n_samples)],
        'CurrencyCode': ['UGX'] * n_samples,
        'CountryCode': ['256'] * n_samples,
        'ProviderId': [f'P{i%3}' for i in range(n_samples)],
        'ProductId': [f'Pr{i%5}' for i in range(n_samples)],
        'ProductCategory': [f'Cat{i%3}' for i in range(n_samples)],
        'ChannelId': [f'Ch{i%2}' for i in range(n_samples)],
        'Amount': np.random.uniform(100, 10000, n_samples),
        'Value': np.random.uniform(100, 10000, n_samples),
        'PricingStrategy': np.random.randint(0, 5, n_samples),
        'FraudResult': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'TransactionStartTime': pd.date_range('2023-01-01', periods=n_samples, freq='h')
    }
    
    df = pd.DataFrame(data)
    
    # Save to temp location
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)
    
    # Update settings to use this temp path
    # Note: modifying global settings might affect other tests if run in parallel, 
    # but pytest usually runs sequentially or isolated. 
    # A safer way is to mock, but here we pass data_path to Trainer.
    
    return str(data_path)

def test_full_pipeline(synthetic_data, tmp_path):
    """
    Test the full pipeline:
    1. Train models using synthetic data.
    2. Save artifacts.
    3. Load predictor.
    4. Make a prediction.
    """
    # 1. Train
    # Use a temp directory for models to avoid overwriting real ones
    original_models_dir = settings.MODELS_DIR
    settings.MODELS_DIR = tmp_path / "models"
    settings.MODELS_DIR.mkdir()
    
    try:
        trainer = ModelTrainer(data_path=synthetic_data, experiment_name="test_experiment")
        
        # Train just one model for speed
        X_train, X_test, y_train, y_test = trainer.prepare_data()
        trainer.train_logistic_regression(X_train, y_train, use_grid_search=False)
        trainer.save_models()
        
        # 2. Verify artifacts exist
        assert (settings.MODELS_DIR / "logistic_regression.pkl").exists()
        assert (settings.MODELS_DIR / "preprocessor.pkl").exists()
        
        # 3. Load Predictor
        # We explicitly pass the paths to ensure we load what we just saved
        predictor = ModelPredictor(
            model_path=str(settings.MODELS_DIR / "logistic_regression.pkl"),
            preprocessor_path=str(settings.MODELS_DIR / "preprocessor.pkl")
        )
        
        # 4. Predict
        input_data = {
            'Amount': 500.0,
            'Value': 500.0,
            'PricingStrategy': 2,
            'ProductCategory': 'Cat1',
            'ChannelId': 'Ch1',
            'ProviderId': 'P1',
            'TransactionId': 'Test1',
            'BatchId': 'B1',
            'AccountId': 'A1',
            'SubscriptionId': 'S1',
            'CustomerId': 'C1',
            'ProductId': 'Pr1',
            'TransactionStartTime': '2023-01-01T12:00:00Z',
            'CurrencyCode': 'UGX',
            'CountryCode': '256'
        }
        
        result = predictor.predict_single(input_data)
        
        assert 'prediction' in result
        assert 'fraud_probability' in result
        assert isinstance(result['prediction'], int)
        
    finally:
        # Restore settings
        settings.MODELS_DIR = original_models_dir
