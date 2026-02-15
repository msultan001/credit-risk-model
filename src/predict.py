"""
Prediction Module
Load trained models and make predictions.
Refactored for strict type checking and config integration.
"""

import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Any, Optional
from pathlib import Path

from src.config import settings

class ModelPredictor:
    """Make predictions using trained models."""
    
    def __init__(self, model_path: Optional[str] = None, preprocessor_path: Optional[str] = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model. Defaults to xgboost.pkl in settings.MODELS_DIR.
            preprocessor_path: Path to saved preprocessor. Defaults to preprocessor.pkl in settings.MODELS_DIR.
        """
        if model_path is None:
            model_path = str(settings.MODELS_DIR / "xgboost.pkl")
        if preprocessor_path is None:
            preprocessor_path = str(settings.MODELS_DIR / "preprocessor.pkl")
            
        self.model = self.load_artifact(model_path)
        self.preprocessor = self.load_artifact(preprocessor_path)
        
    @staticmethod
    def load_artifact(path: str) -> Any:
        """Load pickled artifact."""
        try:
            with open(path, 'rb') as f:
                artifact = pickle.load(f)
            print(f"Loaded artifact from {path}")
            return artifact
        except FileNotFoundError:
            raise FileNotFoundError(f"Artifact not found at {path}")
    
    def preprocess_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data.
        
        Args:
            data: Input DataFrame.
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        # Handle missing values
        data = self.preprocessor.handle_missing_values(data)
        
        # Encode categorical features
        data = self.preprocessor.encode_categorical_features(data, fit=False)
        
        # Scale features
        data = self.preprocessor.scale_features(data, fit=False)
        
        # Select features used in training
        if self.preprocessor.feature_columns:
            available_features = [col for col in self.preprocessor.feature_columns if col in data.columns]
            # Ensure order matches
            data = data[available_features]
            
            # Check for missing features? 
            # Ideally we should raise warning or error if features are missing, but for now we proceed.
        
        return data
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make class predictions."""
        data_processed = self.preprocess_input(data)
        predictions = self.model.predict(data_processed)
        return predictions
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Make probability predictions."""
        data_processed = self.preprocess_input(data)
        probabilities = self.model.predict_proba(data_processed)
        return probabilities
    
    def predict_single(self, transaction: Dict[str, Any]) -> Dict[str, Union[int, float]]:
        """Predict for a single transaction."""
        # Convert to DataFrame
        df = pd.DataFrame([transaction])
        
        # Make prediction
        prediction = self.predict(df)[0]
        probabilities = self.predict_proba(df)[0]
        
        return {
            'prediction': int(prediction),
            'fraud_probability': float(probabilities[1]),
            'legitimate_probability': float(probabilities[0])
        }
    
    def predict_batch(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Union[int, float]]]:
        """Predict for batch of transactions."""
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Make predictions
        predictions = self.predict(df)
        probabilities = self.predict_proba(df)
        
        # Format results
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            results.append({
                'transaction_index': i,
                'prediction': int(pred),
                'fraud_probability': float(proba[1]),
                'legitimate_probability': float(proba[0])
            })
        
        return results

if __name__ == "__main__":
    try:
        predictor = ModelPredictor()
        
        # Sample transaction
        sample_transaction = {
            'Amount': 1000,
            'Value': 1000,
            'ProductCategory': 'airtime',
            'ChannelId': 'ChannelId_3',
            'ProviderId': 'ProviderId_6',
            'CurrencyCode': 'UGX',
            'CountryCode': '256',
            'PricingStrategy': 2,
            'TransactionStartTime': '2023-01-01T00:00:00Z' 
        }
        
        result = predictor.predict_single(sample_transaction)
        print(f"\nPrediction Result:")
        print(f"Fraud: {result['prediction']}")
        print(f"Fraud Probability: {result['fraud_probability']:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure models are trained and saved before running prediction.")
