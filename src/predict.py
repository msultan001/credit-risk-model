"""
Prediction Module
Load trained models and make predictions
"""

import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Union
from pathlib import Path


class ModelPredictor:
    """Make predictions using trained models"""
    
    def __init__(self, model_path: str, preprocessor_path: str):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model
            preprocessor_path: Path to saved preprocessor
        """
        self.model = self.load_model(model_path)
        self.preprocessor = self.load_preprocessor(preprocessor_path)
        
    @staticmethod
    def load_model(model_path: str):
        """Load trained model"""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Loaded model from {model_path}")
        return model
    
    @staticmethod
    def load_preprocessor(preprocessor_path: str):
        """Load preprocessor"""
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Loaded preprocessor from {preprocessor_path}")
        return preprocessor
    
    def preprocess_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data
        
        Args:
            data: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
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
            data = data[available_features]
        
        return data
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            data: Input DataFrame
            
        Returns:
            Array of predictions (0 or 1)
        """
        data_processed = self.preprocess_input(data)
        predictions = self.model.predict(data_processed)
        return predictions
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities
        
        Args:
            data: Input DataFrame
            
        Returns:
            Array of prediction probabilities
        """
        data_processed = self.preprocess_input(data)
        probabilities = self.model.predict_proba(data_processed)
        return probabilities
    
    def predict_single(self, transaction: Dict) -> Dict[str, Union[int, float]]:
        """
        Predict for a single transaction
        
        Args:
            transaction: Dictionary with transaction features
            
        Returns:
            Dictionary with prediction and probability
        """
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
    
    def predict_batch(self, transactions: List[Dict]) -> List[Dict]:
        """
        Predict for batch of transactions
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            List of prediction dictionaries
        """
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
    # Example usage
    predictor = ModelPredictor(
        model_path='models/xgboost.pkl',
        preprocessor_path='models/preprocessor.pkl'
    )
    
    # Sample transaction
    sample_transaction = {
        'Amount': 1000,
        'Value': 1000,
        'ProductCategory': 'airtime',
        'ChannelId': 'ChannelId_3',
        'ProviderId': 'ProviderId_6',
        'CurrencyCode': 'UGX',
        'CountryCode': '256',
        'PricingStrategy': 2
    }
    
    result = predictor.predict_single(sample_transaction)
    print(f"\nPrediction Result:")
    print(f"Fraud: {result['prediction']}")
    print(f"Fraud Probability: {result['fraud_probability']:.4f}")
