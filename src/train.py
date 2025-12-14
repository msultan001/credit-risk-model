"""
Model Training Pipeline
Train and evaluate multiple credit risk models
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

from data_processing import DataLoader, FeatureEngineer, DataPreprocessor


class ModelTrainer:
    """Train and evaluate credit risk models"""
    
    def __init__(self, data_path: str):
        """
        Initialize ModelTrainer
        
        Args:
            data_path: Path to training data
        """
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.preprocessor = None
        
    def prepare_data(self):
        """Load and prepare data for training"""
        # Load data
        loader = DataLoader(self.data_path)
        data = loader.load_data()
        loader.validate_data()
        
        # Feature engineering
        engineer = FeatureEngineer(data)
        data = engineer.engineer_all_features()
        
        # Preprocessing
        self.preprocessor = DataPreprocessor()
        data = self.preprocessor.handle_missing_values(data)
        data = self.preprocessor.encode_categorical_features(data, fit=True)
        data = self.preprocessor.scale_features(data, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_features_target(data)
        
        # Handle class imbalance with SMOTE
        print("Applying SMOTE for class balance...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {X_train_balanced.shape}, Fraud rate: {y_train_balanced.mean():.4f}")
        
        return X_train_balanced, X_test, y_train_balanced, y_test
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model"""
        print("\n" + "="*50)
        print("Training Logistic Regression...")
        print("="*50)
        
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        
        self.models['logistic_regression'] = model
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        print("\n" + "="*50)
        print("Training Random Forest...")
        print("="*50)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        print("\n" + "="*50)
        print("Training XGBoost...")
        print("="*50)
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        self.results[model_name] = metrics
        
        # Print results
        print(f"\n{model_name} Results:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return metrics
    
    def train_all_models(self):
        """Train and evaluate all models"""
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Train models
        lr_model = self.train_logistic_regression(X_train, y_train)
        self.evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')
        
        rf_model = self.train_random_forest(X_train, y_train)
        self.evaluate_model(rf_model, X_test, y_test, 'Random Forest')
        
        xgb_model = self.train_xgboost(X_train, y_train)
        self.evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
        
        # Print comparison
        self.print_comparison()
        
        # Save models
        self.save_models()
        
    def print_comparison(self):
        """Print model comparison"""
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        df_results = pd.DataFrame(self.results).T
        print(df_results.to_string())
        
    def save_models(self, output_dir: str = 'models'):
        """
        Save trained models and preprocessor
        
        Args:
            output_dir: Directory to save models
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save each model
        for model_name, model in self.models.items():
            filepath = f"{output_dir}/{model_name}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {model_name} to {filepath}")
        
        # Save preprocessor
        preprocessor_path = f"{output_dir}/preprocessor.pkl"
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        print(f"Saved preprocessor to {preprocessor_path}")


if __name__ == "__main__":
    # Train models
    trainer = ModelTrainer('data/raw/data.csv')
    trainer.train_all_models()
