"""
Model Training Pipeline with MLflow Tracking.
Train and evaluate multiple credit risk models with experiment tracking.
Refactored for configuration management and modularity.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from src.data_processing import DataLoader, FeatureEngineer, DataPreprocessor
from src.config import settings

class ModelTrainer:
    """Train and evaluate credit risk models with MLflow tracking."""
    
    def __init__(self, data_path: str = str(settings.RAW_DATA_PATH), experiment_name: str = settings.EXPERIMENT_NAME):
        """
        Initialize ModelTrainer.
        
        Args:
            data_path: Path to training data.
            experiment_name: MLflow experiment name.
        """
        self.data_path = data_path
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        self.best_params: Dict[str, Dict[str, Any]] = {}
        self.preprocessor: Optional[DataPreprocessor] = None
        
        # Initialize MLflow
        try:
            mlflow.set_experiment(experiment_name)
            print(f"MLflow experiment set to: {experiment_name}")
        except Exception as e:
            print(f"Warning: Could not set MLflow experiment: {e}")
        
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load and prepare data for training.
        
        Returns:
            X_train_balanced, X_test, y_train_balanced, y_test
        """
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
        smote = SMOTE(random_state=settings.RANDOM_STATE)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {X_train_balanced.shape}, Fraud rate: {y_train_balanced.mean():.4f}")
        
        return X_train_balanced, X_test, y_train_balanced, y_test
    
    def train_logistic_regression(self, X_train, y_train, use_grid_search: bool = True):
        """Train Logistic Regression model with hyperparameter tuning."""
        print("\n" + "="*50)
        print("Training Logistic Regression...")
        print("="*50)
        
        with mlflow.start_run(run_name="Logistic_Regression", nested=True):
            if use_grid_search:
                param_grid = {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l2'],
                    'max_iter': [1000, 2000]
                }
                
                base_model = LogisticRegression(random_state=settings.RANDOM_STATE, class_weight='balanced')
                grid_search = GridSearchCV(
                    base_model, 
                    param_grid, 
                    cv=settings.CV_FOLDS, 
                    scoring=settings.SCORING_METRIC,
                    n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(X_train, y_train)
                
                model = grid_search.best_estimator_
                self.best_params['logistic_regression'] = grid_search.best_params_
                
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metric("cv_best_score", grid_search.best_score_)
            else:
                model = LogisticRegression(
                    random_state=settings.RANDOM_STATE,
                    max_iter=1000,
                    class_weight='balanced'
                )
                model.fit(X_train, y_train)
                mlflow.log_params({'C': 1.0, 'penalty': 'l2', 'max_iter': 1000})
            
            mlflow.sklearn.log_model(model, "model")
            self.models['logistic_regression'] = model
            return model
    
    def train_random_forest(self, X_train, y_train, use_grid_search: bool = True):
        """Train Random Forest model with hyperparameter tuning."""
        print("\n" + "="*50)
        print("Training Random Forest...")
        print("="*50)
        
        with mlflow.start_run(run_name="Random_Forest", nested=True):
            if use_grid_search:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5],
                }
                
                base_model = RandomForestClassifier(random_state=settings.RANDOM_STATE, n_jobs=-1)
                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=settings.CV_FOLDS,
                    scoring=settings.SCORING_METRIC,
                    n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(X_train, y_train)
                
                model = grid_search.best_estimator_
                self.best_params['random_forest'] = grid_search.best_params_
                
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metric("cv_best_score", grid_search.best_score_)
            else:
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=settings.RANDOM_STATE,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                mlflow.log_params({'n_estimators': 100, 'max_depth': 10})
            
            mlflow.sklearn.log_model(model, "model")
            self.models['random_forest'] = model
            return model
    
    def train_xgboost(self, X_train, y_train, use_random_search: bool = True):
        """Train XGBoost model with hyperparameter tuning."""
        print("\n" + "="*50)
        print("Training XGBoost...")
        print("="*50)
        
        with mlflow.start_run(run_name="XGBoost", nested=True):
            # Calculate scale_pos_weight for imbalanced data
            # Note: SMOTE balances the training data, so scale_pos_weight might be 1.0 here
            # But kept for robustness if SMOTE is disabled
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            
            if use_random_search:
                param_distributions = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                }
                
                base_model = xgb.XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    random_state=settings.RANDOM_STATE,
                    n_jobs=-1
                )
                
                random_search = RandomizedSearchCV(
                    base_model,
                    param_distributions,
                    n_iter=10,
                    cv=settings.CV_FOLDS,
                    scoring=settings.SCORING_METRIC,
                    random_state=settings.RANDOM_STATE,
                    n_jobs=-1,
                    verbose=1
                )
                random_search.fit(X_train, y_train)
                
                model = random_search.best_estimator_
                self.best_params['xgboost'] = random_search.best_params_
                
                mlflow.log_params(random_search.best_params_)
                mlflow.log_metric("cv_best_score", random_search.best_score_)
            else:
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight,
                    random_state=settings.RANDOM_STATE,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                mlflow.log_params({
                    'n_estimators': 100,
                    'max_depth': 6, 
                    'learning_rate': 0.1, 
                    'scale_pos_weight': scale_pos_weight
                })
            
            mlflow.xgboost.log_model(model, "model")
            self.models['xgboost'] = model
            return model
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict[str, float]:
        """Evaluate model performance and log metrics to MLflow."""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        self.results[model_name] = metrics
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"Eval_{model_name}", nested=True):
            mlflow.log_metrics(metrics)
        
        print(f"\n{model_name} Results:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return metrics
    
    def train_all_models(self, use_hyperparameter_tuning: bool = True):
        """Train and evaluate all models."""
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        with mlflow.start_run(run_name="Full_Training_Pipeline"):
            # Train models
            lr_model = self.train_logistic_regression(X_train, y_train, use_grid_search=use_hyperparameter_tuning)
            self.evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')
            
            rf_model = self.train_random_forest(X_train, y_train, use_grid_search=use_hyperparameter_tuning)
            self.evaluate_model(rf_model, X_test, y_test, 'Random Forest')
            
            xgb_model = self.train_xgboost(X_train, y_train, use_random_search=use_hyperparameter_tuning)
            self.evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
            
            self.print_comparison()
            self.save_models()
    
    def print_comparison(self):
        """Print model comparison."""
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        if self.results:
            df_results = pd.DataFrame(self.results).T
            print(df_results.to_string())
        else:
            print("No results to display.")
        
    def save_models(self):
        """Save trained models and preprocessor."""
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            filepath = settings.MODELS_DIR / f"{model_name}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {model_name} to {filepath}")
        
        if self.preprocessor:
            preprocessor_path = settings.MODELS_DIR / "preprocessor.pkl"
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(self.preprocessor, f)
            print(f"Saved preprocessor to {preprocessor_path}")

if __name__ == "__main__":
    trainer = ModelTrainer(experiment_name=settings.EXPERIMENT_NAME)
    trainer.train_all_models(use_hyperparameter_tuning=True)
