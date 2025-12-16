"""
Model Training Pipeline with MLflow Tracking
Train and evaluate multiple credit risk models with experiment tracking
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from data_processing import DataLoader, FeatureEngineer, DataPreprocessor


class ModelTrainer:
    """Train and evaluate credit risk models with MLflow tracking"""
    
    def __init__(self, data_path: str, experiment_name: str = "credit-risk-modeling"):
        """
        Initialize ModelTrainer
        
        Args:
            data_path: Path to training data
            experiment_name: MLflow experiment name
        """
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.best_params = {}
        self.preprocessor = None
        
        # Initialize MLflow
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment: {experiment_name}")
        
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
    
    def train_logistic_regression(self, X_train, y_train, use_grid_search=True):
        """Train Logistic Regression model with hyperparameter tuning"""
        print("\n" + "="*50)
        print("Training Logistic Regression...")
        print("="*50)
        
        with mlflow.start_run(run_name="Logistic_Regression"):
            if use_grid_search:
                # Hyperparameter grid
                param_grid = {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l2'],
                    'max_iter': [1000, 2000]
                }
                
                base_model = LogisticRegression(random_state=42, class_weight='balanced')
                grid_search = GridSearchCV(
                    base_model, 
                    param_grid, 
                    cv=3, 
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(X_train, y_train)
                
                model = grid_search.best_estimator_
                self.best_params['logistic_regression'] = grid_search.best_params_
                
                # Log best parameters
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metric("cv_best_score", grid_search.best_score_)
            else:
                model = LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    class_weight='balanced'
                )
                model.fit(X_train, y_train)
                mlflow.log_params({
                    'C': 1.0,
                    'penalty': 'l2',
                    'max_iter': 1000
                })
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            self.models['logistic_regression'] = model
            return model
    
    def train_random_forest(self, X_train, y_train, use_grid_search=True):
        """Train Random Forest model with hyperparameter tuning"""
        print("\n" + "="*50)
        print("Training Random Forest...")
        print("="*50)
        
        with mlflow.start_run(run_name="Random_Forest"):
            if use_grid_search:
                # Hyperparameter grid
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5],
                }
                
                base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=3,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(X_train, y_train)
                
                model = grid_search.best_estimator_
                self.best_params['random_forest'] = grid_search.best_params_
                
                # Log best parameters
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metric("cv_best_score", grid_search.best_score_)
            else:
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                mlflow.log_params({
                    'n_estimators': 100,
                    'max_depth': 10
                })
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            self.models['random_forest'] = model
            return model
    
    def train_xgboost(self, X_train, y_train, use_random_search=True):
        """Train XGBoost model with hyperparameter tuning"""
        print("\n" + "="*50)
        print("Training XGBoost...")
        print("="*50)
        
        with mlflow.start_run(run_name="XGBoost"):
            # Calculate scale_pos_weight for imbalanced data
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            
            if use_random_search:
                # Hyperparameter distributions
                param_distributions = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                }
                
                base_model = xgb.XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1
                )
                
                random_search = RandomizedSearchCV(
                    base_model,
                    param_distributions,
                    n_iter=10,
                    cv=3,
                    scoring='roc_auc',
                    random_state=42,
                    n_jobs=-1,
                    verbose=1
                )
                random_search.fit(X_train, y_train)
                
                model = random_search.best_estimator_
                self.best_params['xgboost'] = random_search.best_params_
                
                # Log best parameters
                mlflow.log_params(random_search.best_params_)
                mlflow.log_metric("cv_best_score", random_search.best_score_)
            else:
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                mlflow.log_params({
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'scale_pos_weight': scale_pos_weight
                })
            
            # Log model
            mlflow.xgboost.log_model(model, "model")
            
            self.models['xgboost'] = model
            return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate model performance and log metrics to MLflow
        
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
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        
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
    
    def train_all_models(self, use_hyperparameter_tuning=True):
        """Train and evaluate all models"""
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Train Logistic Regression
        lr_model = self.train_logistic_regression(X_train, y_train, use_grid_search=use_hyperparameter_tuning)
        self.evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')
        
        # Train Random Forest
        rf_model = self.train_random_forest(X_train, y_train, use_grid_search=use_hyperparameter_tuning)
        self.evaluate_model(rf_model, X_test, y_test, 'Random Forest')
        
        # Train XGBoost
        xgb_model = self.train_xgboost(X_train, y_train, use_random_search=use_hyperparameter_tuning)
        self.evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
        
        # Print comparison
        self.print_comparison()
        
        # Register best model
        self.register_best_model()
        
        # Save models
        self.save_models()
        
    def print_comparison(self):
        """Print model comparison"""
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        df_results = pd.DataFrame(self.results).T
        print(df_results.to_string())
        
    def register_best_model(self):
        """Register the best performing model in MLflow Model Registry"""
        if not self.results:
            print("No models to register")
            return
        
        # Find best model based on ROC-AUC
        best_model_name = max(self.results.items(), key=lambda x: x[1]['roc_auc'])[0]
        best_roc_auc = self.results[best_model_name]['roc_auc']
        
        print(f"\n{'='*70}")
        print(f"Best Model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
        print(f"{'='*70}")
        
        # Register model
        try:
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, "credit-risk-best-model")
            print(f"Registered {best_model_name} in MLflow Model Registry")
        except Exception as e:
            print(f"Note: Could not register model in registry: {e}")
            print("This is expected if MLflow tracking server is not configured.")
    
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
    # Train models with MLflow tracking
    trainer = ModelTrainer('data/raw/data.csv')
    trainer.train_all_models(use_hyperparameter_tuning=True)
    
    print("\n" + "="*70)
    print("To view MLflow results, run: mlflow ui")
    print("Then open http://localhost:5000 in your browser")
    print("="*70)
