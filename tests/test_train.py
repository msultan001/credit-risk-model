"""
Unit Tests for Model Training Pipeline.
Tests ModelTrainer class mocking external dependencies.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.train import ModelTrainer
from src.config import settings

@pytest.fixture
def mock_data():
    """Create mock training data."""
    X = pd.DataFrame(np.random.rand(10, 5), columns=[f'f{i}' for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 10), name='target')
    return X, X, y, y

class TestModelTrainer:
    """Test ModelTrainer class."""
    
    @patch('src.train.DataLoader')
    @patch('src.train.FeatureEngineer')
    @patch('src.train.DataPreprocessor')
    @patch('src.train.SMOTE')
    def test_prepare_data(self, mock_smote, mock_preprocessor_cls, mock_engineer_cls, mock_loader_cls):
        """Test data preparation pipeline."""
        # Setup mocks
        mock_loader = mock_loader_cls.return_value
        mock_loader.load_data.return_value = pd.DataFrame({'a': [1]})
        
        mock_engineer = mock_engineer_cls.return_value
        mock_engineer.engineer_all_features.return_value = pd.DataFrame({'a': [1], 'target': [0]})
        
        mock_preprocessor = mock_preprocessor_cls.return_value
        mock_preprocessor.handle_missing_values.return_value = pd.DataFrame({'a': [1], 'target': [0]})
        mock_preprocessor.encode_categorical_features.return_value = pd.DataFrame({'a': [1], 'target': [0]})
        mock_preprocessor.scale_features.return_value = pd.DataFrame({'a': [1], 'target': [0]})
        
        # X_train, X_test, y_train, y_test
        mock_preprocessor.prepare_features_target.return_value = (
            pd.DataFrame({'a': [1]}), pd.DataFrame({'a': [1]}),
            pd.Series([0]), pd.Series([0])
        )
        
        mock_smote_instance = mock_smote.return_value
        mock_smote_instance.fit_resample.return_value = (
            pd.DataFrame({'a': [1]}), pd.Series([0])
        )
        
        trainer = ModelTrainer(data_path="dummy.csv")
        X_train, X_test, y_train, y_test = trainer.prepare_data()
        
        assert X_train is not None
        mock_loader.load_data.assert_called_once()
        mock_preprocessor.prepare_features_target.assert_called_once()
    
    @patch('mlflow.start_run')
    @patch('mlflow.sklearn.log_model')
    @patch('mlflow.log_params')
    def test_train_logistic_regression(self, mock_log_params, mock_log_model, mock_start_run, mock_data):
        """Test logistic regression training (fast/no grid search)."""
        X_train, _, y_train, _ = mock_data
        
        trainer = ModelTrainer(data_path="dummy.csv")
        model = trainer.train_logistic_regression(X_train, y_train, use_grid_search=False)
        
        assert model is not None
        assert 'logistic_regression' in trainer.models
        mock_log_model.assert_called()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
