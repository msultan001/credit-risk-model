"""
Unit Tests for Data Processing Module
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_processing import DataLoader, FeatureEngineer, DataPreprocessor


# Fixtures
@pytest.fixture
def sample_data():
    """Create sample transaction data for testing"""
    data = {
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5'],
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3'],
        'Amount': [1000, 2000, 500, 1500, 3000],
        'Value': [1000, 2000, 500, 1500, 3000],
        'FraudResult': [0, 0, 1, 0, 1],
        'ProductCategory': ['airtime', 'data', 'airtime', 'utility', 'airtime'],
        'ChannelId': ['Ch1', 'Ch2', 'Ch1', 'Ch3', 'Ch1'],
        'ProviderId': ['P1', 'P2', 'P1', 'P3', 'P1'],
        'CurrencyCode': ['UGX', 'UGX', 'UGX', 'UGX', 'UGX'],
        'CountryCode': ['256', '256', '256', '256', '256'],
        'PricingStrategy': [2, 2, 4, 2, 4]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_file(tmp_path, sample_data):
    """Create a temporary CSV file"""
    csv_file = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_file, index=False)
    return str(csv_file)


# DataLoader Tests
class TestDataLoader:
    """Test DataLoader class"""
    
    def test_load_data(self, sample_csv_file):
        """Test data loading from CSV"""
        loader = DataLoader(sample_csv_file)
        data = loader.load_data()
        
        assert data is not None
        assert len(data) == 5
        assert 'TransactionId' in data.columns
    
    def test_validate_data_success(self, sample_csv_file):
        """Test validation with valid data"""
        loader = DataLoader(sample_csv_file)
        loader.load_data()
        
        assert loader.validate_data() == True
    
    def test_validate_data_missing_columns(self, tmp_path):
        """Test validation with missing columns"""
        # Create incomplete data
        incomplete_data = pd.DataFrame({
            'TransactionId': ['T1', 'T2'],
            'Amount': [1000, 2000]
        })
        csv_file = tmp_path / "incomplete.csv"
        incomplete_data.to_csv(csv_file, index=False)
        
        loader = DataLoader(str(csv_file))
        loader.load_data()
        
        with pytest.raises(ValueError):
            loader.validate_data()


# FeatureEngineer Tests
class TestFeatureEngineer:
    """Test FeatureEngineer class"""
    
    def test_create_customer_aggregates(self, sample_data):
        """Test customer aggregate features"""
        engineer = FeatureEngineer(sample_data)
        agg_data = engineer.create_customer_aggregates()
        
        assert 'CustomerId' in agg_data.columns
        assert 'Amount_sum' in agg_data.columns
        assert 'Amount_mean' in agg_data.columns
        assert len(agg_data) == 3  # 3 unique customers
    
    def test_create_transaction_features(self, sample_data):
        """Test transaction-level features"""
        engineer = FeatureEngineer(sample_data)
        featured_data = engineer.create_transaction_features()
        
        assert 'AmountValueDiff' in featured_data.columns
        assert 'AmountValueRatio' in featured_data.columns
        assert 'ProductDiversity' in featured_data.columns
        assert len(featured_data) == len(sample_data)
    
    def test_engineer_all_features(self, sample_data):
        """Test complete feature engineering pipeline"""
        engineer = FeatureEngineer(sample_data)
        featured_data = engineer.engineer_all_features()
        
        # Check all features were created
        assert 'AmountValueDiff' in featured_data.columns
        assert 'ProductDiversity' in featured_data.columns
        assert 'Amount_sum' in featured_data.columns


# DataPreprocessor Tests
class TestDataPreprocessor:
    """Test DataPreprocessor class"""
    
    def test_handle_missing_values(self, sample_data):
        """Test missing value handling"""
        # Add missing values
        data_with_missing = sample_data.copy()
        data_with_missing.loc[0, 'Amount'] = np.nan
        data_with_missing.loc[1, 'ProductCategory'] = np.nan
        
        preprocessor = DataPreprocessor()
        cleaned_data = preprocessor.handle_missing_values(data_with_missing)
        
        assert cleaned_data['Amount'].isnull().sum() == 0
        assert cleaned_data['ProductCategory'].isnull().sum() == 0
    
    def test_encode_categorical_features(self, sample_data):
        """Test categorical encoding"""
        preprocessor = DataPreprocessor()
        encoded_data = preprocessor.encode_categorical_features(sample_data, fit=True)
        
        # Check that categorical columns are now numeric
        assert encoded_data['ProductCategory'].dtype in [np.int32, np.int64]
        assert encoded_data['ChannelId'].dtype in [np.int32, np.int64]
    
    def test_scale_features(self, sample_data):
        """Test feature scaling"""
        preprocessor = DataPreprocessor()
        scaled_data = preprocessor.scale_features(sample_data, fit=True)
        
        # Check that Amount is scaled (mean should be close to 0)
        assert abs(scaled_data['Amount'].mean()) < 1
    
    def test_prepare_features_target(self, sample_data):
        """Test train-test split"""
        engineer = FeatureEngineer(sample_data)
        featured_data = engineer.engineer_all_features()
        
        preprocessor = DataPreprocessor()
        featured_data = preprocessor.handle_missing_values(featured_data)
        featured_data = preprocessor.encode_categorical_features(featured_data, fit=True)
        
        X_train, X_test, y_train, y_test = preprocessor.prepare_features_target(
            featured_data, test_size=0.4
        )
        
        # Check split proportions
        total_records = len(featured_data)
        assert len(X_train) == int(total_records * 0.6)
        assert len(X_test) == total_records - len(X_train)
        
        # Check target distribution
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)


# Integration Tests
class TestIntegration:
    """Integration tests for complete pipeline"""
    
    def test_full_preprocessing_pipeline(self, sample_data):
        """Test complete preprocessing pipeline"""
        # Feature engineering
        engineer = FeatureEngineer(sample_data)
        data = engineer.engineer_all_features()
        
        # Preprocessing
        preprocessor = DataPreprocessor()
        data = preprocessor.handle_missing_values(data)
        data = preprocessor.encode_categorical_features(data, fit=True)
        data = preprocessor.scale_features(data, fit=True)
        
        X_train, X_test, y_train, y_test = preprocessor.prepare_features_target(data)
        
        # Verify output
        assert X_train is not None
        assert X_test is not None
        assert len(X_train) > 0
        assert len(X_test) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
