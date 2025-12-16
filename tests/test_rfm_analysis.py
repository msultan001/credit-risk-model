"""
Unit Tests for RFM Analysis Module
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rfm_analysis import RFMAnalyzer


@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for RFM testing"""
    np.random.seed(42)
    
    # Create transactions for 5 customers
    data = {
        'TransactionId': [f'T{i}' for i in range(1, 16)],
        'CustomerId': ['C1', 'C1', 'C1', 'C2', 'C2', 'C3', 'C3', 'C3', 'C3', 'C4', 'C5', 'C5', 'C5', 'C5', 'C5'],
        'Amount': [100, 200, 150, 500, 300, 50, 75, 60, 80, 1000, 25, 30, 35, 40, 45],
        'Value': [100, 200, 150, 500, 300, 50, 75, 60, 80, 1000, 25, 30, 35, 40, 45],
        'TransactionStartTime': pd.date_range('2023-01-01', periods=15, freq='2D')
    }
    return pd.DataFrame(data)


class TestRFMAnalyzer:
    """Test RFM Analysis functionality"""
    
    def test_rfm_initialization(self):
        """Test RFM analyzer initialization"""
        analyzer = RFMAnalyzer(snapshot_date='2023-12-31')
        assert analyzer.snapshot_date == '2023-12-31'
        assert analyzer.scaler is not None
        assert analyzer.kmeans is None  # Not fitted yet
    
    def test_calculate_rfm(self, sample_transaction_data):
        """Test RFM calculation"""
        analyzer = RFMAnalyzer(snapshot_date='2023-02-01')
        rfm = analyzer.calculate_rfm(sample_transaction_data)
        
        # Check output structure
        assert 'CustomerId' in rfm.columns
        assert 'Recency' in rfm.columns
        assert 'Frequency' in rfm.columns
        assert 'Monetary' in rfm.columns
        
        # Should have 5 unique customers
        assert len(rfm) == 5
        
        # All values should be non-negative
        assert (rfm['Recency'] >= 0).all()
        assert (rfm['Frequency'] > 0).all()
        assert (rfm['Monetary'] > 0).all()
    
    def test_preprocess_rfm(self, sample_transaction_data):
        """Test RFM preprocessing/scaling"""
        analyzer = RFMAnalyzer()
        rfm = analyzer.calculate_rfm(sample_transaction_data)
        rfm_scaled = analyzer.preprocess_rfm(rfm)
        
        # Scaled features should have mean ~0 and std ~1
        assert abs(rfm_scaled['Recency'].mean()) < 1
        assert abs(rfm_scaled['Frequency'].mean()) < 1
        assert abs(rfm_scaled['Monetary'].mean()) < 1
    
    def test_cluster_customers(self, sample_transaction_data):
        """Test K-Means clustering"""
        analyzer = RFMAnalyzer()
        rfm = analyzer.calculate_rfm(sample_transaction_data)
        rfm_scaled = analyzer.preprocess_rfm(rfm)
        rfm_clustered = analyzer.cluster_customers(rfm_scaled, n_clusters=3, random_state=42)
        
        # Should have cluster assignments
        assert 'RFM_Cluster' in rfm_clustered.columns
        
        # Clusters should be 0, 1, 2
        unique_clusters = rfm_clustered['RFM_Cluster'].unique()
        assert len(unique_clusters) <= 3
        assert all(c in [0, 1, 2] for c in unique_clusters)
    
    def test_create_risk_target(self, sample_transaction_data):
        """Test proxy target creation"""
        analyzer = RFMAnalyzer()
        rfm = analyzer.calculate_rfm(sample_transaction_data)
        rfm_scaled = analyzer.preprocess_rfm(rfm)
        rfm_clustered = analyzer.cluster_customers(rfm_scaled, n_clusters=3, random_state=42)
        rfm_with_target = analyzer.create_risk_target(rfm_clustered)
        
        # Should have binary target
        assert 'is_high_risk' in rfm_with_target.columns
        assert set(rfm_with_target['is_high_risk'].unique()).issubset({0, 1})
        
        # At least one customer should be high risk
        assert rfm_with_target['is_high_risk'].sum() > 0
    
    def test_merge_target_to_transactions(self, sample_transaction_data):
        """Test merging proxy target back to transactions"""
        analyzer = RFMAnalyzer()
        rfm = analyzer.calculate_rfm(sample_transaction_data)
        rfm_scaled = analyzer.preprocess_rfm(rfm)
        rfm_clustered = analyzer.cluster_customers(rfm_scaled, n_clusters=3, random_state=42)
        rfm_with_target = analyzer.create_risk_target(rfm_clustered)
        
        transactions_with_target = analyzer.merge_target_to_transactions(
            sample_transaction_data,
            rfm_with_target
        )
        
        # Should have same number of transactions
        assert len(transactions_with_target) == len(sample_transaction_data)
        
        # Should have proxy target column
        assert 'is_high_risk' in transactions_with_target.columns
        
        # No missing values
        assert transactions_with_target['is_high_risk'].isnull().sum() == 0
    
    def test_full_rfm_pipeline(self, sample_transaction_data):
        """Test complete RFM pipeline"""
        analyzer = RFMAnalyzer(snapshot_date='2023-02-01')
        
        transactions_with_target, rfm_results = analyzer.full_rfm_pipeline(
            sample_transaction_data,
            n_clusters=3,
            random_state=42
        )
        
        # Check transaction output
        assert len(transactions_with_target) == len(sample_transaction_data)
        assert 'is_high_risk' in transactions_with_target.columns
        
        # Check RFM output
        assert 'RFM_Cluster' in rfm_results.columns
        assert 'is_high_risk' in rfm_results.columns
        assert len(rfm_results) == 5  # 5 customers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
