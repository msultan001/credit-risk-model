"""
RFM Analysis Module
Calculate Recency, Frequency, Monetary metrics and create proxy target variable.
Refactored to use centralized configuration and improved type hinting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.config import settings

class RFMAnalyzer:
    """
    RFM (Recency, Frequency, Monetary) Analysis for customer segmentation.
    Creates a proxy target variable 'is_high_risk' based on RFM clustering.
    """
    
    def __init__(self, snapshot_date: Optional[str] = None):
        """
        Initialize RFM Analyzer.
        
        Args:
            snapshot_date: Reference date for recency calculation (format: YYYY-MM-DD).
                         If None, uses the maximum transaction date in dataset + 1 day.
        """
        self.snapshot_date = snapshot_date
        self.scaler = StandardScaler()
        self.kmeans: Optional[KMeans] = None
        self.rfm_data: Optional[pd.DataFrame] = None
        
    def calculate_rfm(
        self, 
        df: pd.DataFrame,
        customer_col: str = 'CustomerId',
        date_col: str = settings.DATE_COL,
        amount_col: str = 'Amount'
    ) -> pd.DataFrame:
        """
        Calculate RFM metrics per customer.
        
        Args:
            df: Transaction DataFrame.
            customer_col: Name of customer ID column.
            date_col: Name of transaction date column.
            amount_col: Name of transaction amount column.
            
        Returns:
            pd.DataFrame: DataFrame with RFM metrics per customer.
        """
        # Ensure datetime format
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Determine snapshot date
        if self.snapshot_date is None:
            snapshot_dt = df[date_col].max() + timedelta(days=1)
        else:
            snapshot_dt = pd.to_datetime(self.snapshot_date)
        
        print(f"RFM Analysis Snapshot Date: {snapshot_dt}")
        
        # Calculate RFM metrics
        # lambda x: (snapshot_dt - x.max()).days  <-- Recency
        rfm = df.groupby(customer_col).agg({
            date_col: lambda x: (snapshot_dt - x.max()).days,
            customer_col: 'count',
            amount_col: 'sum'
        }).rename(columns={
            date_col: 'Recency',
            customer_col: 'Frequency',
            amount_col: 'Monetary'
        }).reset_index()
        
        # Handle negative recency (transactions unlikely after snapshot, but possible if data dirty)
        rfm['Recency'] = rfm['Recency'].clip(lower=0)
        
        print(f"RFM calculated for {len(rfm)} customers")
        print(f"Recency range: {rfm['Recency'].min()}-{rfm['Recency'].max()} days")
        print(f"Frequency range: {rfm['Frequency'].min()}-{rfm['Frequency'].max()} transactions")
        print(f"Monetary range: {rfm['Monetary'].min():.2f}-{rfm['Monetary'].max():.2f}")
        
        self.rfm_data = rfm
        return rfm
    
    def preprocess_rfm(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale RFM features before clustering.
        
        Args:
            rfm_df: DataFrame with RFM metrics.
            
        Returns:
            pd.DataFrame: DataFrame with scaled RFM features.
        """
        rfm_scaled = rfm_df.copy()
        
        # Scale RFM features
        features_to_scale = ['Recency', 'Frequency', 'Monetary']
        rfm_scaled[features_to_scale] = self.scaler.fit_transform(rfm_df[features_to_scale])
        
        print("RFM features scaled successfully")
        return rfm_scaled
    
    def cluster_customers(
        self, 
        rfm_scaled: pd.DataFrame,
        n_clusters: int = 3,
        random_state: int = settings.RANDOM_STATE
    ) -> pd.DataFrame:
        """
        Perform K-Means clustering on scaled RFM features.
        
        Args:
            rfm_scaled: DataFrame with scaled RFM features.
            n_clusters: Number of clusters (default: 3).
            random_state: Random seed for reproducibility.
            
        Returns:
            pd.DataFrame: DataFrame with cluster assignments.
        """
        # Extract features for clustering
        X = rfm_scaled[['Recency', 'Frequency', 'Monetary']].values
        
        # K-Means clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        rfm_scaled['RFM_Cluster'] = self.kmeans.fit_predict(X)
        
        print(f"K-Means clustering complete with {n_clusters} clusters")
        print(f"Cluster distribution:\n{rfm_scaled['RFM_Cluster'].value_counts().sort_index()}")
        
        return rfm_scaled
    
    def create_risk_target(
        self, 
        rfm_clustered: pd.DataFrame,
        high_risk_cluster: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Create binary 'is_high_risk' target variable.
        Strategy: Customers with high recency, low frequency, and low monetary value
        are considered high risk.
        
        Args:
            rfm_clustered: DataFrame with cluster assignments.
            high_risk_cluster: Specific cluster to mark as high risk.
                              If None, automatically determine based on RFM profile.
            
        Returns:
            pd.DataFrame: DataFrame with 'is_high_risk' column.
        """
        rfm_with_target = rfm_clustered.copy()
        
        if high_risk_cluster is None:
            # Analyze cluster characteristics to identify high-risk cluster
            # High risk = High Recency (inactive) + Low Frequency + Low Monetary
            cluster_analysis = rfm_with_target.groupby('RFM_Cluster').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean'
            })
            
            print("\nCluster Analysis (scaled features):")
            print(cluster_analysis)
            
            # Score each cluster: high recency is bad (+), high frequency/monetary is good (-)
            # We want to find the cluster with Max(Recency - Frequency - Monetary)
            # Since features are scaled, this simple linear combination works reasonably well for basic ranking.
            cluster_analysis['Risk_Score'] = (
                cluster_analysis['Recency'] - 
                cluster_analysis['Frequency'] - 
                cluster_analysis['Monetary']
            )
            
            high_risk_cluster = int(cluster_analysis['Risk_Score'].idxmax())
            print(f"\nAutomatically identified high-risk cluster: {high_risk_cluster}")
        
        # Create binary target
        rfm_with_target['is_high_risk'] = (
            rfm_with_target['RFM_Cluster'] == high_risk_cluster
        ).astype(int)
        
        risk_count = rfm_with_target['is_high_risk'].sum()
        risk_rate = rfm_with_target['is_high_risk'].mean()
        
        print(f"\nHigh-risk customers: {risk_count} ({risk_rate:.2%})")
        
        return rfm_with_target
    
    def merge_target_to_transactions(
        self,
        transactions_df: pd.DataFrame,
        rfm_with_target: pd.DataFrame,
        customer_col: str = 'CustomerId'
    ) -> pd.DataFrame:
        """
        Merge proxy target back into main transaction dataset.
        
        Args:
            transactions_df: Original transaction DataFrame.
            rfm_with_target: RFM DataFrame with 'is_high_risk' column.
            customer_col: Name of customer ID column.
            
        Returns:
            pd.DataFrame: Transaction DataFrame with 'is_high_risk' column.
        """
        # Select only customer ID and target
        target_mapping = rfm_with_target[[customer_col, 'is_high_risk']].copy()
        
        # Merge back to transactions
        transactions_with_target = transactions_df.merge(
            target_mapping,
            on=customer_col,
            how='left'
        )
        
        # Fill any missing values (shouldn't happen but safety check)
        transactions_with_target['is_high_risk'].fillna(0, inplace=True)
        transactions_with_target['is_high_risk'] = transactions_with_target['is_high_risk'].astype(int)
        
        print(f"\nProxy target merged to {len(transactions_with_target)} transactions")
        print(f"High-risk transaction rate: {transactions_with_target['is_high_risk'].mean():.2%}")
        
        return transactions_with_target
    
    def full_rfm_pipeline(
        self,
        df: pd.DataFrame,
        customer_col: str = 'CustomerId',
        date_col: str = settings.DATE_COL,
        amount_col: str = 'Amount',
        n_clusters: int = 3,
        random_state: int = settings.RANDOM_STATE
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute complete RFM analysis pipeline.
        
        Args:
            df: Transaction DataFrame.
            customer_col: Customer ID column.
            date_col: Date column.
            amount_col: Amount column.
            n_clusters: Number of clusters.
            random_state: Random seed.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (transactions_with_target, rfm_with_target)
        """
        print("="*70)
        print("RFM ANALYSIS PIPELINE")
        print("="*70)
        
        # Step 1: Calculate RFM
        rfm = self.calculate_rfm(df, customer_col, date_col, amount_col)
        
        # Step 2: Preprocess/Scale RFM
        rfm_scaled = self.preprocess_rfm(rfm)
        
        # Step 3: K-Means Clustering
        rfm_clustered = self.cluster_customers(rfm_scaled, n_clusters, random_state)
        
        # Step 4: Create proxy target
        rfm_with_target = self.create_risk_target(rfm_clustered)
        
        # Step 5: Merge back to transactions
        transactions_with_target = self.merge_target_to_transactions(df, rfm_with_target, customer_col)
        
        print("="*70)
        print("RFM ANALYSIS COMPLETE")
        print("="*70)
        
        return transactions_with_target, rfm_with_target
