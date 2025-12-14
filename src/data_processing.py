"""
Data Processing Module
Handles data loading, feature engineering, and preprocessing for credit risk modeling
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataLoader:
    """Load and validate raw transaction data"""
    
    def __init__(self, filepath: str):
        """
        Initialize DataLoader
        
        Args:
            filepath: Path to CSV data file
        """
        self.filepath = filepath
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Returns:
            DataFrame with raw data
        """
        self.data = pd.read_csv(self.filepath)
        print(f"Loaded {len(self.data)} transactions from {self.filepath}")
        return self.data
    
    def validate_data(self) -> bool:
        """
        Validate data structure and required columns
        
        Returns:
            True if validation passes
        """
        required_columns = [
            'TransactionId', 'Amount', 'Value', 'FraudResult',
            'ProductCategory', 'ChannelId', 'ProviderId'
        ]
        
        missing_cols = [col for col in required_columns if col not in self.data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print("Data validation passed")
        return True


class FeatureEngineer:
    """Create derived features for credit risk modeling"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize FeatureEngineer
        
        Args:
            data: Input DataFrame
        """
        self.data = data.copy()
        
    def create_customer_aggregates(self) -> pd.DataFrame:
        """
        Create customer-level aggregated features
        
        Returns:
            DataFrame with customer aggregates
        """
        # Customer transaction statistics
        customer_stats = self.data.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'std', 'count', 'min', 'max'],
            'Value': ['sum', 'mean'],
            'FraudResult': ['sum', 'mean']
        }).reset_index()
        
        # Flatten column names
        customer_stats.columns = ['_'.join(col).strip('_') for col in customer_stats.columns.values]
        customer_stats.rename(columns={'CustomerId_': 'CustomerId'}, inplace=True)
        
        return customer_stats
    
    def create_transaction_features(self) -> pd.DataFrame:
        """
        Create transaction-level features
        
        Returns:
            DataFrame with new features
        """
        df = self.data.copy()
        
        # Amount vs Value difference (fees/markup)
        df['AmountValueDiff'] = df['Value'] - df['Amount']
        df['AmountValueRatio'] = df['Value'] / (df['Amount'] + 1)  # Avoid division by zero
        
        # Time-based features
        if 'TransactionStartTime' in df.columns:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
            df['TransactionHour'] = df['TransactionStartTime'].dt.hour
            df['TransactionDayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
            df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        
        # Product category diversity per customer
        product_diversity = df.groupby('CustomerId')['ProductCategory'].nunique().reset_index()
        product_diversity.columns = ['CustomerId', 'ProductDiversity']
        df = df.merge(product_diversity, on='CustomerId', how='left')
        
        return df
    
    def engineer_all_features(self) -> pd.DataFrame:
        """
        Apply all feature engineering steps
        
        Returns:
            DataFrame with engineered features
        """
        # Transaction-level features
        df = self.create_transaction_features()
        
        # Customer aggregates
        customer_agg = self.create_customer_aggregates()
        
        # Merge customer aggregates back to transaction level
        df = df.merge(
            customer_agg,
            on='CustomerId',
            how='left',
            suffixes=('', '_cust')
        )
        
        print(f"Feature engineering complete: {df.shape[1]} features")
        return df


class DataPreprocessor:
    """Preprocess data for machine learning"""
    
    def __init__(self):
        """Initialize DataPreprocessor"""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        
        # Fill numeric missing values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: Input DataFrame
            fit: Whether to fit encoders (True for training, False for inference)
            
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        
        categorical_cols = ['ProductCategory', 'ChannelId', 'ProviderId', 'CurrencyCode', 'CountryCode']
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in self.label_encoders[col].classes_ else 'unknown'
                )
                df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df: Input DataFrame
            fit: Whether to fit scaler
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        # Exclude ID columns and target
        exclude_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 
                       'CustomerId', 'ProductId', 'FraudResult']
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return df
    
    def prepare_features_target(
        self,
        df: pd.DataFrame,
        target_col: str = 'FraudResult',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare features and target for modeling
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col 
                       and col not in ['TransactionId', 'BatchId', 'AccountId', 
                                      'SubscriptionId', 'CustomerId', 'ProductId',
                                      'TransactionStartTime']]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Store feature columns
        self.feature_columns = feature_cols
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Fraud rate - Train: {y_train.mean():.4f}, Test: {y_test.mean():.4f}")
        
        return X_train, X_test, y_train, y_test
