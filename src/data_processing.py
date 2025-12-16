"""
Data Processing Module with sklearn.pipeline
Handles data loading, feature engineering, and preprocessing for credit risk modeling
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


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


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract temporal features from TransactionStartTime"""
    
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.datetime_col in X.columns:
            X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
            X['TransactionHour'] = X[self.datetime_col].dt.hour
            X['TransactionDayOfWeek'] = X[self.datetime_col].dt.dayofweek
            X['TransactionMonth'] = X[self.datetime_col].dt.month
            X['TransactionYear'] = X[self.datetime_col].dt.year
        return X


class AggregateFeatureCreator(BaseEstimator, TransformerMixin):
    """Create customer-level aggregate features"""
    
    def __init__(self):
        self.customer_stats = None
    
    def fit(self, X, y=None):
        # Calculate aggregates during fit
        if 'CustomerId' in X.columns:
            self.customer_stats = X.groupby('CustomerId').agg({
                'Amount': ['sum', 'mean', 'std', 'count', 'min', 'max'],
                'Value': ['sum', 'mean'],
            }).reset_index()
            
            # Flatten column names
            self.customer_stats.columns = ['_'.join(col).strip('_') for col in self.customer_stats.columns.values]
            self.customer_stats.rename(columns={'CustomerId_': 'CustomerId'}, inplace=True)
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.customer_stats is not None and 'CustomerId' in X.columns:
            X = X.merge(self.customer_stats, on='CustomerId', how='left', suffixes=('', '_agg'))
        return X


class WoEIVTransformer(BaseEstimator, TransformerMixin):
    """Weight of Evidence and Information Value transformation"""
    
    def __init__(self, target_col='FraudResult', categorical_cols=None):
        self.target_col = target_col
        self.categorical_cols = categorical_cols or []
        self.woe_encoders = {}
        
    def fit(self, X, y=None):
        """Fit WoE encoders for categorical variables"""
        try:
            from xverse.transformer import WOE
            
            if self.target_col in X.columns:
                for col in self.categorical_cols:
                    if col in X.columns:
                        woe = WOE()
                        # WOE requires both X and y
                        temp_df = X[[col]].copy()
                        woe.fit(temp_df, X[self.target_col])
                        self.woe_encoders[col] = woe
        except ImportError:
            print("Warning: xverse not installed. Skipping WoE transformation.")
        except Exception as e:
            print(f"Warning: WoE fitting failed: {e}")
        
        return self
    
    def transform(self, X):
        """Transform categorical variables using WoE"""
        X = X.copy()
        
        for col, woe in self.woe_encoders.items():
            if col in X.columns:
                try:
                    temp_df = X[[col]].copy()
                    transformed = woe.transform(temp_df)
                    X[f'{col}_WoE'] = transformed[col]
                except Exception as e:
                    print(f"Warning: WoE transform failed for {col}: {e}")
        
        return X


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
            df['TransactionYear'] = df['TransactionStartTime'].dt.year
        
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
    """Preprocess data for machine learning using sklearn.pipeline"""
    
    def __init__(self, use_woe=True):
        """
        Initialize DataPreprocessor
        
        Args:
            use_woe: Whether to use WoE/IV transformation
        """
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.pipeline = None
        self.use_woe = use_woe
        
    def build_preprocessing_pipeline(
        self, 
        numeric_features: List[str],
        categorical_features: List[str]
    ) -> Pipeline:
        """
        Build sklearn pipeline for preprocessing
        
        Args:
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
            
        Returns:
            Configured preprocessing pipeline
        """
        # Numeric transformer
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical transformer (Label Encoding)
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])
        
        self.pipeline = pipeline
        return pipeline
        
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
                       'CustomerId', 'ProductId', 'FraudResult', 'is_high_risk']
        
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
                                      'TransactionStartTime', 'FraudResult', 'is_high_risk']]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Store feature columns
        self.feature_columns = feature_cols
        
        # Determine if stratification is safe
        n_samples = len(X)
        n_test_samples = int(n_samples * test_size)
        n_train_samples = n_samples - n_test_samples
        
        # Check if we have enough samples for stratification
        # Need at least 2 samples per class in both train and test
        can_stratify = True
        if n_test_samples < 2 or n_train_samples < 2:
            can_stratify = False
        else:
            # Check class distribution
            class_counts = y.value_counts()
            if class_counts.min() < 2:
                can_stratify = False
        
        # Train-test split with conditional stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if can_stratify else None
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Fraud rate - Train: {y_train.mean():.4f}, Test: {y_test.mean():.4f}")
        
        return X_train, X_test, y_train, y_test
