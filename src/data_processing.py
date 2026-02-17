"""
Data Processing Module
Handles data loading, feature engineering, and preprocessing for credit risk modeling.
Refactored to use centralized configuration and improved type hinting.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import settings

class DataLoader:
    """Load and validate raw transaction data."""
    
    def __init__(self, filepath: str = str(settings.RAW_DATA_PATH)):
        """
        Initialize DataLoader.
        
        Args:
            filepath: Path to CSV data file. Defaults to settings.RAW_DATA_PATH.
        """
        self.filepath = filepath
        self.data: Optional[pd.DataFrame] = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            pd.DataFrame: DataFrame with raw data.
        """
        try:
            self.data = pd.read_csv(self.filepath)
            print(f"Loaded {len(self.data)} transactions from {self.filepath}")
            return self.data
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.filepath}")
    
    def validate_data(self) -> bool:
        """
        Validate data structure and required columns.
        
        Returns:
            bool: True if validation passes.
            
        Raises:
            ValueError: If required columns are missing.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        missing_cols = [col for col in settings.REQUIRED_COLUMNS if col not in self.data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print("Data validation passed")
        return True


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract temporal features from datetime column."""
    
    def __init__(self, datetime_col: str = settings.DATE_COL):
        self.datetime_col = datetime_col
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.datetime_col in X.columns:
            X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
            X['TransactionHour'] = X[self.datetime_col].dt.hour
            X['TransactionDayOfWeek'] = X[self.datetime_col].dt.dayofweek
            X['TransactionMonth'] = X[self.datetime_col].dt.month
            X['TransactionYear'] = X[self.datetime_col].dt.year
        return X


class AggregateFeatureCreator(BaseEstimator, TransformerMixin):
    """Create customer-level aggregate features."""
    
    def __init__(self):
        self.customer_stats: Optional[pd.DataFrame] = None
    
    def fit(self, X: pd.DataFrame, y=None):
        # Calculate aggregates during fit
        if 'CustomerId' in X.columns:
            # Check which columns exist before aggregating
            agg_dict = {}
            if 'Amount' in X.columns:
                agg_dict['Amount'] = ['sum', 'mean', 'std', 'count', 'min', 'max']
            if 'Value' in X.columns:
                agg_dict['Value'] = ['sum', 'mean']
                
            if agg_dict:
                self.customer_stats = X.groupby('CustomerId').agg(agg_dict).reset_index()
                
                # Flatten column names
                self.customer_stats.columns = ['_'.join(col).strip('_') for col in self.customer_stats.columns.values]
                self.customer_stats.rename(columns={'CustomerId_': 'CustomerId'}, inplace=True)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.customer_stats is not None and 'CustomerId' in X.columns:
            X = X.merge(self.customer_stats, on='CustomerId', how='left', suffixes=('', '_agg'))
        return X


class WoEIVTransformer(BaseEstimator, TransformerMixin):
    """Weight of Evidence and Information Value transformation."""
    
    def __init__(self, target_col: str = settings.TARGET_COL, categorical_cols: List[str] = None):
        self.target_col = target_col
        self.categorical_cols = categorical_cols or settings.CATEGORICAL_COLS
        self.woe_encoders = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit WoE encoders for categorical variables."""
        if not settings.USE_WOE:
            return self
            
        try:
            from xverse.transformer import WOE
            
            if self.target_col in X.columns:
                for col in self.categorical_cols:
                    if col in X.columns:
                        woe = WOE()
                        # WOE requires both X and y. Here X usually contains target if passed as single DF
                        # But standard sklearn fit takes X, y.
                        # We'll assume if y is None, target is in X, otherwise use y.
                        
                        if y is None:
                            target = X[self.target_col]
                            features = X[[col]]
                        else:
                            target = y
                            features = X[[col]]

                        try:
                            woe.fit(features, target)
                            self.woe_encoders[col] = woe
                        except Exception as e:
                             print(f"Failed to fit WoE for {col}: {e}")

        except ImportError:
            print("Warning: xverse not installed. Skipping WoE transformation.")
        except Exception as e:
            print(f"Warning: WoE fitting failed: {e}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical variables using WoE."""
        if not self.woe_encoders:
            return X
            
        X = X.copy()
        
        for col, woe in self.woe_encoders.items():
            if col in X.columns:
                try:
                    temp_df = X[[col]].copy()
                    transformed = woe.transform(temp_df)
                    # Check if transformed is a DataFrame or key-value
                    if isinstance(transformed, pd.DataFrame):
                         # xverse usually returns the dataframe with replaced values or new columns
                         # We want to add a specific column
                         if col in transformed.columns:
                             X[f'{col}_WoE'] = transformed[col]
                except Exception as e:
                    print(f"Warning: WoE transform failed for {col}: {e}")
        
        return X


class FeatureEngineer:
    """Create derived features for credit risk modeling."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        
    def create_transaction_features(self) -> pd.DataFrame:
        """Create transaction-level features."""
        df = self.data.copy()
        
        # Amount vs Value difference (fees/markup)
        if 'Value' in df.columns and 'Amount' in df.columns:
            df['AmountValueDiff'] = df['Value'] - df['Amount']
            df['AmountValueRatio'] = df['Value'] / (df['Amount'] + 1)  # Avoid division by zero
        
        # Interaction Features
        if settings.CREATE_INTERACTION_FEATURES and 'PricingStrategy' in df.columns and 'Amount' in df.columns:
             df['Amount_Pricing_Interaction'] = df['Amount'] * df['PricingStrategy']

        # Time-based features
        if settings.DATE_COL in df.columns:
            df[settings.DATE_COL] = pd.to_datetime(df[settings.DATE_COL])
            df['TransactionHour'] = df[settings.DATE_COL].dt.hour
            df['TransactionDayOfWeek'] = df[settings.DATE_COL].dt.dayofweek
            df['TransactionMonth'] = df[settings.DATE_COL].dt.month
            df['TransactionYear'] = df[settings.DATE_COL].dt.year
        
        # Product category diversity per customer
        if 'CustomerId' in df.columns and 'ProductCategory' in df.columns:
            product_diversity = df.groupby('CustomerId')['ProductCategory'].nunique().reset_index()
            product_diversity.columns = ['CustomerId', 'ProductDiversity']
            df = df.merge(product_diversity, on='CustomerId', how='left')
        
        return df
    
    def engineer_all_features(self) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        # Transaction-level features
        df = self.create_transaction_features()
        
        # Use simple aggregation for compatibility with existing tests/logic
        # For more complex pipelines, use AggregateFeatureCreator
        if 'CustomerId' in df.columns:
             agg_cols = {}
             if 'Amount' in df.columns:
                 agg_cols['Amount'] = ['sum', 'mean', 'std', 'count', 'min', 'max']
             if 'Value' in df.columns:
                 agg_cols['Value'] = ['sum', 'mean']
             if 'FraudResult' in df.columns:
                 agg_cols['FraudResult'] = ['sum', 'mean']
             
             if agg_cols:
                customer_stats = df.groupby('CustomerId').agg(agg_cols).reset_index()
                customer_stats.columns = ['_'.join(col).strip('_') for col in customer_stats.columns.values]
                customer_stats.rename(columns={'CustomerId_': 'CustomerId'}, inplace=True)
                
                df = df.merge(
                    customer_stats,
                    on='CustomerId',
                    how='left',
                    suffixes=('', '_cust')
                )
        
        print(f"Feature engineering complete: {df.shape[1]} features")
        return df


class DataPreprocessor:
    """Preprocess data for machine learning using sklearn.pipeline."""
    
    def __init__(self, use_woe: bool = settings.USE_WOE):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns: Optional[List[str]] = None
        self.pipeline: Optional[Pipeline] = None
        self.use_woe = use_woe
        
    def build_preprocessing_pipeline(
        self, 
        numeric_features: List[str],
        categorical_features: List[str]
    ) -> Pipeline:
        """Build sklearn pipeline for preprocessing."""
        
        # Numeric transformer
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical transformer (Label Encoding logic is custom, but for pipeline we can use generic)
        # For sklearn pipeline compatibility, usually OneHotEncoder or OrdinalEncoder is used.
        # But existing code used LabelEncoding. We will keep it simple here.
        # This function was illustrative in original code.
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            # We would typically add encoding here
        ])
        
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
        """Handle missing values."""
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
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown', inplace=True)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()
        
        categorical_cols = [col for col in settings.CATEGORICAL_COLS if col in df.columns]
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                # Ensure strings
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    # Note: LabelEncoder isn't great for unseen data.
                    # Best practice: use a dedicated category encoder or handle 'unknown'.
                    # Here we map unknown to a known class or error, but let's try to handle gracefully.
                    known_classes = set(self.label_encoders[col].classes_)
                    df[col] = df[col].astype(str).apply(lambda x: x if x in known_classes else list(known_classes)[0]) # Fallback to first class
                    df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        df = df.copy()
        
        exclude_cols = settings.ID_COLS + [settings.TARGET_COL, 'is_high_risk']
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not numeric_cols:
            return df

        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return df
    
    def prepare_features_target(
        self,
        df: pd.DataFrame,
        target_col: str = settings.TARGET_COL,
        test_size: float = settings.TEST_SIZE,
        random_state: int = settings.RANDOM_STATE
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare features and target for modeling."""
        
        # Identify feature columns (exclude IDs and Target)
        # Using a deny-list approach
        exclude_cols = settings.ID_COLS + [settings.DATE_COL, target_col, 'is_high_risk']
                        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target_col]
        
        self.feature_columns = feature_cols
        
        # Check stratification
        n_samples = len(X)
        n_test_samples = int(n_samples * test_size)
        n_train_samples = n_samples - n_test_samples
        
        can_stratify = True
        if n_test_samples < 2 or n_train_samples < 2:
            can_stratify = False
        else:
            if y.value_counts().min() < 2:
                can_stratify = False
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if can_stratify else None
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
