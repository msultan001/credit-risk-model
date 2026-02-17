"""
Configuration Management Module
Uses Pydantic for robust setting management and environment variable loading.
"""

from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """
    Application Settings
    """
    # Project Paths
    DATA_DIR: Path = Field(default=BASE_DIR / "data")
    RAW_DATA_PATH: Path = Field(default=BASE_DIR / "data/raw/data.csv")
    PROCESSED_DATA_DIR: Path = Field(default=BASE_DIR / "data/processed")
    MODELS_DIR: Path = Field(default=BASE_DIR / "models")
    REPORTS_DIR: Path = Field(default=BASE_DIR / "reports")
    
    # Data Processing
    REQUIRED_COLUMNS: List[str] = [
        'TransactionId', 'Amount', 'Value', 'FraudResult',
        'ProductCategory', 'ChannelId', 'ProviderId', 'PricingStrategy'
    ]
    CATEGORICAL_COLS: List[str] = [
        'ProductCategory', 'ChannelId', 'ProviderId', 'CurrencyCode', 'CountryCode'
    ]
    NUMERIC_COLS: List[str] = ['Amount', 'Value', 'PricingStrategy']
    DATE_COL: str = 'TransactionStartTime'
    TARGET_COL: str = 'FraudResult'
    
    # Validation & Preprocessing
    ID_COLS: List[str] = [
        'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 
        'CustomerId', 'ProductId'
    ]
    UNKNOWN_CAT_LABEL: str = 'unknown'
    
    # Model Training
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 3
    SCORING_METRIC: str = 'roc_auc'
    EXPERIMENT_NAME: str = "credit-risk-modeling"
    
    # Feature Engineering
    CREATE_INTERACTION_FEATURES: bool = True
    USE_WOE: bool = True
    
    # Dashboard
    DASHBOARD_TITLE: str = "Credit Risk Assessment Dashboard"
    
    class Config:
        case_sensitive = True
        env_file = ".env"

# Create global settings instance
settings = Settings()

# Ensure directories exist
for path in [settings.DATA_DIR, settings.PROCESSED_DATA_DIR, settings.MODELS_DIR, settings.REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
