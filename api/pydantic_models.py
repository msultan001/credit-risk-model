"""
Pydantic Models for API Request/Response Validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class TransactionInput(BaseModel):
    """Input schema for transaction prediction"""
    
    Amount: float = Field(..., description="Transaction amount")
    Value: float = Field(..., description="Transaction value")
    ProductCategory: str = Field(..., description="Product category")
    ChannelId: str = Field(..., description="Channel ID")
    ProviderId: str = Field(..., description="Provider ID")
    CurrencyCode: str = Field(default="UGX", description="Currency code")
    CountryCode: str = Field(default="256", description="Country code")
    PricingStrategy: int = Field(default=2, description="Pricing strategy")
    
    class Config:
        json_schema_extra = {
            "example": {
                "Amount": 1000,
                "Value": 1000,
                "ProductCategory": "airtime",
                "ChannelId": "ChannelId_3",
                "ProviderId": "ProviderId_6",
                "CurrencyCode": "UGX",
                "CountryCode": "256",
                "PricingStrategy": 2
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for prediction response"""
    
    prediction: int = Field(..., description="0 = Legitimate, 1 = Fraud")
    fraud_probability: float = Field(..., description="Probability of fraud")
    legitimate_probability: float = Field(..., description="Probability of legitimate transaction")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH")


class BatchTransactionInput(BaseModel):
    """Input schema for batch predictions"""
    
    transactions: List[TransactionInput]


class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions"""
    
    predictions: List[PredictionOutput]
    total_transactions: int
    fraud_count: int
    fraud_rate: float


class ModelInfo(BaseModel):
    """Model information schema"""
    
    model_name: str
    version: str
    features_count: int
    training_date: Optional[str] = None
