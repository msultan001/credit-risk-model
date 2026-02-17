"""
FastAPI Application for Credit Risk Prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from predict import ModelPredictor
from api.pydantic_models import (
    TransactionInput, PredictionOutput,
    BatchTransactionInput, BatchPredictionOutput,
    ModelInfo
)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for fraud detection and credit risk assessment",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (will be loaded on startup)
predictor = None


@app.on_event("startup")
async def load_model():
    """Load model on startup - try MLflow registry first, then fallback to pickle"""
    global predictor
    try:
        # Try loading from MLflow Model Registry first
        try:
            import mlflow
            model_name = "credit-risk-best-model"
            model_version = "latest"
            
            # Load from MLflow Model Registry
            model_uri = f"models:/{model_name}/{model_version}"
            print(f"Attempting to load model from MLflow: {model_uri}")
            
            # For now, fall through to pickle loading
            # MLflow registry integration would require proper MLflow server setup
            raise Exception("MLflow registry not configured, using pickle fallback")
            
        except Exception as mlflow_error:
            print(f"MLflow loading failed: {mlflow_error}")
            print("Falling back to pickle files...")
            
            # Fallback to pickle files
            predictor = ModelPredictor(
                model_path='models/xgboost.pkl',
                preprocessor_path='models/preprocessor.pkl'
            )
            print("Model loaded successfully from pickle files")
            
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("API will start but predictions will fail until model is trained")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Credit Risk Prediction API is running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    model_loaded = predictor is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "api_version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict_transaction(transaction: TransactionInput):
    """
    Make prediction for a single transaction
    
    Args:
        transaction: Transaction input data
        
    Returns:
        Prediction result with fraud probability
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Convert to dict
        transaction_dict = transaction.dict()
        
        # Make prediction
        result = predictor.predict_single(transaction_dict)
        
        # Determine risk level
        fraud_prob = result['fraud_probability']
        if fraud_prob < 0.3:
            risk_level = "LOW"
        elif fraud_prob < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        return PredictionOutput(
            prediction=result['prediction'],
            fraud_probability=result['fraud_probability'],
            legitimate_probability=result['legitimate_probability'],
            risk_level=risk_level
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(batch_input: BatchTransactionInput):
    """
    Make predictions for multiple transactions
    
    Args:
        batch_input: Batch of transactions
        
    Returns:
        Batch prediction results
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Convert to list of dicts
        transactions = [t.dict() for t in batch_input.transactions]
        
        # Make batch prediction
        results = predictor.predict_batch(transactions)
        
        # Format output
        predictions = []
        fraud_count = 0
        
        for result in results:
            fraud_prob = result['fraud_probability']
            
            # Determine risk level
            if fraud_prob < 0.3:
                risk_level = "LOW"
            elif fraud_prob < 0.7:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            
            predictions.append(PredictionOutput(
                prediction=result['prediction'],
                fraud_probability=result['fraud_probability'],
                legitimate_probability=result['legitimate_probability'],
                risk_level=risk_level
            ))
            
            if result['prediction'] == 1:
                fraud_count += 1
        
        total = len(predictions)
        fraud_rate = fraud_count / total if total > 0 else 0
        
        return BatchPredictionOutput(
            predictions=predictions,
            total_transactions=total,
            fraud_count=fraud_count,
            fraud_rate=fraud_rate
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        feature_count = len(predictor.preprocessor.feature_columns) if predictor.preprocessor.feature_columns else 0
        
        return ModelInfo(
            model_name="XGBoost Fraud Detector",
            version="1.0.0",
            features_count=feature_count
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
