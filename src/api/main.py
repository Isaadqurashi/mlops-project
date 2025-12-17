from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from typing import List

app = FastAPI(title="Stock Prediction API", version="1.0.0")

# Global variables to store models
models = {}

class PredictionInput(BaseModel):
    symbol: str = "AAPL"
    sma_20: float
    sma_50: float
    rsi: float
    macd: float

class PredictionOutput(BaseModel):
    prediction: float
    model_type: str
    symbol: str

@app.on_event("startup")
def load_models():
    """Load models on startup."""
    from pathlib import Path
    
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    model_dir = BASE_DIR / "models"
    
    print(f"Loading models from: {model_dir}")

    if not model_dir.exists():
        print(f"Models directory not found at {model_dir}")
        return

    # iterate over subdirs (symbols)
    for symbol_dir in model_dir.iterdir():
        if symbol_dir.is_dir():
            symbol = symbol_dir.name
            print(f"Found symbol directory: {symbol}")
            
            # Load Regression
            reg_path = symbol_dir / "regression_model.pkl"
            if reg_path.exists():
                try:
                    key = f"regression_{symbol}"
                    models[key] = joblib.load(reg_path)
                    print(f"Loaded {key} from {reg_path}")
                    
                    # Keep legacy 'regression' key pointing to AAPL for backward compat
                    if 'regression' not in models or symbol == "AAPL":
                         models['regression'] = models[key]
                except Exception as e:
                     print(f"Failed to load {reg_path}: {e}")

            # Load Classification
            clf_path = symbol_dir / "classification_model.pkl"
            if clf_path.exists():
                try:
                    key = f"classification_{symbol}"
                    models[key] = joblib.load(clf_path)
                    print(f"Loaded {key} from {clf_path}")
                    
                    if 'classification' not in models or symbol == "AAPL":
                         models['classification'] = models[key]
                except Exception as e:
                     print(f"Failed to load {clf_path}: {e}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": list(models.keys())}

@app.post("/predict/price", response_model=PredictionOutput)
def predict_price(input_data: PredictionInput):
    symbol = input_data.symbol
    model_key = f"regression_{symbol}"
    
    # Fallback to generic 'regression' if specific symbol not found
    if model_key not in models:
        if 'regression' in models:
            model_key = 'regression'
        else:
            raise HTTPException(status_code=503, detail=f"Regression model for {symbol} not loaded")
    
    features = [[input_data.sma_20, input_data.sma_50, input_data.rsi, input_data.macd]]
    prediction = models[model_key].predict(features)[0]
    return {"prediction": prediction, "model_type": str(type(models[model_key])), "symbol": symbol}

@app.post("/predict/direction", response_model=PredictionOutput)
def predict_direction(input_data: PredictionInput):
    symbol = input_data.symbol
    model_key = f"classification_{symbol}"
    
    if model_key not in models:
        if 'classification' in models:
             model_key = 'classification'
        else:
            raise HTTPException(status_code=503, detail=f"Classification model for {symbol} not loaded")
    
    features = [[input_data.sma_20, input_data.sma_50, input_data.rsi, input_data.macd]]
    prediction = models[model_key].predict(features)[0]
    return {"prediction": float(prediction), "model_type": str(type(models[model_key])), "symbol": symbol}

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    if 'regression' not in models:
        raise HTTPException(status_code=503, detail="Regression model not loaded")
    
    try:
        df = pd.read_csv(file.file)
        required_cols = ['sma_20', 'sma_50', 'rsi', 'macd']
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(status_code=400, detail=f"CSV must contain columns: {required_cols}")
        
        features = df[required_cols]
        predictions = models['regression'].predict(features)
        
        results = df.copy()
        results['predicted_price'] = predictions
        
        # Return as JSON records
        return results.to_dict(orient="records")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
