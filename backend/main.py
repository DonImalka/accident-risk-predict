import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os

# Initialize FastAPI app
app = FastAPI(title="Accident Risk Prediction API")

# Enable CORS so frontend can access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for the model
model = None

# Load model at startup
@app.on_event("startup")
def load_model():
    global model
    try:
        model_path = "final_model.pkl"
        if not os.path.exists(model_path):
             print(f"Model file not found at {model_path}")
             return
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

# Define Input Schema using Pydantic with specific feature names
class PredictionInput(BaseModel):
    road_type: str = Field(..., description="Type of road")
    num_lanes: int = Field(..., description="Number of lanes")
    curvature: float = Field(..., description="Road curvature")
    speed_limit: float = Field(..., description="Speed limit")
    lighting: str = Field(..., description="Lighting conditions")
    weather: str = Field(..., description="Weather conditions")
    road_signs_present: int = Field(..., description="1 if signs present, 0 otherwise")
    public_road: int = Field(..., description="1 if public road, 0 otherwise")
    time_of_day: str = Field(..., description="Time of day")
    holiday: int = Field(..., description="1 if holiday, 0 otherwise")
    school_season: int = Field(..., description="1 if school season, 0 otherwise")
    num_reported_accidents: int = Field(..., description="Number of reported accidents")

# Prediction Endpoint
@app.post("/predict")
def predict_risk(input_data: PredictionInput):
    global model
    if model is None:
        # Attempt to load if not loaded (fallback)
        load_model()
        if model is None:
            raise HTTPException(status_code=500, detail="Model not active. Please check server logs.")
    
    try:
        # Convert input to Pandas DataFrame with correct column names
        data = {
            'road_type': [input_data.road_type],
            'num_lanes': [input_data.num_lanes],
            'curvature': [input_data.curvature],
            'speed_limit': [input_data.speed_limit],
            'lighting': [input_data.lighting],
            'weather': [input_data.weather],
            'road_signs_present': [input_data.road_signs_present],
            'public_road': [input_data.public_road],
            'time_of_day': [input_data.time_of_day],
            'holiday': [input_data.holiday],
            'school_season': [input_data.school_season],
            'num_reported_accidents': [input_data.num_reported_accidents]
        }
        
        df = pd.DataFrame(data)
        
        # Predict
        prediction = model.predict(df)
        
        # Handle prediction output
        if isinstance(prediction, (np.ndarray, list)):
            risk_value = float(prediction[0])
        else:
            risk_value = float(prediction)
        
        # Ensure risk is within reasonable bounds (0-1) if the model doesn't guarantee it
        # But for raw regression output, we just return what the model says.
        # If the user wanted strict probability, we might clip it, but "risk" can be a score.
        
        return {"accident_risk": risk_value}
        
    except Exception as e:
        # Handle invalid inputs and runtime errors gracefully
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)
