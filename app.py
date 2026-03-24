from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

# Load saved artifacts
model = joblib.load("artifacts/car_price_model.pkl")
feature_names = joblib.load("artifacts/feature_names.pkl")

app = FastAPI(title="Car Price Predictor")

# Define what input the API expects
class CarFeatures(BaseModel):
    Engine_HP: float
    Engine_Cylinders: float
    highway_MPG: int
    city_mpg: int
    Popularity: int
    Year: int
    Vehicle_Size: int  # 0=Compact, 1=Midsize, 2=Large

@app.get("/")
def home():
    return {"status": "Car Price Predictor is running"}

@app.post("/predict")
def predict(car: CarFeatures):
    # Build a dataframe from input
    data = {
        "Engine HP": car.Engine_HP,
        "Engine Cylinders": car.Engine_Cylinders,
        "highway MPG": car.highway_MPG,
        "city mpg": car.city_mpg,
        "Popularity": car.Popularity,
        "Year": car.Year,
        "Vehicle Size": car.Vehicle_Size,
    }

    # Create dataframe with all expected columns, fill missing with 0
    df_input = pd.DataFrame([data])
    df_input = df_input.reindex(columns=feature_names, fill_value=0)

    # Predict and reverse log transform
    prediction_log = model.predict(df_input)[0]
    prediction = np.expm1(prediction_log)

    return {
        "predicted_price": f"${prediction:,.0f}"
    }