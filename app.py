from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

import pandas as pd



# 加载已训练的 XGBoost 模型
with open("best_xgb_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# 初始化 FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"message": "FastAPI is running!"}

# 定义请求数据格式
class CarFeatures(BaseModel):
    Manufacturer: int
    Model: int
    Category: int
    Leather_interior: int # Leather interior
    Fuel_type: int # Fuel type
    Engine_volume: float # Engine volume
    Mileage: float
    Cylinders: float
    Gear_box_type: int # Gear box type
    Drive_wheels: int #Drive wheels
    Doors: int
    Wheel: int
    Color: int
    Airbags: int
    Car_Age: int # Car Age
    Drive_4x4: bool
    Drive_Front: bool
    Drive_Rear: bool
    Gear_box_Automatic: bool
    Gear_box_Manual: bool
    Gear_box_Tiptronic: bool
    Gear_box_Variator: bool
    Fuel_CNG: bool
    Fuel_Diesel: bool
    Fuel_Hybrid: bool
    Fuel_Hydrogen: bool
    Fuel_LPG: bool
    Fuel_Petrol: bool
    Fuel_Plug_in_Hybrid: bool #Fuel_Plug-in Hybrid 
    Levy_log: float

# 预测 API 端点

@app.post("/predict")
def predict_price(features: CarFeatures):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([features.dict()])

        # Column name mapping to match model feature names
        column_mapping = {
            "Leather_interior": "Leather interior",
            "Fuel_type": "Fuel type",
            "Gear_box_type": "Gear box type",
            "Engine_volume": "Engine volume",
            "Drive_wheels": "Drive wheels",
            "Car_Age": "Car Age",
            "Fuel_Plug_in_Hybrid": "Fuel_Plug-in Hybrid"
        }

        # Apply column renaming
        input_df.rename(columns=column_mapping, inplace=True)

        # Ensure input matches model's expected features
        model_features = model.feature_names_in_
        input_df = input_df[model_features]  # Keep only necessary columns

        # Make prediction
        predicted_price = model.predict(input_df)[0]

        return {"predicted_price": float(predicted_price)}

    except Exception as e:
        return {"error": str(e)}

print("✅ FastAPI Endpoints Created Successfully!")



