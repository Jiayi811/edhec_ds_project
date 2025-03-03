# Car Price Prediction with Machine Learning

## Project Description
This project aims to predict car prices based on user-provided features using a machine learning model. We processed real-world car data, trained an **XGBoost model**, and deployed an API using **FastAPI**. The model provides an estimated price for a car based on its attributes.

## Dataset Information
- **Source**: [Kaggle - Car Prices Dataset](https://www.kaggle.com/datasets/sidharth178/car-prices-dataset)
- **Key Features**: `Levy`, `Mileage`, `Engine_volume`, `Manufacturer`, `Model`, `Category`, `Fuel_type`, `Gear_box_type`, etc.
- **Preprocessing**: Handling missing values, feature encoding, scaling, and outlier removal.

## Model & Performance
- **Algorithm Used**: XGBoost
- **Final Model Performance**:
  - **RMSE**: 6429.74
  - **R² Score**: 0.8019
- **Best Hyperparameters**:
  ```json
  {
    "colsample_bytree": 0.7380,
    "gamma": 0.3476,
    "learning_rate": 0.0132,
    "max_depth": 11,
    "min_child_weight": 1,
    "n_estimators": 744,
    "subsample": 0.7843
  }
  ```

## API Usage (FastAPI)
### Run API Locally
```sh
uvicorn main:app --reload
```

### API Endpoint
- **POST /predict/**
  - **Input (JSON format)**:
    ```json
    {
      "Levy": 0,
      "Mileage": 10000,
      "Engine_volume": 2.0,
      "Manufacturer": "Toyota",
      "Model": "Corolla",
      "Category": "Sedan",
      "Leather_interior": "Yes",
      "Fuel_type": "Petrol",
      "Gear_box_type": "Manual",
      "Drive_wheels": "FWD",
      "Doors": 4,
      "Wheel": "16",
      "Color": "Blue"
    }
    ```
  - **Output (Predicted Price JSON)**:
    ```json
    {
      "predicted_price": 15400.75
    }
    ```

## Installation & Setup
### Clone Repository
```sh
git clone https://github.com/Jiayi811/edhec_ds_project.git
cd edhec_ds_project
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Deployment
- **Platform**: Render https://edhec-ds-project.onrender.com

## Project Structure
```
📂 edhec_ds_project/
 ┣ 📜 car_price_model.ipynb    # Jupyter Notebook (Data Analysis & Model Training)
 ┣ 📜 app.py                   # FastAPI script for prediction API
 ┣ 📜 requirements.txt         # Dependencies
 ┣ 📜 README.md               # Project documentation
 ┣ 📜 best_xgb_model.pkl       # Trained model file
 ┣ 📜 train.csv                # Dataset
 ┣ 📜 Report.docx              # Project report
 ┗ 📂 images/                  # API success screenshots
```

## Contributors & Contact
- **Author**: Jiayi QIAN， Shujuan ZHU
- **Contact**: jiayi.qian@edhec.com, shujuan.zhu@edhec.com



