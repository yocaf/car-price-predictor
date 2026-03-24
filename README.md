# 🚗 Car Price Predictor

A machine learning API that predicts car prices based on vehicle specifications.

## Results
- **Model**: XGBoost
- **R²**: 0.9910
- **Average Price Error**: $3,259

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- FastAPI, Uvicorn
- MLflow, SHAP
- Git, GitHub

## Project Structure
- `eda.ipynb` — data exploration, cleaning, feature engineering, model training
- `app.py` — FastAPI prediction server
- `artifacts/` — saved model and encodings

## How to Run Locally
```bash
git clone https://github.com/yocaf/car-price-predictor.git
cd car-price-predictor
python -m venv env
source env/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```
Then open http://127.0.0.1:8000/docs to test the API.

## What I Learned
- Full ML pipeline from raw data to deployed API
- Feature engineering: target encoding, one-hot encoding, log transformation
- Model comparison: Linear Regression vs Random Forest vs XGBoost
- Hyperparameter tuning with GridSearchCV
- Experiment tracking with MLflow
- Model explainability with SHAP
