from pathlib import Path
import pandas as pd

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
pipeline = joblib.load(ROOT / "churn_model.joblib")

app = FastAPI(title="B2B SaaS Churn API")

class Account(BaseModel):
    industry: str
    company_size: str
    plan_tier: str
    billing_cycle: str

@app.post("/predict")
def predict(account: Account):
    score = pipeline.predict_proba([account.model_dump()])[0, 1]
    return {"churn_score": float(score)}

@app.post("/predict-params")
def get_predict_params():
    vectorizer = pipeline.named_steps["dictvectorizer"]
    options = {field: [] for field in Account.model_fields}

    for feature in vectorizer.get_feature_names_out():
        field, separator, value = feature.partition("=")
        if separator and field in options:
            options[field].append(value)

    return options