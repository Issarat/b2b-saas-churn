from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["industry", "company_size", "plan_tier", "billing_cycle"]
TARGET = "is_churned"

def main():
    csv_path = next((ROOT / "data").rglob("MP_Synthetics_50000_Rows.csv"))
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    df_full_train, _ = train_test_split(df, test_size=0.2, random_state=11)
    df_train, _ = train_test_split(
        df_full_train, test_size=0.25, random_state=11
    )

    pipeline = make_pipeline(
        DictVectorizer(sparse=False),
        LogisticRegression(),
    )

    records = df_train[FEATURES].to_dict(orient="records")
    pipeline.fit(records, df_train[TARGET])

    artifact_path = ROOT / "churn_model.joblib"
    joblib.dump(pipeline, artifact_path)
    print(f"Saved model to {artifact_path}")

if __name__ == "__main__":
    main()