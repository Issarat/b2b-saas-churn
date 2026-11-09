# B2B SaaS Customer Churn Classification

[Machine Learning Zoomcamp midterm project](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/projects). This project compares a machine
learning model with an existing churn risk score and serves account-level
predictions through a Dockerized FastAPI API.

## Problem Description

Customer Success teams have limited time to review accounts. If the team can
review only 10% of its accounts, can a machine learning model identify more
churned accounts than the existing risk score?

The dataset contains 50,000 synthetic B2B SaaS accounts. The target is
`is_churned`, where `1` indicates churn and `0` indicates no churn.
The API model uses four account features: `industry`, `company_size`,
`plan_tier`, and `billing_cycle`.

## Key Findings

- Enterprise accounts contribute **75.78% of total monthly recurring revenue
  (MRR)** and **68.89% of the MRR associated with churned accounts**.
- Starter has the highest account churn rate at **2.28%**.

## Held-Out Test Results

Recall@10% is the percentage of all churned accounts found among the
10% of accounts ranked highest. Both methods reviewed 1,000 accounts
from the same test set, which contained 183 churned accounts.
A fixed random seed was used to break score ties.

| Method                      | Churned accounts found | Recall@10% | Average Precision |
| --------------------------- | ---------------------: | ---------: | ----------------: |
| Selected ML model           |              16 of 183 |      8.74% |            0.0193 |
| Existing risk score         |              42 of 183 |     22.95% |            0.0763 |
| Random selection (expected) |            18.3 of 183 |     10.00% |                 — |

The existing risk score found **26 more churned accounts** under the same
review budget. The evaluation measures ranking performance on a synthetic
snapshot. Evaluating future churn would require dated account snapshots
and a defined prediction window.

See [the notebook](b2b-saas-predict.ipynb) for exploratory analysis,
charts, and model comparisons.

## Train the Model

Download [the dataset from Kaggle](https://www.kaggle.com/datasets/mpsynthetics/b2b-saas-churn-dataset)
and extract `MP_Synthetics_50000_Rows.csv` into `data/`.

From the repository root, run the following commands with Python 3.13.

```bash
python -m pip install -r requirements.txt
python src/train.py
```

The training script writes `churn_model.joblib` to the repository root.
Generate this file before building the Docker image.

## Run the API with Docker

```bash
docker build -t b2b-saas-churn .
docker run --rm -p 8000:8000 b2b-saas-churn
```

Open [Swagger UI](http://127.0.0.1:8000/docs) to try the endpoints.

| Method | Endpoint          | Purpose                                                     |
| ------ | ----------------- | ----------------------------------------------------------- |
| POST   | `/predict`        | Return a churn score for one account                        |
| POST   | `/predict-params` | List the account categories recognized by the trained model |

Example JSON body for `POST /predict`

```json
{
  "industry": "HealthTech",
  "company_size": "11-50",
  "plan_tier": "Starter",
  "billing_cycle": "Annual"
}
```

Example response

```json
{
  "churn_score": 0.019541941788806555
}
```

Use the category values returned by `/predict-params` for your trained model.
The exact score can vary with the training configuration.

<!-- After verifying the public deployment, add a Cloud Demo section
     with the public Swagger URL. -->
