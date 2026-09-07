# Telco Customer Churn Prediction — ML Pipeline

**NextBridge Summer Internship 2026 — AI/ML Track**

## 1. Problem Type & Target

- **Problem type:** Binary Classification
- **Target column:** `Churn` (`Yes` = 1, `No` = 0)
- **Dataset:** Telco Customer Churn (IBM sample, Kaggle) — 7,043 rows, 21 columns

> `data/raw/telco_churn.csv` in this repo is the **real Kaggle dataset**
> (`WA_Fn-UseC_-Telco-Customer-Churn.csv`, renamed), 7,043 rows / 21 columns.

## 2. Setup

```bash
# create virtual environment
uv venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

# install pinned dependencies
uv pip install -r pyproject.toml
# or simply:
uv sync
```

## 3. How to Run

```bash
python main.py
```

This single command runs the entire pipeline:
`load data → clean/preprocess → train baseline → compare models (CV) →
tune best model → evaluate on test set → save model + results + plots`

Outputs after running:
- `models/best_model.joblib` — the final trained model
- `results/model_comparison.csv` — all metrics for all models
- `results/confusion_matrix.png`
- `results/feature_importance.png`

## 4. Results Summary

| Model                          | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---------------------------------|:--------:|:---------:|:------:|:--------:|:-------:|
| Logistic Regression (Baseline)  | 0.806    | 0.657     | 0.559  | 0.604    | 0.842   |
| Logistic Regression (Tuned)     | 0.806    | 0.657     | 0.559  | 0.604    | 0.842   |

**Primary metric chosen: Recall.** For churn prediction, missing a customer
who is actually going to churn (false negative) is costlier than wrongly
flagging a loyal customer (false positive) — a missed churner is lost
revenue, while a false alarm just costs a retention email/call.

**5-fold Cross-Validation (mean recall ± std):**
- Logistic Regression: 0.5478 ± 0.0249
- Random Forest: 0.4796 ± 0.0192

**Recommendation:** Logistic Regression performed best on the primary
metric (recall) and was therefore selected as the final model. A paired
t-test on the per-fold CV scores of the top two models gave p = 0.0016
(< 0.05), meaning the difference between Logistic Regression and Random
Forest is statistically significant, not just random noise.

## 5. Required Analysis Answers

1. **Best model & improvement over baseline:** Logistic Regression was both
   the baseline and best model in this run (CV recall 0.548 vs Random
   Forest's 0.480) — see `results/model_comparison.csv`.
2. **Statistically meaningful?** Yes — paired t-test on the 5 CV folds gave
   p = 0.0016 (< 0.05), so the difference is real, not noise.
3. **Most important features:** See `results/feature_importance.png` —
   generated from the model's coefficients.
4. **Where the model fails most:** See `results/confusion_matrix.png` for
   the breakdown of false positives vs false negatives — customers on
   month-to-month contracts with short tenure show the highest error rate,
   likely because their behavior is more unpredictable than long-term
   contract customers.

## 6. Project Structure

```
ml-algorithms-task2/
├── data/
│   ├── raw/                # original CSV (not pushed to git)
│   └── processed/          # cleaned, feature-ready data
├── src/
│   ├── settings.py         # all hardcoded values/paths/constants
│   ├── preprocessing.py    # build_features(), get_train_test_split()
│   ├── modeling.py         # train_and_evaluate(), tuning logic
│   ├── evaluation.py       # metrics, confusion matrix, plots
│   └── utils.py            # shared helpers
├── models/
│   └── best_model.joblib   # final trained model (not pushed to git)
├── results/
│   ├── model_comparison.csv
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── reports/
│   └── final_report.pdf    # stakeholder-facing summary
├── main.py                 # single entry point
├── pyproject.toml          # pinned dependencies
└── README.md
```

## 7. Key Design Decisions (justifications)

- **One-Hot Encoding** for all categorical columns (Contract, PaymentMethod,
  InternetService, etc.) because none of them have a natural order.
- **StandardScaler** applied only to numeric columns, and only needed for
  Logistic Regression (distance/gradient based). Random Forest doesn't need
  scaling since it splits on thresholds, not distances.
- **Stratified train/test split (80/20)** to keep the ~27-39% churn ratio
  consistent in both sets, avoiding a misleadingly high accuracy.
- **5-fold Cross-Validation** instead of a single train/test score, to get
  a more reliable estimate (mean ± std) of each model's performance.
- **GridSearchCV on Random Forest** (`n_estimators`, `max_depth`,
  `min_samples_split`) because it was a strong candidate; a small grid keeps
  tuning fast without overfitting to noise.
- **Recall as primary metric** because missing a churner is costlier than a
  false alarm in a retention business context.
