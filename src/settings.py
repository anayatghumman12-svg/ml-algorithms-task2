# ---------- FILE PATHS ----------
RAW_DATA_PATH = "data/raw/telco_churn.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_data.csv"
BEST_MODEL_PATH = "models/best_model.joblib"
RESULTS_CSV_PATH = "results/model_comparison.csv"
CONFUSION_MATRIX_PATH = "results/confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = "results/feature_importance.png"
REPORT_PATH = "reports/final_report.pdf"

# ---------- DATASET INFO ----------
TARGET_COLUMN = "Churn"          # column jo predict karni hai
ID_COLUMN = "customerID"          # ye column model mein use nahi hogi
PROBLEM_TYPE = "Binary Classification"  # Churn = Yes/No

# ---------- SPLIT SETTINGS ----------
TEST_SIZE = 0.20      # 80/20 split
RANDOM_STATE = 42     # taake results har baar same aayen (reproducibility)

# ---------- CROSS VALIDATION ----------
CV_FOLDS = 5           # 5-fold cross validation

# ---------- PRIMARY METRIC ----------
# Churn prediction mein "recall" zyada important hai kyunki jo customer
# churn karega usay miss karna (false negative) zyada costly hai
# bajaye is ke ke hum kisi loyal customer ko galti se churn-risk keh dein.
PRIMARY_METRIC = "recall"

# ---------- HYPERPARAMETER GRID (Random Forest) ----------
RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}
