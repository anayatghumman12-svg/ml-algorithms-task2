import logging

import joblib

from src import settings, utils, preprocessing, modeling, evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Pura ML pipeline yahan se chalta hai, step by step."""

    # Step 0: Zaroori folders bana lo (agar pehle se nahi hain)
    utils.ensure_folders_exist(["data/processed", "models", "results", "reports"])

    # Step 1: Data load karo
    logger.info("STEP 1: Loading data...")
    raw_df = preprocessing.load_data(settings.RAW_DATA_PATH)

    # Step 2: Features (X) aur target (y) banao
    logger.info("STEP 2: Building features...")
    X, y = preprocessing.build_features(raw_df)

    # cleaned data ko processed folder mein bhi save kar lete hain
    X.assign(**{settings.TARGET_COLUMN: y}).to_csv(settings.PROCESSED_DATA_PATH, index=False)

    # Step 3: Train/Test split (stratified, kyunki churn imbalanced hai)
    logger.info("STEP 3: Splitting data...")
    X_train, X_test, y_train, y_test = preprocessing.get_train_test_split(X, y)

    # Step 4: Numeric columns ko scale karo (Logistic Regression ke liye)
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    numeric_cols = [col for col in numeric_cols if col in X_train.columns]
    X_train_scaled, X_test_scaled = preprocessing.scale_numeric_features(
        X_train, X_test, numeric_cols
    )

    # Step 5: Baseline model (Logistic Regression) + sab models compare karo
    logger.info("STEP 4: Comparing models with cross-validation...")
    cv_results = modeling.compare_models(X_train_scaled, y_train)

    # Step 6: Best model choose karo (jiska mean CV score sabse zyada hai)
    best_model_name = max(cv_results, key=lambda name: cv_results[name]["mean_score"])
    logger.info(f"Best model based on CV: {best_model_name}")

    # Step 7: Best model ko tune karo (GridSearchCV)
    logger.info("STEP 5: Tuning best model...")
    tuned_model, best_params, tuned_score = modeling.tune_best_model(
        best_model_name, X_train_scaled, y_train
    )

    # Step 8: Test set par final evaluation
    logger.info("STEP 6: Final evaluation on test set...")
    y_pred = tuned_model.predict(X_test_scaled)
    y_proba = tuned_model.predict_proba(X_test_scaled)[:, 1]
    final_metrics = evaluation.get_metrics(y_test, y_pred, y_proba)
    final_metrics["model"] = f"{best_model_name} (Tuned)"

    # Baseline (Logistic Regression, bina tuning) ka bhi test set par result
    baseline_model = modeling.get_models()["Logistic Regression"]
    baseline_model.fit(X_train_scaled, y_train)
    baseline_pred = baseline_model.predict(X_test_scaled)
    baseline_proba = baseline_model.predict_proba(X_test_scaled)[:, 1]
    baseline_metrics = evaluation.get_metrics(y_test, baseline_pred, baseline_proba)
    baseline_metrics["model"] = "Logistic Regression (Baseline)"

    # Step 9: Results table save karo
    evaluation.save_results_table([baseline_metrics, final_metrics])

    # Step 10: Confusion matrix aur feature importance save karo
    evaluation.save_confusion_matrix(y_test, y_pred, best_model_name)
    evaluation.save_feature_importance(tuned_model, X_train_scaled.columns)

    # Step 11: Statistical comparison (paired t-test) between top 2 models
    lr_folds = cv_results["Logistic Regression"]["fold_scores"]
    rf_folds = cv_results["Random Forest"]["fold_scores"]
    stat_result = evaluation.compare_two_models_statistically(lr_folds, rf_folds)
    logger.info(f"Statistical comparison: {stat_result}")

    # Step 12: Best model ko disk par save karo
    joblib.dump(tuned_model, settings.BEST_MODEL_PATH)
    logger.info(f"Best model saved to {settings.BEST_MODEL_PATH}")

    logger.info("Pipeline finished successfully!")
    logger.info(f"Best params: {best_params}")
    logger.info(f"Final metrics: {final_metrics}")


if __name__ == "__main__":
    main()
