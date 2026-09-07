
import logging

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_metrics(y_true, y_pred, y_proba) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    return metrics


def save_confusion_matrix(y_true, y_pred, model_name: str,
                           path: str = settings.CONFUSION_MATRIX_PATH) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {path}")


def save_feature_importance(model, feature_names, path: str = settings.FEATURE_IMPORTANCE_PATH,
                             top_n: int = 10) -> pd.DataFrame:

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Logistic Regression ke liye absolute coefficient value use karte hain
        importance = abs(model.coef_[0])
    else:
        logger.warning("Model has no feature_importances_ or coef_ attribute.")
        return pd.DataFrame()

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values(by="importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    ax.set_title(f"Top {top_n} Important Features")
    ax.set_xlabel("Importance")

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Feature importance plot saved to {path}")

    return importance_df


def compare_two_models_statistically(fold_scores_model_a: list,
                                      fold_scores_model_b: list) -> dict:
    t_stat, p_value = ttest_rel(fold_scores_model_a, fold_scores_model_b)

    # 0.05 ek common threshold hai statistics mein significance ke liye
    if p_value < 0.05:
        conclusion = "Farq statistically significant hai (p < 0.05), sirf noise nahi."
    else:
        conclusion = "Farq statistically significant nahi hai (p >= 0.05), ho sakta hai noise ho."

    return {"t_statistic": t_stat, "p_value": p_value, "conclusion": conclusion}


def save_results_table(results: list, path: str = settings.RESULTS_CSV_PATH) -> None:
    df = pd.DataFrame(results)
    df = df.sort_values(by=settings.PRIMARY_METRIC, ascending=False)
    df.to_csv(path, index=False)
    logger.info(f"Results table saved to {path}")
