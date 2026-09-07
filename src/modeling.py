import logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

from src import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=settings.RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            random_state=settings.RANDOM_STATE
        ),
    }
    return models


def train_and_evaluate(model, X, y) -> dict:
    try:
        scores = cross_val_score(
            model, X, y,
            cv=settings.CV_FOLDS,
            scoring=settings.PRIMARY_METRIC,
        )
    except Exception as e:
        logger.error(f"Cross-validation failed: {e}")
        raise

    result = {
        "mean_score": scores.mean(),
        "std_score": scores.std(),
        "fold_scores": scores.tolist(),
    }
    logger.info(
        f"{model.__class__.__name__} -> mean {settings.PRIMARY_METRIC}: "
        f"{result['mean_score']:.4f} (+/- {result['std_score']:.4f})"
    )
    return result


def compare_models(X, y) -> dict:
    models = get_models()
    all_results = {}

    for name, model in models.items():
        logger.info(f"Evaluating: {name}")
        all_results[name] = train_and_evaluate(model, X, y)

    return all_results


def tune_best_model(model_name: str, X, y):
    if model_name != "Random Forest":
        # is task mein sirf Random Forest ke liye grid diya gaya hai
        logger.info(f"No tuning grid defined for {model_name}, skipping tuning.")
        model = get_models()[model_name]
        model.fit(X, y)
        return model, {}, None

    base_model = RandomForestClassifier(random_state=settings.RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=settings.RF_PARAM_GRID,
        scoring=settings.PRIMARY_METRIC,
        cv=settings.CV_FOLDS,
        n_jobs=-1,
    )

    logger.info("Starting GridSearchCV for Random Forest...")
    grid_search.fit(X, y)

    logger.info(f"Best params: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
