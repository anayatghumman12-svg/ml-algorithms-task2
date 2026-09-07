"""
preprocessing.py
-----------------
Is file mein data load karna, clean karna, aur model ke liye ready
(features + target) banana wale simple functions hain.
"""

import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(path: str = settings.RAW_DATA_PATH) -> pd.DataFrame:
    """
    CSV file ko load karta hai aur check karta hai ke file exist karti hai
    aur data khaali nahi hai.

    Args:
        path (str): CSV file ka path.

    Returns:
        pd.DataFrame: loaded dataset.
    """
    if not os.path.exists(path):
        logger.error(f"File not found at: {path}")
        raise FileNotFoundError(f"Dataset not found at {path}. Please place the CSV there.")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error while reading CSV: {e}")
        raise

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning: TotalCharges ko numeric banana, missing values ko
    handle karna, aur customerID (jo model ke liye useless hai) hatana.

    Args:
        df (pd.DataFrame): raw dataframe.

    Returns:
        pd.DataFrame: cleaned dataframe.
    """
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
        logger.info("TotalCharges cleaned and converted to numeric.")

    if settings.ID_COLUMN in df.columns:
        df = df.drop(columns=[settings.ID_COLUMN])

    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Dataframe se X (features) aur y (target) alag karta hai, categorical
    columns ko one-hot encode karta hai.

    Justification:
    - One-Hot Encoding: Contract, PaymentMethod, InternetService jaise columns
      mein koi natural order nahi hai, is liye label/ordinal encoding galat
      hoga. One-hot sahi choice hai.
    - Scaling: Logistic Regression distance/gradient based hai, is liye
      numeric features ko scale karna zaroori hai. Random Forest ko scaling
      ki zaroorat nahi.

    Args:
        df (pd.DataFrame): cleaned dataframe.

    Returns:
        tuple[pd.DataFrame, pd.Series]: (X features, y target)
    """
    df = clean_data(df)

    y = df[settings.TARGET_COLUMN].map({"Yes": 1, "No": 0})
    X = df.drop(columns=[settings.TARGET_COLUMN])

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    logger.info(f"Features built. X shape: {X.shape}, y shape: {y.shape}")
    return X, y


def scale_numeric_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                            numeric_cols: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Numeric columns ko StandardScaler se scale karta hai (sirf Logistic
    Regression ke liye zaroori hai). Scaler sirf train data par fit hota hai
    taake test data par data leakage na ho.

    Args:
        X_train (pd.DataFrame): training features.
        X_test (pd.DataFrame): test features.
        numeric_cols (list): scale karne wale numeric column names.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: scaled X_train, X_test
    """
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    logger.info("Numeric features scaled using StandardScaler.")
    return X_train, X_test


def get_train_test_split(X: pd.DataFrame, y: pd.Series):
    """
    Data ko train aur test sets mein split karta hai (80/20).
    stratify=y use kiya hai taake dono sets mein churn ka ratio same rahe.

    Args:
        X (pd.DataFrame): features.
        y (pd.Series): target.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
        stratify=y,
    )
    logger.info(f"Train/Test split done. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test