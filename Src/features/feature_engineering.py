from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
OUTPUT_DIR = ARTIFACTS_DIR / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"


def load_joined_dataset() -> pd.DataFrame:
    dataset_path = DATA_DIR / "joined_clients.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} not found. Run Src/pipelines/join_datasets.py first.")
    dataset = pd.read_csv(dataset_path)
    print(f"Dataset loaded with shape {dataset.shape}")
    return dataset


def split_data(df: pd.DataFrame) -> Tuple[Tuple[pd.DataFrame, pd.Series], ...]:
    X = df.drop(columns=["target", "client_id"], errors="ignore")
    y = df["target"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def build_feature_pipeline(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ],
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ],
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
    )
    return preprocessor


def add_sample_weights(y: pd.Series) -> pd.Series:
    weights = compute_sample_weight(class_weight="balanced", y=y)
    return pd.Series(weights, name="sample_weight")


def _to_dataframe(matrix, feature_names) -> pd.DataFrame:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return pd.DataFrame(matrix, columns=feature_names)


def materialize_datasets(preprocessor: ColumnTransformer, splits: Dict[str, Tuple[pd.DataFrame, pd.Series]]) -> None:
    feature_names = None
    for split_name, (X_split, y_split) in splits.items():
        if split_name == "train":
            transformed = preprocessor.fit_transform(X_split)
            feature_names = preprocessor.get_feature_names_out()
            joblib.dump(preprocessor, PREPROCESSOR_PATH)
        else:
            transformed = preprocessor.transform(X_split)
        if feature_names is None:
            raise RuntimeError("Feature names unavailable. Ensure training split is processed first.")
        X_frame = _to_dataframe(transformed, feature_names)
        X_path = OUTPUT_DIR / f"X_{split_name}.parquet"
        y_path = OUTPUT_DIR / f"y_{split_name}.parquet"
        X_frame.to_parquet(X_path, index=False)
        y_split.to_frame("target").to_parquet(y_path, index=False)
        if split_name == "train":
            weights = add_sample_weights(y_split)
            weights.to_parquet(OUTPUT_DIR / "sample_weights_train.parquet", index=False)
        print(f"Wrote {split_name} split: {X_frame.shape}")


def run_feature_engineering() -> None:
    df = load_joined_dataset()
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = split_data(df)
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()
    preprocessor = build_feature_pipeline(categorical_cols, numeric_cols)
    splits = {
        "train": (X_train, y_train),
        "valid": (X_valid, y_valid),
        "test": (X_test, y_test),
    }
    materialize_datasets(preprocessor, splits)
    print(f"Preprocessor saved to {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    run_feature_engineering()
