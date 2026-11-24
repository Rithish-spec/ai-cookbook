import os

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import anomaly_store
from proper_training_size import (
    filename,
    random_state,
    simulate_train_sizes,
    target_column,
    test_size,
)

TRACKING_URI = "sqlite:///" + os.path.join(os.path.dirname(__file__), "mlflow.db")
EXPERIMENT_NAME = "training-size-test"
# Features that leak the target (OBLIGATION ~= AMOUNT_PAID + REMAINING_AMT) or are
# precomputed predictions. Set to [] to keep them.
DROP_FEATURES = ["AMOUNT_PAID", "REMAINING_AMT", "expected"]


def make_model():
    """Single place to swap the (currently unknown) model."""
    return xgb.XGBRegressor(
        objective="reg:squarederror", random_state=random_state, n_jobs=1
    )


def prepare(df):
    """Drop anomalies using the persistent store. Returns (clean_df, clean_keys, anomaly_keys)."""
    is_anomaly, keys = anomaly_store.sync(df)
    mlflow.log_metrics({"rows_anomalous": int(is_anomaly.sum())})
    mlflow.log_param("anomaly_rules_fingerprint", anomaly_store.rules_fingerprint())
    mlflow.log_artifacts(anomaly_store.STORE_DIR, artifact_path="anomaly_store")
    clean = df[~is_anomaly].drop(columns=[c for c in DROP_FEATURES if c in df.columns])
    clean_keys = keys[~is_anomaly].reset_index(drop=True)
    print(f"Kept {len(clean)} of {len(df)} rows after anomaly filtering.")
    return clean.reset_index(drop=True), clean_keys, set(keys[is_anomaly])


def train(df):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="train-size-test"):
        return _train(df)


def _train(df):
    mlflow.log_params(
        {
            "model": type(make_model()).__name__,
            "target": target_column,
            "test_size": test_size,
            "random_state": random_state,
            "dropped_features": ",".join(DROP_FEATURES),
        }
    )
    clean, clean_keys, anomaly_keys = prepare(df)
    mlflow.log_metrics({"rows_total": len(df), "rows_clean": len(clean)})

    # 1. Find the smallest sufficient training size (clean data only).
    results, stop_fraction = simulate_train_sizes(clean, model_factory=make_model)
    print(results.to_string(index=False))
    mlflow.log_table(results, "train_size_results.json")
    for step, row in results.iterrows():
        mlflow.log_metric("size_search_rmse", row["rmse"], step=int(row["training_rows"]))
    fraction = stop_fraction if stop_fraction is not None else results.iloc[0]["train_size"]
    print(f"Using train fraction {fraction:.1%} of clean data.")

    # 2. Train the final model on that many clean rows; evaluate on clean holdout.
    X = pd.get_dummies(clean.drop(columns=[target_column]))
    y = clean[target_column]
    train_idx, test_idx = train_test_split(
        np.arange(len(clean)), test_size=test_size, random_state=random_state
    )
    n = min(len(train_idx), max(1, int(len(clean) * fraction)))
    train_idx = np.random.default_rng(random_state).permutation(train_idx)[:n]

    # Hard guarantee: nothing flagged as an anomaly is in the training set.
    leaked = anomaly_keys & set(clean_keys.iloc[train_idx])
    assert not leaked, f"{len(leaked)} anomalous rows ended up in training data"

    model = make_model().fit(X.iloc[train_idx], y.iloc[train_idx])
    rmse = float(np.sqrt(mean_squared_error(y.iloc[test_idx], model.predict(X.iloc[test_idx]))))
    print(f"Trained on {n} rows; holdout RMSE = {rmse:,.2f}")

    mlflow.log_params({"chosen_train_fraction": round(fraction, 4), "training_rows": n})
    mlflow.log_metric("holdout_rmse", rmse)
    mlflow.xgboost.log_model(model, name="model", input_example=X.iloc[train_idx].head(5))
    print(f"Logged run to MLflow experiment {EXPERIMENT_NAME!r} at {TRACKING_URI}")
    return model


if __name__ == "__main__":
    train(pd.read_excel(filename))
