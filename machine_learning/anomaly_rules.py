"""Anomaly rules: each rule is f(df, params) -> boolean Series (True = anomalous).

Rules must be row-local (judge a row on its own plus frozen `params`), so the
anomaly store can evaluate only new rows. Dataset-wide statistics are computed
once in fit_params() and frozen in the store. Editing this file changes its
fingerprint, which makes the store re-evaluate everything on the next run.
(Exact duplicate rows are handled by the store itself.)
"""
import numpy as np
import pandas as pd

TARGET = "OBLIGATION"
OUTLIER_IQR_MULTIPLIER = 3.0   # on log1p(target); larger = fewer outliers flagged
ACCOUNTING_TOLERANCE = 1.0     # allowed |OBLIGATION - AMOUNT_PAID - REMAINING_AMT|


def fit_params(df, target=TARGET):
    """Dataset-wide statistics the rules need, frozen into the store."""
    y = pd.to_numeric(df[target], errors="coerce")
    logy = np.log1p(y.clip(lower=0))
    q1, q3 = logy.quantile([0.25, 0.75])
    iqr = q3 - q1
    return {
        "log_lower": float(q1 - OUTLIER_IQR_MULTIPLIER * iqr),
        "log_upper": float(q3 + OUTLIER_IQR_MULTIPLIER * iqr),
    }


def missing_or_nonfinite_target(df, params, target=TARGET):
    return ~np.isfinite(pd.to_numeric(df[target], errors="coerce"))


def negative_target(df, params, target=TARGET):
    return pd.to_numeric(df[target], errors="coerce") < 0


def accounting_mismatch(df, params, target=TARGET):
    if not {target, "AMOUNT_PAID", "REMAINING_AMT"}.issubset(df.columns):
        return pd.Series(False, index=df.index)
    gap = df[target] - df["AMOUNT_PAID"] - df["REMAINING_AMT"]
    return gap.abs() > ACCOUNTING_TOLERANCE


def extreme_target_outlier(df, params, target=TARGET):
    y = pd.to_numeric(df[target], errors="coerce")
    logy = np.log1p(y.clip(lower=0))
    return (logy < params["log_lower"]) | (logy > params["log_upper"])


RULES = [
    missing_or_nonfinite_target,
    negative_target,
    accounting_mismatch,
    extreme_target_outlier,
]
