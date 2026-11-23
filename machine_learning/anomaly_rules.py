"""Anomaly rules: each rule returns a boolean Series (True = anomalous row).

Edit RULES to add/remove rules. Rows flagged by ANY rule are kept out of
training (and out of the evaluation set, so scores reflect normal data).
"""
import numpy as np
import pandas as pd

TARGET = "OBLIGATION"
OUTLIER_IQR_MULTIPLIER = 3.0   # on log1p(target); larger = fewer outliers flagged
ACCOUNTING_TOLERANCE = 1.0     # allowed |OBLIGATION - AMOUNT_PAID - REMAINING_AMT|


def missing_or_nonfinite_target(df, target=TARGET):
    y = pd.to_numeric(df[target], errors="coerce")
    return ~np.isfinite(y)


def negative_target(df, target=TARGET):
    return pd.to_numeric(df[target], errors="coerce") < 0


def accounting_mismatch(df, target=TARGET):
    needed = {target, "AMOUNT_PAID", "REMAINING_AMT"}
    if not needed.issubset(df.columns):
        return pd.Series(False, index=df.index)
    gap = df[target] - df["AMOUNT_PAID"] - df["REMAINING_AMT"]
    return gap.abs() > ACCOUNTING_TOLERANCE


def extreme_target_outlier(df, target=TARGET):
    y = pd.to_numeric(df[target], errors="coerce")
    logy = np.log1p(y.clip(lower=0))
    q1, q3 = logy.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (logy < q1 - OUTLIER_IQR_MULTIPLIER * iqr) | (
        logy > q3 + OUTLIER_IQR_MULTIPLIER * iqr
    )


def duplicate_rows(df, target=TARGET):
    return df.duplicated(keep="first")


RULES = [
    missing_or_nonfinite_target,
    negative_target,
    accounting_mismatch,
    extreme_target_outlier,
    duplicate_rows,
]


def flag_anomalies(df, rules=RULES):
    """Return a DataFrame of one boolean column per rule plus 'is_anomaly'."""
    flags = pd.DataFrame({r.__name__: r(df).fillna(True) for r in rules}, index=df.index)
    flags["is_anomaly"] = flags.any(axis=1)
    return flags
