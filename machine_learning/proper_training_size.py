import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


filename = os.path.join(os.path.dirname(__file__), "Book1.xlsx")
test_size = 0.2
train_size = 0.8
target_column = "OBLIGATION"
train_size_step = 0.1
minimum_train_size = 0.1
baseline_tolerance = 0.05
required_consecutive_sizes = 2
random_state = 42


def simulate_train_sizes(
    df,
    target=target_column,
    test_fraction=test_size,
    baseline_train_fraction=train_size,
    model_factory=None,
):
    if model_factory is None:
        model_factory = lambda: xgb.XGBRegressor(
            objective="reg:squarederror", random_state=random_state, n_jobs=1
        )

    if target not in df.columns:
        raise ValueError(f"Target column {target!r} is not in the dataset.")

    y = pd.to_numeric(df[target], errors="raise")
    if y.isna().any():
        raise ValueError(f"Target column {target!r} contains missing values.")

    X = pd.get_dummies(df.drop(columns=[target]))
    if X.empty:
        raise ValueError("The dataset must contain at least one feature column.")

    train_indices, test_indices = train_test_split(
        np.arange(len(df)),
        test_size=test_fraction,
        train_size=baseline_train_fraction,
        random_state=random_state,
    )
    nested_train_indices = np.random.default_rng(random_state).permutation(
        train_indices
    )
    fractions = [baseline_train_fraction]
    while fractions[-1] > minimum_train_size:
        next_fraction = round(fractions[-1] - train_size_step, 10)
        fractions.append(max(minimum_train_size, next_fraction))

    results = []
    baseline_rmse = None
    previous_training_count = len(train_indices) + 1
    consecutive_acceptable_sizes = 0
    stop_train_size = None

    for index, fraction in enumerate(fractions):
        training_count = min(len(train_indices), max(1, int(len(df) * fraction)))
        if training_count >= previous_training_count:
            continue

        selected_indices = nested_train_indices[:training_count]
        model = model_factory()
        model.fit(X.iloc[selected_indices], y.iloc[selected_indices])
        predictions = model.predict(X.iloc[test_indices])
        rmse = float(np.sqrt(mean_squared_error(y.iloc[test_indices], predictions)))

        if baseline_rmse is None:
            baseline_rmse = rmse
            within_tolerance = True
        else:
            within_tolerance = rmse <= baseline_rmse * (1 + baseline_tolerance)

        consecutive_acceptable_sizes = (
            consecutive_acceptable_sizes + 1
            if index > 0 and within_tolerance
            else 0
        )
        stop_here = consecutive_acceptable_sizes >= required_consecutive_sizes
        actual_train_size = training_count / len(df)
        results.append(
            {
                "train_size": actual_train_size,
                "training_rows": training_count,
                "rmse": rmse,
                "within_baseline_tolerance": within_tolerance,
                "stop_here": stop_here,
            }
        )
        previous_training_count = training_count

        if stop_here:
            stop_train_size = actual_train_size
            break

    return pd.DataFrame(results), stop_train_size


def main():
    df = pd.read_excel(filename)
    results, stop_train_size = simulate_train_sizes(df)

    print(f"Baseline test_size={test_size:.1%}, train_size={train_size:.1%}")
    print(results.to_string(index=False))

    if stop_train_size is None:
        print(
            "No stopping point met the tolerance down to the minimum tested "
            "train_size."
        )
    else:
        print(
            f"Stop at train_size={stop_train_size:.1%}: two successive smaller "
            f"sizes stayed within {baseline_tolerance:.0%} of baseline RMSE."
        )

    baseline_rmse = results.iloc[0]["rmse"]
    plt.plot(results["train_size"], results["rmse"], marker="o")
    plt.axhline(
        baseline_rmse * (1 + baseline_tolerance),
        color="red",
        linestyle="--",
        label=f"Baseline RMSE + {baseline_tolerance:.0%}",
    )
    plt.xlabel("Training fraction of full dataset")
    plt.ylabel("Test RMSE")
    plt.title("Test error as training data is reduced")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
