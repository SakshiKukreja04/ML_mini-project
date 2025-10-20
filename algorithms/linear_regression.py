import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple, Optional


def load_data(file_path: str, feature_col: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load CSV and return X (DataFrame) and y (Series) for the specified columns.

    Raises FileNotFoundError if file missing.
    """
    df = pd.read_csv(file_path)
    mask = df[feature_col].notna() & df[target_col].notna()
    X = df[[feature_col]][mask].reset_index(drop=True)
    y = df[target_col][mask].reset_index(drop=True)
    return X, y


def regression_errors(y_true: np.ndarray, y_pred: np.ndarray, num_features: int) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    n = len(y_true)
    p = num_features
    # Adjusted R-squared formula
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else float('nan')

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "Adj_R2": adj_r2}


def train_and_evaluate(
    file_path: str,
    feature_col: str = "median_income",
    target_col: str = "median_house_value",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Train a simple linear regression on one feature and return metrics and artifacts.

    Returns:
      metrics: Dict of evaluation metrics
      artifacts: Dict with keys: model, X_test, y_test, y_pred, slope, intercept
    """
    X, y = load_data(file_path, feature_col, target_col)
    # also keep a combined dataframe for plotting and preview
    df_combined = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = LinearRegression()
    model.fit(X_train, y_train)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    y_pred = model.predict(X_test)

    metrics = regression_errors(y_test, y_pred, num_features=X.shape[1])

    artifacts = {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "slope": slope,
        "intercept": intercept,
        "df": df_combined,
    }

    return metrics, artifacts


def plot_results(
    artifacts: Dict[str, object],
    feature_col: str = "median_income",
    target_col: str = "median_house_value",
    plot_type: str = "scatter",
) -> bytes:
    """Create matplotlib plot (multiple types) and return the PNG bytes.

    plot_type: 'scatter' | 'residual' | 'hist'
    """
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]
    y_pred = artifacts["y_pred"]
    model = artifacts.get("model")

    # Ensure arrays
    x_vals = X_test[feature_col].values.ravel()
    y_vals = y_test.values.ravel()
    y_pred_vals = np.array(y_pred).ravel()

    plt.figure(figsize=(10, 7))

    if plot_type == "scatter":
        # scatter with regression line. Use full combined df as background if available
        df = artifacts.get("df") if artifacts is not None else None
        if df is not None and feature_col in df.columns and target_col in df.columns:
            sample_x = df[feature_col].values
            sample_y = df[target_col].values
            plt.scatter(sample_x, sample_y, color="lightgray", alpha=0.6, label=f"All data (n={len(sample_y)})")
        else:
            plt.scatter(x_vals, y_vals, color="lightgray", alpha=0.6, label=f"Data (n={len(y_vals)})")

        # regression line using sorted x
        order = np.argsort(x_vals)
        X_sorted = x_vals[order].reshape(-1, 1)
        y_pred_sorted = model.predict(X_sorted)
        plt.plot(X_sorted, y_pred_sorted, color="red", linewidth=3, label="Regression line", zorder=5)

        # overlay test actuals and predicted
        plt.scatter(x_vals, y_vals, color="orange", edgecolor="k", alpha=0.8, label="Test (actual)", zorder=3)
        plt.scatter(x_vals, y_pred_vals, color="purple", marker="X", s=50, label="Predicted (test)", zorder=4)
        plt.xlabel(feature_col)
        plt.ylabel(target_col)
        plt.title(f"Scatter & Regression - {feature_col} vs {target_col}")
        plt.legend()

    elif plot_type == "residual":
        # residuals vs predicted or feature
        residuals = y_vals - y_pred_vals
        plt.scatter(y_pred_vals, residuals, color="teal", alpha=0.7)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Predicted values")
        plt.ylabel("Residuals (actual - predicted)")
        plt.title("Residuals vs Predicted")
        plt.grid(alpha=0.3)

    elif plot_type == "hist":
        # histogram of actual and predicted
        plt.hist(y_vals, bins=30, alpha=0.6, label="Actual")
        plt.hist(y_pred_vals, bins=30, alpha=0.6, label="Predicted")
        plt.xlabel(target_col)
        plt.ylabel("Frequency")
        plt.title("Distribution: Actual vs Predicted")
        plt.legend()

    else:
        plt.text(0.5, 0.5, f"Unknown plot type: {plot_type}", ha="center")

    plt.grid(alpha=0.2)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.getvalue()


if __name__ == "__main__":
    # Simple command-line test
    import os
    fp = os.path.join(os.path.dirname(__file__), "..", "housing.csv")
    fp = os.path.abspath(fp)
    try:
        metrics, artifacts = train_and_evaluate(fp)
        print("Metrics:\n", metrics)
        png = plot_results(artifacts)
        with open("lr_plot.png", "wb") as f:
            f.write(png)
        print("Saved lr_plot.png")
    except FileNotFoundError:
        print("housing.csv not found; please place it next to the project root.")