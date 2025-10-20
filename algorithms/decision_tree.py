import io
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV


def prepare_data(
    file_path: str,
    target_col: str = "median_house_value",
    q1_override: float = None,
    q2_override: float = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """Load CSV, create price_range target (low/medium/high) from target_col, return dataframe and info.

    Price ranges are created using terciles (33% and 66% percentiles).
    """
    df = pd.read_csv(file_path)
    # drop rows with missing key columns
    before = len(df)
    df = df.dropna(subset=[target_col])
    after = len(df)

    # Create price_range using tertiles (allow overrides)
    q1 = df[target_col].quantile(0.33) if q1_override is None else float(q1_override)
    q2 = df[target_col].quantile(0.66) if q2_override is None else float(q2_override)

    def _bucket(v):
        if v <= q1:
            return "low"
        if v <= q2:
            return "medium"
        return "high"

    df["price_range"] = df[target_col].apply(_bucket)

    # select numeric features except target_col
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    if target_col in numeric:
        numeric.remove(target_col)
    if "price_range" in numeric:
        numeric.remove("price_range")

    # For simplicity, use numeric features only
    X = df[numeric].copy()
    y = df["price_range"].copy()

    preprocessing = {
        "rows_before": before,
        "rows_after": after,
        "dropped_rows": before - after,
        "features_used": numeric,
        "bins": {"q1": float(q1), "q2": float(q2)},
    }

    return X, y, df, preprocessing


def train_and_evaluate(
    file_path: str,
    target_col: str = "median_house_value",
    test_size: float = 0.2,
    random_state: int = 42,
    max_depth: int = None,
    q1_override: float = None,
    q2_override: float = None,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Train a Decision Tree classifier and return metrics and artifacts.

    Artifacts include: model, X_test, y_test, y_pred, df (full preprocessed), preprocessing info, class_names, train/test counts.
    """
    X, y, df_full, preprocessing = prepare_data(
        file_path, target_col=target_col, q1_override=q1_override, q2_override=q2_override
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }

    artifacts = {
        "model": clf,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "df": df_full,
        "preprocessing": preprocessing,
        "class_names": clf.classes_.tolist(),
        "train_count": len(X_train),
        "test_count": len(X_test),
        "confusion_matrix": cm,
    }

    return metrics, artifacts


def plot_tree_image(artifacts: Dict[str, object], feature_names: list = None) -> bytes:
    """Plot decision tree using sklearn's plot_tree and return PNG bytes."""
    clf = artifacts.get("model")
    df = artifacts.get("df")

    if feature_names is None:
        # try to infer feature names from X_test
        X_test = artifacts.get("X_test")
        feature_names = X_test.columns.tolist() if X_test is not None else None

    plt.figure(figsize=(18, 10))
    plot_tree(clf, feature_names=feature_names, class_names=artifacts.get("class_names"), filled=True, fontsize=8)
    plt.title("Decision Tree")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.getvalue()


def plot_confusion_matrix(artifacts: Dict[str, object]) -> bytes:
    cm = artifacts.get("confusion_matrix")
    classes = artifacts.get("class_names")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix for Decision Tree Classifier")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.getvalue()


def train_with_grid_search(
    file_path: str,
    target_col: str = "median_house_value",
    param_grid: Dict = None,
    cv: int = 5,
    random_state: int = 42,
    q1_override: float = None,
    q2_override: float = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Run GridSearchCV over DecisionTreeClassifier and return search results and artifacts.

    Returns (grid_info, artifacts) where grid_info contains 'best_params' and 'cv_results' summary.
    """
    if param_grid is None:
        param_grid = {
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5, 10],
        }

    X, y, df_full, preprocessing = prepare_data(
        file_path, target_col=target_col, q1_override=q1_override, q2_override=q2_override
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    base_clf = DecisionTreeClassifier(random_state=random_state)
    grid = GridSearchCV(base_clf, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=-1, return_train_score=True)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_pred = best.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=best.classes_)

    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }

    artifacts = {
        "model": best,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "df": df_full,
        "preprocessing": preprocessing,
        "class_names": best.classes_.tolist(),
        "train_count": len(X_train),
        "test_count": len(X_test),
        "confusion_matrix": cm,
    }

    grid_info = {
        "best_params": grid.best_params_,
        "best_score": float(grid.best_score_),
        "cv_results": grid.cv_results_,
        "param_grid": param_grid,
    }

    return grid_info, {"metrics": metrics, "artifacts": artifacts}


def plot_cv_heatmap(cv_results: Dict, param_x: str, param_y: str) -> bytes:
    """Plot a heatmap of mean_test_score for a 2D grid defined by param_x and param_y.

    cv_results is the GridSearchCV.cv_results_ dict.
    """
    # Extract parameter arrays
    params = cv_results["params"]
    mean_scores = cv_results["mean_test_score"]

    # unique values
    x_vals = sorted({p[param_x] for p in params})
    y_vals = sorted({p[param_y] for p in params})

    heat = np.zeros((len(y_vals), len(x_vals)))
    # build mapping
    for p, score in zip(params, mean_scores):
        xi = x_vals.index(p[param_x])
        yi = y_vals.index(p[param_y])
        heat[yi, xi] = score

    plt.figure(figsize=(8, 5))
    sns.heatmap(heat, annot=True, xticklabels=x_vals, yticklabels=y_vals, cmap="viridis", fmt=".3f")
    plt.xlabel(param_x)
    plt.ylabel(param_y)
    plt.title("GridSearchCV mean_test_score")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.getvalue()


if __name__ == "__main__":
    import os
    fp = os.path.join(os.path.dirname(__file__), "..", "housing.csv")
    fp = os.path.abspath(fp)
    try:
        metrics, artifacts = train_and_evaluate(fp)
        print("Metrics:\n", metrics)
        with open("dt_confusion.png", "wb") as f:
            f.write(plot_confusion_matrix(artifacts))
        with open("dt_tree.png", "wb") as f:
            f.write(plot_tree_image(artifacts))
        print("Saved dt_confusion.png and dt_tree.png")
    except FileNotFoundError:
        print("housing.csv not found; please place it next to the project root.")
