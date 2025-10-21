import io
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def _build_pipeline(numeric_features, categorical_features, degree=2, alpha=1.0):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),
        ('regressor', Ridge(alpha=alpha))
    ])

    return pipeline


def train_and_evaluate(file_path: str, degree: int = 2, alpha: float = 1.0, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dict, Dict]:
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['median_house_value'])

    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']

    # feature lists
    numeric_features = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income'
    ]
    categorical_features = ['ocean_proximity'] if 'ocean_proximity' in X.columns else []

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pipeline = _build_pipeline(numeric_features, categorical_features, degree=degree, alpha=alpha)

    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Metrics
    def _metrics(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        return {'r2': float(r2), 'mae': float(mae), 'mse': float(mse), 'rmse': rmse}

    metrics_train = _metrics(y_train, y_train_pred)
    metrics_test = _metrics(y_test, y_test_pred)

    artifacts = {
        'model_pipeline': pipeline,
        'X_test': X_test,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'df': df,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'metrics_train': metrics_train,
        'metrics_test': metrics_test,
        'degree': degree,
        'alpha': alpha
    }

    # generate 3D surface plot PNG bytes
    try:
        png = plot_3d_surface(artifacts)
        artifacts['plot_png'] = png
    except Exception:
        artifacts['plot_png'] = None

    # summarize metrics to a single dict for UI convenience
    metrics_summary = {
        'r2_test': metrics_test['r2'],
        'mae_test': metrics_test['mae'],
        'mse_test': metrics_test['mse'],
        'rmse_test': metrics_test['rmse'],
        'r2_train': metrics_train['r2']
    }

    return metrics_summary, artifacts


def plot_3d_surface(artifacts: Dict) -> bytes:
    """Create a 3D surface plot for median_income vs housing_median_age vs median_house_value.

    Holds other features at median/mode from training data.
    """
    pipeline = artifacts.get('model_pipeline')
    X_test = artifacts.get('X_test')
    y_test = artifacts.get('y_test')
    df = artifacts.get('df')

    # constants for other features
    const_values = {}
    for col in artifacts['numeric_features']:
        const_values[col] = X_test[col].median() if col in X_test.columns else 0.0
    if artifacts['categorical_features']:
        cat = artifacts['categorical_features'][0]
        const_values[cat] = X_test[cat].mode()[0] if cat in X_test.columns else None

    # meshgrid
    x_surf, y_surf = np.meshgrid(
        np.linspace(df['median_income'].min(), df['median_income'].max(), 30),
        np.linspace(df['housing_median_age'].min(), df['housing_median_age'].max(), 30)
    )

    surf_data = pd.DataFrame({
        'median_income': x_surf.flatten(),
        'housing_median_age': y_surf.flatten()
    })

    for col, val in const_values.items():
        # skip if already present
        if col not in surf_data.columns:
            surf_data[col] = val

    # ensure columns order aligns with training X (drop target)
    Xcols = df.drop('median_house_value', axis=1).columns
    surf_data = surf_data.reindex(columns=Xcols, fill_value=0)

    predicted_surf = pipeline.predict(surf_data)
    predicted_surf = predicted_surf.reshape(x_surf.shape)

    # plotting
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # sample test data scatter
    test_data = X_test.copy()
    test_data['median_house_value'] = y_test

    sample_size = min(2000, len(test_data))
    df_sample = test_data.sample(n=sample_size, random_state=42)

    ax.scatter(df_sample['median_income'], df_sample['housing_median_age'], df_sample['median_house_value'],
               c='blue', marker='o', alpha=0.3, label='Test Data (Sampled)')

    ax.plot_surface(x_surf, y_surf, predicted_surf, cmap='viridis', alpha=0.7)

    ax.set_xlabel('Median Income')
    ax.set_ylabel('Housing Median Age')
    ax.set_zlabel('Median House Value')
    ax.set_title(f'3D Polynomial Regression (Degree {artifacts.get("degree")}) - Full Model')
    ax.legend()
    ax.view_init(elev=20, azim=120)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def plot_residual_hist(artifacts: Dict) -> bytes:
    """Plot histogram of residuals (y_test - y_pred) and return PNG bytes."""
    y_test = artifacts.get('y_test')
    y_pred = artifacts.get('y_test_pred')
    if y_test is None or y_pred is None:
        raise ValueError('Missing test predictions in artifacts')

    residuals = y_test.values - y_pred

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=40, color='gray', edgecolor='black', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    ax.set_title('Residuals Histogram (Test)')
    ax.set_xlabel('Residual (actual - predicted)')
    ax.set_ylabel('Count')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def plot_actual_vs_pred(artifacts: Dict) -> bytes:
    """Plot actual vs predicted scatter with y=x diagonal and return PNG bytes."""
    y_test = artifacts.get('y_test')
    y_pred = artifacts.get('y_test_pred')
    if y_test is None or y_pred is None:
        raise ValueError('Missing test predictions in artifacts')

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, s=10)
    # diagonal line
    vmin = min(y_test.min(), y_pred.min())
    vmax = max(y_test.max(), y_pred.max())
    ax.plot([vmin, vmax], [vmin, vmax], color='red', linestyle='--')
    ax.set_xlabel('Actual values')
    ax.set_ylabel('Predicted values')
    ax.set_title('Actual vs Predicted (Test)')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


if __name__ == '__main__':
    import os
    fp = os.path.join(os.path.dirname(__file__), '..', 'housing.csv')
    fp = os.path.abspath(fp)
    try:
        metrics, artifacts = train_and_evaluate(fp)
        print('Metrics:', metrics)
        if artifacts.get('plot_png'):
            with open('mv_model_plot.png', 'wb') as f:
                f.write(artifacts.get('plot_png'))
            print('Saved mv_model_plot.png')
    except FileNotFoundError:
        print('housing.csv not found; please add it to project root.')
