"""
SVM Housing Affordability Classifier
=====================================
Optimized implementation of Support Vector Machine for binary classification
of housing affordability using scikit-learn Pipeline and GridSearchCV.

Author: Optimized by GitHub Copilot
Date: October 2025
"""

from typing import Tuple, Dict, Optional, Any
import io
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA


# ============================================================================
# CONSTANTS
# ============================================================================
RANDOM_STATE = 42
CLASS_NAMES = ['Affordable', 'Not Affordable']
PARAM_GRID = {
    'svc__C': [0.1, 1.0, 10.0, 100.0],
    'svc__kernel': ['linear', 'rbf', 'poly'],
    'svc__gamma': ['scale', 'auto']
}
CV_FOLDS = 3


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def load_and_preprocess_data(
    file_path: str,
    drop_geo_features: bool = True
) -> Tuple[pd.DataFrame, pd.Series, list, list, Optional[str]]:
    """
    Load and preprocess the housing dataset for SVC classification.
    
    Performs feature engineering to create derived features and drops
    original columns along with longitude and latitude.
    
    Args:
        file_path: Path to the housing.csv file.
        drop_geo_features: Whether to drop longitude/latitude (default True).
        
    Returns:
        X: Feature DataFrame (after encoding and feature engineering).
        y: Target variable (binary categories: Affordable, Not Affordable).
        class_names: List of class labels.
        feature_names: List of feature column names.
        error: Error message if any, otherwise None.
    """
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        
        # Handle missing values efficiently (vectorized)
        if 'total_bedrooms' in df.columns:
            median_bedrooms = df['total_bedrooms'].median()
            df['total_bedrooms'] = df['total_bedrooms'].fillna(median_bedrooms)
        
        # Feature Engineering: Create derived ratio features
        df['rooms_per_person'] = df['total_rooms'] / df['population']
        df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
        df['population_per_household'] = df['population'] / df['households']
        
        # Handle inf/nan from division efficiently using clip + fillna
        for col in ['rooms_per_person', 'bedrooms_per_room', 'population_per_household']:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].median())
        
        # Create binary target variable using median threshold
        median_price = df['median_house_value'].median()
        y = pd.cut(
            df['median_house_value'], 
            bins=[0, median_price, float('inf')], 
            labels=CLASS_NAMES
        )
        
        # Drop unnecessary columns
        columns_to_drop = [
            'median_house_value',
            'total_rooms', 'total_bedrooms', 'population', 'households'
        ]
        
        if drop_geo_features:
            columns_to_drop.extend(['longitude', 'latitude'])
        
        X = df.drop(columns=columns_to_drop)
        
        # One-hot encode categorical features
        X = pd.get_dummies(X, columns=['ocean_proximity'], drop_first=False)
        
        # Get feature names
        feature_names = X.columns.tolist()
        
        return X, y, CLASS_NAMES, feature_names, None
        
    except Exception as e:
        return None, None, None, None, str(e)


# ============================================================================
# MODEL TRAINING
# ============================================================================

def create_svm_pipeline() -> Pipeline:
    """
    Create an optimized sklearn Pipeline with StandardScaler + SVC.
    
    Returns:
        Pipeline: Sklearn pipeline object.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(random_state=RANDOM_STATE, probability=True))
    ])


def train_and_evaluate(
    file_path: str,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    use_cache: bool = False,
    cache_path: Optional[str] = None,
    use_gridsearch: bool = True,
    best_params: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train and evaluate an SVM classifier using Pipeline + GridSearchCV.
    
    Uses sklearn Pipeline to combine preprocessing and modeling steps.
    Implements efficient GridSearchCV with parallel processing.
    
    Args:
        file_path: Path to the housing.csv file.
        test_size: Proportion of test set (default 0.2).
        random_state: Random seed for reproducibility (default 42).
        use_cache: Whether to load cached model if available (default False).
        cache_path: Path to cache the trained model (default None).
        use_gridsearch: Whether to run GridSearchCV or use pre-defined params (default True).
        best_params: Pre-defined best parameters if use_gridsearch=False (default None).
        
    Returns:
        metrics: Dictionary containing accuracy, classification report, best_params, and best_score.
        artifacts: Dictionary containing model, predictions, training data, and metadata.
    """
    # Check for cached model
    if use_cache and cache_path and os.path.exists(cache_path):
        print(f"Loading cached model from: {cache_path}")
        cached_data = joblib.load(cache_path)
        return cached_data['metrics'], cached_data['artifacts']
    
    # Load and preprocess data
    X, y, class_names, feature_names, error = load_and_preprocess_data(file_path)
    
    if error:
        raise Exception(f"Data loading failed: {error}")
    
    # Convert to numpy arrays for faster processing
    X_values = X.values
    y_values = y.values
    
    # Stratified train-test split for balanced classes
    X_train, X_test, y_train, y_test = train_test_split(
        X_values, y_values, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y_values
    )
    
    # Create pipeline
    pipeline = create_svm_pipeline()
    
    if use_gridsearch:
        # Perform GridSearchCV with parallel processing
        print(f"Running GridSearchCV with {CV_FOLDS}-fold CV and {len(PARAM_GRID['svc__C']) * len(PARAM_GRID['svc__kernel']) * len(PARAM_GRID['svc__gamma'])} parameter combinations...")
        
        grid_search = GridSearchCV(
            pipeline,
            PARAM_GRID,
            cv=CV_FOLDS,
            scoring='accuracy',
            n_jobs=-1,  # Use all available cores
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model and parameters
        best_model = grid_search.best_estimator_
        best_params_raw = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Print GridSearch results with clear formatting
        print("\n" + "=" * 50)
        print("SVM GridSearch Results".center(50))
        print("=" * 50)
        print(f"Best Parameters: {best_params_raw}")
        print(f"Best Cross-Validation Score: {best_score:.4f}")
        print("=" * 50)
    else:
        # Train with pre-defined best parameters (faster, no GridSearch)
        print("\n" + "=" * 50)
        print("Training with Pre-defined Parameters".center(50))
        print("=" * 50)
        
        if best_params is None:
            # Use default best params from previous runs
            best_params = {'C': 100.0, 'gamma': 'scale', 'kernel': 'rbf'}
        
        print(f"Using Parameters: {best_params}")
        print("=" * 50)
        
        # Set pipeline parameters directly
        pipeline.set_params(
            svc__C=best_params['C'],
            svc__kernel=best_params['kernel'],
            svc__gamma=best_params['gamma']
        )
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        best_model = pipeline
        best_params_raw = {f'svc__{k}': v for k, v in best_params.items()}
        best_score = None  # No CV score when not using GridSearch
        grid_search = None
    
    # Make predictions on test set
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    class_report = classification_report(
        y_test, y_pred, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Extract best SVC parameters for display
    best_svc_params = {
        k.replace('svc__', ''): v 
        for k, v in best_params_raw.items()
    }
    
    # Print evaluation results
    print("\n" + "=" * 50)
    print("Evaluation Results".center(50))
    print("=" * 50)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    for class_name in class_names:
        if class_name in class_report:
            print(f"\n{class_name}:")
            print(f"  Precision: {class_report[class_name]['precision']:.4f}")
            print(f"  Recall:    {class_report[class_name]['recall']:.4f}")
            print(f"  F1-Score:  {class_report[class_name]['f1-score']:.4f}")
    print("=" * 50)
    
    metrics = {
        'accuracy_score': accuracy,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'best_params': best_svc_params,
        'best_score': best_score
    }
    
    # Build model params string
    model_params = f"SVC(C={best_svc_params['C']}, kernel='{best_svc_params['kernel']}', gamma='{best_svc_params['gamma']}')"
    
    # Extract the scaler and SVC from pipeline for visualization
    scaler = best_model.named_steps['scaler']
    svc_model = best_model.named_steps['svc']
    
    # Scale data for visualizations
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Store artifacts
    artifacts = {
        'pipeline': best_model,
        'scaler': scaler,
        'svc_model': svc_model,
        'X_train': X_train,
        'X_train_scaled': X_train_scaled,
        'y_train': y_train,
        'X_test': X_test,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'y_pred': y_pred,
        'model_params': model_params,
        'features': feature_names,
        'class_names': class_names,
        'best_params': best_svc_params,
        'random_state': random_state,
        'grid_search': grid_search
    }
    
    # Cache model if requested
    if cache_path:
        print(f"Caching model to: {cache_path}")
        joblib.dump({'metrics': metrics, 'artifacts': artifacts}, cache_path)
    
    return metrics, artifacts


# ============================================================================
# PREDICTION
# ============================================================================

def predict_affordability(
    pipeline: Pipeline,
    feature_dict: Dict[str, float],
    feature_names: list
) -> Tuple[str, np.ndarray]:
    """
    Predict housing affordability for new data.
    
    Args:
        pipeline: Trained sklearn Pipeline.
        feature_dict: Dictionary of feature values.
        feature_names: Expected feature names in order.
        
    Returns:
        prediction: Class label ('Affordable' or 'Not Affordable').
        probabilities: Prediction probabilities for each class.
    """
    # Create input array in correct feature order
    input_array = np.array([[feature_dict.get(f, 0.0) for f in feature_names]])
    
    # Make prediction
    prediction = pipeline.predict(input_array)[0]
    probabilities = pipeline.predict_proba(input_array)[0]
    
    return prediction, probabilities


# ============================================================================
# VISUALIZATION
# ============================================================================

# Import visualization functions from separate module
try:
    # Try relative import first (when imported as part of algorithms package)
    from .svm_visualizations import plot_results
except (ImportError, ValueError):
    # Fall back to direct import (when running as standalone script)
    try:
        from svm_visualizations import plot_results
    except ImportError:
        # Last resort: try importing from algorithms package
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from svm_visualizations import plot_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """Test the optimized SVM classifier."""
    
    # Construct path to housing.csv
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "housing.csv")
    cache_path = os.path.join(os.path.dirname(__file__), "svm_model_cache.pkl")
    
    print("=" * 80)
    print("OPTIMIZED SVM HOUSING AFFORDABILITY CLASSIFIER")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  - Random State: {RANDOM_STATE}")
    print(f"  - CV Folds: {CV_FOLDS}")
    print(f"  - Parameter Grid: {len(PARAM_GRID['svc__C']) * len(PARAM_GRID['svc__kernel']) * len(PARAM_GRID['svc__gamma'])} combinations")
    print(f"  - Parallel Processing: Enabled (n_jobs=-1)")
    print(f"  - Model Caching: Enabled")
    
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    # Option 1: Train with GridSearchCV (slower, finds best params)
    # metrics, artifacts = train_and_evaluate(file_path, use_cache=False, cache_path=cache_path, use_gridsearch=True)
    
    # Option 2: Train with pre-defined best params (faster, no GridSearch)
    metrics, artifacts = train_and_evaluate(
        file_path, 
        use_cache=False, 
        cache_path=cache_path,
        use_gridsearch=False,
        best_params={'C': 100.0, 'gamma': 'scale', 'kernel': 'rbf'}
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\n✓ Best Parameters: {metrics['best_params']}")
    print(f"✓ Best CV Score: {metrics['best_score']:.4f}")
    print(f"✓ Test Set Accuracy: {metrics['accuracy_score']:.4f}")
    
    print("\nClassification Report:")
    print("-" * 80)
    class_report = metrics['classification_report']
    for class_name in CLASS_NAMES:
        if class_name in class_report:
            print(f"\n{class_name}:")
            print(f"  Precision: {class_report[class_name]['precision']:.4f}")
            print(f"  Recall:    {class_report[class_name]['recall']:.4f}")
            print(f"  F1-Score:  {class_report[class_name]['f1-score']:.4f}")
            print(f"  Support:   {class_report[class_name]['support']}")
    
    print(f"\n✓ Overall Accuracy: {class_report['accuracy']:.4f}")
    print(f"✓ Macro Avg F1-Score: {class_report['macro avg']['f1-score']:.4f}")
    print(f"✓ Weighted Avg F1-Score: {class_report['weighted avg']['f1-score']:.4f}")
    
    print("\n" + "=" * 80)
    print("MODEL INFO")
    print("=" * 80)
    print(f"  Model: {artifacts['model_params']}")
    print(f"  Features: {len(artifacts['features'])}")
    print(f"  Train Size: {len(artifacts['y_train'])}")
    print(f"  Test Size: {len(artifacts['y_test'])}")
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("VISUALIZATIONS")
    print("=" * 80)
    
    plot_configs = [
        ('confusion_matrix', 'svm_confusion_matrix.png', 'Confusion Matrix'),
        ('decision_boundary', 'svm_decision_boundary.png', 'Decision Boundary'),
        ('support_vectors_2d', 'svm_support_vectors_2d.png', 'Support Vectors with Margins'),
        ('learning_curve', 'svm_learning_curve.png', 'Learning Curve')
    ]
    
    for plot_type, filename, description in plot_configs:
        print(f"\n{description}...")
        png_data = plot_results(artifacts, plot_type=plot_type)
        output_path = os.path.join(os.path.dirname(__file__), filename)
        with open(output_path, 'wb') as f:
            f.write(png_data)
        print(f"  ✓ Saved: {output_path}")
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE!")
    print("=" * 80)
    print(f"\n✓ Model cached to: {cache_path}")
    print(f"✓ All visualizations saved to: {os.path.dirname(__file__)}")
    print("\nTo reuse cached model, run with use_cache=True")


