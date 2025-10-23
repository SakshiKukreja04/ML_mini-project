"""
Ensemble Learning Module: Before and After Comparison
=====================================================

This module implements a "before and after" ensemble learning comparison,
using the same preprocessing and structure as decision_tree.py for consistency.

Two-Step Workflow:
1. find_best_params(): Slow GridSearchCV to find optimal Random Forest parameters
2. train_and_evaluate(): Fast training using found parameters to compare models

Bagging Approach:
- BEFORE: Single Decision Tree (from decision_tree.py)
- AFTER: Random Forest (bagging multiple decision trees)

Both models use the same:
- Data preprocessing (via decision_tree.prepare_data)
- Train/test split
- Target encoding (price_range: low/medium/high)
- Evaluation metrics

Author: GitHub Copilot
Date: October 23, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from typing import Dict, Tuple, List, Any
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

# Import preprocessing from decision_tree module
try:
    from algorithms.decision_tree import prepare_data
    DECISION_TREE_AVAILABLE = True
except ImportError:
    try:
        from decision_tree import prepare_data
        DECISION_TREE_AVAILABLE = True
    except ImportError:
        DECISION_TREE_AVAILABLE = False
        print("Warning: Could not import prepare_data from decision_tree module")


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================



def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Load and preprocess the housing dataset using decision_tree.prepare_data().
    
    This ensures both Decision Tree and Random Forest use identical preprocessing:
    - Same feature selection (numeric only)
    - Same target encoding (low/medium/high price ranges)
    - Same quantile splits (33% and 66% percentiles)
    - Handles NaN values by dropping rows
    
    Args:
        file_path: Path to the housing.csv file
    
    Returns:
        X: Feature DataFrame (numeric features only, no NaN values)
        y: Target Series (3 categories: 'low', 'medium', 'high')
        feature_names: List of feature names
    """
    if not DECISION_TREE_AVAILABLE:
        raise ImportError("decision_tree module not available. Cannot use prepare_data().")
    
    # Use decision_tree's preprocessing for consistency
    X, y, df_full, preprocessing = prepare_data(file_path, target_col='median_house_value')
    
    # Handle NaN values (critical for PCA and other algorithms)
    print(f"Checking for NaN values...")
    nan_count_before = X.isna().sum().sum()
    if nan_count_before > 0:
        print(f"  Found {nan_count_before} NaN values in features")
        # Drop rows with any NaN values
        valid_indices = ~X.isna().any(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]
        print(f"  Dropped {(~valid_indices).sum()} rows with NaN values")
        print(f"  Remaining samples: {len(X)}")
    else:
        print(f"  No NaN values found")
    
    # Final verification
    assert X.isna().sum().sum() == 0, "NaN values still present after cleaning!"
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    print(f"Preprocessing info: {preprocessing}")
    print(f"Features used: {feature_names}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_names


# ============================================================================
# STEP 1: FIND BEST PARAMETERS (SLOW - RUN ONCE)
# ============================================================================

def find_best_params(file_path: str, random_state: int = 42) -> Dict[str, Any]:
    """
    Find the best hyperparameters for Random Forest using GridSearchCV.
    
    This is the SLOW step that should be run once to find optimal parameters.
    
    Args:
        file_path: Path to the housing.csv file
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary containing:
            - best_params: Best parameters found by GridSearchCV
            - best_score: Best cross-validation score achieved
    """
    print("=" * 80)
    print("STEP 1: Finding Best Parameters with GridSearchCV")
    print("=" * 80)
    
    # Load and preprocess data
    X, y, feature_names = load_and_preprocess_data(file_path)
    print(f"\nDataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target classes: {y.value_counts().to_dict()}")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    print(f"\nParameter grid: {param_grid}")
    print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
    
    # Initialize Random Forest
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    
    # Initialize and run GridSearchCV
    print("\nRunning GridSearchCV (this will take several minutes)...")
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X, y)
    
    # Extract results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print("\n" + "=" * 80)
    print("GridSearchCV Complete!")
    print("=" * 80)
    print(f"Best Parameters: {best_params}")
    print(f"Best CV Score: {best_score:.4f} ({best_score*100:.2f}%)")
    print("=" * 80)
    
    return {
        'best_params': best_params,
        'best_score': best_score
    }


# ============================================================================
# STEP 2: TRAIN AND EVALUATE (FAST - USE FOUND PARAMETERS)
# ============================================================================

def train_and_evaluate(
    file_path: str,
    rf_params: Dict[str, Any],
    base_tree_max_depth: int = 10,
    random_state: int = 42
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Train and evaluate both baseline and ensemble models using found parameters.
    
    This is the FAST step that compares "before" (single tree) vs "after" (Random Forest).
    
    Args:
        file_path: Path to the housing.csv file
        rf_params: Best parameters found by find_best_params()
        base_tree_max_depth: Max depth for baseline Decision Tree
        random_state: Random seed for reproducibility
    
    Returns:
        metrics: Dictionary containing performance metrics for both models
        artifacts: Dictionary containing trained models and data for visualization
    """
    print("=" * 80)
    print("STEP 2: Training and Evaluating Models")
    print("=" * 80)
    
    try:
        # Load and preprocess data
        X, y, feature_names = load_and_preprocess_data(file_path)
        print(f"\nDataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Encode labels for PCA visualization
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Initialize metrics and artifacts dictionaries
        metrics = {
            'single_tree': {},
            'random_forest': {}
        }
        artifacts = {
            'single_tree': {},
            'random_forest': {}
        }
        
        # ========================================================================
        # BEFORE: Train Single Decision Tree (Baseline)
        # ========================================================================
        print("\n" + "-" * 80)
        print("Training BASELINE: Single Decision Tree")
        print("-" * 80)
        
        tree_model = DecisionTreeClassifier(
            max_depth=base_tree_max_depth,
            random_state=random_state
        )
        tree_model.fit(X_train, y_train)
        
        # Predictions
        y_pred_tree = tree_model.predict(X_test)
        
        # Metrics
        tree_accuracy = accuracy_score(y_test, y_pred_tree)
        tree_precision, tree_recall, tree_f1, _ = precision_recall_fscore_support(
            y_test, y_pred_tree, average='weighted', zero_division=0
        )
        
        print(f"Baseline Tree Accuracy: {tree_accuracy:.4f} ({tree_accuracy*100:.2f}%)")
        
        metrics['single_tree'] = {
            'accuracy': tree_accuracy,
            'precision': tree_precision,
            'recall': tree_recall,
            'f1_score': tree_f1,
            'classification_report': classification_report(y_test, y_pred_tree, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred_tree)
        }
        
        artifacts['single_tree'] = {
            'model': tree_model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred_tree,
            'features': feature_names,
            'class_names': tree_model.classes_.tolist(),  # Use actual classes from model
            'X_train_processed': X_train,
            'y_train_encoded': y_train_encoded,
            'label_encoder': label_encoder,
            'accuracy': tree_accuracy  # Add accuracy for scatter plot
        }
        
        # ========================================================================
        # AFTER: Train Random Forest (Ensemble)
        # ========================================================================
        print("\n" + "-" * 80)
        print("Training ENSEMBLE: Random Forest")
        print("-" * 80)
        print(f"Using parameters: {rf_params}")
        
        rf_model = RandomForestClassifier(
            **rf_params,
            random_state=random_state,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Predictions
        y_pred_rf = rf_model.predict(X_test)
        
        # Metrics
        rf_accuracy = accuracy_score(y_test, y_pred_rf)
        rf_precision, rf_recall, rf_f1, _ = precision_recall_fscore_support(
            y_test, y_pred_rf, average='weighted', zero_division=0
        )
        
        print(f"Random Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
        print(f"Improvement: {(rf_accuracy - tree_accuracy)*100:.2f}%")
        
        metrics['random_forest'] = {
            'accuracy': rf_accuracy,
            'precision': rf_precision,
            'recall': rf_recall,
            'f1_score': rf_f1,
            'classification_report': classification_report(y_test, y_pred_rf, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred_rf),
            'best_params': rf_params,
            'feature_importances': rf_model.feature_importances_
        }
        
        artifacts['random_forest'] = {
            'model': rf_model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred_rf,
            'features': feature_names,
            'class_names': rf_model.classes_.tolist(),  # Use actual classes from model
            'X_train_processed': X_train,
            'y_train_encoded': y_train_encoded,
            'label_encoder': label_encoder,
            'accuracy': rf_accuracy  # Add accuracy for scatter plot
        }
        
        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"Baseline (Single Tree): {tree_accuracy*100:.2f}%")
        print(f"Ensemble (Random Forest): {rf_accuracy*100:.2f}%")
        print(f"Improvement: +{(rf_accuracy - tree_accuracy)*100:.2f}%")
        print("=" * 80)
        
        return metrics, artifacts
    
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Return error in metrics
        return {
            'single_tree': {'error': str(e)},
            'random_forest': {'error': str(e)}
        }, {}


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_results(
    artifacts: Dict[str, Dict[str, Any]],
    model_type: str,
    plot_type: str
) -> io.BytesIO:
    """
    Generate optimized visualizations for before/after ensemble comparison.
    
    Args:
        artifacts: Dictionary containing model artifacts
        model_type: 'single_tree' or 'random_forest'
        plot_type: Type of plot to generate:
            - 'confusion_matrix': Normalized confusion matrix heatmap
            - 'learning_curve': Accuracy vs estimators (RF) or bias/variance (Tree)
            - 'feature_importance': Top 10 feature importance (Random Forest only)
            - 'decision_boundary_pca': Decision boundary with probability shading
            - 'probability_density': Model confidence distribution (KDE plot)
    
    Returns:
        BytesIO buffer containing the plot image
    """
    if model_type not in artifacts:
        raise ValueError(f"Model type '{model_type}' not found in artifacts")
    
    model_artifacts = artifacts[model_type]
    
    # Set consistent style and color palette
    plt.style.use('seaborn-v0_8-whitegrid')
    
    if plot_type == 'confusion_matrix':
        return _plot_confusion_matrix(model_artifacts, model_type)
    elif plot_type == 'learning_curve':
        return _plot_learning_curve(model_artifacts, model_type)
    elif plot_type == 'feature_importance':
        if model_type != 'random_forest':
            raise ValueError("Feature importance is only available for Random Forest")
        return _plot_feature_importance(model_artifacts)
    elif plot_type == 'decision_boundary_pca':
        return _plot_decision_boundary_pca(model_artifacts, model_type)
    elif plot_type == 'probability_density':
        return _plot_probability_density(model_artifacts, model_type)
    elif plot_type == 'decision_boundary_scatter':
        return _plot_decision_boundary_scatter(model_artifacts, model_type)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")


def _plot_confusion_matrix(artifacts: Dict[str, Any], model_type: str) -> io.BytesIO:
    """Plot normalized confusion matrix heatmap with percentages."""
    y_test = artifacts['y_test']
    y_pred = artifacts['y_pred']
    class_names = artifacts['class_names']
    
    # Calculate both raw and normalized confusion matrices
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Create annotations with both count and percentage
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})'
    
    sns.heatmap(
        cm_normalized, annot=annotations, fmt='', cmap='coolwarm',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax, cbar_kws={'label': 'Normalized Rate'},
        vmin=0, vmax=1, linewidths=0.5, linecolor='gray'
    )
    
    # Add subtitle based on model type
    model_name = "Single Decision Tree (Before)" if model_type == 'single_tree' else "Random Forest (After)"
    title = f"Confusion Matrix: {model_name}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf


def _plot_learning_curve(artifacts: Dict[str, Any], model_type: str) -> io.BytesIO:
    """
    Plot learning curves:
    - Random Forest: Accuracy vs Number of Estimators (demonstrates ensemble effect)
    - Single Tree: Training vs Validation (shows overfitting)
    """
    model = artifacts['model']
    X_train = artifacts['X_train']
    y_train = artifacts['y_train']
    X_test = artifacts['X_test']
    y_test = artifacts['y_test']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if model_type == 'random_forest':
        # For Random Forest: Show accuracy vs number of estimators
        n_estimators = model.n_estimators
        estimator_range = range(1, min(n_estimators + 1, 201), max(1, n_estimators // 20))
        
        train_scores = []
        test_scores = []
        
        for n in estimator_range:
            # Use subset of trees
            temp_model = RandomForestClassifier(
                n_estimators=n,
                max_depth=model.max_depth,
                max_features=model.max_features,
                min_samples_split=model.min_samples_split,
                min_samples_leaf=model.min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )
            temp_model.fit(X_train, y_train)
            train_scores.append(temp_model.score(X_train, y_train))
            test_scores.append(temp_model.score(X_test, y_test))
        
        ax.plot(estimator_range, train_scores, 'o-', color='#2E86AB', 
                label='Training Accuracy', linewidth=2.5, markersize=6)
        ax.plot(estimator_range, test_scores, 'o-', color='#A23B72', 
                label='Test Accuracy', linewidth=2.5, markersize=6)
        
        ax.set_xlabel('Number of Trees in Forest', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy Score', fontsize=12, fontweight='bold')
        ax.set_title('Ensemble Effect: Random Forest (After)', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
    else:  # single_tree
        # For Single Tree: Show traditional learning curve (training size vs accuracy)
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            cv=3,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy',
            random_state=42
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.plot(train_sizes, train_mean, 'o-', color='#2E86AB', 
                label='Training Score', linewidth=2.5, markersize=6)
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.2, color='#2E86AB')
        
        ax.plot(train_sizes, val_mean, 'o-', color='#A23B72', 
                label='Validation Score', linewidth=2.5, markersize=6)
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.2, color='#A23B72')
        
        # Highlight the gap (overfitting indicator)
        ax.fill_between(train_sizes, train_mean, val_mean, alpha=0.15, color='red', 
                        label='Overfitting Gap')
        
        ax.set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy Score', fontsize=12, fontweight='bold')
        ax.set_title('Bias/Variance: Single Decision Tree (Before)', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf


def _plot_feature_importance(artifacts: Dict[str, Any]) -> io.BytesIO:
    """Plot top 10 feature importances with color-coded magnitude."""
    model = artifacts['model']
    feature_names = artifacts['features']
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Select top 10 features
    top_n = min(10, len(feature_names))
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Color coding: High importance = dark red, Low = light yellow
    colors = plt.cm.YlOrRd(np.linspace(0.4, 0.95, top_n))
    bars = ax.barh(range(top_n), top_importances, color=colors, edgecolor='black', linewidth=1.2)
    
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features, fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Top Features Driving Ensemble Model (RF Only)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, top_importances)):
        ax.text(importance + 0.005, i, f'{importance:.4f}', 
                va='center', fontsize=10, fontweight='bold')
    
    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, 
                               norm=plt.Normalize(vmin=top_importances.min(), 
                                                 vmax=top_importances.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label('Importance Magnitude', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf


def _plot_decision_boundary_pca(artifacts: Dict[str, Any], model_type: str) -> io.BytesIO:
    """
    Plot decision boundary on 2D PCA projection.
    
    This visualization helps understand how each model separates the data:
    - A jagged, complex boundary indicates overfitting (common in single trees)
    - A smoother boundary indicates better generalization (typical of Random Forest)
    
    Note: Uses a subset of data points for faster plotting.
    """
    # Get data
    X_train_processed = artifacts['X_train_processed']
    y_train_encoded = artifacts['y_train_encoded']
    model = artifacts['model']
    class_names = artifacts['class_names']
    
    # Reduce data points for faster plotting (use 2000 samples max)
    max_plot_samples = 2000
    if len(X_train_processed) > max_plot_samples:
        print(f"Reducing data points from {len(X_train_processed)} to {max_plot_samples} for plotting")
        sample_indices = np.random.RandomState(42).choice(len(X_train_processed), max_plot_samples, replace=False)
        X_train_plot = X_train_processed.iloc[sample_indices]
        y_train_plot = y_train_encoded[sample_indices]
    else:
        X_train_plot = X_train_processed
        y_train_plot = y_train_encoded
    
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_train_plot)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Re-train model on 2D PCA data
    if model_type == 'single_tree':
        # Get original max_depth parameter
        max_depth = model.max_depth
        pca_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    else:  # random_forest
        # Get original parameters
        n_estimators = model.n_estimators
        max_depth = model.max_depth
        max_features = model.max_features
        min_samples_split = model.min_samples_split
        min_samples_leaf = model.min_samples_leaf
        
        pca_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
    
    pca_model.fit(X_pca, y_train_plot)
    
    # Create mesh grid for decision boundary
    # Use larger step size to reduce memory usage (0.1 instead of 0.02)
    h = 0.1  # step size in the mesh (larger = faster, less memory)
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    
    # Limit the mesh grid size to prevent memory issues
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Adjust step size if range is too large
    max_points_per_axis = 200  # Maximum 200x200 = 40k points
    h_x = max(h, x_range / max_points_per_axis)
    h_y = max(h, y_range / max_points_per_axis)
    
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h_x), 
        np.arange(y_min, y_max, h_y)
    )
    
    print(f"Mesh grid shape: {xx.shape} (total points: {xx.size})")
    
    # Predict class probabilities on mesh grid for probability shading
    Z_proba = pca_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = pca_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Get max probability (confidence) for each point
    Z_confidence = np.max(Z_proba, axis=1).reshape(xx.shape)
    
    # Create color map
    cmap_bold = ['#E74C3C', '#2ECC71', '#3498DB']  # Red, Green, Blue
    
    # Plot
    fig, ax = plt.subplots(figsize=(11, 8))
    
    # Plot decision boundary with probability shading (alpha varies by confidence)
    contour = ax.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm', levels=np.arange(-0.5, 3.5, 1))
    
    # Add probability contours
    contour_lines = ax.contour(xx, yy, Z_confidence, levels=[0.5, 0.7, 0.9], 
                               colors='black', linewidths=[1, 1.5, 2], 
                               linestyles=['dotted', 'dashed', 'solid'], alpha=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=9, fmt='%.1f conf')
    
    # Plot training data (using the reduced y_train_plot)
    for idx, label in enumerate(np.unique(y_train_plot)):
        mask = y_train_plot == label
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=cmap_bold[idx], label=class_names[idx],
            edgecolors='white', linewidth=1, s=60, alpha=0.8, marker='o'
        )
    
    # Set labels and title
    model_name = "Single Decision Tree (Before)" if model_type == 'single_tree' else "Random Forest (After)"
    title = f"Decision Boundary: {model_name}\nPCA - Class Separation View"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                  fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add info box
    boundary_type = "Jagged (Overfitting)" if model_type == 'single_tree' else "Smooth (Generalization)"
    info_text = f'Boundary: {boundary_type}\n'
    info_text += f'PCA Variance: {pca.explained_variance_ratio_.sum():.1%}\n'
    info_text += f'Confidence Lines: 50%, 70%, 90%'
    ax.text(
        0.02, 0.98, info_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, edgecolor='black', linewidth=1.5)
    )
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf


def _plot_probability_density(artifacts: Dict[str, Any], model_type: str) -> io.BytesIO:
    """
    Plot KDE of predicted probabilities to show model confidence.
    
    Random Forest outputs smoother, more confident predictions compared to
    single decision trees which tend to be more extreme (0 or 1).
    """
    model = artifacts['model']
    X_test = artifacts['X_test']
    y_test = artifacts['y_test']
    class_names = artifacts['class_names']
    
    # Get predicted probabilities
    y_proba = model.predict_proba(X_test)
    
    # Get the maximum probability (confidence) for each prediction
    max_proba = np.max(y_proba, axis=1)
    
    # Get probabilities for each class
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Model Confidence Distribution: {model_type.replace("_", " ").title()}\n' + 
                 ('Before (Single Tree)' if model_type == 'single_tree' else 'After (Random Forest)'),
                 fontsize=15, fontweight='bold', y=0.995)
    
    # Plot 1: Overall confidence distribution
    ax1 = axes[0, 0]
    ax1.hist(max_proba, bins=50, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.2)
    sns.kdeplot(max_proba, ax=ax1, color='darkblue', linewidth=3, label='KDE')
    ax1.axvline(max_proba.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {max_proba.mean():.3f}')
    ax1.set_xlabel('Maximum Predicted Probability (Confidence)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Overall Prediction Confidence', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Per-class probability distribution
    ax2 = axes[0, 1]
    colors = ['#E74C3C', '#2ECC71', '#3498DB']
    for idx, class_name in enumerate(class_names):
        class_proba = y_proba[:, idx]
        sns.kdeplot(class_proba, ax=ax2, color=colors[idx], linewidth=2.5, 
                   label=f'{class_name} (mean={class_proba.mean():.2f})', fill=True, alpha=0.3)
    ax2.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax2.set_title('Per-Class Probability Distributions', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: Confidence by correctness
    ax3 = axes[1, 0]
    y_pred = model.predict(X_test)
    correct_mask = (y_pred == y_test)
    
    correct_proba = max_proba[correct_mask]
    incorrect_proba = max_proba[~correct_mask]
    
    ax3.hist([correct_proba, incorrect_proba], bins=30, label=['Correct', 'Incorrect'], 
            color=['green', 'red'], alpha=0.6, edgecolor='black', linewidth=1)
    ax3.set_xlabel('Prediction Confidence', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax3.set_title('Confidence: Correct vs Incorrect Predictions', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 4: Confidence statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    📊 Confidence Statistics
    
    Overall:
    • Mean Confidence: {max_proba.mean():.3f}
    • Median Confidence: {np.median(max_proba):.3f}
    • Std Dev: {max_proba.std():.3f}
    
    Correct Predictions ({len(correct_proba)}):
    • Mean Confidence: {correct_proba.mean():.3f}
    • High Confidence (>0.8): {(correct_proba > 0.8).sum()} ({(correct_proba > 0.8).sum()/len(correct_proba)*100:.1f}%)
    
    Incorrect Predictions ({len(incorrect_proba)}):
    • Mean Confidence: {incorrect_proba.mean():.3f}
    • High Confidence (>0.8): {(incorrect_proba > 0.8).sum()} ({(incorrect_proba > 0.8).sum()/len(incorrect_proba)*100 if len(incorrect_proba) > 0 else 0:.1f}%)
    
    Interpretation:
    {'✅ High mean confidence indicates model certainty' if max_proba.mean() > 0.7 else '⚠️ Low confidence suggests uncertainty'}
    {'✅ Correct predictions more confident than incorrect' if correct_proba.mean() > incorrect_proba.mean() else '⚠️ Model is overconfident in errors'}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', 
                                         alpha=0.8, edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf


def _plot_decision_boundary_scatter(artifacts: Dict[str, Any], model_type: str) -> io.BytesIO:
    """
    Create standalone decision boundary scatter plot for BEFORE/AFTER comparison.
    
    This creates clean, separate visualizations showing:
    - Colored decision regions (via contourf)
    - Training data as scatter points
    - Clear title with ACTUAL model accuracy (from full-dimensional data)
    - Simple, publication-ready format
    
    Note: The decision boundary shown is based on 2D PCA projection for visualization,
    but the accuracy displayed is from the actual model trained on all features.
    Uses a subset of data points for faster plotting.
    """
    # Get data
    X_train_processed = artifacts['X_train_processed']
    y_train_encoded = artifacts['y_train_encoded']
    X_test = artifacts['X_test']
    y_test = artifacts['y_test']
    model = artifacts['model']
    class_names = artifacts['class_names']
    
    # Get ACTUAL model accuracy from full-dimensional data
    actual_accuracy = artifacts.get('accuracy', model.score(X_test, y_test))
    
    # Reduce data points for faster plotting (use 2000 samples max)
    max_plot_samples = 2000
    if len(X_train_processed) > max_plot_samples:
        print(f"Reducing scatter plot data from {len(X_train_processed)} to {max_plot_samples} points")
        sample_indices = np.random.RandomState(42).choice(len(X_train_processed), max_plot_samples, replace=False)
        X_train_plot = X_train_processed.iloc[sample_indices]
        y_train_plot = y_train_encoded[sample_indices]
    else:
        X_train_plot = X_train_processed
        y_train_plot = y_train_encoded
    
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train_plot)
    X_test_pca = pca.transform(X_test)
    
    # Encode test labels
    label_encoder = artifacts.get('label_encoder')
    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.fit(y_train_plot)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Re-train model on 2D PCA data
    if model_type == 'single_tree':
        max_depth = model.max_depth
        pca_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    else:  # random_forest
        n_estimators = model.n_estimators
        max_depth = model.max_depth
        max_features = model.max_features
        min_samples_split = model.min_samples_split
        min_samples_leaf = model.min_samples_leaf
        
        pca_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
    
    pca_model.fit(X_train_pca, y_train_plot)
    
    # Calculate PCA accuracy (for reference, but we'll display actual accuracy)
    pca_accuracy = pca_model.score(X_test_pca, y_test_encoded)
    
    # Create mesh grid for decision boundary
    h = 0.1  # step size
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    
    # Limit mesh size
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_points_per_axis = 200
    h_x = max(h, x_range / max_points_per_axis)
    h_y = max(h, y_range / max_points_per_axis)
    
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h_x),
        np.arange(y_min, y_max, h_y)
    )
    
    # Predict on mesh grid
    Z = pca_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Define colors for decision regions and scatter points
    region_colors = ['#FFE5E5', '#E5FFE5', '#E5E5FF']  # Light red, green, blue
    scatter_colors = ['#FF4444', '#44FF44', '#4444FF']  # Bold red, green, blue
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot decision regions (colored background)
    cmap = ListedColormap(region_colors)
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=cmap, levels=np.arange(-0.5, 3.5, 1))
    
    # Plot training data as scatter points (using reduced dataset)
    for idx, label in enumerate(np.unique(y_train_plot)):
        mask = y_train_plot == label
        ax.scatter(
            X_train_pca[mask, 0], X_train_pca[mask, 1],
            c=scatter_colors[idx], label=class_names[idx],
            edgecolors='black', linewidth=1.5, s=80, alpha=0.9, marker='o'
        )
    
    # Set title based on model type
    if model_type == 'single_tree':
        title = f"Before: Single Decision Tree\n(Actual Accuracy: {actual_accuracy:.4f})"
    else:
        title = f"After: Random Forest\n(Actual Accuracy: {actual_accuracy:.4f})"
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Feature 1 (PCA Component 1)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Feature 2 (PCA Component 2)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black', shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Add accuracy text in top-left corner (showing ACTUAL accuracy from full model)
    accuracy_text = f'Full Model Accuracy: {actual_accuracy:.4f} ({actual_accuracy*100:.2f}%)'
    ax.text(0.02, 0.98, accuracy_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                     edgecolor='black', linewidth=2))
    
    # Add explanation text
    if model_type == 'single_tree':
        explanation = f'Baseline: Simple decision tree\nMay show overfitting (jagged boundaries)\n\n2D PCA Accuracy: {pca_accuracy:.4f}'
    else:
        explanation = f'Ensemble: Random Forest\nSmooth boundaries, better generalization\n\n2D PCA Accuracy: {pca_accuracy:.4f}'
    
    ax.text(0.98, 0.02, explanation, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, 
                     edgecolor='black', linewidth=1.5))
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf


def save_decision_boundary_plots(artifacts: Dict[str, Dict[str, Any]], save_dir: str = '.') -> Tuple[str, str]:
    """
    Save both BEFORE and AFTER decision boundary plots as PNG files.
    
    Args:
        artifacts: Dictionary containing both 'single_tree' and 'random_forest' artifacts
        save_dir: Directory to save the PNG files (default: current directory)
    
    Returns:
        Tuple of (before_path, after_path) - paths to saved PNG files
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate BEFORE plot (Single Tree)
    print("Generating BEFORE plot (Single Decision Tree)...")
    before_buffer = _plot_decision_boundary_scatter(artifacts['single_tree'], 'single_tree')
    before_path = os.path.join(save_dir, 'before_tree.png')
    with open(before_path, 'wb') as f:
        f.write(before_buffer.getvalue())
    print(f"✅ Saved: {before_path}")
    
    # Generate AFTER plot (Random Forest)
    print("Generating AFTER plot (Random Forest)...")
    after_buffer = _plot_decision_boundary_scatter(artifacts['random_forest'], 'random_forest')
    after_path = os.path.join(save_dir, 'after_rf.png')
    with open(after_path, 'wb') as f:
        f.write(after_buffer.getvalue())
    print(f"✅ Saved: {after_path}")
    
    return before_path, after_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Demonstrate the two-step workflow:
    1. Find best parameters (slow, run once)
    2. Train models with those parameters (fast, can run multiple times)
    """
    import os
    
    # Path to housing data
    file_path = os.path.join("..", "housing.csv")
    
    print("\n" + "=" * 80)
    print("ENSEMBLE LEARNING: TWO-STEP WORKFLOW DEMONSTRATION")
    print("=" * 80)
    
    # STEP 1: Find best parameters (slow)
    print("\n>>> STEP 1: Finding best parameters...")
    param_results = find_best_params(file_path, random_state=42)
    best_params = param_results['best_params']
    best_score = param_results['best_score']
    
    print(f"\n>>> Best parameters found:")
    for key, value in best_params.items():
        print(f"    {key}: {value}")
    print(f"\n>>> Best CV score: {best_score:.4f} ({best_score*100:.2f}%)")
    
    # STEP 2: Train models with best parameters (fast)
    print("\n>>> STEP 2: Training models with best parameters...")
    metrics, artifacts = train_and_evaluate(
        file_path,
        rf_params=best_params,
        base_tree_max_depth=10,
        random_state=42
    )
    
    # Display comparison
    if 'error' not in metrics['single_tree']:
        print("\n" + "=" * 80)
        print("FINAL COMPARISON")
        print("=" * 80)
        print(f"Baseline (Single Tree) Accuracy: {metrics['single_tree']['accuracy']*100:.2f}%")
        print(f"Ensemble (Random Forest) Accuracy: {metrics['random_forest']['accuracy']*100:.2f}%")
        print(f"Improvement: +{(metrics['random_forest']['accuracy'] - metrics['single_tree']['accuracy'])*100:.2f}%")
        print("=" * 80)
        
        print("\n>>> Classification Reports:")
        print("\nBaseline (Single Tree):")
        print(metrics['single_tree']['classification_report'])
        print("\nEnsemble (Random Forest):")
        print(metrics['random_forest']['classification_report'])
