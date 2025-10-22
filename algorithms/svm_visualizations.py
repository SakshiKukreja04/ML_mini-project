"""
SVM Visualization Module
=========================
Optimized visualization functions for Support Vector Machine classification.
Separated for better code organization and maintainability.

Author: Optimized by GitHub Copilot
Date: October 2025
"""

from typing import Dict, Any
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve


# Constants
RANDOM_STATE = 42
CV_FOLDS = 3


def plot_confusion_matrix(artifacts: Dict[str, Any]) -> bytes:
    """
    Generate confusion matrix visualization.
    
    Args:
        artifacts: Dictionary containing model artifacts.
        
    Returns:
        PNG bytes of the generated plot.
    """
    y_test = artifacts['y_test']
    y_pred = artifacts['y_pred']
    class_names = artifacts['class_names']
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Count'}, ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix - SVM Classification', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf.getvalue()


def plot_decision_boundary(artifacts: Dict[str, Any]) -> bytes:
    """
    Generate 2D decision boundary visualization using PCA.
    
    Args:
        artifacts: Dictionary containing model artifacts.
        
    Returns:
        PNG bytes of the generated plot.
    """
    X_test_scaled = artifacts['X_test_scaled']
    y_test = artifacts['y_test']
    class_names = artifacts['class_names']
    best_params = artifacts['best_params']
    
    # Use PCA to reduce to 2 dimensions
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_test_pca = pca.fit_transform(X_test_scaled)
    
    # Train a new SVC on 2D PCA data for visualization
    model_2d = SVC(
        C=best_params['C'],
        kernel=best_params['kernel'],
        gamma=best_params['gamma'],
        random_state=RANDOM_STATE
    )
    model_2d.fit(X_test_pca, y_test)
    
    # Create meshgrid
    h = 0.02
    x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
    y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    
    # Predict on meshgrid
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Map predictions to numeric
    class_to_num = {cls: idx for idx, cls in enumerate(class_names)}
    Z_numeric = np.array([class_to_num[z] for z in Z]).reshape(xx.shape)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors
    colors = ['#FFB3B3', '#B3D9FF']
    
    # Plot decision regions
    ax.contourf(xx, yy, Z_numeric, alpha=0.4, cmap=plt.cm.RdBu, levels=1)
    
    # Plot test points
    for idx, cls in enumerate(class_names):
        mask = y_test == cls
        ax.scatter(
            X_test_pca[mask, 0], X_test_pca[mask, 1],
            c=colors[idx], label=cls, edgecolors='black',
            linewidth=1.5, s=50, alpha=0.8
        )
    
    # Labels and title
    ax.set_xlabel('PCA Component 1', fontsize=12)
    ax.set_ylabel('PCA Component 2', fontsize=12)
    ax.set_title(
        f'SVM Decision Boundary (2D PCA) - {best_params["kernel"].upper()} kernel',
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='best', title='Price Category')
    
    # Add explained variance
    explained_var = pca.explained_variance_ratio_
    ax.text(
        0.02, 0.98,
        f'Explained Variance: {explained_var[0]:.2%} + {explained_var[1]:.2%} = {sum(explained_var):.2%}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf.getvalue()


def plot_support_vectors(artifacts: Dict[str, Any]) -> bytes:
    """
    Generate support vectors visualization with margins.
    
    Shows the decision boundary, margins, and support vectors using
    linear kernel on PCA-reduced data for clear visualization.
    
    Args:
        artifacts: Dictionary containing model artifacts.
        
    Returns:
        PNG bytes of the generated plot.
    """
    X_train_scaled = artifacts['X_train_scaled']
    y_train = artifacts['y_train']
    class_names = artifacts['class_names']
    best_params = artifacts['best_params']
    
    # Use PCA to reduce to 2 dimensions
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    # Train linear SVC on 2D data for clear margin visualization
    model_2d = SVC(
        C=best_params['C'],
        kernel='linear',
        random_state=RANDOM_STATE
    )
    model_2d.fit(X_train_pca, y_train)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create meshgrid
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    
    # Get decision function values
    Z = model_2d.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    ax.contour(xx, yy, Z, colors='black', levels=[0], linewidths=2, linestyles='solid')
    ax.contour(xx, yy, Z, colors='black', levels=[-1, 1], linewidths=1.5, linestyles='dashed')
    ax.contourf(xx, yy, Z, levels=[-1000, 0, 1000], colors=['#FFB3B3', '#B3D9FF'], alpha=0.3)
    
    # Plot training points (sample for clarity)
    colors_data = ['#FF0000', '#0000FF']
    for idx, cls in enumerate(class_names):
        mask = y_train == cls
        # Sample data points for cleaner visualization
        sample_size = min(500, np.sum(mask))
        sample_indices = np.random.choice(np.where(mask)[0], sample_size, replace=False)
        ax.scatter(
            X_train_pca[sample_indices, 0], X_train_pca[sample_indices, 1],
            c=colors_data[idx], s=30, alpha=0.6,
            edgecolors='black', linewidth=0.5,
            label=f'{cls} (sample)', marker='o'
        )
    
    # Highlight only 3-4 support vectors for clarity
    support_vectors = model_2d.support_vectors_
    n_support_total = len(support_vectors)
    
    # Select 4 representative support vectors (2 from each side of decision boundary)
    n_show = min(4, n_support_total)
    if n_support_total > 4:
        # Select support vectors closest to the decision boundary
        decision_values = model_2d.decision_function(support_vectors)
        abs_decision = np.abs(decision_values)
        closest_indices = np.argsort(abs_decision)[:n_show]
        support_vectors_display = support_vectors[closest_indices]
    else:
        support_vectors_display = support_vectors
    
    ax.scatter(
        support_vectors_display[:, 0], support_vectors_display[:, 1],
        s=200, linewidth=3, facecolors='none',
        edgecolors='green', label=f'Support Vectors (showing {len(support_vectors_display)})', marker='o'
    )
    
    # Labels and title
    ax.set_xlabel('PCA Component 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('PCA Component 2', fontsize=14, fontweight='bold')
    ax.set_title(
        'SVM Decision Boundary with Support Vectors & Margins',
        fontsize=16, fontweight='bold'
    )
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # Add info boxes
    info_text = f'Total Support Vectors: {n_support_total}\nDisplayed: {len(support_vectors_display)}\nC: {best_params["C"]}\nKernel: linear'
    ax.text(
        0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7)
    )
    
    explanation = 'Solid line: Decision boundary\nDashed lines: Margins\nGreen circles: Support vectors'
    ax.text(
        0.98, 0.02, explanation, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf.getvalue()


def plot_learning_curve(artifacts: Dict[str, Any]) -> bytes:
    """
    Generate learning curve visualization.
    
    Shows how model performance changes with training set size.
    
    Args:
        artifacts: Dictionary containing model artifacts.
        
    Returns:
        PNG bytes of the generated plot.
    """
    pipeline = artifacts['pipeline']
    X_train = artifacts['X_train']
    y_train = artifacts['y_train']
    
    # Define training sizes
    train_sizes = np.linspace(0.1, 1.0, 5)
    
    # Calculate learning curve with parallel processing
    print("Calculating learning curve...")
    train_sizes_abs, train_scores, val_scores = learning_curve(
        pipeline,
        X_train,
        y_train,
        train_sizes=train_sizes,
        cv=CV_FOLDS,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    # Calculate statistics
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training scores
    ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(
        train_sizes_abs, train_mean - train_std, train_mean + train_std,
        alpha=0.2, color='blue'
    )
    
    # Plot validation scores
    ax.plot(train_sizes_abs, val_mean, 'o-', color='green', label='Cross-Validation Score')
    ax.fill_between(
        train_sizes_abs, val_mean - val_std, val_mean + val_std,
        alpha=0.2, color='green'
    )
    
    # Labels and title
    ax.set_xlabel('Training Examples', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Learning Curve (SVM)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add info text
    best_params = artifacts['best_params']
    info_text = f"C={best_params['C']}, kernel={best_params['kernel']}, gamma={best_params['gamma']}"
    ax.text(
        0.02, 0.02, info_text, transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf.getvalue()


def plot_results(artifacts: Dict[str, Any], plot_type: str = 'confusion_matrix') -> bytes:
    """
    Generate visualization for SVM classification results.
    
    Dispatcher function that calls the appropriate plotting function.
    
    Args:
        artifacts: Dictionary containing model artifacts.
        plot_type: Type of plot to generate ('confusion_matrix', 'decision_boundary',
                   'support_vectors_2d', 'learning_curve').
        
    Returns:
        PNG bytes of the generated plot.
        
    Raises:
        ValueError: If plot_type is not supported.
    """
    plot_functions = {
        'confusion_matrix': plot_confusion_matrix,
        'decision_boundary': plot_decision_boundary,
        'support_vectors_2d': plot_support_vectors,
        'learning_curve': plot_learning_curve
    }
    
    if plot_type not in plot_functions:
        raise ValueError(
            f"Unsupported plot_type: {plot_type}. "
            f"Choose from: {', '.join(plot_functions.keys())}"
        )
    
    return plot_functions[plot_type](artifacts)
