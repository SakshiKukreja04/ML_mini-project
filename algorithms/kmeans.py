"""
K-Means Clustering Module
Performs unsupervised learning using K-Means clustering algorithm.
Includes elbow method, silhouette analysis, and comprehensive visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings('ignore')

# Import the prepare_data function from decision_tree for consistent preprocessing
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from decision_tree import prepare_data


def load_data(data_path='housing.csv'):
    """
    Load and preprocess the dataset using the same preprocessing as decision tree.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file containing the dataset
        
    Returns:
    --------
    X_scaled : numpy.ndarray
        Scaled feature matrix ready for clustering
    y : pandas.Series
        Original target labels (for comparison purposes)
    df_full : pandas.DataFrame
        Full preprocessed dataframe
    feature_names : list
        Names of the features used
    """
    print("=" * 80)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("=" * 80)
    
    # Use the same preprocessing as decision tree for consistency
    X, y, df_full, preprocessing_info = prepare_data(data_path)
    
    # Handle any NaN values that might exist
    if X.isnull().any().any():
        print(f"\n⚠️ Warning: Found {X.isnull().sum().sum()} NaN values. Dropping rows with NaN...")
        mask = ~X.isnull().any(axis=1)
        X = X[mask]
        y = y[mask]
        df_full = df_full[mask]
    
    # Store feature names
    feature_names = X.columns.tolist()
    
    print(f"\n✓ Dataset loaded successfully")
    print(f"  - Total samples: {len(X)}")
    print(f"  - Number of features: {len(feature_names)}")
    print(f"  - Features: {', '.join(feature_names)}")
    print(f"  - Target categories: {y.unique().tolist()}")
    print(f"  - Target distribution:\n{y.value_counts()}")
    
    # Scale the features for K-Means (important for distance-based algorithms)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n✓ Features standardized using StandardScaler")
    print(f"  - Mean: ~0, Std: ~1 for all features")
    
    return X_scaled, y, df_full, feature_names


def perform_kmeans(X_scaled, k_range=range(2, 11), optimal_k=None):
    """
    Perform K-Means clustering with elbow method and silhouette analysis.
    
    Parameters:
    -----------
    X_scaled : numpy.ndarray
        Scaled feature matrix
    k_range : range
        Range of cluster numbers to test
    optimal_k : int, optional
        If provided, use this value for final clustering. Otherwise, auto-detect.
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'inertias': list of inertia values for each k
        - 'silhouette_scores': list of silhouette scores for each k
        - 'optimal_k': chosen number of clusters
        - 'final_model': fitted KMeans model with optimal_k
        - 'cluster_labels': cluster assignments for each sample
        - 'centroids': cluster centroids
        - 'final_inertia': inertia of the final model
        - 'final_silhouette': silhouette score of the final model
    """
    print("\n" + "=" * 80)
    print("STEP 2: ELBOW METHOD AND SILHOUETTE ANALYSIS")
    print("=" * 80)
    
    inertias = []
    silhouette_scores_list = []
    
    print(f"\nTesting K-Means for k = {k_range.start} to {k_range.stop - 1}...")
    print("-" * 80)
    
    # Test different values of k
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(X_scaled)
        
        inertia = kmeans.inertia_
        silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
        
        inertias.append(inertia)
        silhouette_scores_list.append(silhouette_avg)
        
        print(f"k = {k:2d} | Inertia: {inertia:12,.2f} | Silhouette Score: {silhouette_avg:.4f}")
    
    # Determine optimal k using elbow method (rate of decrease in inertia)
    if optimal_k is None:
        # Calculate the rate of change (second derivative approximation)
        inertia_diff = np.diff(inertias)
        inertia_diff2 = np.diff(inertia_diff)
        
        # Find the elbow point (where rate of decrease slows down significantly)
        # Use silhouette score as a tiebreaker
        optimal_k_elbow = k_range.start + np.argmax(inertia_diff2) + 2
        optimal_k_silhouette = k_range.start + np.argmax(silhouette_scores_list)
        
        # Prefer silhouette score if it's close to elbow point
        if abs(optimal_k_elbow - optimal_k_silhouette) <= 1:
            optimal_k = optimal_k_silhouette
        else:
            optimal_k = optimal_k_elbow
    
    print("-" * 80)
    print(f"\n✓ Optimal number of clusters determined: k = {optimal_k}")
    print(f"  - Method: Elbow method combined with silhouette analysis")
    print(f"  - Best silhouette score: {max(silhouette_scores_list):.4f} at k = {k_range.start + np.argmax(silhouette_scores_list)}")
    
    # Train final model with optimal k
    print("\n" + "=" * 80)
    print(f"STEP 3: TRAINING FINAL K-MEANS MODEL (k = {optimal_k})")
    print("=" * 80)
    
    final_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
    cluster_labels = final_model.fit_predict(X_scaled)
    
    final_inertia = final_model.inertia_
    final_silhouette = silhouette_score(X_scaled, cluster_labels)
    
    print(f"\n✓ Model trained successfully")
    print(f"  - Number of clusters: {optimal_k}")
    print(f"  - Inertia (within-cluster sum of squares): {final_inertia:,.2f}")
    print(f"  - Silhouette Score: {final_silhouette:.4f}")
    print(f"  - Number of iterations: {final_model.n_iter_}")
    
    # Analyze cluster sizes
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\n  Cluster Distribution:")
    for cluster_id, count in zip(unique, counts):
        percentage = (count / len(cluster_labels)) * 100
        print(f"    - Cluster {cluster_id}: {count:,} samples ({percentage:.1f}%)")
    
    results = {
        'inertias': inertias,
        'silhouette_scores': silhouette_scores_list,
        'k_range': k_range,
        'optimal_k': optimal_k,
        'final_model': final_model,
        'cluster_labels': cluster_labels,
        'centroids': final_model.cluster_centers_,
        'final_inertia': final_inertia,
        'final_silhouette': final_silhouette
    }
    
    return results


def visualize_results(X_scaled, results, feature_names, y=None, save_plots=False, output_dir='output'):
    """
    Create comprehensive visualizations for K-Means clustering results.
    
    Parameters:
    -----------
    X_scaled : numpy.ndarray
        Scaled feature matrix
    results : dict
        Results dictionary from perform_kmeans()
    feature_names : list
        Names of the features
    y : pandas.Series, optional
        Original target labels for comparison
    save_plots : bool
        Whether to save plots to files
    output_dir : str
        Directory to save plots
    """
    print("\n" + "=" * 80)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create a 2x2 subplot layout
    fig = plt.figure(figsize=(16, 12))
    
    # ========== PLOT 1: Elbow Curve ==========
    ax1 = plt.subplot(2, 2, 1)
    k_range = results['k_range']
    inertias = results['inertias']
    optimal_k = results['optimal_k']
    
    ax1.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, 
                label=f'Optimal k = {optimal_k}')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12, fontweight='bold')
    ax1.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Annotate the elbow point
    elbow_idx = optimal_k - k_range.start
    ax1.annotate(f'Elbow\n({optimal_k}, {inertias[elbow_idx]:,.0f})',
                xy=(optimal_k, inertias[elbow_idx]),
                xytext=(optimal_k + 1, inertias[elbow_idx] + (max(inertias) - min(inertias)) * 0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold', color='red')
    
    print("\n✓ Plot 1: Elbow curve created")
    
    # ========== PLOT 2: Silhouette Score ==========
    ax2 = plt.subplot(2, 2, 2)
    silhouette_scores = results['silhouette_scores']
    
    ax2.plot(list(k_range), silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
                label=f'Optimal k = {optimal_k}')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Annotate the best silhouette score
    best_silhouette_idx = np.argmax(silhouette_scores)
    best_k = k_range.start + best_silhouette_idx
    ax2.annotate(f'Best\n({best_k}, {silhouette_scores[best_silhouette_idx]:.3f})',
                xy=(best_k, silhouette_scores[best_silhouette_idx]),
                xytext=(best_k + 1, silhouette_scores[best_silhouette_idx] - 0.05),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, fontweight='bold', color='green')
    
    print("✓ Plot 2: Silhouette score analysis created")
    
    # ========== PLOT 3: Cluster Scatter (PCA 2D) ==========
    ax3 = plt.subplot(2, 2, 3)
    
    # Use PCA to reduce to 2D for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    cluster_labels = results['cluster_labels']
    
    # Plot each cluster with different colors
    colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))
    for cluster_id in range(optimal_k):
        cluster_mask = cluster_labels == cluster_id
        ax3.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1],
                   c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Project centroids to PCA space and plot
    centroids_pca = pca.transform(results['centroids'])
    ax3.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               c='red', marker='X', s=300, edgecolors='black', linewidth=2,
               label='Centroids', zorder=5)
    
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                   fontsize=12, fontweight='bold')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                   fontsize=12, fontweight='bold')
    ax3.set_title(f'K-Means Clusters (k={optimal_k}) - PCA Visualization', 
                  fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)
    
    print("✓ Plot 3: Cluster scatter plot with centroids created")
    
    # ========== PLOT 4: Cluster Distribution ==========
    ax4 = plt.subplot(2, 2, 4)
    
    unique, counts = np.unique(cluster_labels, return_counts=True)
    bars = ax4.bar(unique, counts, color=colors[:optimal_k], edgecolor='black', linewidth=1.5)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({count/len(cluster_labels)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax4.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax4.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    ax4.set_xticks(unique)
    ax4.grid(True, alpha=0.3, axis='y')
    
    print("✓ Plot 4: Cluster distribution created")
    
    plt.tight_layout()
    
    if save_plots:
        filepath = os.path.join(output_dir, 'kmeans_analysis.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plots saved to: {filepath}")
    
    plt.show()
    
    # ========== ADDITIONAL PLOT: Comparison with True Labels (if available) ==========
    if y is not None:
        print("\n" + "-" * 80)
        print("BONUS: Comparing Clusters with Supervised Labels")
        print("-" * 80)
        
        fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot clusters colored by true labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        scatter1 = ax5.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, 
                              cmap='viridis', alpha=0.6, s=50, 
                              edgecolors='black', linewidth=0.5)
        ax5.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                   c='red', marker='X', s=300, edgecolors='black', linewidth=2,
                   label='K-Means Centroids', zorder=5)
        
        ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                      fontsize=12, fontweight='bold')
        ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                      fontsize=12, fontweight='bold')
        ax5.set_title('True Labels (Supervised)', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax5)
        cbar1.set_label('Price Category', fontsize=11, fontweight='bold')
        cbar1.set_ticks(range(len(label_encoder.classes_)))
        cbar1.set_ticklabels(label_encoder.classes_)
        
        # Plot clusters colored by cluster labels
        scatter2 = ax6.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                              cmap='tab10', alpha=0.6, s=50,
                              edgecolors='black', linewidth=0.5)
        ax6.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                   c='red', marker='X', s=300, edgecolors='black', linewidth=2,
                   label='Centroids', zorder=5)
        
        ax6.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                      fontsize=12, fontweight='bold')
        ax6.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                      fontsize=12, fontweight='bold')
        ax6.set_title(f'K-Means Clusters (k={optimal_k}) - Unsupervised', 
                     fontsize=14, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar2 = plt.colorbar(scatter2, ax=ax6)
        cbar2.set_label('Cluster ID', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            filepath2 = os.path.join(output_dir, 'kmeans_vs_supervised.png')
            plt.savefig(filepath2, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison plot saved to: {filepath2}")
        
        plt.show()
        
        print("\n✓ Comparison visualization created")


def compare_with_supervised(results, y):
    """
    Compare K-Means clustering results with supervised learning outcomes.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from perform_kmeans()
    y : pandas.Series
        Original target labels
    """
    print("\n" + "=" * 80)
    print("STEP 5: COMPARISON WITH SUPERVISED MODELS")
    print("=" * 80)
    
    cluster_labels = results['cluster_labels']
    
    print("\n📊 Interpretability Analysis:")
    print("-" * 80)
    
    # Analyze how clusters relate to true labels
    comparison_df = pd.DataFrame({
        'Cluster': cluster_labels,
        'True_Label': y
    })
    
    print("\nCluster composition by true price categories:")
    cross_tab = pd.crosstab(comparison_df['Cluster'], 
                            comparison_df['True_Label'], 
                            normalize='index') * 100
    
    print("\nPercentage of each price category within each cluster:")
    print(cross_tab.round(2))
    
    print("\n" + "-" * 80)
    print("🔍 Key Insights:")
    print("-" * 80)
    
    insights = []
    
    # Check cluster purity
    for cluster_id in range(results['optimal_k']):
        cluster_data = comparison_df[comparison_df['Cluster'] == cluster_id]
        dominant_label = cluster_data['True_Label'].mode()[0]
        purity = (cluster_data['True_Label'] == dominant_label).sum() / len(cluster_data) * 100
        
        insights.append(
            f"  • Cluster {cluster_id}: {purity:.1f}% are '{dominant_label}' price category"
        )
    
    for insight in insights:
        print(insight)
    
    print("\n" + "-" * 80)
    print("📈 Supervised vs Unsupervised Learning:")
    print("-" * 80)
    
    comparison_text = f"""
  Supervised Models (Decision Tree, Random Forest):
    ✓ Use labeled data (price categories) for training
    ✓ Goal: Predict the price category of new laptops
    ✓ Evaluation: Accuracy, precision, recall, F1-score
    ✓ Interpretability: Feature importance, decision rules
    
  Unsupervised Model (K-Means):
    ✓ No labeled data used - discovers patterns independently
    ✓ Goal: Group similar laptops based on feature similarity
    ✓ Evaluation: Silhouette score ({results['final_silhouette']:.4f}), inertia
    ✓ Interpretability: Cluster characteristics, centroids
    
  Key Differences:
    • K-Means found {results['optimal_k']} natural groupings in the data
    • These clusters may or may not align with price categories
    • Useful for: Market segmentation, anomaly detection, pattern discovery
    • Supervised models are better for: Direct price prediction tasks
    
  When to use K-Means:
    • Exploring data without preconceived categories
    • Finding customer/product segments for marketing
    • Discovering hidden patterns in laptop specifications
    • Preprocessing step for supervised learning (cluster-based features)
"""
    
    print(comparison_text)
    
    print("=" * 80)


def analyze_cluster_characteristics(X_scaled, results, feature_names):
    """
    Analyze and display the characteristics of each cluster.
    
    Parameters:
    -----------
    X_scaled : numpy.ndarray
        Scaled feature matrix
    results : dict
        Results dictionary from perform_kmeans()
    feature_names : list
        Names of the features
    """
    print("\n" + "=" * 80)
    print("STEP 6: CLUSTER CHARACTERISTICS ANALYSIS")
    print("=" * 80)
    
    cluster_labels = results['cluster_labels']
    centroids = results['centroids']
    
    # Convert back to DataFrame for easier analysis
    X_df = pd.DataFrame(X_scaled, columns=feature_names)
    X_df['Cluster'] = cluster_labels
    
    print("\n📋 Centroid Characteristics (Standardized Values):")
    print("-" * 80)
    
    centroids_df = pd.DataFrame(centroids, columns=feature_names)
    centroids_df.index.name = 'Cluster'
    
    print(centroids_df.round(3))
    
    print("\n" + "-" * 80)
    print("🎯 Cluster Interpretations:")
    print("-" * 80)
    
    for cluster_id in range(results['optimal_k']):
        print(f"\n  Cluster {cluster_id}:")
        
        # Find top 3 features with highest values in this cluster
        centroid_values = centroids[cluster_id]
        top_features_idx = np.argsort(centroid_values)[-3:][::-1]
        
        print(f"    Dominant characteristics:")
        for idx in top_features_idx:
            value = centroid_values[idx]
            feature = feature_names[idx]
            if value > 0.5:
                descriptor = "Very High"
            elif value > 0:
                descriptor = "High"
            elif value > -0.5:
                descriptor = "Low"
            else:
                descriptor = "Very Low"
            print(f"      • {descriptor} {feature} (z-score: {value:.2f})")
    
    print("\n" + "=" * 80)


def run_kmeans_analysis(data_path='housing.csv', k_range=range(2, 11), 
                       optimal_k=None, save_plots=False, output_dir='output'):
    """
    Main function to run complete K-Means clustering analysis.
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset
    k_range : range
        Range of cluster numbers to test
    optimal_k : int, optional
        Force a specific number of clusters
    save_plots : bool
        Whether to save plots to files
    output_dir : str
        Directory to save plots
        
    Returns:
    --------
    results : dict
        Complete results dictionary with all analysis outputs
    """
    print("\n" + "=" * 80)
    print("K-MEANS CLUSTERING ANALYSIS")
    print("=" * 80)
    print(f"Dataset: {data_path}")
    print(f"Testing k from {k_range.start} to {k_range.stop - 1}")
    print("=" * 80)
    
    # Step 1: Load and preprocess data
    X_scaled, y, df_full, feature_names = load_data(data_path)
    
    # Step 2: Perform K-Means with elbow method
    results = perform_kmeans(X_scaled, k_range, optimal_k)
    
    # Step 3: Visualize results
    visualize_results(X_scaled, results, feature_names, y, save_plots, output_dir)
    
    # Step 4: Compare with supervised learning
    compare_with_supervised(results, y)
    
    # Step 5: Analyze cluster characteristics
    analyze_cluster_characteristics(X_scaled, results, feature_names)
    
    print("\n" + "=" * 80)
    print("✅ K-MEANS ANALYSIS COMPLETE")
    print("=" * 80)
    
    # Add additional data to results for return
    results['X_scaled'] = X_scaled
    results['y'] = y
    results['df_full'] = df_full
    results['feature_names'] = feature_names
    
    return results


# Main execution
if __name__ == "__main__":
    # Run the complete analysis
    results = run_kmeans_analysis(
        data_path='housing.csv',
        k_range=range(2, 11),
        optimal_k=None,  # Auto-detect optimal k
        save_plots=True,
        output_dir='output'
    )
    
    print("\n" + "=" * 80)
    print("FINAL METRICS SUMMARY")
    print("=" * 80)
    print(f"  • Optimal k: {results['optimal_k']}")
    print(f"  • Final Silhouette Score: {results['final_silhouette']:.4f}")
    print(f"  • Final Inertia: {results['final_inertia']:,.2f}")
    print(f"  • Total samples analyzed: {len(results['cluster_labels']):,}")
    print(f"  • Number of features: {len(results['feature_names'])}")
    print("=" * 80)
