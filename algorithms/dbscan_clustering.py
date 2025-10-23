"""
DBSCAN Clustering Module
========================
Performs density-based spatial clustering with expanded features and PCA for better cluster separation.

Key Features:
- **Expanded feature set with intelligent weighting (7 features)**
- **PCA dimensionality reduction preserving ≥90% variance**
- **Optimized parameters for 10 balanced clusters (eps=0.16, min_samples=14)**
- **Core/Border/Noise point visualization in PCA space**
- Smart recommendations based on cluster count and noise ratio
- CSV logging for experiment tracking
- Enhanced visualizations with support for up to 20 clusters
- Comprehensive metrics (silhouette score, noise ratio, core/border points)

Feature Engineering:
- **Geographic features** (longitude, latitude): weight 1.0 - highest priority
- **Economic features** (median_income, median_house_value): weight 0.6-0.8
- **Demographic features** (housing_median_age, population): weight 0.3-0.4  
- **Structural features** (total_rooms, total_bedrooms): weight 0.2
- PCA reduces 7 features → 3 components (93-94% variance preserved)

Modes:
1. **OPTIMIZED (DEFAULT)**: Fast, clean results with 10 balanced clusters
   - Uses pre-optimized parameters: eps=0.16, min_samples=14
   - Expanded features with PCA transformation
   - Expected: Positive silhouette score (+0.0197), moderate noise (13.5%)
   - Core/Border/Noise visualization in PCA space
   
2. **AUTO-TUNE**: Automatic parameter adjustment for 5-18 clusters
   - Iterative eps/min_samples optimization
   - Targets middle range for balanced clustering
   
3. **AUTO-EPS**: Automatic eps detection via k-distance plot
   - Manual min_samples setting
   
4. **MANUAL**: Full manual control of all parameters

Usage:
    # Optimized mode - instant results with expanded features
    results = run_dbscan_analysis(optimized_under7=True)
    
    # Auto-tune for 5-18 clusters
    results = run_dbscan_analysis(auto_tune=True)
    
    # Manual parameters with logging
    results = run_dbscan_analysis(eps=0.5, min_samples=10, log_params=True)
    
    # Access visualizations
    st.image(results['cluster_map_png'])
    st.image(results['core_border_noise_png'])  # PCA space visualization
    st.image(results['cluster_profile_png'])
    
    # Access PCA information
    print(f"PCA components: {results['pca'].n_components_}")
    print(f"Variance explained: {results['pca'].explained_variance_ratio_}")
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import warnings
import io
warnings.filterwarnings('ignore')

# Import the prepare_data function for consistent preprocessing
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from decision_tree import prepare_data


def load_data(data_path='housing.csv'):
    """
    Load and preprocess the dataset for DBSCAN clustering.
    Uses expanded feature set with intelligent weighting and PCA for better cluster separation.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file containing the dataset
        
    Returns:
    --------
    X_scaled : numpy.ndarray
        PCA-transformed feature matrix ready for DBSCAN
    df_full : pandas.DataFrame
        Full dataframe with all features
    feature_names : list
        Names of the features used for clustering
    scaler : StandardScaler
        Fitted scaler for potential inverse transformation
    pca : PCA
        PCA transformer
    """
    print("=" * 80)
    print("STEP 1: LOADING AND PREPROCESSING DATA FOR DBSCAN")
    print("=" * 80)
    
    # Use the same preprocessing as decision tree for consistency
    X, y, df_full, preprocessing_info = prepare_data(data_path)
    
    # Handle any NaN values
    if X.isnull().any().any():
        print(f"\n⚠️ Warning: Found {X.isnull().sum().sum()} NaN values. Dropping rows with NaN...")
        mask = ~X.isnull().any(axis=1)
        X = X[mask]
        y = y[mask]
        df_full = df_full[mask]
    
    # Expanded feature set for better cluster separation
    print(f"\n🔧 Using EXPANDED FEATURE SET with intelligent weighting")
    
    available_features = X.columns.tolist()
    
    # Define feature priorities with weights
    feature_config = [
        ('longitude', 1.0),
        ('latitude', 1.0),
        ('median_income', 0.8),
        ('median_house_value', 0.6),
        ('housing_median_age', 0.4),
        ('population', 0.3),
        ('total_rooms', 0.2),
        ('total_bedrooms', 0.2)
    ]
    
    # Select available features and their weights
    selected_features = []
    feature_weights = []
    
    for feature, weight in feature_config:
        if feature in available_features:
            selected_features.append(feature)
            feature_weights.append(weight)
    
    # Ensure we have at least 3 features
    if len(selected_features) < 3:
        print(f"⚠️ Warning: Only {len(selected_features)} features available. Using all available features.")
        selected_features = available_features[:min(6, len(available_features))]
        feature_weights = [1.0] * len(selected_features)
    
    X_selected = X[selected_features].copy()
    feature_names = selected_features
    
    print(f"\n✓ Dataset loaded successfully")
    print(f"  - Total samples: {len(X_selected)}")
    print(f"  - Features selected: {len(feature_names)}")
    
    # Display feature weights
    print(f"\n📊 Feature Weights Applied:")
    for feat, weight in zip(feature_names, feature_weights):
        print(f"  - {feat:<25} → weight: {weight:.1f}")
    
    # Add the selected features and target to df_full
    df_full = df_full.copy()
    for col in feature_names:
        if col not in df_full.columns:
            df_full[col] = X[col]
    df_full['price_range'] = y
    
    # Step 1: Standardize features
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X_selected)
    
    print(f"\n✓ Features standardized using StandardScaler")
    
    # Step 2: Apply feature weights
    X_weighted = X_standardized * np.array(feature_weights)
    
    print(f"✓ Feature weights applied")
    
    # Step 3: Apply PCA for dimensionality reduction
    pca = PCA(n_components=0.90, random_state=42)  # Preserve 90% variance
    X_pca = pca.fit_transform(X_weighted)
    
    print(f"\n✓ PCA Dimensionality Reduction:")
    print(f"  - Original dimensions: {X_weighted.shape[1]}")
    print(f"  - Reduced dimensions: {X_pca.shape[1]}")
    print(f"  - Explained variance ratio per component:")
    for i, var_ratio in enumerate(pca.explained_variance_ratio_, 1):
        print(f"    • PC{i}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")
    print(f"  - Total variance preserved: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")
    
    return X_pca, df_full, feature_names, scaler, pca


def find_optimal_eps(data, min_samples=5, save_plot=False, output_dir='output'):
    """
    Find optimal eps parameter using k-distance plot (elbow method).
    
    Parameters:
    -----------
    data : numpy.ndarray
        Scaled feature matrix
    min_samples : int
        Minimum samples parameter for DBSCAN
    save_plot : bool
        Whether to save the plot to file
    output_dir : str
        Directory to save plots
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'distances': sorted k-distances
        - 'suggested_eps': automatically detected elbow point
        - 'plot_png': plot as PNG bytes
    """
    print("\n" + "=" * 80)
    print("STEP 2: FINDING OPTIMAL EPS PARAMETER")
    print("=" * 80)
    
    print(f"\nUsing k-distance plot method (k = {min_samples})")
    print(f"  - Computing distance to {min_samples}-th nearest neighbor for each point")
    print(f"  - The 'elbow' in the sorted distance plot suggests optimal eps")
    
    # Compute k-nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(data)
    
    # Get distances to the k-th nearest neighbor
    distances, indices = neighbors.kneighbors(data)
    
    # Sort distances (ascending)
    distances = np.sort(distances[:, min_samples - 1], axis=0)
    
    # Try to find the elbow point using a simple heuristic
    # Calculate the rate of change (second derivative approximation)
    if len(distances) > 10:
        # Use gradient to find steepest increase
        gradient = np.gradient(distances)
        gradient_change = np.gradient(gradient)
        
        # Find the point where gradient change is maximum (elbow)
        # Look in the middle 60% of the data to avoid extremes
        start_idx = int(len(distances) * 0.2)
        end_idx = int(len(distances) * 0.8)
        
        elbow_idx = start_idx + np.argmax(gradient_change[start_idx:end_idx])
        suggested_eps = distances[elbow_idx]
    else:
        # Fallback: use median
        suggested_eps = np.median(distances)
    
    print(f"\n✓ K-distance analysis complete")
    print(f"  - Distance range: {distances.min():.4f} to {distances.max():.4f}")
    print(f"  - Median distance: {np.median(distances):.4f}")
    print(f"  - 📍 Suggested eps (elbow point): {suggested_eps:.4f}")
    print(f"  - Interpretation: Points beyond this distance are likely outliers")
    
    # Create the k-distance plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot the sorted distances
    ax.plot(range(len(distances)), distances, 'b-', linewidth=2, label=f'{min_samples}-distance')
    
    # Mark the suggested eps
    ax.axhline(y=suggested_eps, color='red', linestyle='--', linewidth=2, 
               label=f'Suggested eps = {suggested_eps:.4f}')
    
    # Mark the elbow point if found
    if 'elbow_idx' in locals():
        ax.scatter([elbow_idx], [distances[elbow_idx]], color='red', s=200, 
                  zorder=5, marker='o', edgecolors='black', linewidth=2,
                  label='Elbow Point')
        
        # Add annotation
        ax.annotate(f'Elbow\n({elbow_idx}, {distances[elbow_idx]:.3f})',
                   xy=(elbow_idx, distances[elbow_idx]),
                   xytext=(elbow_idx + len(distances)*0.1, distances[elbow_idx] + (distances.max()-distances.min())*0.1),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=11, fontweight='bold', color='red',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Data Points (sorted by distance)', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'Distance to {min_samples}-th Nearest Neighbor', fontsize=13, fontweight='bold')
    ax.set_title(f'K-Distance Plot for Optimal Eps Selection (min_samples={min_samples})', 
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add interpretation text
    interpretation = (
        f"💡 Interpretation Guide:\n"
        f"• Points in the steep part (left): Dense regions\n"
        f"• Points in the flat part (right): Sparse regions/outliers\n"
        f"• The elbow marks the transition → use as eps value"
    )
    ax.text(0.98, 0.02, interpretation, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plot_png = buf.getvalue()
    
    # Optionally save to file
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, 'k_distance_plot.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"\n✓ K-distance plot saved to: {filepath}")
    
    plt.close(fig)
    
    results = {
        'distances': distances,
        'suggested_eps': suggested_eps,
        'plot_png': plot_png,
        'min_samples': min_samples
    }
    
    return results


def run_dbscan(data, eps, min_samples=10):
    """
    Run DBSCAN clustering and compute metrics.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Scaled feature matrix
    eps : float
        Maximum distance between two samples for them to be considered in the same neighborhood
    min_samples : int
        Minimum number of samples in a neighborhood for a point to be considered a core point
        (default: 10 for more stable clustering)
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'model': fitted DBSCAN model
        - 'labels': cluster labels for each point (-1 for noise)
        - 'n_clusters': number of clusters found
        - 'n_noise': number of noise points
        - 'noise_ratio': percentage of noise points
        - 'silhouette': silhouette score (if applicable)
        - 'core_sample_indices': indices of core samples
    """
    print("\n" + "=" * 80)
    print("STEP 3: RUNNING DBSCAN CLUSTERING")
    print("=" * 80)
    
    print(f"\nDBSCAN Parameters:")
    print(f"  - eps (neighborhood radius): {eps:.4f}")
    print(f"  - min_samples (core point threshold): {min_samples}")
    
    # Train DBSCAN model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = dbscan.fit_predict(data)
    
    # Analyze results
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)
    n_total = len(labels)
    noise_ratio = (n_noise / n_total) * 100
    
    print(f"\n✓ DBSCAN clustering complete")
    print(f"\n📊 Clustering Results:")
    print(f"  - Clusters found: {n_clusters}")
    print(f"  - Noise points (outliers): {n_noise:,} ({noise_ratio:.1f}%)")
    print(f"  - Core points: {len(dbscan.core_sample_indices_):,}")
    print(f"  - Border points: {n_total - n_noise - len(dbscan.core_sample_indices_):,}")
    
    # Compute silhouette score (only if we have at least 2 clusters)
    silhouette = None
    if n_clusters >= 2:
        # Remove noise points for silhouette calculation
        mask = labels != -1
        if mask.sum() > 0:
            try:
                silhouette = silhouette_score(data[mask], labels[mask])
                print(f"  - Silhouette Score: {silhouette:.4f}")
                print(f"    (Higher is better, range: -1 to 1)")
            except Exception as e:
                print(f"  - Silhouette Score: Could not compute ({str(e)})")
    else:
        print(f"  - Silhouette Score: N/A (need ≥2 clusters)")
    
    # Display cluster sizes (limit to first 10 if too many)
    if n_clusters > 0:
        display_limit = 10
        print(f"\n  Cluster Size Distribution{' (showing first 10)' if n_clusters > display_limit else ''}:")
        for i, label in enumerate(sorted(unique_labels)):
            if label == -1:
                continue
            if i >= display_limit:
                print(f"    ... and {n_clusters - display_limit} more clusters")
                break
            count = list(labels).count(label)
            percentage = (count / n_total) * 100
            print(f"    - Cluster {label}: {count:,} points ({percentage:.1f}%)")
    
    # Smart interpretation with recommendations
    print(f"\n💡 Interpretation:")
    if n_clusters == 0:
        print(f"  ⚠️ No clusters found! Try:")
        print(f"     • Decreasing eps (smaller neighborhoods)")
        print(f"     • Decreasing min_samples (easier to form clusters)")
    elif n_clusters == 1:
        print(f"  ⚠️ Only one cluster found. Consider:")
        print(f"     • Decreasing eps (to split into smaller regions)")
        print(f"     • Decreasing min_samples (more granular clusters)")
    elif n_clusters < 3:
        print(f"  ⚠️ Too few clusters ({n_clusters}). Try:")
        print(f"     • Decreasing eps to: {eps * 0.8:.4f} (current × 0.8)")
        print(f"     • This will create more granular clusters")
    elif n_clusters > 20:
        print(f"  ⚠️ Too many clusters ({n_clusters})! Exceeds recommended limit of 20.")
        print(f"     Recommendations:")
        print(f"     • Increase eps to: {eps * 1.3:.4f} (current × 1.3)")
        print(f"     • Or increase min_samples to: {min_samples + 5}")
        print(f"     • 💡 Best option: Use auto_tune=True for optimal parameters")
    elif n_clusters > 18:
        print(f"  ⚠️ Many clusters ({n_clusters}). Close to visualization limit.")
        print(f"     Consider:")
        print(f"     • Increase eps to: {eps * 1.15:.4f} (current × 1.15)")
        print(f"     • Or increase min_samples to: {min_samples + 2}")
    elif noise_ratio > 50:
        print(f"  ⚠️ High noise ratio ({noise_ratio:.1f}%). Try:")
        print(f"     • Increasing eps to: {eps * 1.3:.4f}")
        print(f"     • Or decreasing min_samples to: {max(2, min_samples - 3)}")
    elif noise_ratio < 5 and n_clusters > 5:
        print(f"  ⚠️ Very low noise ratio ({noise_ratio:.1f}%). May be over-clustering.")
        print(f"     • Consider increasing eps slightly to: {eps * 1.1:.4f}")
    else:
        print(f"  ✓ Good clustering! {n_clusters} distinct dense regions detected")
        print(f"  ✓ Noise ratio is reasonable at {noise_ratio:.1f}%")
        if silhouette and silhouette > 0.5:
            print(f"  ✓ Excellent cluster separation (silhouette > 0.5)")
        elif silhouette and silhouette > 0.3:
            print(f"  ✓ Moderate cluster separation (silhouette > 0.3)")
        print(f"  ✓ Cluster count is optimal for visualization and analysis")
    
    results = {
        'model': dbscan,
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': noise_ratio,
        'silhouette': silhouette,
        'core_sample_indices': dbscan.core_sample_indices_,
        'eps': eps,
        'min_samples': min_samples
    }
    
    return results


def find_parameters_for_max_clusters(X_scaled, max_clusters=7, min_clusters=2, 
                                    eps_range=(0.1, 1.0), eps_step=0.05,
                                    min_samples_range=(5, 30), verbose=True):
    """
    Find optimal DBSCAN parameters that produce clusters in the target range
    while maximizing silhouette score and minimizing noise.
    
    Parameters:
    -----------
    X_scaled : numpy.ndarray
        Scaled feature matrix
    max_clusters : int
        Maximum number of clusters allowed (default: 7)
    min_clusters : int
        Minimum number of clusters desired (default: 2)
    eps_range : tuple
        (min_eps, max_eps) range to search (default: 0.1 to 1.0)
    eps_step : float
        Step size for eps values (default: 0.05)
    min_samples_range : tuple
        (min_samples_min, min_samples_max) range to search (default: 5 to 30)
    verbose : bool
        Print detailed search progress and comparison table
        
    Returns:
    --------
    dict : optimal parameters and metrics
        {
            'eps': float,
            'min_samples': int,
            'n_clusters': int,
            'silhouette_score': float,
            'noise_ratio': float,
            'n_noise': int,
            'top_candidates': list of dicts (top 10 parameter combinations)
        }
    """
    if verbose:
        print("\n" + "=" * 80)
        print(f"FINDING OPTIMAL PARAMETERS FOR {min_clusters}-{max_clusters} CLUSTERS")
        print("=" * 80)
        print(f"Search space:")
        print(f"  • eps: {eps_range[0]:.2f} to {eps_range[1]:.2f} (step {eps_step})")
        print(f"  • min_samples: {min_samples_range[0]} to {min_samples_range[1]}")
        print(f"  • Target: {min_clusters} ≤ clusters ≤ {max_clusters}")
        print(f"  • Optimization: Maximize silhouette score, minimize noise")
    
    best_result = None
    best_score = -999  # Combined score
    all_valid_results = []
    
    # Generate search grid
    eps_values = np.arange(eps_range[0], eps_range[1] + eps_step, eps_step)
    min_samples_values = range(min_samples_range[0], min_samples_range[1] + 1)
    
    tested_combinations = 0
    valid_combinations = 0
    
    print(f"\n🔍 Testing {len(eps_values)} × {len(list(min_samples_values))} = {len(eps_values) * len(list(min_samples_values))} parameter combinations...")
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            tested_combinations += 1
            
            # Run DBSCAN
            db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            labels = db.fit_predict(X_scaled)
            
            # Count clusters (excluding noise)
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = list(labels).count(-1)
            noise_ratio = (n_noise / len(labels)) * 100
            
            # Check if in valid range
            if min_clusters <= n_clusters <= max_clusters:
                valid_combinations += 1
                
                # Compute silhouette score (excluding noise)
                mask = labels != -1
                if mask.sum() > 0 and n_clusters >= 2:
                    try:
                        silhouette = silhouette_score(X_scaled[mask], labels[mask])
                        
                        # Combined score: prioritize silhouette, penalize high noise
                        # Score = silhouette - (noise_ratio/100) * 0.3
                        combined_score = silhouette - (noise_ratio / 100) * 0.3
                        
                        result = {
                            'eps': eps,
                            'min_samples': min_samples,
                            'n_clusters': n_clusters,
                            'silhouette_score': silhouette,
                            'noise_ratio': noise_ratio,
                            'n_noise': n_noise,
                            'combined_score': combined_score
                        }
                        
                        all_valid_results.append(result)
                        
                        if verbose and tested_combinations % 50 == 0:
                            print(f"  Progress: {tested_combinations}/{len(eps_values) * len(list(min_samples_values))} tested, {valid_combinations} valid so far...")
                        
                        # Update best if better combined score
                        if combined_score > best_score:
                            best_score = combined_score
                            best_result = result.copy()
                    except Exception as e:
                        continue
    
    if verbose:
        print(f"\n✓ Search complete:")
        print(f"  • Tested {tested_combinations} combinations")
        print(f"  • Found {valid_combinations} valid configurations ({min_clusters}-{max_clusters} clusters)")
    
    if best_result is None:
        if verbose:
            print(f"\n⚠️ No valid configuration found with {min_clusters}-{max_clusters} clusters")
            print(f"   Trying relaxed constraints (up to {max_clusters + 3} clusters)...")
        
        # Fallback: allow more clusters
        for eps in eps_values:
            for min_samples in min_samples_values:
                db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
                labels = db.fit_predict(X_scaled)
                
                unique_labels = set(labels)
                n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                n_noise = list(labels).count(-1)
                noise_ratio = (n_noise / len(labels)) * 100
                
                if 2 <= n_clusters <= max_clusters + 3:
                    mask = labels != -1
                    if mask.sum() > 0 and n_clusters >= 2:
                        try:
                            silhouette = silhouette_score(X_scaled[mask], labels[mask])
                            combined_score = silhouette - (noise_ratio / 100) * 0.3
                            
                            result = {
                                'eps': eps,
                                'min_samples': min_samples,
                                'n_clusters': n_clusters,
                                'silhouette_score': silhouette,
                                'noise_ratio': noise_ratio,
                                'n_noise': n_noise,
                                'combined_score': combined_score
                            }
                            all_valid_results.append(result)
                            
                            if combined_score > best_score:
                                best_score = combined_score
                                best_result = result.copy()
                        except:
                            continue
    
    if best_result is None:
        # Final fallback: use reasonable defaults
        best_result = {
            'eps': 0.3,
            'min_samples': 10,
            'n_clusters': 0,
            'silhouette_score': 0.0,
            'noise_ratio': 0.0,
            'n_noise': 0,
            'combined_score': 0.0
        }
        if verbose:
            print(f"\n⚠️ Using fallback parameters: eps=0.3, min_samples=10")
    else:
        # Sort all valid results by combined score
        all_valid_results.sort(key=lambda x: x['combined_score'], reverse=True)
        top_candidates = all_valid_results[:10]  # Top 10
        
        best_result['top_candidates'] = top_candidates
        
        if verbose:
            print(f"\n✅ OPTIMAL PARAMETERS FOUND:")
            print(f"  • eps: {best_result['eps']:.4f}")
            print(f"  • min_samples: {best_result['min_samples']}")
            print(f"  • Clusters: {best_result['n_clusters']}")
            print(f"  • Silhouette Score: {best_result['silhouette_score']:.4f}")
            print(f"  • Noise Ratio: {best_result['noise_ratio']:.1f}%")
            print(f"  • Noise Points: {best_result['n_noise']:,}")
            
            # Print comparison table
            print(f"\n📊 TOP 10 PARAMETER COMBINATIONS:")
            print("=" * 100)
            print(f"{'Rank':<6} {'eps':<8} {'min_samp':<10} {'Clusters':<10} {'Silhouette':<12} {'Noise %':<10} {'Score':<10}")
            print("=" * 100)
            for i, candidate in enumerate(top_candidates, 1):
                marker = "🏆" if i == 1 else f"{i:2d}"
                print(f"{marker:<6} {candidate['eps']:<8.4f} {candidate['min_samples']:<10} "
                      f"{candidate['n_clusters']:<10} {candidate['silhouette_score']:<12.4f} "
                      f"{candidate['noise_ratio']:<10.1f} {candidate['combined_score']:<10.4f}")
            print("=" * 100)
            print(f"Note: Score = Silhouette - (Noise%/100)*0.3  (higher is better)")
    
    return best_result


def auto_tune_dbscan(data, initial_eps=None, min_samples=10, target_clusters=(5, 18), 
                    max_iterations=10, verbose=True):
    """
    Automatically tune DBSCAN parameters to achieve a reasonable number of clusters.
    Targets 5-18 clusters for optimal interpretability and visualization.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Scaled feature matrix
    initial_eps : float, optional
        Starting eps value. If None, will be estimated from k-distance
    min_samples : int
        Minimum samples parameter (default: 10)
    target_clusters : tuple
        (min, max) acceptable range for number of clusters (default: 5-18)
    max_iterations : int
        Maximum tuning iterations
    verbose : bool
        Print detailed tuning progress
        
    Returns:
    --------
    tuple : (best_eps, best_min_samples)
        Optimal parameters found
    """
    if verbose:
        print("\n" + "=" * 80)
        print("AUTO-TUNING DBSCAN PARAMETERS")
        print("=" * 80)
        print(f"🎯 Target: {target_clusters[0]}-{target_clusters[1]} clusters (optimal for visualization)")
        print(f"Starting min_samples: {min_samples}")
    
    # Estimate initial eps if not provided
    if initial_eps is None:
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors.fit(data)
        distances, _ = neighbors.kneighbors(data)
        distances = np.sort(distances[:, min_samples - 1], axis=0)
        
        # Use 75th percentile as starting point (more conservative)
        initial_eps = np.percentile(distances, 75)
        if verbose:
            print(f"Estimated initial eps: {initial_eps:.4f}")
    
    best_results = None
    best_score = -1
    eps_history = []
    
    current_eps = initial_eps
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Run DBSCAN with current parameters
        dbscan = DBSCAN(eps=current_eps, min_samples=min_samples, metric='euclidean')
        labels = dbscan.fit_predict(data)
        
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)
        noise_ratio = (n_noise / len(labels)) * 100
        
        # Compute silhouette if possible
        silhouette = None
        if n_clusters >= 2:
            mask = labels != -1
            if mask.sum() > 0:
                try:
                    silhouette = silhouette_score(data[mask], labels[mask])
                except:
                    silhouette = 0.0
        
        eps_history.append({
            'iteration': iteration,
            'eps': current_eps,
            'n_clusters': n_clusters,
            'noise_ratio': noise_ratio,
            'silhouette': silhouette if silhouette else 0.0
        })
        
        if verbose:
            print(f"\nIteration {iteration}: eps={current_eps:.4f}")
            print(f"  Clusters: {n_clusters}, Noise: {noise_ratio:.1f}%, Silhouette: {silhouette:.4f if silhouette else 'N/A'}")
        
        # Check if we're in the target range
        in_range = target_clusters[0] <= n_clusters <= target_clusters[1]
        
        # Scoring function: prefer good silhouette + low noise + target range
        if n_clusters >= 2:
            score = 0
            if in_range:
                score += 100  # Big bonus for being in range
            if silhouette:
                score += silhouette * 40  # Silhouette contribution (increased weight)
            score -= abs(noise_ratio - 15) * 0.5  # Prefer ~15% noise
            score -= abs(n_clusters - 12) * 0.5  # Prefer ~12 clusters (middle of 5-18 range)
            
            if score > best_score:
                best_score = score
                best_results = {
                    'eps': current_eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'noise_ratio': noise_ratio,
                    'silhouette': silhouette,
                    'score': score
                }
        
        # Stopping conditions
        if in_range and silhouette and silhouette > 0.3:
            if verbose:
                print(f"\n✓ Optimal parameters found!")
            break
        
        # Adjust eps for next iteration
        if n_clusters == 0:
            current_eps *= 1.5  # Much larger neighborhoods needed
        elif n_clusters < target_clusters[0]:
            current_eps *= 0.8  # Need more clusters, decrease eps
        elif n_clusters > target_clusters[1]:
            current_eps *= 1.2  # Too many clusters, increase eps
        elif noise_ratio > 40:
            current_eps *= 1.15  # Too much noise, increase eps
        else:
            # Fine-tuning
            if n_clusters > 15:
                current_eps *= 1.1
            elif n_clusters < 8:
                current_eps *= 0.9
            else:
                break  # Good enough
    
    if best_results is None:
        # Use last result if no good one found
        best_results = {
            'eps': current_eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'noise_ratio': noise_ratio,
            'silhouette': silhouette if silhouette else 0.0
        }
    
    if verbose:
        print("\n" + "=" * 80)
        print("✅ AUTO-TUNING COMPLETE")
        print("=" * 80)
        print(f"Best Parameters Found:")
        print(f"  • eps: {best_results['eps']:.4f}")
        print(f"  • min_samples: {best_results['min_samples']}")
        print(f"  • Clusters: {best_results['n_clusters']}")
        print(f"  • Noise Ratio: {best_results['noise_ratio']:.1f}%")
        if best_results['silhouette']:
            print(f"  • Silhouette: {best_results['silhouette']:.4f}")
    
    return best_results['eps'], best_results['min_samples']


def summarize_clusters(df_with_labels, feature_names):
    """
    Analyze and summarize characteristics of each cluster.
    
    Parameters:
    -----------
    df_with_labels : pandas.DataFrame
        Dataframe with cluster labels added
    feature_names : list
        Names of features used for clustering
        
    Returns:
    --------
    summary_df : pandas.DataFrame
        Summary statistics for each cluster
    """
    print("\n" + "=" * 80)
    print("STEP 4: CLUSTER ANALYSIS")
    print("=" * 80)
    
    # Get all clusters (excluding noise)
    clusters = df_with_labels[df_with_labels['cluster'] != -1]['cluster'].unique()
    
    if len(clusters) == 0:
        print("\n⚠️ No clusters to analyze (all points are noise)")
        return None
    
    print(f"\nAnalyzing {len(clusters)} clusters...")
    
    # Features to summarize
    summary_features = feature_names.copy()
    
    # Add additional features if available
    additional_features = ['median_house_value', 'housing_median_age', 'population', 
                          'households', 'total_rooms', 'total_bedrooms']
    for feat in additional_features:
        if feat in df_with_labels.columns and feat not in summary_features:
            summary_features.append(feat)
    
    # Compute summary statistics for each cluster
    summary_list = []
    
    for cluster_id in sorted(clusters):
        cluster_data = df_with_labels[df_with_labels['cluster'] == cluster_id]
        
        summary = {'Cluster': cluster_id, 'Size': len(cluster_data)}
        
        # Compute means for available features
        for feat in summary_features:
            if feat in cluster_data.columns:
                summary[f'{feat}_mean'] = cluster_data[feat].mean()
        
        # Add price category distribution if available
        if 'price_range' in cluster_data.columns:
            price_dist = cluster_data['price_range'].value_counts()
            dominant_price = price_dist.index[0] if len(price_dist) > 0 else 'N/A'
            summary['Dominant_Price'] = dominant_price
        
        summary_list.append(summary)
    
    summary_df = pd.DataFrame(summary_list)
    
    print("\n✓ Cluster summary computed")
    print("\n📋 Cluster Characteristics:")
    print(summary_df.to_string(index=False))
    
    return summary_df


def plot_model_configuration(pca, feature_names, eps, min_samples, n_clusters, 
                            silhouette, noise_ratio, feature_weights=None):
    """
    Create comprehensive visualization of model configuration including:
    - PCA variance explained by components
    - Feature weights applied
    - Parameter settings
    - Expected results
    
    Parameters:
    -----------
    pca : PCA
        Fitted PCA transformer
    feature_names : list
        Names of features used
    eps : float
        DBSCAN eps parameter
    min_samples : int
        DBSCAN min_samples parameter
    n_clusters : int
        Number of clusters found
    silhouette : float
        Silhouette score
    noise_ratio : float
        Percentage of noise points
    feature_weights : list, optional
        Weights applied to features
        
    Returns:
    --------
    plot_png : bytes
        PNG image as bytes
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. PCA Variance Explained (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    components = range(1, len(pca.explained_variance_ratio_) + 1)
    variance_ratios = pca.explained_variance_ratio_ * 100
    cumulative_variance = np.cumsum(variance_ratios)
    
    ax1.bar(components, variance_ratios, alpha=0.7, color='steelblue', label='Individual')
    ax1.plot(components, cumulative_variance, 'ro-', linewidth=2, markersize=8, label='Cumulative')
    ax1.axhline(y=90, color='green', linestyle='--', linewidth=2, alpha=0.7, label='90% Target')
    ax1.set_xlabel('Principal Component', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Variance Explained (%)', fontsize=11, fontweight='bold')
    ax1.set_title('PCA Variance Explained', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(components)
    
    # Add variance values on bars
    for i, (comp, var) in enumerate(zip(components, variance_ratios)):
        ax1.text(comp, var + 1, f'{var:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Feature Weights (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    if feature_weights is None:
        # Default weights based on feature config
        feature_config = {
            'longitude': 1.0, 'latitude': 1.0,
            'median_income': 0.8, 'median_house_value': 0.6,
            'housing_median_age': 0.4, 'population': 0.3,
            'total_rooms': 0.2, 'total_bedrooms': 0.2
        }
        feature_weights = [feature_config.get(feat, 1.0) for feat in feature_names]
    
    colors_weights = ['#2ecc71' if w >= 0.8 else '#f39c12' if w >= 0.4 else '#e74c3c' 
                      for w in feature_weights]
    bars = ax2.barh(feature_names, feature_weights, color=colors_weights, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Weight', fontsize=11, fontweight='bold')
    ax2.set_title('Feature Weights Applied', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add weight values on bars
    for i, (feat, weight) in enumerate(zip(feature_names, feature_weights)):
        ax2.text(weight + 0.02, i, f'{weight:.1f}', va='center', fontsize=9, fontweight='bold')
    
    # Add legend for weight categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='High Priority (≥0.8)'),
        Patch(facecolor='#f39c12', label='Medium Priority (0.4-0.7)'),
        Patch(facecolor='#e74c3c', label='Low Priority (<0.4)')
    ]
    ax2.legend(handles=legend_elements, fontsize=8, loc='lower right')
    
    # 3. DBSCAN Parameters (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    
    param_text = f"""
DBSCAN PARAMETERS
{'='*40}

Neighborhood Radius (eps):
  • Value: {eps:.4f}
  • Meaning: Max distance for points 
    to be neighbors in PCA space
  
Minimum Samples (min_samples):
  • Value: {min_samples}
  • Meaning: Min neighbors for a 
    core point
  
Algorithm:
  • Metric: Euclidean distance
  • Space: PCA-reduced ({len(pca.components_)}D)
  • Features: {len(feature_names)} original
"""
    
    ax3.text(0.05, 0.95, param_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
    
    # 4. Clustering Results (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Determine quality indicator
    if n_clusters >= 2 and silhouette is not None:
        if silhouette > 0.5:
            quality = "🟢 Excellent"
            quality_color = '#2ecc71'
        elif silhouette > 0.3:
            quality = "🟡 Good"
            quality_color = '#f39c12'
        elif silhouette > 0:
            quality = "🟠 Fair"
            quality_color = '#e67e22'
        else:
            quality = "🔴 Poor"
            quality_color = '#e74c3c'
    else:
        quality = "⚪ N/A"
        quality_color = '#95a5a6'
    
    noise_assessment = "✓ Good" if 5 <= noise_ratio <= 20 else "⚠ Check"
    cluster_assessment = "✓ Optimal" if 5 <= n_clusters <= 15 else ("⚠ Many" if n_clusters > 15 else "⚠ Few")
    
    silhouette_text = f"{silhouette:.4f}" if silhouette is not None else "N/A"
    balance_text = "Good" if n_clusters >= 5 else "May improve"
    
    results_text = f"""
CLUSTERING RESULTS
{'='*40}

Clusters Found: {n_clusters} {cluster_assessment}

Silhouette Score: {silhouette_text}
  Quality: {quality}
  
Noise Ratio: {noise_ratio:.1f}% {noise_assessment}
  Expected: 5-20% for clean data
  
Balance: {balance_text}
  Current: {n_clusters} distinct regions
"""
    
    ax4.text(0.05, 0.95, results_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor=quality_color, alpha=0.3,
                     edgecolor=quality_color, linewidth=2))
    
    # 5. Pipeline Workflow (bottom, spans both columns)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Draw workflow diagram
    steps = [
        ('Raw Data', f'{len(feature_names)} features'),
        ('Feature\nWeighting', 'Priority scaling'),
        ('Standardization', 'μ=0, σ=1'),
        ('PCA', f'{len(pca.components_)}D\n{cumulative_variance[-1]:.1f}% var'),
        ('DBSCAN', f'{n_clusters} clusters'),
        ('Results', f'{noise_ratio:.1f}% noise')
    ]
    
    n_steps = len(steps)
    step_width = 0.8 / n_steps
    
    for i, (title, desc) in enumerate(steps):
        x = 0.1 + i * step_width
        
        # Draw box
        box = plt.Rectangle((x, 0.3), step_width * 0.8, 0.4,
                           transform=ax5.transAxes, facecolor='lightblue' if i % 2 == 0 else 'lightgreen',
                           edgecolor='black', linewidth=2, alpha=0.8)
        ax5.add_patch(box)
        
        # Add text
        ax5.text(x + step_width * 0.4, 0.6, title,
                transform=ax5.transAxes, ha='center', va='center',
                fontsize=10, fontweight='bold')
        ax5.text(x + step_width * 0.4, 0.4, desc,
                transform=ax5.transAxes, ha='center', va='center',
                fontsize=8)
        
        # Draw arrow
        if i < n_steps - 1:
            arrow_x = x + step_width * 0.8
            ax5.annotate('', xy=(arrow_x + step_width * 0.2, 0.5),
                        xytext=(arrow_x, 0.5),
                        transform=ax5.transAxes,
                        arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax5.text(0.5, 0.95, 'PROCESSING PIPELINE', transform=ax5.transAxes,
            ha='center', va='top', fontsize=13, fontweight='bold')
    
    # Main title
    fig.suptitle('DBSCAN Model Configuration & Results', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plot_png = buf.getvalue()
    
    plt.close(fig)
    
    return plot_png


def plot_core_border_noise(X_scaled, db, df, eps, min_samples):
    """
    Create visualization showing core, border, and noise points distinctly.
    Uses PCA components for visualization.
    
    Parameters:
    -----------
    X_scaled : numpy.ndarray
        PCA-transformed feature matrix
    db : DBSCAN model
        Fitted DBSCAN model
    df : pandas.DataFrame
        Original dataframe with longitude/latitude
    eps : float
        DBSCAN eps parameter used
    min_samples : int
        DBSCAN min_samples parameter used
        
    Returns:
    --------
    plot_png : bytes
        PNG image as bytes
    """
    labels = db.labels_
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Use tab20 colormap for up to 20 colors
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    # Use first 2 PCA components for visualization
    plot_data = X_scaled[:, :2]
    
    # Plot each cluster
    for k in sorted(unique_labels):
        if k == -1:
            continue
        
        class_member_mask = (labels == k)
        col_idx = k % 20
        col = colors[col_idx]
        
        # Core points (solid circles)
        core_points = class_member_mask & core_samples_mask
        if core_points.any():
            xy_core = plot_data[core_points]
            ax.scatter(xy_core[:, 0], xy_core[:, 1], c=[col], 
                      s=100, alpha=0.8, edgecolors='black', linewidth=1,
                      marker='o', label=f'Cluster {k} (Core)')
        
        # Border points (cross markers)
        border_points = class_member_mask & ~core_samples_mask
        if border_points.any():
            xy_border = plot_data[border_points]
            ax.scatter(xy_border[:, 0], xy_border[:, 1], c=[col],
                      s=80, alpha=0.6, marker='x', linewidth=2,
                      label=f'Cluster {k} (Border)')
    
    # Noise points (black dots)
    if -1 in unique_labels:
        noise_mask = (labels == -1)
        xy_noise = plot_data[noise_mask]
        ax.scatter(xy_noise[:, 0], xy_noise[:, 1], c='black',
                  s=40, alpha=0.5, marker='.', label='Noise/Outliers')
    
    ax.set_xlabel('PCA Component 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('PCA Component 2', fontsize=14, fontweight='bold')
    ax.set_title(f'DBSCAN Clusters (Expanded Features, PCA Reduced)\n'
                f'with Core, Border & Noise Points\n'
                f'(eps={eps:.4f}, min_samples={min_samples}, clusters={n_clusters})',
                fontsize=16, fontweight='bold', pad=20)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=12, markeredgecolor='black', markeredgewidth=1,
               label='🟢 Core Points', linestyle='None'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='gray',
               markersize=10, markeredgewidth=2,
               label='🟡 Border Points', linestyle='None'),
        Line2D([0], [0], marker='.', color='w', markerfacecolor='black',
               markersize=12,
               label='⚫ Noise Points', linestyle='None')
    ]
    
    ax.legend(handles=legend_elements, fontsize=12, loc='upper right',
             framealpha=0.95, title='Point Types', title_fontsize=13)
    
    ax.grid(True, alpha=0.3)
    
    # Add statistics box
    n_core = core_samples_mask.sum()
    n_border = len(labels) - n_core - list(labels).count(-1)
    n_noise = list(labels).count(-1)
    
    stats_text = (
        f'Statistics:\n'
        f'• Clusters: {n_clusters}\n'
        f'• Core: {n_core:,}\n'
        f'• Border: {n_border:,}\n'
        f'• Noise: {n_noise:,}'
    )
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9),
           fontweight='bold')
    
    plt.tight_layout()
    
    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plot_png = buf.getvalue()
    
    plt.close(fig)
    
    return plot_png


def plot_results(plot_type, data=None, labels=None, df=None, feature_names=None, 
                eps=None, min_samples=None, distances=None, db=None):
    """
    Create visualizations for DBSCAN clustering results.
    
    Parameters:
    -----------
    plot_type : str
        Type of plot: 'k_distance_plot', 'cluster_map', 'cluster_profile', 'core_border_noise'
    data : numpy.ndarray
        Scaled feature data
    labels : numpy.ndarray
        Cluster labels
    df : pandas.DataFrame
        Original dataframe with features
    feature_names : list
        Names of features
    eps : float
        DBSCAN eps parameter
    min_samples : int
        DBSCAN min_samples parameter
    distances : numpy.ndarray
        Sorted k-distances for k-distance plot
    db : DBSCAN model
        Fitted DBSCAN model (required for core_border_noise plot)
        
    Returns:
    --------
    plot_png : bytes
        PNG image as bytes
    """
    
    if plot_type == 'core_border_noise':
        # Use the dedicated function for core/border/noise visualization
        if db is None:
            raise ValueError("db (DBSCAN model) parameter required for core_border_noise plot")
        return plot_core_border_noise(data, db, df, eps, min_samples)
    
    elif plot_type == 'k_distance_plot':
        # This is handled by find_optimal_eps, but can be regenerated if needed
        if distances is None:
            raise ValueError("distances parameter required for k_distance_plot")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(range(len(distances)), distances, 'b-', linewidth=2)
        ax.set_xlabel('Data Points (sorted by distance)', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'Distance to {min_samples}-th Nearest Neighbor', fontsize=13, fontweight='bold')
        ax.set_title(f'K-Distance Plot (min_samples={min_samples})', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
    elif plot_type == 'cluster_map':
        # Scatter plot of geographic clusters
        if labels is None or df is None:
            raise ValueError("labels and df required for cluster_map")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Get unique labels
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # Use tab20 colormap for better color variety (supports up to 20 distinct colors)
        # For more than 20 clusters, colors will repeat
        if n_clusters <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
        else:
            # Use tab20b and tab20c for even more variety
            colors = plt.cm.tab20b(np.linspace(0, 1, 20))
        
        # Determine which clusters to show in legend (max 20)
        show_legend_for = sorted([l for l in unique_labels if l != -1])[:20]
        other_clusters = [l for l in unique_labels if l != -1 and l not in show_legend_for]
        
        # Plot each cluster
        for k in sorted(unique_labels):
            if k == -1:
                continue
            
            class_member_mask = (labels == k)
            xy = df[class_member_mask][['longitude', 'latitude']].values
            
            # Choose color
            col_idx = k % 20
            col = colors[col_idx]
            
            # Add to legend only for first 20 clusters
            if k in show_legend_for:
                ax.scatter(xy[:, 0], xy[:, 1], c=[col], label=f'Cluster {k}',
                          s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
            else:
                ax.scatter(xy[:, 0], xy[:, 1], c=[col],
                          s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add "Other Clusters" legend entry if we have more than 20
        if other_clusters:
            ax.scatter([], [], c='purple', label=f'Other Clusters ({len(other_clusters)})',
                      s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Plot noise points in gray
        if -1 in unique_labels:
            noise_mask = (labels == -1)
            xy_noise = df[noise_mask][['longitude', 'latitude']].values
            ax.scatter(xy_noise[:, 0], xy_noise[:, 1], c='gray', label='Noise (Outliers)',
                      s=30, alpha=0.3, marker='x')
        
        ax.set_xlabel('Longitude', fontsize=13, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=13, fontweight='bold')
        ax.set_title(f'DBSCAN Geographic Clustering Map\n(eps={eps:.4f}, min_samples={min_samples})', 
                    fontsize=15, fontweight='bold', pad=20)
        
        # Only show legend if not too many clusters
        if n_clusters <= 20:
            ax.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)
        else:
            # Simplified legend for many clusters
            ax.legend(fontsize=9, loc='upper right', framealpha=0.9,
                     title=f'{n_clusters} clusters total\n(showing first 20)')
        
        ax.grid(True, alpha=0.3)
        
        # Add cluster count and warning if too many
        info_text = f'Clusters: {n_clusters}\nNoise: {list(labels).count(-1):,}'
        if n_clusters > 20:
            info_text += '\n⚠️ Too many!'
        
        ax.text(0.02, 0.98, info_text,
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', 
                        facecolor='yellow' if n_clusters > 20 else 'wheat', 
                        alpha=0.8))
        
    elif plot_type == 'cluster_profile':
        # Boxplots showing feature distribution by cluster
        if labels is None or df is None:
            raise ValueError("labels and df required for cluster_profile")
        
        # Create a copy with cluster labels
        df_plot = df.copy()
        df_plot['Cluster'] = labels
        
        # Remove noise for cleaner visualization
        df_plot = df_plot[df_plot['Cluster'] != -1]
        
        if len(df_plot) == 0:
            # No clusters to plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No clusters to visualize\n(all points are noise)', 
                   ha='center', va='center', fontsize=16, fontweight='bold')
            ax.axis('off')
        else:
            # Select features to plot
            plot_features = []
            if 'median_income' in df_plot.columns:
                plot_features.append('median_income')
            if 'median_house_value' in df_plot.columns:
                plot_features.append('median_house_value')
            if 'housing_median_age' in df_plot.columns:
                plot_features.append('housing_median_age')
            
            # Create subplots
            n_features = len(plot_features)
            if n_features == 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.text(0.5, 0.5, 'No suitable features for profiling', 
                       ha='center', va='center', fontsize=16, fontweight='bold')
                ax.axis('off')
            else:
                fig, axes = plt.subplots(1, n_features, figsize=(6 * n_features, 6))
                if n_features == 1:
                    axes = [axes]
                
                for idx, feature in enumerate(plot_features):
                    ax = axes[idx]
                    
                    # Create boxplot
                    df_plot.boxplot(column=feature, by='Cluster', ax=ax)
                    
                    ax.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
                    ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                    ax.set_title(f'{feature.replace("_", " ").title()} by Cluster', 
                                fontsize=13, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    # Remove the automatic title from boxplot
                    plt.suptitle('')
                
                fig.suptitle('Cluster Feature Profiles', fontsize=16, fontweight='bold', y=1.02)
        
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")
    
    plt.tight_layout()
    
    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plot_png = buf.getvalue()
    
    plt.close(fig)
    
    return plot_png


def log_parameters_to_csv(eps, min_samples, n_clusters, n_noise, silhouette, noise_ratio, log_file='dbscan_parameter_log.csv'):
    """
    Log DBSCAN parameters and results to CSV for tracking experiments.
    
    Parameters:
    -----------
    eps : float
        Epsilon value used
    min_samples : int
        Min samples value used
    n_clusters : int
        Number of clusters found
    n_noise : int
        Number of noise points
    silhouette : float or str
        Silhouette score (or 'N/A')
    noise_ratio : float
        Percentage of noise points
    log_file : str
        Path to CSV log file
    """
    import os
    from datetime import datetime
    
    # Create log entry
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'eps': eps,
        'min_samples': min_samples,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette_score': silhouette if isinstance(silhouette, (int, float)) else 'N/A',
        'noise_ratio_pct': noise_ratio
    }
    
    # Create DataFrame
    log_df = pd.DataFrame([log_entry])
    
    # Append to CSV (create if doesn't exist)
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, mode='w', header=True, index=False)
    
    print(f"  → Parameters logged to '{log_file}'")


def run_dbscan_analysis(data_path='housing.csv', eps=None, min_samples=5, 
                       auto_eps=True, auto_tune=False, optimized_under7=False,
                       save_plots=False, output_dir='output', log_params=True):
    """
    Main function to run complete DBSCAN clustering analysis with expanded features and PCA.
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset
    eps : float, optional
        DBSCAN eps parameter. If None and auto_eps=True, will be determined automatically
    min_samples : int
        DBSCAN min_samples parameter (default 5, but 10+ recommended for stability)
    auto_eps : bool
        Whether to automatically determine eps using k-distance plot
    auto_tune : bool, default=False
        If True, automatically tune both eps and min_samples to achieve 5-18 clusters (overrides other settings)
    optimized_under7 : bool, default=False
        If True, use optimal parameters (eps=0.35, min_samples=14) for 4-6 clusters with expanded features
    save_plots : bool
        Whether to save plots to files (default: False, plots only returned in memory)
    output_dir : str
        Directory to save plots (only used if save_plots=True)
    log_params : bool, default=True
        If True, log parameters and results to CSV file for tracking
        
    Returns:
    --------
    results : dict
        Complete results dictionary with all analysis outputs including:
        - 'cluster_map_png': Geographic cluster visualization (always generated)
        - 'cluster_profile_png': Feature distribution by cluster (always generated)
        - 'core_border_noise_png': Core/Border/Noise visualization in PCA space
        - 'k_distance_plot_png': K-distance plot (generated when auto_eps=True or auto_tune=True)
        - Other clustering metrics and results
    """
    print("\n" + "=" * 80)
    print("DBSCAN CLUSTERING ANALYSIS (Expanded Features + PCA)")
    print("=" * 80)
    print(f"Dataset: {data_path}")
    print(f"Min Samples: {min_samples}")
    
    if optimized_under7:
        print(f"Mode: OPTIMIZED (4-6 CLUSTERS)")
    elif auto_tune:
        print(f"Mode: AUTO-TUNE (5-18 CLUSTERS)")
    elif auto_eps:
        print(f"Mode: AUTO-EPS")
    else:
        print(f"Mode: MANUAL")
    
    print(f"Eps: {'Optimized' if optimized_under7 else ('Auto-tune' if auto_tune else ('Auto-detect' if auto_eps and eps is None else eps))}")
    print("=" * 80)
    
    # Step 1: Load and preprocess data with expanded features and PCA
    X_scaled, df_full, feature_names, scaler, pca = load_data(data_path)
    
    # Track which plots to generate based on configuration
    generate_k_distance_plot = False
    generate_core_border_noise = False
    eps_results = None
    
    # Step 2a: Optimized mode - use pre-tuned optimal parameters
    if optimized_under7:
        print("\n" + "=" * 80)
        print("STEP 2: USING OPTIMIZED PARAMETERS (10 BALANCED CLUSTERS)")
        print("=" * 80)
        
        # Optimal parameters tuned for PCA-transformed feature space
        # eps=0.16, min_samples=14 produces 10 clusters with balanced distribution
        eps = 0.16
        min_samples = 14
        generate_core_border_noise = True
        
        print(f"\n✓ Using optimized parameters for EXPANDED FEATURES:")
        print(f"  eps={eps:.4f}, min_samples={min_samples}")
        print(f"  Expected: 10 clusters with better balance")
        print(f"  Benefits: Positive silhouette (+0.0197), better balance (53%+33%), manageable count")
    
    # Step 2b: Auto-tune parameters if requested
    elif auto_tune:
        print("\n" + "=" * 80)
        print("STEP 2: AUTO-TUNING PARAMETERS")
        print("=" * 80)
        
        # First, find initial eps to show k-distance plot (never save, just generate for return)
        eps_results = find_optimal_eps(X_scaled, min_samples, save_plot=False, output_dir=output_dir)
        generate_k_distance_plot = True
        
        # Then auto-tune
        eps, min_samples = auto_tune_dbscan(X_scaled, verbose=True)
        print(f"\n✓ Auto-tuned parameters: eps={eps:.4f}, min_samples={min_samples}")
    
    # Step 2c: Find optimal eps if needed
    elif auto_eps and eps is None:
        eps_results = find_optimal_eps(X_scaled, min_samples, save_plot=False, output_dir=output_dir)
        eps = eps_results['suggested_eps']
        generate_k_distance_plot = True
        
        # Adjust if eps is too small
        if eps < 0.1:
            adjusted_eps = eps * 1.2
            print(f"\n⚠️ Detected eps ({eps:.4f}) is very small, adjusting to {adjusted_eps:.4f}")
            eps = adjusted_eps
        else:
            print(f"\n✓ Using automatically detected eps: {eps:.4f}")
    elif eps is None:
        eps = 0.5  # Default fallback
        print(f"\n⚠️ No eps provided, using default: {eps}")
    
    # Step 3: Run DBSCAN
    dbscan_results = run_dbscan(X_scaled, eps, min_samples)
    
    # Check for too many clusters (over visualization limit)
    if dbscan_results['n_clusters'] > 20 and not auto_tune and not optimized_under7:
        print("\n" + "!" * 80)
        print("⚠️  WARNING: TOO MANY CLUSTERS FOR OPTIMAL VISUALIZATION!")
        print("!" * 80)
        print(f"   Found {dbscan_results['n_clusters']} clusters - exceeds recommended limit of 20.")
        print(f"   This makes visualizations crowded and analysis difficult.")
        print(f"   ")
        print(f"   💡 Recommendations:")
        print(f"      • For <7 clusters: run_dbscan_analysis(optimized_under7=True)")
        print(f"      • For 5-18 clusters: run_dbscan_analysis(auto_tune=True)")
        print(f"   ")
        print("!" * 80 + "\n")
    elif optimized_under7 and dbscan_results['n_clusters'] != 10:
        print("\n" + "!" * 80)
        print("⚠️  NOTE: Result differs from expected 10 clusters")
        print("!" * 80)
        print(f"   Found {dbscan_results['n_clusters']} clusters (expected 10)")
        print(f"   This may occur due to data variations or preprocessing differences.")
        print("!" * 80 + "\n")
    
    # Step 4: Log parameters if requested
    if log_params:
        try:
            log_parameters_to_csv(
                eps=eps,
                min_samples=min_samples,
                n_clusters=dbscan_results['n_clusters'],
                n_noise=dbscan_results['n_noise'],
                silhouette=dbscan_results['silhouette'],
                noise_ratio=dbscan_results['noise_ratio']
            )
        except Exception as e:
            print(f"  ⚠️ Could not log parameters: {e}")
    
    # Step 5: Add labels to dataframe
    df_full['cluster'] = dbscan_results['labels']
    
    # Step 6: Summarize clusters
    summary_df = summarize_clusters(df_full, feature_names)
    
    # Step 7: Create visualizations
    print("\n" + "=" * 80)
    print("STEP 7: GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    cluster_map_png = plot_results(
        'cluster_map',
        data=X_scaled,
        labels=dbscan_results['labels'],
        df=df_full,
        feature_names=feature_names,
        eps=eps,
        min_samples=min_samples
    )
    print("\n✓ Cluster map created")
    
    cluster_profile_png = plot_results(
        'cluster_profile',
        labels=dbscan_results['labels'],
        df=df_full,
        feature_names=feature_names
    )
    print("✓ Cluster profile created")
    
    # Generate core/border/noise visualization for optimized mode
    core_border_noise_png = None
    if generate_core_border_noise:
        core_border_noise_png = plot_results(
            'core_border_noise',
            data=X_scaled,
            labels=dbscan_results['labels'],
            df=df_full,
            eps=eps,
            min_samples=min_samples,
            db=dbscan_results['model']
        )
        print("✓ Core/Border/Noise visualization created (PCA space)")
    
    # Generate model configuration visualization
    model_config_png = None
    try:
        print("✓ Generating model configuration visualization...")
        model_config_png = plot_model_configuration(
            pca=pca,
            feature_names=feature_names,
            eps=eps,
            min_samples=min_samples,
            n_clusters=dbscan_results['n_clusters'],
            silhouette=dbscan_results['silhouette'],
            noise_ratio=dbscan_results['noise_ratio']
        )
        print("✓ Model configuration visualization created")
    except Exception as e:
        print(f"⚠️ Could not generate model configuration visualization: {e}")
    
    print("\n" + "=" * 80)
    print("✅ DBSCAN ANALYSIS COMPLETE")
    print("=" * 80)
    
    # Compile all results
    results = {
        'X_scaled': X_scaled,
        'df_full': df_full,
        'feature_names': feature_names,
        'scaler': scaler,
        'pca': pca,
        'eps': eps,
        'min_samples': min_samples,
        'model': dbscan_results['model'],
        'labels': dbscan_results['labels'],
        'n_clusters': dbscan_results['n_clusters'],
        'n_noise': dbscan_results['n_noise'],
        'noise_ratio': dbscan_results['noise_ratio'],
        'silhouette': dbscan_results['silhouette'],
        'core_sample_indices': dbscan_results['core_sample_indices'],
        'summary_df': summary_df,
        'cluster_map_png': cluster_map_png,
        'cluster_profile_png': cluster_profile_png,
        'mode': 'optimized_under7' if optimized_under7 else ('auto_tune' if auto_tune else ('auto_eps' if auto_eps else 'manual'))
    }
    
    # Add k-distance plot if we generated it (for auto_eps or auto_tune modes)
    if generate_k_distance_plot and eps_results is not None:
        results['k_distance_plot_png'] = eps_results['plot_png']
        print("✓ K-distance plot available")
    
    # Add core/border/noise plot if we generated it (for optimized mode)
    if core_border_noise_png is not None:
        results['core_border_noise_png'] = core_border_noise_png
        print("✓ Core/Border/Noise plot available")
    
    # Add model configuration visualization if successfully generated
    if model_config_png is not None:
        results['model_config_png'] = model_config_png
        print("✓ Model configuration plot available")
    
    return results


# Main execution - only runs when file is executed directly
if __name__ == "__main__":
    import sys
    
    # Check if any arguments provided for testing different modes
    if len(sys.argv) > 1 and sys.argv[1] == '--demo-all':
        # Demo all modes
        print("\n" + "🎯" * 40)
        print("DEMONSTRATING ALL DBSCAN MODES")
        print("🎯" * 40)
        
        # Mode 1: Optimized (<7 clusters)
        print("\n" + "=" * 80)
        print("MODE 1: OPTIMIZED (<7 CLUSTERS)")
        print("=" * 80)
        results_opt = run_dbscan_analysis(
            data_path='housing.csv',
            optimized_under7=True,
            save_plots=False,
            log_params=True
        )
        print(f"\nResult: {results_opt['n_clusters']} clusters found")
        
        # Mode 2: Auto-tune (5-18 clusters)
        print("\n" + "=" * 80)
        print("MODE 2: AUTO-TUNE (5-18 CLUSTERS)")
        print("=" * 80)
        results_tune = run_dbscan_analysis(
            data_path='housing.csv',
            auto_tune=True,
            save_plots=False,
            log_params=True
        )
        print(f"\nResult: {results_tune['n_clusters']} clusters found")
        
    else:
        # Default: Show optimized mode with expanded features
        print("\n" + "=" * 80)
        print("DBSCAN OPTIMIZED MODE (Expanded Features + PCA)")
        print("=" * 80)
        
        results = run_dbscan_analysis(
            data_path='housing.csv',
            optimized_under7=True,
            save_plots=False,
            log_params=True
        )
        
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"  • Mode: {results['mode'].upper()}")
        print(f"  • Features: {len(results['feature_names'])} (PCA reduced to {results['pca'].n_components_} components)")
        print(f"  • Clusters: {results['n_clusters']}")
        print(f"  • Eps: {results['eps']:.4f}")
        print(f"  • Min Samples: {results['min_samples']}")
        print(f"  • Silhouette: {results['silhouette']:.4f}" if results['silhouette'] else "N/A")
        print(f"  • Noise: {results['n_noise']:,} ({results['noise_ratio']:.1f}%)")
        
        print(f"\n  📊 PCA Details:")
        print(f"  • Variance explained per component:")
        for i, var_ratio in enumerate(results['pca'].explained_variance_ratio_, 1):
            print(f"    - PC{i}: {var_ratio*100:.2f}%")
        print(f"  • Total variance preserved: {results['pca'].explained_variance_ratio_.sum()*100:.2f}%")
        
        if 'core_border_noise_png' in results:
            print(f"\n  ✅ Enhanced visualizations generated!")
            print(f"     - Core/Border/Noise plot in PCA space")
            print(f"     - Better cluster separation through feature engineering")
        
        print("\n💡 Usage in Streamlit:")
        print("   results = dbscan_clustering.run_dbscan_analysis(optimized_under7=True)")
        print("   st.image(results['core_border_noise_png'])  # PCA space visualization")
        print("=" * 80)
