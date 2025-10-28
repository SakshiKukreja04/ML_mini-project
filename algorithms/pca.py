# Integrated PCA logic from user-provided Colab code

# Streamlit-native PCA workflow
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

def run_pca_streamlit():
	# Use compute_pca_results to get figures + metrics, then streamlit-display
	st.header("PCA Analysis for Housing Data")
	results = compute_pca_results()
	if results is None:
		return
	fig1 = results['figs'][0]
	fig2 = results['figs'][1]
	components_for_95 = results['components_for_95']
	variance_for_2 = results['variance_for_2']

	st.subheader("Explained Variance by Principal Component")
	st.pyplot(fig1)
	st.metric("Components for 95% variance", components_for_95)
	st.metric("Variance explained by first 2 components", f"{variance_for_2:.2%}")

	st.subheader("PCA Visualization for Housing Data")
	st.pyplot(fig2)
	st.success("PCA analysis complete!")


def compute_pca_results(data_path: str = 'housing.csv', n_components_viz: int = 2, scale: bool = True):
	"""Compute PCA figures and metrics without Streamlit calls.
	Returns a dict with keys: figs (list of matplotlib.Figure), components_for_95, variance_for_2,
	plus artifacts: scaler (or None), feature_names (list), pca_model (2-component PCA), pca_full (full PCA), X_pca (numpy array)
	"""
	# Try reading the provided path; if not found, try resolving relative to the
	# current working directory (project root). Return None if file still not found.
	try:
		data = pd.read_csv(data_path)
	except FileNotFoundError:
		alt_path = os.path.join(os.getcwd(), data_path)
		if alt_path != data_path:
			try:
				data = pd.read_csv(alt_path)
			except FileNotFoundError:
				return None
		else:
			return None

	X = data.drop('median_house_value', axis=1)
	y = data['median_house_value']

	imputer = SimpleImputer(strategy='median')
	X_imputed = X.copy()
	# make sure column exists
	if 'total_bedrooms' in X_imputed.columns:
		X_imputed['total_bedrooms'] = imputer.fit_transform(X[['total_bedrooms']])

	X_processed = pd.get_dummies(X_imputed, columns=['ocean_proximity'], drop_first=True)
	feature_names = list(X_processed.columns)

	if scale:
		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X_processed)
	else:
		scaler = None
		X_scaled = X_processed.values.astype(float)

	# Analysis PCA (all components)
	pca_analysis = PCA()
	pca_analysis.fit(X_scaled)
	explained_variance = pca_analysis.explained_variance_ratio_
	cumulative_variance = np.cumsum(explained_variance)
	n_components = len(explained_variance)

	fig1, ax1 = plt.subplots(figsize=(10, 6))
	sns.barplot(
		x=list(range(1, n_components + 1)), y=explained_variance,
		alpha=0.8, color='g', ax=ax1, label='Individual Explained Variance'
	)
	ax1.set_xlabel('Principal Component')
	ax1.set_ylabel('Explained Variance Ratio', color='g')
	ax1.tick_params(axis='y', labelcolor='g')
	ax2 = ax1.twinx()
	ax2.plot(
		range(0, n_components), cumulative_variance,
		marker='o', linestyle='--', color='b', label='Cumulative Explained Variance'
	)
	ax2.set_ylabel('Cumulative Explained Variance', color='b')
	ax2.tick_params(axis='y', labelcolor='b')
	ax2.set_ylim([0, 1.05])
	plt.title('Explained Variance by Principal Component (Corrected)', fontsize=16)
	fig1.tight_layout()

	components_for_95 = np.argmax(cumulative_variance >= 0.95) + 1
	variance_for_2 = cumulative_variance[1] if len(cumulative_variance) >= 2 else float(cumulative_variance[-1])

	# PCA for visualization (n_components_viz components)
	pca = PCA(n_components=n_components_viz)
	X_pca = pca.fit_transform(X_scaled)
	pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components_viz)])
	# keep median_house_value for coloring if present
	if 'median_house_value' in data.columns and n_components_viz >= 2:
		pca_df['median_house_value'] = y.values

	# Create plot for first two components if available
	fig2 = plt.figure(figsize=(10, 8))
	if n_components_viz >= 2:
		plt.scatter(
			pca_df['PC1'],
			pca_df['PC2'],
			c=pca_df.get('median_house_value', None),
			cmap='jet',
			alpha=0.5
		)
		if 'median_house_value' in pca_df.columns:
			plt.colorbar(label='Median House Value')
		plt.axhline(0, color='black', linewidth=1.2)
		plt.axvline(0, color='black', linewidth=1.2)
		x_min, x_max = plt.xlim()
		y_min, y_max = plt.ylim()
		max_lim = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
		plt.xlim(-max_lim, max_lim)
		plt.ylim(-max_lim, max_lim)
		plt.plot([-max_lim, max_lim], [-max_lim, max_lim], color='black', linewidth=1, linestyle='--')
		plt.plot([-max_lim, max_lim], [max_lim, -max_lim], color='black', linewidth=1, linestyle='--')
		plt.text(max_lim * 0.9, max_lim * 0.05, 'PC1', fontsize=12, color='black', fontweight='bold')
		plt.text(max_lim * 0.05, max_lim * 0.9, 'PC2', fontsize=12, color='black', fontweight='bold')
		plt.xlabel('Principal Component 1')
		plt.ylabel('Principal Component 2')
		plt.title('PCA Visualization for Housing Data', fontsize=16)
		plt.grid(True)
	else:
		# simple line plot for single component
		plt.plot(pca_df[f'PC1'])
		plt.title('PCA (1 component)')

	return {
		'figs': [fig1, fig2],
		'components_for_95': int(components_for_95),
		'variance_for_2': float(variance_for_2),
		'scaler': scaler,
		'feature_names': feature_names,
		'pca_model': pca,
		'pca_full': pca_analysis,
		'X_pca': X_pca,
		'pca_df': pca_df
	}

if __name__ == "__main__":
	run_pca_streamlit()
