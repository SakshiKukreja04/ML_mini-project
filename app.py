import streamlit as st
from typing import List
import os
import numpy as np

# Attempt to import algorithm modules; provide a user-friendly message if deps are missing
try:
    from algorithms import linear_regression
    _LR_AVAILABLE = True
except Exception as e:  # ImportError or missing dependency inside the module
    linear_regression = None
    _LR_AVAILABLE = False
    _LR_IMPORT_ERROR = e

try:
    from algorithms import decision_tree
    _DT_AVAILABLE = True
except Exception as e:
    decision_tree = None
    _DT_AVAILABLE = False
    _DT_IMPORT_ERROR = e
try:
    from algorithms import multivariate_nonlinear
    _MVNL_AVAILABLE = True
except Exception as e:
    multivariate_nonlinear = None
    _MVNL_AVAILABLE = False
    _MVNL_IMPORT_ERROR = e

try:
    from algorithms import support_vector_machine
    _SVM_AVAILABLE = True
except Exception as e:
    support_vector_machine = None
    _SVM_AVAILABLE = False
    _SVM_IMPORT_ERROR = e

try:
    from algorithms.ensemble_learning import (
        find_best_params as find_best_ensemble_params,
        train_and_evaluate as train_and_evaluate_ensemble,
        plot_results as plot_results_ensemble
    )
    _ENSEMBLE_AVAILABLE = True
except Exception as e:
    _ENSEMBLE_AVAILABLE = False
    _ENSEMBLE_IMPORT_ERROR = e

try:
    from algorithms import kmeans
    _KMEANS_AVAILABLE = True
except Exception as e:
    kmeans = None
    _KMEANS_AVAILABLE = False
    _KMEANS_IMPORT_ERROR = e

try:
    from algorithms import dbscan_clustering
    _DBSCAN_AVAILABLE = True
except Exception as e:
    dbscan_clustering = None
    _DBSCAN_AVAILABLE = False
    _DBSCAN_IMPORT_ERROR = e

# -----------------------------
# Header / Page Config
# -----------------------------
st.set_page_config(page_title="ML Dashboard 🧠", page_icon="🧠", layout="wide")


@st.cache_data
def load_data_placeholder() -> None:
    """Placeholder for future data-loading logic.

    Returns None for now. Kept to demonstrate caching decorator.
    """
    return None


def sidebar_controls(algorithms: List[str]):
    """Renders the sidebar with logo/title, algorithm selector and dataset options."""
    with st.sidebar:
        st.markdown("# ML Mini-Project 🧠")
        st.markdown("---")

        # Use a centered emoji as a lightweight app logo placeholder
        st.markdown("<div style='text-align:center; font-size:48px;'>🏠</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; font-weight:600;'>House Price Predictor</div>", unsafe_allow_html=True)

        algo = st.selectbox("Select algorithm 🔍", options=algorithms)

        st.markdown(
            "Using bundled sample dataset. To use a different dataset, replace 'housing.csv' in the project root.\n\n"
        )

        # No upload option — app uses the bundled sample dataset by default
        dataset_mode = "Use sample dataset"
        uploaded_file = None

        st.markdown("---")
        st.markdown("Built with Streamlit 🧠")
    return algo, dataset_mode, uploaded_file


def main():
    # Available algorithms
    algorithms = [
        "Linear Regression",
        "Multivariate Linear Regression",
        "Decision Tree Classifier",
        "Support Vector Machine",
        "Ensemble Learning (Bagging/Boosting)",
        "K-Means Clustering",
        "DBSCAN Clustering",
    "PCA",
    ]

    # Sidebar
    selected_algo, dataset_mode, uploaded_file = sidebar_controls(algorithms)

    # Visualization Gallery toggle: show a full-dashboard view if requested
    st_session = st.session_state
    if "show_gallery" not in st_session:
        st_session["show_gallery"] = False

    # Add a checkbox in the sidebar to toggle the Visualization Gallery
    # (we add it here to ensure it's rendered after the other sidebar elements)
    with st.sidebar:
        show_gallery = st.checkbox("full dashboard", value=st_session["show_gallery"])
        st_session["show_gallery"] = show_gallery

    # Header area
    st.title("Machine Learning Dashboard 📊")
    st.markdown(f"### You selected: **{selected_algo}**")

    st.markdown("Use the tabs to configure and run models, or use the Overview quick actions to run sample workflows.")

    # Universal top-level Run Model button (runs the currently selected algorithm)
    # Hide the Run Model button when the full dashboard (visualization gallery) is enabled
    if not show_gallery:
        if st.button("Run Model"):
            # Provide immediate feedback
            st.info(f"Running {selected_algo}...")
            try:
                if selected_algo == "Linear Regression" and _LR_AVAILABLE:
                    data_path = os.path.join(os.getcwd(), "housing.csv")
                    metrics, artifacts = linear_regression.train_and_evaluate(data_path)
                    st.success("Linear Regression training complete")
                    st.session_state["latest_metrics"] = metrics
                    st.session_state["latest_plot_lr"] = linear_regression.plot_results(artifacts)
                    st.session_state["latest_artifacts"] = artifacts

                elif selected_algo == "Decision Tree Classifier" and _DT_AVAILABLE:
                    data_path = os.path.join(os.getcwd(), "housing.csv")
                    dt_metrics, dt_artifacts = decision_tree.train_and_evaluate(data_path, max_depth=5)
                    st.success("Decision Tree training complete")
                    st.session_state["dt_metrics"] = dt_metrics
                    st.session_state["dt_artifacts"] = dt_artifacts
                    st.session_state["dt_confusion_png"] = decision_tree.plot_confusion_matrix(dt_artifacts)
                    st.session_state["dt_tree_png"] = decision_tree.plot_tree_image(dt_artifacts)

                elif selected_algo == "Multivariate Linear Regression" and _MVNL_AVAILABLE:
                    data_path = os.path.join(os.getcwd(), "housing.csv")
                    m_metrics, m_artifacts = multivariate_nonlinear.train_and_evaluate(data_path)
                    st.success("Multivariate model training complete")
                    st.session_state['mvnl_metrics'] = m_metrics
                    st.session_state['mvnl_artifacts'] = m_artifacts
                    st.session_state['mvnl_plot'] = m_artifacts.get('plot_png')
                
                elif selected_algo == "Support Vector Machine":
                    st.info("Please go to the 'Model Configuration ⚙️' tab to run the SVM (SVC) model.")
                    st.session_state['last_run_algo'] = "Support Vector Machine"
                
                elif selected_algo == "Ensemble Learning (Bagging/Boosting)":
                    data_path = os.path.join(os.getcwd(), "housing.csv")
                    
                    # Check if we have cached best parameters
                    if 'rf_best_params' not in st.session_state:
                        st.info("🔍 Running GridSearchCV to find best parameters (first time only)...")
                        with st.spinner("Finding best parameters... This may take a few minutes."):
                            param_results = find_best_ensemble_params(data_path, random_state=42)
                            st.session_state['rf_best_params'] = param_results['best_params']
                            st.session_state['rf_best_score'] = param_results['best_score']
                            st.success(f"✅ Found best parameters! CV Score: {param_results['best_score']:.2%}")
                    
                    # Now train with the best parameters
                    best_params = st.session_state['rf_best_params']
                    with st.spinner("Training Ensemble models..."):
                        metrics, artifacts = train_and_evaluate_ensemble(
                            data_path,
                            rf_params=best_params,
                            base_tree_max_depth=10,  # Default depth
                            random_state=42
                        )
                        
                        if "error" not in metrics['single_tree']:
                            st.session_state['ensemble_metrics'] = metrics
                            st.session_state['ensemble_artifacts'] = artifacts
                            st.session_state['last_run_algo'] = "Ensemble Learning (Bagging/Boosting)"
                            st.success("✅ Models trained successfully!")
                        else:
                            st.error(f"Error: {metrics['single_tree']['error']}")
                
                elif selected_algo == "K-Means Clustering":
                    if not _KMEANS_AVAILABLE:
                        st.warning(f"K-Means module not available: {_KMEANS_IMPORT_ERROR}")
                    else:
                        data_path = os.path.join(os.getcwd(), "housing.csv")
                        with st.spinner("Running K-Means clustering analysis..."):
                            try:
                                # Run K-Means analysis with default parameters
                                results = kmeans.run_kmeans_analysis(
                                    data_path=data_path,
                                    k_range=range(2, 11),
                                    optimal_k=None,  # Auto-detect
                                    save_plots=False,
                                    output_dir='output'
                                )
                                st.session_state['kmeans_results'] = results
                                st.session_state['last_run_algo'] = "K-Means Clustering"
                                st.success("✅ K-Means clustering analysis complete!")
                            except Exception as e:
                                st.error(f"K-Means clustering failed: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                
                elif selected_algo == "DBSCAN Clustering":
                    if not _DBSCAN_AVAILABLE:
                        st.warning(f"DBSCAN module not available: {_DBSCAN_IMPORT_ERROR}")
                    else:
                        data_path = os.path.join(os.getcwd(), "housing.csv")
                        with st.spinner("Running DBSCAN clustering analysis (optimized parameters)..."):
                            try:
                                # Run DBSCAN analysis with pre-optimized parameters
                                results = dbscan_clustering.run_dbscan_analysis(
                                    data_path=data_path,
                                    optimized_under7=True,  # Use pre-optimized parameters (eps=0.16, min_samples=14)
                                    save_plots=False,
                                    log_params=True
                                )
                                st.session_state['dbscan_results'] = results
                                st.session_state['last_run_algo'] = "DBSCAN Clustering"
                                st.success(f"✅ DBSCAN clustering complete! Found {results['n_clusters']} clusters (eps={results['eps']:.2f}, min_samples={results['min_samples']}) with silhouette score: {results['silhouette']:.4f}")
                            except Exception as e:
                                st.error(f"DBSCAN clustering failed: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                
                elif selected_algo == "PCA":
                    # Run PCA immediately and show a preview and metrics in the header area.
                    st.info(f"Running {selected_algo}...")
                    with st.spinner("Running PCA workflow..."):
                        try:
                            # import locally to avoid import-time Streamlit calls
                            from algorithms.pca import compute_pca_results
                            data_path = os.path.join(os.getcwd(), "housing.csv")
                            results = compute_pca_results(data_path=data_path)
                            if results is None:
                                st.error("housing.csv not found in project root or PCA failed.")
                            else:
                                # Store metrics for Overview / Metrics panel
                                st.session_state['pca_results'] = {
                                    'components_for_95': results.get('components_for_95'),
                                    'variance_for_2': results.get('variance_for_2')
                                }
                                # Store returned matplotlib figures so Overview can render them
                                st.session_state['pca_figs'] = results.get('figs', [])
                                st.session_state['pca_ready'] = True

                                # Show a compact preview of returned figures in the header area
                                figs = results.get('figs', [])
                                try:
                                    import io

                                    if len(figs) >= 2:
                                        c1, c2 = st.columns(2)
                                        buf = io.BytesIO()
                                        figs[0].savefig(buf, format='png', dpi=120, bbox_inches='tight')
                                        buf.seek(0)
                                        c1.image(buf, use_column_width=True)

                                        buf2 = io.BytesIO()
                                        figs[1].savefig(buf2, format='png', dpi=120, bbox_inches='tight')
                                        buf2.seek(0)
                                        c2.image(buf2, use_column_width=True)
                                    elif len(figs) == 1:
                                        buf = io.BytesIO()
                                        figs[0].savefig(buf, format='png', dpi=120, bbox_inches='tight')
                                        buf.seek(0)
                                        st.image(buf, use_column_width=True)
                                    else:
                                        st.info("No figures returned from PCA.")
                                except Exception:
                                    # Fallback: use st.pyplot if image conversion fails
                                    for f in figs:
                                        st.pyplot(f)

                                st.success("PCA analysis complete — preview shown above. Full visuals available in Overview tab.")
                        except Exception as e:
                            st.error(f"PCA run failed: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    st.warning(f"Selected algorithm not available or has missing dependencies: {selected_algo}")
            except FileNotFoundError:
                st.error("housing.csv not found in project root.")
            except Exception as e:
                st.error(f"Training failed: {e}")

    # If the gallery toggle is enabled, show a grid of visualization placeholders
    # Quick actions for Decision Tree
    if selected_algo == "Decision Tree Classifier":
        with st.expander("Quick actions — Decision Tree", expanded=False):
            st.write("Run a quick Decision Tree classifier and view basic metrics.")
            if not _DT_AVAILABLE:
                st.warning("Decision Tree module not available (missing dependencies).")
            else:
                max_depth_quick = st.slider("Max depth (quick)", min_value=1, max_value=30, value=5)
                if st.button("Run Decision Tree (quick)"):
                    st.info("Training Decision Tree Classifier...")
                    try:
                        dt_metrics, dt_artifacts = decision_tree.train_and_evaluate(
                            os.path.join(os.getcwd(), "housing.csv"), max_depth=max_depth_quick
                        )
                        st.success("Decision Tree training complete")

                        # Store for Overview visualizations
                        st.session_state["dt_metrics"] = dt_metrics
                        st.session_state["dt_artifacts"] = dt_artifacts
                        st.session_state["dt_confusion_png"] = decision_tree.plot_confusion_matrix(dt_artifacts)
                        st.session_state["dt_tree_png"] = decision_tree.plot_tree_image(dt_artifacts)

                        # Nicely formatted output: metrics table and preprocessing summary
                        st.subheader("Training summary")
                        cols1, cols2 = st.columns([2, 1])
                        with cols1:
                            st.markdown("**Model metrics**")
                            import pandas as _pd

                            metrics_df = _pd.DataFrame.from_dict(dt_metrics, orient="index", columns=["value"]) \
                                .reset_index().rename(columns={"index": "metric"})
                            # format percentages
                            metrics_df["value"] = metrics_df["value"].apply(lambda x: round(x, 4) if isinstance(x, float) else x)
                            st.table(metrics_df)

                        with cols2:
                            st.markdown("**Data / Preprocessing**")
                            st.write(f"Train records: {dt_artifacts.get('train_count')}")
                            st.write(f"Test records: {dt_artifacts.get('test_count')}")
                            prep = dt_artifacts.get("preprocessing", {})
                            st.write(f"Rows before: {prep.get('rows_before')}")
                            st.write(f"Rows after: {prep.get('rows_after')}")

                        # Suggestions removed per user request
                    except FileNotFoundError:
                        st.error("housing.csv not found in project root.")
    if show_gallery:
        st.header("House Price Prediction Dashboard — Overview 🏠📈")

        # Project Info Card
        with st.container():
            st.subheader("📊 California Housing Price Analysis")
            col_title1, col_title2 = st.columns([3,1])
            with col_title1:
                st.markdown("""
                **Project Goal:** Develop an AI-powered system to analyze California housing market dynamics 
                and predict price patterns using multiple machine learning approaches.
                """)
            with col_title2:
                st.markdown("""
                **Dataset**: 1990 California Housing
                - 20,640 entries
                - 10 features
                """)
        
        # Problem & Solution Cards
        col_left, col_right = st.columns([1,1])
        with col_left:
            st.markdown("""
            ### 🎯 Problem Statement
            
            California's housing market faces several challenges:
            
            1. **Complex Price Drivers**
               - Location-based variations
               - Economic indicators
               - Demographic patterns
            
            2. **Market Analysis Gaps**
               - Non-linear relationships
               - Regional clustering needs
               - Price anomaly detection
            
            3. **Decision Support**
               - Affordability assessment
               - Investment guidance
               - Policy planning
            """)
        
        with col_right:
            st.markdown("""
            ### 💡 Our Solution
            
            Multi-model machine learning approach:
            
            1. **Price Prediction**
               - Decision Trees (72% acc)
               - SVM Classification (84% acc)
               - Random Forest (80% acc)
            
            2. **Market Segmentation**
               - K-Means (4 segments)
               - DBSCAN (anomaly detection)
            
            3. **Feature Analysis**
               - PCA dimensionality reduction
               - Feature importance ranking
            """)

        # Key Metrics Dashboard
        st.markdown("---")
        st.markdown("### 📈 Model Performance Dashboard")
        
        # Classification Models
        st.markdown("#### Classification Performance")
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric(
                "Support Vector Machine",
                "84%",
                "+12%",
                help="Best performing model - RBF kernel with optimized parameters"
            )
        with metric_cols[1]:
            st.metric(
                "Random Forest",
                "80%",
                "+8%",
                help="Ensemble method with feature importance insights"
            )
        with metric_cols[2]:
            st.metric(
                "Decision Tree",
                "72%",
                "baseline",
                help="Interpretable baseline model"
            )
        with metric_cols[3]:
            st.metric(
                "Cross-Val Score",
                "0.82",
                help="Average cross-validation score across models"
            )

        # Clustering Insights
        st.markdown("#### Clustering Analysis")
        cluster_cols = st.columns(4)
        
        with cluster_cols[0]:
            st.metric(
                "K-Means Segments",
                "4",
                help="Optimal number of market segments"
            )
        with cluster_cols[1]:
            st.metric(
                "K-Means Silhouette",
                "0.52",
                help="Cluster separation quality (0-1 scale)"
            )
        with cluster_cols[2]:
            st.metric(
                "DBSCAN Clusters",
                "10",
                help="Automatically detected density-based clusters"
            )
        with cluster_cols[3]:
            st.metric(
                "DBSCAN Silhouette",
                "0.60",
                help="Density-based cluster quality"
            )

        # Key Findings Section
        st.markdown("---")
        st.markdown("### 🔍 Key Findings & Impact")
        
        finding_cols = st.columns(3)
        
        with finding_cols[0]:
            st.markdown("""
            #### 📈 Market Insights
            
            - **4 distinct** market segments identified
            - **15% premium** for coastal properties
            - Income correlates strongly (0.68) with price
            - Geographic clusters show clear patterns
            """)
            
        with finding_cols[1]:
            st.markdown("""
            #### 🎯 Model Performance
            
            - SVM achieves **84% accuracy**
            - Random Forest provides reliable **80% accuracy**
            - DBSCAN detected **10 micro-markets**
            - PCA explains **85% variance** with 3 components
            """)
            
        with finding_cols[2]:
            st.markdown("""
            #### 💡 Applications
            
            - Automated market segmentation
            - Price anomaly detection
            - Investment opportunity spotting
            - Policy impact assessment
            """)

        # Implementation Details
        st.markdown("---")
        st.markdown("### 🛠️ Technical Implementation")
        
        impl_cols = st.columns([2,3])
        
        with impl_cols[0]:
            st.markdown("""
            #### Architecture
            
            1. **Data Pipeline**
               - Automated preprocessing
               - Feature engineering
               - Cross-validation
            
            2. **Model Integration**
               - Ensemble methods
               - Hyperparameter optimization
               - Real-time prediction
            """)
            
        with impl_cols[1]:
            st.markdown("""
            #### Key Features
            
            - **Interactive Dashboard**: Real-time model comparison and visualization
            - **Multiple Algorithms**: Classification, clustering, and dimensionality reduction
            - **Automated Insights**: Key metric tracking and anomaly detection
            - **Prediction API**: Single and batch prediction capabilities
            - **Custom Visualization**: Geographic clustering and price distribution maps
            """)

        st.markdown("---")
        st.info("Gallery mode: use the sidebar toggle to switch back to the tabbed view for model configuration, training, and prediction.")
    else:
        # Tabs for main content
        tabs = st.tabs(["Overview / Visualization 📈", "Model Configuration ⚙️", "Prediction 🔮"])

        # Overview / Visualization Tab
        with tabs[0]:
            st.header("Overview & Visualization")

            with st.expander("Dataset summary (expand)", expanded=True):
                st.write("Dataset mode:", dataset_mode)
                if uploaded_file:
                    st.write("Uploaded file:", uploaded_file.name)
                    try:
                        df_preview = None
                        uploaded_file.seek(0)
                        df_preview = __import__('pandas').read_csv(uploaded_file)
                        st.dataframe(df_preview.head(5))
                    except Exception:
                        st.write("Could not preview uploaded file.")
                else:
                    # attempt to preview bundled housing.csv
                    default_path = os.path.join(os.getcwd(), "housing.csv")
                    if os.path.exists(default_path):
                        try:
                            import pandas as _pd

                            df_preview = _pd.read_csv(default_path)
                            st.dataframe(df_preview.head(5))
                        except Exception:
                            st.write("Could not read housing.csv for preview.")
                    else:
                        st.write("No dataset preview available. Upload a CSV or add 'housing.csv' to project root.")

            st.markdown("### Visualizations")
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("#### Visualization")
                # Quick-run button at the top (single entry point).
                if selected_algo == "Multivariate Linear Regression" and _MVNL_AVAILABLE:
                    if st.button("Run Multivariate (quick)", key="overview_mvnl_quick"):
                        # Use session state flags to ensure messages render in the
                        # placeholder below the button and above the visualization.
                        st.session_state['mvnl_status'] = 'running'
                        st.session_state['mvnl_status_msg'] = 'Running multivariate model (quick)...'
                        try:
                            data_path = os.path.join(os.getcwd(), "housing.csv")
                            mv_metrics, mv_art = multivariate_nonlinear.train_and_evaluate(data_path)
                            st.session_state['mvnl_metrics'] = mv_metrics
                            st.session_state['mvnl_artifacts'] = mv_art
                            st.session_state['mvnl_plot'] = mv_art.get('plot_png')
                            try:
                                st.session_state['mvnl_resid_png'] = multivariate_nonlinear.plot_residual_hist(mv_art)
                            except Exception:
                                st.session_state['mvnl_resid_png'] = None
                            try:
                                st.session_state['mvnl_actual_png'] = multivariate_nonlinear.plot_actual_vs_pred(mv_art)
                            except Exception:
                                st.session_state['mvnl_actual_png'] = None
                            st.session_state['mvnl_status'] = 'complete'
                            st.session_state['mvnl_status_msg'] = 'Multivariate quick run complete'
                        except FileNotFoundError:
                            st.session_state['mvnl_status'] = 'error'
                            st.session_state['mvnl_status_msg'] = "housing.csv not found in project root."

                # Message placeholder (between button and visualization)
                message_placeholder = st.empty()
                # Visualization placeholder (below status messages)
                graph_placeholder = st.empty()

                # If PCA is selected, render PCA visuals into the graph placeholder
                if selected_algo == "PCA":
                    # When PCA has been run (from the top 'Run Model' button), display plots in the Overview placeholder.
                    if st.session_state.get('pca_ready'):
                        figs = st.session_state.get('pca_figs', [])

                        if not figs:
                            graph_placeholder.info("PCA results available but no figures found.")
                        else:
                            # Provide a compact selector with only two PCA-relevant choices
                            plot_type = st.selectbox(
                                "Show plot (optional)",
                                options=["Variance", "PCA visualization"],
                                index=0,
                                key='overview_pca_plot_selector'
                            )

                            # Always render the selected plot into the graph placeholder (plot must be shown)
                            if plot_type == "Variance":
                                # variance bar + cumulative line is stored as figs[0]
                                if len(figs) >= 1:
                                    graph_placeholder.pyplot(figs[0])
                                else:
                                    graph_placeholder.info("Variance plot not available.")
                            else:
                                # PCA 2-component scatter is stored as figs[1]
                                if len(figs) >= 2:
                                    graph_placeholder.pyplot(figs[1])
                                else:
                                    # If only one fig available, show it
                                    graph_placeholder.pyplot(figs[0])

                # Render status message if present
                status = st.session_state.get('mvnl_status')
                status_msg = st.session_state.get('mvnl_status_msg')
                if status == 'running' and status_msg:
                    message_placeholder.info(status_msg)
                elif status == 'complete' and status_msg:
                    message_placeholder.success(status_msg)
                elif status == 'error' and status_msg:
                    message_placeholder.error(status_msg)
                # If Decision Tree artifacts exist or current selection is DT, show DT visuals
                # Only show Decision Tree visualizations when the Decision Tree algorithm is selected
                if selected_algo == "Decision Tree Classifier" and "dt_artifacts" in st.session_state:
                    # Always offer visualization options for Decision Tree
                    dt_options = ["confusion_matrix", "heatmap", "tree"]

                    dt_choice = st.selectbox("DT visualization", options=dt_options, index=0)
                    if dt_choice == "confusion_matrix":
                        if "dt_confusion_png" in st.session_state:
                            graph_placeholder.image(st.session_state["dt_confusion_png"], use_container_width=True)
                        else:
                            graph_placeholder.info("Confusion matrix not available yet. Run the Decision Tree model.")
                    elif dt_choice == "heatmap":
                        # Prefer stored heatmap bytes
                        if "dt_grid_heatmap_png" in st.session_state:
                            graph_placeholder.image(st.session_state["dt_grid_heatmap_png"], use_container_width=True)
                        elif "dt_grid_info" in st.session_state:
                            # attempt to regenerate from stored cv_results
                            try:
                                cvr = st.session_state["dt_grid_info"].get("cv_results")
                                if cvr and isinstance(cvr.get("params"), list) and len(cvr.get("params")) > 0:
                                    keys = list(cvr.get("params")[0].keys())
                                    if len(keys) >= 2:
                                        heat_png = decision_tree.plot_cv_heatmap(cvr, keys[0], keys[1])
                                        graph_placeholder.image(heat_png, use_container_width=True)
                                    else:
                                        graph_placeholder.info("Heatmap unavailable: GridSearch used fewer than 2 tunable parameters.")
                                else:
                                    graph_placeholder.info("GridSearchCV results not available. Run GridSearch first from Model Configuration.")
                            except Exception as e:
                                graph_placeholder.error(f"Failed to generate heatmap: {e}")
                        else:
                            graph_placeholder.info("Heatmap not available. Run GridSearch first from Model Configuration.")
                    elif dt_choice == "tree":
                        if "dt_tree_png" in st.session_state:
                            graph_placeholder.image(st.session_state["dt_tree_png"], use_container_width=True)
                        else:
                            graph_placeholder.info("Decision tree image not available yet. Run the Decision Tree model.")
                else:
                    # Visualization type selector (defaults to scatter) for regression
                    # When Multivariate model is selected, show only multivariate-related plot options
                    if selected_algo == "Multivariate Linear Regression":
                        if not _MVNL_AVAILABLE:
                            st.warning("Multivariate module not available (missing dependencies).")
                        else:
                            # Quick run handled by the overview button above (single entry)

                            plot_type = st.selectbox("Plot type", options=["multivariate_3d", "residual_hist", "actual_vs_pred"], index=0)

                            if plot_type == "multivariate_3d":
                                if "mvnl_plot" in st.session_state and st.session_state.get('mvnl_plot') is not None:
                                    graph_placeholder.image(st.session_state.get('mvnl_plot'), use_container_width=True)
                                else:
                                    graph_placeholder.info("3D plot not available. Run the Multivariate model from Model Configuration or click Run Multivariate (quick).")
                            elif plot_type == "residual_hist":
                                if "mvnl_resid_png" in st.session_state and st.session_state.get('mvnl_resid_png') is not None:
                                    graph_placeholder.image(st.session_state.get('mvnl_resid_png'), use_container_width=True)
                                else:
                                    graph_placeholder.info("Residual histogram not available. Run the Multivariate model to generate test predictions.")
                            else:
                                if "mvnl_actual_png" in st.session_state and st.session_state.get('mvnl_actual_png') is not None:
                                    graph_placeholder.image(st.session_state.get('mvnl_actual_png'), use_container_width=True)
                                else:
                                    graph_placeholder.info("Actual vs Predicted plot not available. Run the Multivariate model to generate test predictions.")
                    
                    elif selected_algo == "Support Vector Machine" or st.session_state.get('last_run_algo') == "Support Vector Machine":
                        # Support Vector Machine Visualization - Display all four plots
                        if 'svm_artifacts' in st.session_state and _SVM_AVAILABLE:
                            st.write("#### Support Vector Machine Visualizations (Binary Classification: Affordable vs Not Affordable)")
                            
                            # Display plots in columns
                            col_plot1, col_plot2 = st.columns(2)
                            
                            with col_plot1:
                                st.write("##### Confusion Matrix")
                                plot_bytes_cm = support_vector_machine.plot_results(st.session_state['svm_artifacts'], plot_type="confusion_matrix")
                                st.image(plot_bytes_cm, use_container_width=True)
                            
                            with col_plot2:
                                st.write("##### Decision Boundary (via PCA)")
                                with st.spinner("Generating PCA plot..."):
                                    plot_bytes_pca = support_vector_machine.plot_results(st.session_state['svm_artifacts'], plot_type="decision_boundary")
                                    st.image(plot_bytes_pca, use_container_width=True)
                            
                            # Support Vectors plot (full width - the new plot you requested)
                            st.write("##### Support Vectors with Margins & Decision Boundary")
                            with st.spinner("Generating Support Vectors plot..."):
                                plot_bytes_sv = support_vector_machine.plot_results(st.session_state['svm_artifacts'], plot_type="support_vectors_2d")
                                st.image(plot_bytes_sv, use_container_width=True)
                            
                            # Learning curve gets full width
                            st.write("##### Learning Curve")
                            with st.spinner("Generating Learning Curve..."):
                                plot_bytes_lc = support_vector_machine.plot_results(st.session_state['svm_artifacts'], plot_type="learning_curve")
                                st.image(plot_bytes_lc, use_container_width=True)
                        else:
                            graph_placeholder.info("Run the Support Vector Machine model from the 'Model Configuration' tab to see results.")
                    
                    elif selected_algo == "Ensemble Learning (Bagging/Boosting)" or st.session_state.get('last_run_algo') == "Ensemble Learning (Bagging/Boosting)":
                        # Ensemble Learning Visualization - Before and After Comparison
                        if 'ensemble_metrics' not in st.session_state:
                            graph_placeholder.info("Run the Ensemble models from the 'Model Configuration' tab to see results.")
                        else:
                            metrics = st.session_state['ensemble_metrics']
                            artifacts = st.session_state['ensemble_artifacts']
                            
                            st.write("#### Ensemble Learning: Decision Tree vs. Random Forest")
                            
                            # --- "Before vs After" Accuracy Metrics ---
                            st.write("##### Test Set Accuracy Comparison")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Baseline (Single Decision Tree)", f"{metrics['single_tree']['accuracy']:.2%}")
                            with col2:
                                st.metric("Ensemble (Random Forest)", f"{metrics['random_forest']['accuracy']:.2%}",
                                         delta=f"{metrics['random_forest']['accuracy'] - metrics['single_tree']['accuracy']:.2%}")

                            st.write("##### Random Forest Best Parameters Used:")
                            # Display the params that were found and used
                            if 'rf_best_params' in st.session_state:
                                st.json(st.session_state['rf_best_params'])
                            else:
                                # Fallback just in case
                                st.json(metrics['random_forest'].get('best_params', {"error": "Params not found in session"}))
                            
                            st.divider()

                            # --- Dropdown for Visualization Selection ---
                            st.write("##### 📊 Select Visualization")
                            plot_type = st.selectbox(
                                "Choose visualization type",
                                options=[
                                    "Decision Boundary (PCA)",
                                    "Decision Boundary Scatter (Publication-Ready)",
                                    "Confusion Matrix",
                                    "Learning Curves",
                                    "Probability Density Distribution",
                                    "Feature Importance"
                                ],
                                index=0,
                                key="ensemble_plot_selector"
                            )
                            
                            st.divider()
                            
                            # Display selected visualization
                            if plot_type == "Decision Boundary (PCA)":
                                st.write("##### 🎯 Decision Boundary - Class Separation View")
                                st.info("📌 2D PCA projection showing decision boundaries with probability contours")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**BEFORE: Single Decision Tree**")
                                    with st.spinner("Generating Tree Boundary Plot..."):
                                        plot_pca_tree = plot_results_ensemble(artifacts, 'single_tree', 'decision_boundary_pca')
                                        st.image(plot_pca_tree, use_container_width=True)
                                with col2:
                                    st.write("**AFTER: Random Forest**")
                                    with st.spinner("Generating RF Boundary Plot..."):
                                        plot_pca_rf = plot_results_ensemble(artifacts, 'random_forest', 'decision_boundary_pca')
                                        st.image(plot_pca_rf, use_container_width=True)
                            
                            elif plot_type == "Decision Boundary Scatter (Publication-Ready)":
                                st.write("##### 🎨 Standalone Decision Boundary (Publication-Ready)")
                                st.info("📌 Clean scatter plots with colored decision regions - perfect for presentations!")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**BEFORE: Single Decision Tree**")
                                    with st.spinner("Creating BEFORE scatter plot..."):
                                        plot_scatter_tree = plot_results_ensemble(artifacts, 'single_tree', 'decision_boundary_scatter')
                                        st.image(plot_scatter_tree, use_container_width=True)
                                        # Add download button for standalone PNG
                                        st.download_button(
                                            label="📥 Download BEFORE (Tree)",
                                            data=plot_scatter_tree,
                                            file_name="before_tree.png",
                                            mime="image/png",
                                            help="Download this plot as PNG for presentations"
                                        )
                                with col2:
                                    st.write("**AFTER: Random Forest**")
                                    with st.spinner("Creating AFTER scatter plot..."):
                                        plot_scatter_rf = plot_results_ensemble(artifacts, 'random_forest', 'decision_boundary_scatter')
                                        st.image(plot_scatter_rf, use_container_width=True)
                                        # Add download button for standalone PNG
                                        st.download_button(
                                            label="📥 Download AFTER (Forest)",
                                            data=plot_scatter_rf,
                                            file_name="after_rf.png",
                                            mime="image/png",
                                            help="Download this plot as PNG for presentations"
                                        )
                            
                            elif plot_type == "Confusion Matrix":
                                st.write("##### 📋 Confusion Matrix - Classification Performance")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**BEFORE: Single Decision Tree**")
                                    plot_cm_tree = plot_results_ensemble(artifacts, 'single_tree', 'confusion_matrix')
                                    st.image(plot_cm_tree, use_container_width=True)
                                with col2:
                                    st.write("**AFTER: Random Forest**")
                                    plot_cm_rf = plot_results_ensemble(artifacts, 'random_forest', 'confusion_matrix')
                                    st.image(plot_cm_rf, use_container_width=True)
                            
                            elif plot_type == "Learning Curves":
                                st.write("##### 📈 Learning Analysis")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**BEFORE: Single Decision Tree**")
                                    with st.spinner("Generating Tree Learning Curve..."):
                                        plot_lc_tree = plot_results_ensemble(artifacts, 'single_tree', 'learning_curve')
                                        st.image(plot_lc_tree, use_container_width=True)
                                with col2:
                                    st.write("**AFTER: Random Forest**")
                                    with st.spinner("Generating RF Ensemble Effect..."):
                                        plot_lc_rf = plot_results_ensemble(artifacts, 'random_forest', 'learning_curve')
                                        st.image(plot_lc_rf, use_container_width=True)
                            
                            elif plot_type == "Probability Density Distribution":
                                st.write("##### 📊 Model Confidence Distribution")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**BEFORE: Single Decision Tree**")
                                    with st.spinner("Analyzing Tree Confidence..."):
                                        plot_prob_tree = plot_results_ensemble(artifacts, 'single_tree', 'probability_density')
                                        st.image(plot_prob_tree, use_container_width=True)
                                with col2:
                                    st.write("**AFTER: Random Forest**")
                                    with st.spinner("Analyzing RF Confidence..."):
                                        plot_prob_rf = plot_results_ensemble(artifacts, 'random_forest', 'probability_density')
                                        st.image(plot_prob_rf, use_container_width=True)
                            
                            elif plot_type == "Feature Importance":
                                st.write("##### 🎯 Feature Importance Analysis")
                                st.info("📌 Feature importance is only available for Random Forest (ensemble method)")
                                plot_fi_rf = plot_results_ensemble(artifacts, 'random_forest', 'feature_importance')
                                st.image(plot_fi_rf, use_container_width=True)
                    
                    elif selected_algo == "K-Means Clustering" or st.session_state.get('last_run_algo') == "K-Means Clustering":
                        # K-Means Clustering Visualization
                        if 'kmeans_results' not in st.session_state:
                            graph_placeholder.info("Run the K-Means clustering analysis from the 'Model Configuration' tab to see results.")
                        else:
                            results = st.session_state['kmeans_results']
                            
                            st.write("#### K-Means Clustering Analysis")
                            st.write(f"**Optimal Clusters (k):** {results['optimal_k']}")
                            
                            # Metrics display
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Optimal k", results['optimal_k'])
                            col2.metric("Silhouette Score", f"{results['final_silhouette']:.4f}")
                            col3.metric("Inertia", f"{results['final_inertia']:,.0f}")
                            
                            st.divider()
                            
                            # Visualization selector
                            viz_type = st.selectbox(
                                "Select Visualization",
                                options=[
                                    "Elbow Curve",
                                    "Silhouette Analysis", 
                                    "Cluster Scatter (PCA)",
                                    "Cluster Distribution",
                                    "Comparison with Supervised"
                                ],
                                index=2
                            )
                            
                            st.write(f"##### {viz_type}")
                            
                            with st.spinner(f"Generating {viz_type}..."):
                                if viz_type == "Elbow Curve":
                                    # Generate elbow curve
                                    import matplotlib.pyplot as plt
                                    import io
                                    
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    k_range = results['k_range']
                                    inertias = results['inertias']
                                    optimal_k = results['optimal_k']
                                    
                                    ax.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
                                    ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, 
                                              label=f'Optimal k = {optimal_k}')
                                    ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
                                    ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12, fontweight='bold')
                                    ax.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
                                    ax.legend(fontsize=11)
                                    ax.grid(True, alpha=0.3)
                                    
                                    elbow_idx = optimal_k - k_range.start
                                    ax.annotate(f'Elbow\n({optimal_k}, {inertias[elbow_idx]:,.0f})',
                                              xy=(optimal_k, inertias[elbow_idx]),
                                              xytext=(optimal_k + 1, inertias[elbow_idx] + (max(inertias) - min(inertias)) * 0.1),
                                              arrowprops=dict(arrowstyle='->', color='red', lw=2),
                                              fontsize=10, fontweight='bold', color='red')
                                    
                                    plt.tight_layout()
                                    buf = io.BytesIO()
                                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                                    buf.seek(0)
                                    st.image(buf, use_container_width=True)
                                    plt.close(fig)
                                
                                elif viz_type == "Silhouette Analysis":
                                    # Generate silhouette analysis
                                    import matplotlib.pyplot as plt
                                    import io
                                    
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    k_range = results['k_range']
                                    silhouette_scores = results['silhouette_scores']
                                    optimal_k = results['optimal_k']
                                    
                                    ax.plot(list(k_range), silhouette_scores, 'go-', linewidth=2, markersize=8)
                                    ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
                                              label=f'Optimal k = {optimal_k}')
                                    ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
                                    ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
                                    ax.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
                                    ax.legend(fontsize=11)
                                    ax.grid(True, alpha=0.3)
                                    
                                    best_silhouette_idx = np.argmax(silhouette_scores)
                                    best_k = k_range.start + best_silhouette_idx
                                    ax.annotate(f'Best\n({best_k}, {silhouette_scores[best_silhouette_idx]:.3f})',
                                              xy=(best_k, silhouette_scores[best_silhouette_idx]),
                                              xytext=(best_k + 1, silhouette_scores[best_silhouette_idx] - 0.05),
                                              arrowprops=dict(arrowstyle='->', color='green', lw=2),
                                              fontsize=10, fontweight='bold', color='green')
                                    
                                    plt.tight_layout()
                                    buf = io.BytesIO()
                                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                                    buf.seek(0)
                                    st.image(buf, use_container_width=True)
                                    plt.close(fig)
                                
                                elif viz_type == "Cluster Scatter (PCA)":
                                    # Generate cluster scatter plot
                                    import matplotlib.pyplot as plt
                                    from sklearn.decomposition import PCA
                                    import io
                                    
                                    X_scaled = results['X_scaled']
                                    cluster_labels = results['cluster_labels']
                                    optimal_k = results['optimal_k']
                                    
                                    # Use PCA to reduce to 2D
                                    pca = PCA(n_components=2, random_state=42)
                                    X_pca = pca.fit_transform(X_scaled)
                                    
                                    fig, ax = plt.subplots(figsize=(12, 8))
                                    
                                    # Plot each cluster
                                    colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))
                                    for cluster_id in range(optimal_k):
                                        cluster_mask = cluster_labels == cluster_id
                                        ax.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1],
                                                 c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                                                 alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
                                    
                                    # Project and plot centroids
                                    centroids_pca = pca.transform(results['centroids'])
                                    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                                             c='red', marker='X', s=300, edgecolors='black', linewidth=2,
                                             label='Centroids', zorder=5)
                                    
                                    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                                                 fontsize=12, fontweight='bold')
                                    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                                                 fontsize=12, fontweight='bold')
                                    ax.set_title(f'K-Means Clusters (k={optimal_k}) - PCA Visualization', 
                                                fontsize=14, fontweight='bold')
                                    ax.legend(fontsize=9, loc='best')
                                    ax.grid(True, alpha=0.3)
                                    
                                    plt.tight_layout()
                                    buf = io.BytesIO()
                                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                                    buf.seek(0)
                                    st.image(buf, use_container_width=True)
                                    plt.close(fig)
                                
                                elif viz_type == "Cluster Distribution":
                                    # Generate cluster distribution bar chart
                                    import matplotlib.pyplot as plt
                                    import io
                                    
                                    cluster_labels = results['cluster_labels']
                                    optimal_k = results['optimal_k']
                                    
                                    unique, counts = np.unique(cluster_labels, return_counts=True)
                                    colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))
                                    
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    bars = ax.bar(unique, counts, color=colors[:optimal_k], edgecolor='black', linewidth=1.5)
                                    
                                    # Add count labels
                                    for bar, count in zip(bars, counts):
                                        height = bar.get_height()
                                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                              f'{count:,}\n({count/len(cluster_labels)*100:.1f}%)',
                                              ha='center', va='bottom', fontsize=10, fontweight='bold')
                                    
                                    ax.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
                                    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
                                    ax.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
                                    ax.set_xticks(unique)
                                    ax.grid(True, alpha=0.3, axis='y')
                                    
                                    plt.tight_layout()
                                    buf = io.BytesIO()
                                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                                    buf.seek(0)
                                    st.image(buf, use_container_width=True)
                                    plt.close(fig)
                                
                                elif viz_type == "Comparison with Supervised":
                                    # Generate comparison plot
                                    import matplotlib.pyplot as plt
                                    from sklearn.decomposition import PCA
                                    from sklearn.preprocessing import LabelEncoder
                                    import io
                                    
                                    X_scaled = results['X_scaled']
                                    cluster_labels = results['cluster_labels']
                                    y = results['y']
                                    optimal_k = results['optimal_k']
                                    
                                    # PCA for visualization
                                    pca = PCA(n_components=2, random_state=42)
                                    X_pca = pca.fit_transform(X_scaled)
                                    
                                    # Encode true labels
                                    label_encoder = LabelEncoder()
                                    y_encoded = label_encoder.fit_transform(y)
                                    
                                    # Create side-by-side comparison
                                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                                    
                                    # Left: True labels
                                    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, 
                                                         cmap='viridis', alpha=0.6, s=50, 
                                                         edgecolors='black', linewidth=0.5)
                                    centroids_pca = pca.transform(results['centroids'])
                                    ax1.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                                              c='red', marker='X', s=300, edgecolors='black', linewidth=2,
                                              label='K-Means Centroids', zorder=5)
                                    
                                    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                                                  fontsize=11, fontweight='bold')
                                    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                                                  fontsize=11, fontweight='bold')
                                    ax1.set_title('True Labels (Supervised)', fontsize=13, fontweight='bold')
                                    ax1.legend(fontsize=9)
                                    ax1.grid(True, alpha=0.3)
                                    
                                    cbar1 = plt.colorbar(scatter1, ax=ax1)
                                    cbar1.set_label('Price Category', fontsize=10, fontweight='bold')
                                    cbar1.set_ticks(range(len(label_encoder.classes_)))
                                    cbar1.set_ticklabels(label_encoder.classes_)
                                    
                                    # Right: Cluster labels
                                    scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                                                         cmap='tab10', alpha=0.6, s=50,
                                                         edgecolors='black', linewidth=0.5)
                                    ax2.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                                              c='red', marker='X', s=300, edgecolors='black', linewidth=2,
                                              label='Centroids', zorder=5)
                                    
                                    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                                                  fontsize=11, fontweight='bold')
                                    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                                                  fontsize=11, fontweight='bold')
                                    ax2.set_title(f'K-Means Clusters (k={optimal_k}) - Unsupervised', 
                                                 fontsize=13, fontweight='bold')
                                    ax2.legend(fontsize=9)
                                    ax2.grid(True, alpha=0.3)
                                    
                                    cbar2 = plt.colorbar(scatter2, ax=ax2)
                                    cbar2.set_label('Cluster ID', fontsize=10, fontweight='bold')
                                    
                                    plt.tight_layout()
                                    buf = io.BytesIO()
                                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                                    buf.seek(0)
                                    st.image(buf, use_container_width=True)
                                    plt.close(fig)
                            
                            # Additional insights
                            st.divider()
                            st.write("##### 🔍 Cluster Analysis Insights")
                            
                            import pandas as pd
                            comparison_df = pd.DataFrame({
                                'Cluster': results['cluster_labels'],
                                'True_Label': results['y']
                            })
                            
                            st.write("**Cluster composition by price categories:**")
                            cross_tab = pd.crosstab(
                                comparison_df['Cluster'], 
                                comparison_df['True_Label'], 
                                normalize='index'
                            ) * 100
                            
                            st.dataframe(cross_tab.round(1), use_container_width=True)
                            
                            st.info("""
                            💡 **Interpretation:**
                            - **Silhouette Score** close to 1: Well-separated clusters
                            - **Silhouette Score** close to 0: Overlapping clusters
                            - **Silhouette Score** negative: Misclassified points
                            - **Inertia**: Lower is better (compact clusters)
                            """)
                    
                    elif selected_algo == "DBSCAN Clustering" or st.session_state.get('last_run_algo') == "DBSCAN Clustering":
                        # DBSCAN Clustering Visualization
                        if 'dbscan_results' not in st.session_state:
                            graph_placeholder.info("Run the DBSCAN clustering analysis from the 'Model Configuration' tab to see results.")
                        else:
                            results = st.session_state['dbscan_results']
                            
                            st.write("#### DBSCAN Clustering Analysis")
                            # Show mode if available
                            mode_display = results.get('mode', 'standard').replace('_', ' ').title()
                            st.write(f"**Density-Based Spatial Clustering** ({mode_display} Mode)")
                            
                            # Display parameters used
                            st.caption(f"Parameters: eps={results['eps']:.4f}, min_samples={results['min_samples']}")
                            
                            # Metrics display
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Clusters", results['n_clusters'])
                            col2.metric("Noise Points", f"{results['n_noise']:,}")
                            col3.metric("Noise Ratio", f"{results['noise_ratio']:.1f}%")
                            if results['silhouette']:
                                col4.metric("Silhouette", f"{results['silhouette']:.4f}")
                            else:
                                col4.metric("Silhouette", "N/A")
                            
                            st.divider()
                            
                            # Visualization selector - include Core/Border/Noise if available
                            viz_options = ["Model Configuration", "Cluster Map (Geographic)", "Cluster Profiles"]
                            
                            # Add Core/Border/Noise visualization if using optimized mode
                            if 'core_border_noise_png' in results:
                                viz_options.insert(2, "Core/Border/Noise Points")
                            
                            # Add K-Distance plot if available
                            if 'k_distance_plot_png' in results or 'dbscan_eps_results' in st.session_state:
                                viz_options.append("K-Distance Plot")
                            
                            viz_type = st.selectbox(
                                "Select Visualization",
                                options=viz_options,
                                index=0
                            )
                            
                            st.write(f"##### {viz_type}")
                            
                            with st.spinner(f"Generating {viz_type}..."):
                                if viz_type == "Model Configuration":
                                    if 'model_config_png' in results:
                                        st.image(results['model_config_png'], use_container_width=True)
                                        
                                        st.info("""
                                        ⚙️ **Model Configuration Overview:**
                                        - **PCA Variance**: Shows how much information each component captures
                                        - **Feature Weights**: Priority given to different features (geographic > economic > demographic)
                                        - **DBSCAN Parameters**: Neighborhood radius (eps) and minimum samples
                                        - **Pipeline Workflow**: Complete data processing flow from raw data to clusters
                                        - **Quality Metrics**: Silhouette score, noise ratio, and cluster count assessment
                                        
                                        This view helps you understand exactly how the model processes your data!
                                        """)
                                    else:
                                        st.warning("Model configuration visualization not available.")
                                
                                elif viz_type == "Cluster Map (Geographic)":
                                    st.image(results['cluster_map_png'], use_container_width=True)
                                    
                                    st.info("""
                                    🗺️ **Geographic Clustering:**
                                    - Colored points: Dense regions (clusters)
                                    - Gray X marks: Noise points (outliers/sparse areas)
                                    - DBSCAN identifies naturally dense geographic areas
                                    """)
                                
                                elif viz_type == "Core/Border/Noise Points":
                                    if 'core_border_noise_png' in results:
                                        st.image(results['core_border_noise_png'], use_container_width=True)
                                        
                                        st.info("""
                                        🎯 **Point Type Classification:**
                                        - 🟢 **Core Points** (circles): Dense cluster centers with ≥min_samples neighbors
                                        - 🟡 **Border Points** (X marks): Edge points within eps of core points
                                        - ⚫ **Noise Points** (dots): Outliers not belonging to any cluster
                                        
                                        This visualization shows the internal structure of DBSCAN clustering!
                                        """)
                                    else:
                                        st.warning("Core/Border/Noise visualization not available for this mode.")
                                
                                elif viz_type == "Cluster Profiles":
                                    st.image(results['cluster_profile_png'], use_container_width=True)
                                    
                                    st.info("""
                                    📊 **Feature Distribution:**
                                    - Shows how features vary across clusters
                                    - Helps understand what makes each cluster unique
                                    - Useful for market segmentation insights
                                    """)
                                
                                elif viz_type == "K-Distance Plot":
                                    # Try to get from results first, then from session state
                                    k_plot_png = results.get('k_distance_plot_png')
                                    if k_plot_png:
                                        st.image(k_plot_png, use_container_width=True)
                                    elif 'dbscan_eps_results' in st.session_state:
                                        st.image(st.session_state['dbscan_eps_results']['plot_png'], use_container_width=True)
                                    else:
                                        st.warning("K-Distance plot not available.")
                                    
                                    st.info("""
                                    📈 **Elbow Detection:**
                                    - Left side: Points in dense regions
                                    - Right side: Sparse/outlier points
                                    - Elbow point: Optimal eps parameter
                                    """)
                            
                            # Cluster summary table
                            st.divider()
                            st.write("##### 🔍 Cluster Characteristics")
                            
                            if results['summary_df'] is not None and len(results['summary_df']) > 0:
                                st.dataframe(results['summary_df'], use_container_width=True, hide_index=True)
                            else:
                                st.warning("No clusters found (all points are noise). Try adjusting eps or min_samples.")
                            
                            # Interpretation guide
                            st.divider()
                            st.info("""
                            💡 **DBSCAN vs K-Means:**
                            
                            **DBSCAN (Density-Based):**
                            - ✅ Finds clusters of arbitrary shape
                            - ✅ Automatically detects outliers
                            - ✅ No need to specify number of clusters
                            - ⚠️ Sensitive to eps and min_samples parameters
                            
                            **K-Means (Centroid-Based):**
                            - ✅ Finds spherical clusters efficiently
                            - ✅ Works well with evenly-sized clusters
                            - ⚠️ Must specify k in advance
                            - ⚠️ Sensitive to outliers
                            
                            **Use DBSCAN when:**
                            - Your data has varying cluster densities
                            - You want to identify outliers/anomalies
                            - Cluster shapes are irregular or non-spherical
                            - You don't know the number of clusters beforehand
                            """)
                    
                    else:
                        # non-multivariate: keep existing scatter/residual/hist behavior
                        # Do not render the generic plot selector when PCA is selected
                        if selected_algo == "PCA":
                            # PCA handled above; nothing to do here
                            pass
                        else:
                            plot_type = st.selectbox("Plot type", options=["scatter", "residual", "hist"], index=0)

                            if "latest_artifacts" in st.session_state and _LR_AVAILABLE:
                                # Render chosen plot type from stored artifacts
                                art = st.session_state["latest_artifacts"]
                                png = linear_regression.plot_results(art, plot_type=plot_type)
                                graph_placeholder.image(png, use_container_width=True)
                            elif "latest_plot_lr" in st.session_state:
                                # fallback to the previously generated default plot
                                graph_placeholder.image(st.session_state["latest_plot_lr"], use_container_width=True)
                            else:
                                graph_placeholder.info("Graph Placeholder — Plotly charts will be placed here.")

            with col2:
                st.markdown("#### Metrics & Model Info")
                metrics_placeholder = st.empty()
                # PCA metrics override: show PCA-specific info when available
                if selected_algo == "PCA" and st.session_state.get('pca_results'):
                    pr = st.session_state.get('pca_results')
                    metrics_placeholder.markdown("**PCA Analysis Results**")
                    metrics_placeholder.write(f"Number of components to explain 95% of variance: {pr.get('components_for_95')}")
                    metrics_placeholder.write(f"Variance explained by first 2 components: {pr.get('variance_for_2'):.2%}")
                elif "latest_metrics" in st.session_state:
                    m = st.session_state["latest_metrics"]
                    # show intercept & slope if available
                    if "latest_artifacts" in st.session_state:
                        art = st.session_state["latest_artifacts"]
                        slope = art.get("slope")
                        intercept = art.get("intercept")
                        st.write(f"Intercept: {intercept:.4f}")
                        st.write(f"Slope: {slope:.4f}")

                    for k, v in m.items():
                        if k == "MSE":
                            metrics_placeholder.write(f"{k}: {v:,.2f}")
                        else:
                            metrics_placeholder.write(f"{k}: {v:.4f}")
                else:
                    metrics_placeholder.write("Metrics will appear here once models run.")
                # Decision Tree metrics intentionally not displayed in Overview (shown in Model Configuration instead)

                if "dt_artifacts" in st.session_state:
                    dt_art = st.session_state["dt_artifacts"]
                    st.markdown("---")
                    st.markdown("**DT Data / Preprocessing**")
                    st.write(f"Train records: {dt_art.get('train_count')}")
                    st.write(f"Test records: {dt_art.get('test_count')}")
                    prep = dt_art.get('preprocessing', {})
                    st.write(f"Rows before drop: {prep.get('rows_before')}")
                    st.write(f"Rows after drop: {prep.get('rows_after')}")
                    st.write("Features used:")
                    st.write(prep.get('features_used'))

                # Show multivariate metrics if present
                if "mvnl_artifacts" in st.session_state:
                    mv = st.session_state["mvnl_artifacts"]
                    st.markdown("---")
                    st.markdown("**Multivariate Non-linear Regression Metrics (test)**")
                    import pandas as _pd
                    mm = mv.get('metrics_test', {})
                    mm_df = _pd.DataFrame.from_dict(mm, orient='index', columns=['value']).reset_index().rename(columns={'index':'metric'})
                    mm_df['value'] = mm_df['value'].apply(lambda x: round(x,4) if isinstance(x, float) else x)
                    st.table(mm_df)
                
                # Show SVM metrics if present
                if "svm_artifacts" in st.session_state:
                    st.markdown("---")
                    st.markdown("**SVM (SVC) Classification Metrics**")
                    import pandas as _pd
                    
                    # Display Accuracy and Best CV Score
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        acc = st.session_state['svm_metrics'].get('accuracy_score')
                        if acc:
                            st.metric("Test Accuracy", f"{acc:.2%}")
                    with col_m2:
                        best_score = st.session_state['svm_metrics'].get('best_score')
                        if best_score:
                            st.metric("Best CV Score", f"{best_score:.2%}")
                    
                    # Display Best Parameters
                    st.write("**Best Parameters:**")
                    best_params = st.session_state['svm_metrics'].get('best_params', {})
                    st.json(best_params)
                    
                    # Display Classification Report
                    report = st.session_state['svm_metrics'].get('classification_report')
                    if report:
                        st.write("**Classification Report:**")
                        report_df = _pd.DataFrame(report).transpose()
                        st.dataframe(report_df.round(3))
                    
                    st.write(f"**Model:** `{st.session_state['svm_artifacts'].get('model_params', 'N/A')}`")

        # Model Configuration Tab
        with tabs[1]:
            st.header("Model Configuration")

            st.markdown("Configure hyperparameters for the selected algorithm:")

            # Dynamic UI: show different controls based on selected algorithm
            if selected_algo == "Linear Regression":
                with st.expander("Linear Regression settings", expanded=True):
                    st.slider("Fit intercept", 0, 1, 1)
                    st.slider("Normalize", 0, 1, 0)

                # Run the Linear Regression pipeline
                if st.button("Run Linear Regression"):
                    st.info("Training Linear Regression...")
                    # file path: prefer uploaded file if present, otherwise use bundled housing.csv
                    if uploaded_file is not None:
                        # save uploaded file to a temp location
                        data_path = os.path.join(os.getcwd(), "uploaded_housing.csv")
                        with open(data_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    else:
                        data_path = os.path.join(os.getcwd(), "housing.csv")

                    try:
                        metrics, artifacts = linear_regression.train_and_evaluate(data_path)
                        st.success("Training complete")
                        # Show metrics
                        st.subheader("Performance Metrics")
                        for k, v in metrics.items():
                            if k == "MSE":
                                st.write(f"{k}: {v:,.2f}")
                            else:
                                st.write(f"{k}: {v:.4f}")

                        # Plot generation (saved to Overview) — do not display in Model Configuration
                        png = linear_regression.plot_results(artifacts)
                        st.session_state["latest_metrics"] = metrics
                        st.session_state["latest_plot_lr"] = png
                        st.session_state["latest_artifacts"] = artifacts
                    except FileNotFoundError:
                        st.error("Dataset not found. Place 'housing.csv' in the project root or upload a CSV.")

            elif selected_algo == "Decision Tree Classifier":
                with st.expander("Decision Tree settings", expanded=True):
                    max_depth = st.slider("Max depth", min_value=1, max_value=50, value=5)
                    st.write("Current settings:", {"max_depth": max_depth})

                    use_grid = st.checkbox("Run GridSearchCV for hyperparameter tuning", value=False)

                    if not _DT_AVAILABLE:
                        st.warning("Decision Tree module not available (missing dependencies).")
                    else:
                        if use_grid:
                            st.markdown("**Grid search settings**")
                            param_preset = st.selectbox("Parameter preset", options=["small", "medium", "large"], index=0)
                            if param_preset == "small":
                                param_grid = {"max_depth": [3, 5, 10], "min_samples_split": [2, 5]}
                            elif param_preset == "medium":
                                param_grid = {"max_depth": [3, 5, 10, 15], "min_samples_split": [2, 5, 10]}
                            else:
                                param_grid = {"max_depth": [3, 5, 10, 15, 20], "min_samples_split": [2, 5, 10, 20]}

                            if st.button("Run GridSearchCV"):
                                st.info("Running GridSearchCV — this may take a while...")
                                try:
                                    grid_info, grid_out = decision_tree.train_with_grid_search(
                                        os.path.join(os.getcwd(), "housing.csv"), param_grid=param_grid
                                    )
                                    st.success("Grid search complete")
                                    st.write("Best params:", grid_info.get("best_params"))
                                    st.write("Best CV score:", round(grid_info.get("best_score", 0), 4))
                                    # save artifacts and metrics
                                    st.session_state["dt_metrics"] = grid_out.get("metrics")
                                    st.session_state["dt_artifacts"] = grid_out.get("artifacts")
                                    st.session_state["dt_confusion_png"] = decision_tree.plot_confusion_matrix(grid_out.get("artifacts"))
                                    st.session_state["dt_tree_png"] = decision_tree.plot_tree_image(grid_out.get("artifacts"))
                                    st.session_state["dt_grid_info"] = grid_info

                                    # Show heatmap for first two params
                                    params = list(param_grid.keys())
                                    if len(params) >= 2:
                                        heat_png = decision_tree.plot_cv_heatmap(grid_info["cv_results"], params[0], params[1])
                                        # store heatmap bytes so Overview can display it later without re-running plotting
                                        st.session_state["dt_grid_heatmap_png"] = heat_png
                                        st.image(heat_png, caption="GridSearchCV heatmap (mean_test_score)", use_container_width=True)
                                except FileNotFoundError:
                                    st.error("housing.csv not found in project root.")
                        else:
                            if st.button("Run Decision Tree"):
                                st.info("Training Decision Tree Classifier...")
                                data_path = os.path.join(os.getcwd(), "housing.csv")
                                try:
                                    dt_metrics, dt_artifacts = decision_tree.train_and_evaluate(data_path, max_depth=max_depth)
                                    st.success("Decision Tree training complete")
                                    st.subheader("DT Metrics")
                                    try:
                                        import pandas as _pd
                                        if isinstance(dt_metrics, dict):
                                            dt_df = _pd.DataFrame.from_dict(dt_metrics, orient='index', columns=['value']).reset_index().rename(columns={'index':'metric'})
                                            dt_df['value'] = dt_df['value'].apply(lambda x: round(x,4) if isinstance(x, float) else x)
                                            st.table(dt_df)
                                        else:
                                            st.write(dt_metrics)
                                    except Exception:
                                        st.write(dt_metrics)
                                    # store artifacts for Overview visualization
                                    st.session_state["dt_metrics"] = dt_metrics
                                    st.session_state["dt_artifacts"] = dt_artifacts
                                    # also store confusion matrix and tree images
                                    cm_png = decision_tree.plot_confusion_matrix(dt_artifacts)
                                    tree_png = decision_tree.plot_tree_image(dt_artifacts)
                                    st.session_state["dt_confusion_png"] = cm_png
                                    st.session_state["dt_tree_png"] = tree_png
                                except FileNotFoundError:
                                    st.error("housing.csv not found in project root.")

            elif selected_algo == "Support Vector Machine":
                with st.expander("SVM settings", expanded=True):
                    if not _SVM_AVAILABLE:
                        st.error(f"Support Vector Machine module could not be loaded: {_SVM_IMPORT_ERROR}")
                    else:
                        st.info("This model uses pre-optimized parameters for fast training (~10 seconds instead of 3 minutes).")
                        st.markdown("**Best Parameters (from previous GridSearch):**")
                        st.write("- C: 100.0")
                        st.write("- Kernel: rbf")
                        st.write("- Gamma: scale")
                        st.write("- Expected Accuracy: ~85%")

                        if st.button("Run Support Vector Machine (Fast - Best Params)", key="run_svm_config"):
                            with st.spinner("Training SVM with best parameters... This will take ~10 seconds."):
                                file_path = os.path.join(os.getcwd(), "housing.csv")
                                try:
                                    # Use pre-defined best parameters (no GridSearch)
                                    metrics, artifacts = support_vector_machine.train_and_evaluate(
                                        file_path,
                                        use_gridsearch=False,
                                        best_params={'C': 100.0, 'gamma': 'scale', 'kernel': 'rbf'}
                                    )
                                    st.session_state['svm_metrics'] = metrics
                                    st.session_state['svm_artifacts'] = artifacts
                                    st.session_state['last_run_algo'] = "Support Vector Machine"
                                    st.success("SVM (SVC) model trained successfully with best parameters (C=100.0, kernel=rbf, gamma=scale)!")
                                    
                                    # Display metrics
                                    import pandas as _pd
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        acc = metrics.get('accuracy_score')
                                        if acc:
                                            st.metric("Test Set Accuracy", f"{acc:.2%}")
                                    with col2:
                                        best_score = metrics.get('best_score')
                                        if best_score is not None:
                                            st.metric("Best CV Score", f"{best_score:.2%}")
                                        else:
                                            st.metric("Best CV Score", "N/A (No GridSearch)")
                                    
                                    st.write("##### Best Parameters Found:")
                                    st.json(metrics.get('best_params'))
                                    
                                    report = metrics.get('classification_report')
                                    if report:
                                        st.write("##### Classification Report")
                                        report_df = _pd.DataFrame(report).transpose()
                                        st.dataframe(report_df.round(3))
                                except FileNotFoundError:
                                    st.error("housing.csv not found in project root.")
                                except Exception as e:
                                    st.error(f"Training failed: {e}")
                                    st.exception(e)
            
            elif selected_algo == "Ensemble Learning (Bagging/Boosting)":
                st.subheader("Ensemble Learning (Bagging/Boosting) Settings")
                if not _ENSEMBLE_AVAILABLE:
                    st.error(f"Ensemble Learning module could not be loaded: {_ENSEMBLE_IMPORT_ERROR}")
                else:
                    file_path = os.path.join(os.getcwd(), "housing.csv")
                    
                    # Show cached parameters if available
                    if 'rf_best_params' in st.session_state:
                        st.success("✅ Using cached best parameters (found via GridSearch)")
                        with st.expander("View Best Parameters", expanded=False):
                            st.json(st.session_state['rf_best_params'])
                            st.write(f"**Best CV Score:** {st.session_state['rf_best_score']:.2%}")
                        
                        if st.button("🔄 Re-run GridSearchCV (Optional)", key="rerun_gridsearch"):
                            with st.spinner("Running GridSearchCV... This will take several minutes."):
                                param_results = find_best_ensemble_params(file_path, random_state=42)
                                st.session_state['rf_best_params'] = param_results['best_params']
                                st.session_state['rf_best_score'] = param_results['best_score']
                                st.success(f"✅ Updated! CV Score: {param_results['best_score']:.2%}")
                                st.rerun()
                    else:
                        st.info("""
                        ℹ️ **First-time setup:** Click the button below to automatically:
                        1. Run GridSearchCV to find best parameters (only once)
                        2. Train both models for comparison
                        
                        Future runs will use cached parameters and be much faster!
                        """)
                    
                    st.divider()
                    
                    # Configuration
                    st.write("#### Model Configuration")
                    base_tree_depth = st.slider(
                        "Baseline Tree Max Depth", 
                        min_value=2, 
                        max_value=20, 
                        value=10, 
                        key="base_tree_depth",
                        help="Set the max_depth for the 'before' Decision Tree. A lower number shows underfitting, a higher number shows overfitting."
                    )

                    # Single unified button
                    if st.button("🚀 Train Ensemble Models", key="run_ensemble_unified", type="primary"):
                        # Step 1: Find best params if not cached
                        if 'rf_best_params' not in st.session_state:
                            st.info("🔍 Finding best parameters (first time only)...")
                            with st.spinner("Running GridSearchCV... This may take a few minutes."):
                                param_results = find_best_ensemble_params(file_path, random_state=42)
                                st.session_state['rf_best_params'] = param_results['best_params']
                                st.session_state['rf_best_score'] = param_results['best_score']
                                st.success(f"✅ Found best parameters! CV Score: {param_results['best_score']:.2%}")
                        
                        # Step 2: Train models
                        best_params = st.session_state['rf_best_params']
                        with st.spinner("Training Baseline and Ensemble models..."):
                            metrics, artifacts = train_and_evaluate_ensemble(
                                file_path,
                                rf_params=best_params,
                                base_tree_max_depth=base_tree_depth,
                                random_state=42
                            )
                            
                            if "error" in metrics['single_tree']:
                                st.error(f"Error during training: {metrics['single_tree']['error']}")
                            else:
                                st.session_state['ensemble_metrics'] = metrics
                                st.session_state['ensemble_artifacts'] = artifacts
                                st.session_state['last_run_algo'] = "Ensemble Learning (Bagging/Boosting)"
                                
                                st.success("✅ Models trained successfully!")
                                
                                # Display metrics
                                st.write("#### Test Set Accuracy")
                                col1, col2 = st.columns(2)
                                col1.metric("Baseline (Single Decision Tree)", f"{metrics['single_tree']['accuracy']:.2%}")
                                col2.metric("Ensemble (Random Forest)", f"{metrics['random_forest']['accuracy']:.2%}")

            elif selected_algo == "Random Forest":
                with st.expander("Random Forest settings", expanded=True):
                    n_estimators = st.slider("n_estimators", min_value=10, max_value=1000, value=100, step=10)
                    max_features = st.selectbox("max_features", options=["auto", "sqrt", "log2"])  # placeholder
                    st.write({"n_estimators": n_estimators, "max_features": max_features})

            elif selected_algo == "K-Means Clustering":
                st.subheader("K-Means Clustering Settings")
                if not _KMEANS_AVAILABLE:
                    st.error(f"K-Means module could not be loaded: {_KMEANS_IMPORT_ERROR}")
                else:
                    st.info("K-Means is an unsupervised learning algorithm that groups similar data points into clusters.")
                    
                    file_path = os.path.join(os.getcwd(), "housing.csv")
                    
                    # Configuration
                    st.write("#### Clustering Configuration")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        k_min = st.number_input("Minimum k to test", min_value=2, max_value=10, value=2, 
                                               help="Minimum number of clusters to test")
                        optimal_k_override = st.number_input("Force specific k (0 = auto-detect)", min_value=0, max_value=20, value=0,
                                                            help="Set to 0 to auto-detect optimal k using elbow method")
                    
                    with col2:
                        k_max = st.number_input("Maximum k to test", min_value=3, max_value=20, value=11,
                                               help="Maximum number of clusters to test")
                        save_plots_checkbox = st.checkbox("Save plots to 'output' folder", value=False)
                    
                    st.divider()
                    
                    # Show previous results if available
                    if 'kmeans_results' in st.session_state:
                        st.success("✅ K-Means analysis previously completed")
                        results = st.session_state['kmeans_results']
                        
                        with st.expander("View Previous Results Summary", expanded=False):
                            col_a, col_b, col_c = st.columns(3)
                            col_a.metric("Optimal k", results['optimal_k'])
                            col_b.metric("Silhouette Score", f"{results['final_silhouette']:.4f}")
                            col_c.metric("Inertia", f"{results['final_inertia']:,.0f}")
                    
                    # Run button
                    if st.button("🚀 Run K-Means Clustering Analysis", key="run_kmeans_config", type="primary"):
                        with st.spinner("Running K-Means clustering analysis... This may take a minute."):
                            try:
                                k_range = range(k_min, k_max)
                                optimal_k_param = None if optimal_k_override == 0 else optimal_k_override
                                
                                results = kmeans.run_kmeans_analysis(
                                    data_path=file_path,
                                    k_range=k_range,
                                    optimal_k=optimal_k_param,
                                    save_plots=save_plots_checkbox,
                                    output_dir='output'
                                )
                                
                                st.session_state['kmeans_results'] = results
                                st.session_state['last_run_algo'] = "K-Means Clustering"
                                st.success("✅ K-Means clustering analysis complete!")
                                
                                # Display metrics
                                st.write("#### Clustering Results")
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Optimal k", results['optimal_k'])
                                col2.metric("Silhouette Score", f"{results['final_silhouette']:.4f}")
                                col3.metric("Inertia", f"{results['final_inertia']:,.0f}")
                                
                                st.write("##### Cluster Distribution")
                                import pandas as pd
                                unique, counts = np.unique(results['cluster_labels'], return_counts=True)
                                cluster_df = pd.DataFrame({
                                    'Cluster ID': unique,
                                    'Sample Count': counts,
                                    'Percentage': [f"{(c/len(results['cluster_labels'])*100):.1f}%" for c in counts]
                                })
                                st.dataframe(cluster_df, use_container_width=True)
                                
                                if save_plots_checkbox:
                                    st.info("📁 Plots saved to the 'output' folder")
                                
                            except Exception as e:
                                st.error(f"K-Means clustering failed: {e}")
                                import traceback
                                st.exception(e)

            elif selected_algo == "DBSCAN Clustering":
                st.subheader("DBSCAN Clustering Settings")
                if not _DBSCAN_AVAILABLE:
                    st.error(f"DBSCAN module could not be loaded: {_DBSCAN_IMPORT_ERROR}")
                else:
                    st.info("DBSCAN is a density-based clustering algorithm that identifies dense regions and outliers in geographic data.")
                    
                    file_path = os.path.join(os.getcwd(), "housing.csv")
                    
                    # Two-step process
                    st.write("#### 📊 Step 1: Find Optimal Eps Parameter")
                    st.markdown("""
                    The **eps** parameter defines the maximum distance between two samples for them to be considered in the same neighborhood.
                    Use the k-distance plot to identify the optimal eps value (the "elbow" in the curve).
                    """)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        min_samples_step1 = st.number_input(
                            "Min Samples (for eps detection)", 
                            min_value=2, 
                            max_value=20, 
                            value=5,
                            help="Minimum number of points to form a dense region"
                        )
                    
                    with col2:
                        if st.button("🔍 Find Optimal Eps", key="find_eps_button"):
                            with st.spinner("Computing k-distance plot..."):
                                try:
                                    # Load data (now returns 5 values including pca)
                                    X_scaled, df_full, feature_names, scaler, pca = dbscan_clustering.load_data(file_path)
                                    
                                    # Find optimal eps
                                    eps_results = dbscan_clustering.find_optimal_eps(
                                        X_scaled, 
                                        min_samples=min_samples_step1,
                                        save_plot=False
                                    )
                                    
                                    st.session_state['dbscan_eps_results'] = eps_results
                                    st.session_state['dbscan_data'] = {
                                        'X_scaled': X_scaled,
                                        'df_full': df_full,
                                        'feature_names': feature_names
                                    }
                                    
                                    st.success(f"✅ Suggested eps: {eps_results['suggested_eps']:.4f}")
                                    
                                except Exception as e:
                                    st.error(f"Error finding optimal eps: {e}")
                                    import traceback
                                    st.exception(e)
                    
                    # Display k-distance plot if available
                    if 'dbscan_eps_results' in st.session_state:
                        st.write("##### K-Distance Plot")
                        st.image(st.session_state['dbscan_eps_results']['plot_png'], use_container_width=True)
                        
                        suggested_eps = st.session_state['dbscan_eps_results']['suggested_eps']
                        st.info(f"💡 **Recommended eps value:** {suggested_eps:.4f}")
                    
                    st.divider()
                    
                    # Step 2: Run DBSCAN with chosen parameters
                    st.write("#### 🎯 Step 2: Run DBSCAN Clustering")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Pre-fill with suggested eps if available
                        default_eps = st.session_state.get('dbscan_eps_results', {}).get('suggested_eps', 0.5)
                        eps_input = st.number_input(
                            "Eps (neighborhood radius)",
                            min_value=0.01,
                            max_value=10.0,
                            value=float(default_eps),
                            format="%.4f",
                            help="Maximum distance between two samples in the same neighborhood"
                        )
                    
                    with col2:
                        min_samples_input = st.number_input(
                            "Min Samples",
                            min_value=2,
                            max_value=50,
                            value=min_samples_step1,
                            help="Minimum points to form a dense region"
                        )
                    
                    # Show previous results if available
                    if 'dbscan_results' in st.session_state:
                        st.success("✅ DBSCAN analysis previously completed")
                        results = st.session_state['dbscan_results']
                        
                        with st.expander("View Previous Results Summary", expanded=False):
                            col_a, col_b, col_c = st.columns(3)
                            col_a.metric("Clusters Found", results['n_clusters'])
                            col_b.metric("Noise Points", f"{results['n_noise']:,}")
                            col_c.metric("Noise Ratio", f"{results['noise_ratio']:.1f}%")
                    
                    if st.button("🚀 Run DBSCAN Clustering", key="run_dbscan_config", type="primary"):
                        with st.spinner("Running DBSCAN clustering analysis..."):
                            try:
                                # Use cached data if available, otherwise load
                                if 'dbscan_data' in st.session_state:
                                    X_scaled = st.session_state['dbscan_data']['X_scaled']
                                    df_full = st.session_state['dbscan_data']['df_full']
                                    feature_names = st.session_state['dbscan_data']['feature_names']
                                else:
                                    # Load data (now returns 5 values including pca)
                                    X_scaled, df_full, feature_names, scaler, pca = dbscan_clustering.load_data(file_path)
                                
                                # Run DBSCAN
                                dbscan_results = dbscan_clustering.run_dbscan(X_scaled, eps_input, min_samples_input)
                                
                                # Add labels to dataframe
                                df_full['cluster'] = dbscan_results['labels']
                                
                                # Summarize clusters
                                summary_df = dbscan_clustering.summarize_clusters(df_full, feature_names)
                                
                                # Create visualizations
                                cluster_map_png = dbscan_clustering.plot_results(
                                    'cluster_map',
                                    data=X_scaled,
                                    labels=dbscan_results['labels'],
                                    df=df_full,
                                    feature_names=feature_names,
                                    eps=eps_input,
                                    min_samples=min_samples_input
                                )
                                
                                cluster_profile_png = dbscan_clustering.plot_results(
                                    'cluster_profile',
                                    labels=dbscan_results['labels'],
                                    df=df_full,
                                    feature_names=feature_names
                                )
                                
                                # Store results
                                results = {
                                    'X_scaled': X_scaled,
                                    'df_full': df_full,
                                    'feature_names': feature_names,
                                    'eps': eps_input,
                                    'min_samples': min_samples_input,
                                    'model': dbscan_results['model'],
                                    'labels': dbscan_results['labels'],
                                    'n_clusters': dbscan_results['n_clusters'],
                                    'n_noise': dbscan_results['n_noise'],
                                    'noise_ratio': dbscan_results['noise_ratio'],
                                    'silhouette': dbscan_results['silhouette'],
                                    'core_sample_indices': dbscan_results['core_sample_indices'],
                                    'summary_df': summary_df,
                                    'cluster_map_png': cluster_map_png,
                                    'cluster_profile_png': cluster_profile_png
                                }
                                
                                st.session_state['dbscan_results'] = results
                                st.session_state['last_run_algo'] = "DBSCAN Clustering"
                                
                                st.success("✅ DBSCAN clustering complete!")
                                
                                # Display metrics
                                st.write("#### Clustering Results")
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Clusters", results['n_clusters'])
                                col2.metric("Noise Points", f"{results['n_noise']:,}")
                                col3.metric("Noise Ratio", f"{results['noise_ratio']:.1f}%")
                                if results['silhouette']:
                                    col4.metric("Silhouette Score", f"{results['silhouette']:.4f}")
                                else:
                                    col4.metric("Silhouette Score", "N/A")
                                
                                # Display cluster summary
                                if summary_df is not None and len(summary_df) > 0:
                                    st.write("##### Cluster Summary")
                                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                                
                            except Exception as e:
                                st.error(f"DBSCAN clustering failed: {e}")
                                import traceback
                                st.exception(e)

            elif selected_algo == "K-Means":
                with st.expander("K-Means settings", expanded=True):
                    n_clusters = st.slider("n_clusters", min_value=2, max_value=50, value=8)
                    init = st.selectbox("init", options=["k-means++", "random"])  # placeholder
                    st.write({"n_clusters": n_clusters, "init": init})

            elif selected_algo == "DBSCAN":
                with st.expander("DBSCAN settings", expanded=True):
                    eps = st.number_input("eps", min_value=0.01, max_value=10.0, value=0.5)
                    min_samples = st.number_input("min_samples", min_value=1, max_value=50, value=5)
                    st.write({"eps": eps, "min_samples": min_samples})

            elif selected_algo == "PCA":
                with st.expander("PCA settings", expanded=True):
                    st.markdown("Configure PCA and run a custom PCA workflow.")
                    n_components_viz = st.slider(
                        "Number of components for visualization (2 for 2D)",
                        min_value=1, max_value=10, value=2, step=1, key='pca_cfg_ncomp'
                    )
                    scale_features = st.checkbox("Scale features (StandardScaler)", value=True, key='pca_cfg_scale')

                    st.write({"n_components_viz": n_components_viz, "scale": scale_features})

                    if st.button("Run PCA (with config)", key="run_pca_config"):
                        st.info("Running PCA with configured settings...")
                        try:
                            from algorithms.pca import compute_pca_results
                            data_path = os.path.join(os.getcwd(), "housing.csv")
                            results = compute_pca_results(data_path=data_path, n_components_viz=n_components_viz, scale=scale_features)
                            if results is None:
                                st.error("housing.csv not found in project root or PCA failed.")
                            else:
                                st.session_state['pca_results'] = {
                                    'components_for_95': results.get('components_for_95'),
                                    'variance_for_2': results.get('variance_for_2')
                                }
                                st.session_state['pca_figs'] = results.get('figs', [])
                                # store artifacts for predictions
                                st.session_state['pca_artifacts'] = {
                                    'scaler': results.get('scaler'),
                                    'feature_names': results.get('feature_names'),
                                    'pca_model': results.get('pca_model'),
                                    'pca_full': results.get('pca_full'),
                                    'X_pca': results.get('X_pca'),
                                    'pca_df': results.get('pca_df')
                                }
                                st.session_state['pca_ready'] = True
                                st.success("PCA run complete — visuals available in Overview and Metrics.")
                        except Exception as e:
                            st.error(f"PCA run failed: {e}")
            elif selected_algo == "Multivariate Linear Regression":
                with st.expander("Multivariate Non-linear Regression settings", expanded=True):
                    degree = st.slider("Polynomial degree", min_value=1, max_value=4, value=2)
                    alpha = st.number_input("Ridge alpha", min_value=0.0, value=1.0, format="%.2f")
                    st.write({"degree": degree, "alpha": alpha})

                if not _MVNL_AVAILABLE:
                    st.warning("Multivariate module not available (missing deps).")
                else:
                    if st.button("Run Multivariate Non-linear model"):
                        st.info("Training multivariate polynomial Ridge model...")
                        try:
                            data_path = os.path.join(os.getcwd(), "housing.csv")
                            m_metrics, m_artifacts = multivariate_nonlinear.train_and_evaluate(data_path, degree=degree, alpha=alpha)
                            st.success("Training complete")
                            st.session_state['mvnl_metrics'] = m_metrics
                            st.session_state['mvnl_artifacts'] = m_artifacts
                            st.session_state['mvnl_plot'] = m_artifacts.get('plot_png')
                            st.subheader("Model metrics (test)")
                            try:
                                import pandas as _pd
                                if isinstance(m_metrics, dict):
                                    mm_df = _pd.DataFrame.from_dict(m_metrics, orient='index', columns=['value']).reset_index().rename(columns={'index':'metric'})
                                    mm_df['value'] = mm_df['value'].apply(lambda x: round(x,4) if isinstance(x, float) else x)
                                    st.table(mm_df)
                                else:
                                    st.write(m_metrics)
                            except Exception:
                                # Fallback to plain write if something unexpected occurs
                                st.write(m_metrics)
                        except FileNotFoundError:
                            st.error("housing.csv not found in project root.")
                # Removed stray PCA settings from inside the Multivariate block.

            else:
                st.info("Algorithm configuration will appear here.")

                st.markdown("---")
                st.button("Save configuration 💾")

        # Prediction Tab
        with tabs[2]:
            st.header("Prediction")

            # Decision Tree prediction UI
            if selected_algo == "Decision Tree Classifier":
                st.markdown("### Decision Tree — Predict price_range (low/medium/high)")
                st.markdown("You can adjust how price ranges are created by changing the lower/upper percentile cutoffs.\n"
                            "After retraining, provide features and click Predict to see the class label.")

                import pandas as _pd

                default_path = os.path.join(os.getcwd(), "housing.csv")
                if not os.path.exists(default_path):
                    st.error("housing.csv not found in project root. Add the file to use Decision Tree predictions.")
                else:
                    # percentile controls
                    col_a, col_b = st.columns(2)
                    with col_a:
                        lower_pct = st.slider("Lower percentile for 'low' cutoff", min_value=5, max_value=45, value=33)
                    with col_b:
                        upper_pct = st.slider("Upper percentile for 'high' cutoff", min_value=55, max_value=95, value=66)

                    retrain = st.button("Retrain Decision Tree with these cutoffs")

                    # If user retrains, compute q thresholds and retrain
                    if retrain:
                        try:
                            df_all = _pd.read_csv(default_path)
                            q1_val = float(df_all['median_house_value'].quantile(lower_pct / 100.0))
                            q2_val = float(df_all['median_house_value'].quantile(upper_pct / 100.0))
                            st.info(f"Retraining with q1={q1_val:.2f} (pct={lower_pct}), q2={q2_val:.2f} (pct={upper_pct})")
                            dt_metrics, dt_artifacts = decision_tree.train_and_evaluate(
                                default_path, max_depth=5, q1_override=q1_val, q2_override=q2_val
                            )
                            st.success("Retrain complete")
                            st.session_state['dt_metrics'] = dt_metrics
                            st.session_state['dt_artifacts'] = dt_artifacts
                            st.session_state['dt_confusion_png'] = decision_tree.plot_confusion_matrix(dt_artifacts)
                            st.session_state['dt_tree_png'] = decision_tree.plot_tree_image(dt_artifacts)
                            # store chosen cutoffs so prediction uses same bins
                            st.session_state['dt_bins'] = {'q1': q1_val, 'q2': q2_val, 'pct1': lower_pct, 'pct2': upper_pct}
                        except Exception as e:
                            st.error(f"Retrain failed: {e}")

                    # If artifacts exist, show quick prediction UI
                    if 'dt_artifacts' in st.session_state:
                        art = st.session_state['dt_artifacts']
                        model = art.get('model')
                        features = art.get('preprocessing', {}).get('features_used', [])

                        st.markdown("#### Single example prediction")
                        lng = st.number_input("longitude", value=-122.3, format="%.3f")
                        lat = st.number_input("latitude", value=37.9, format="%.3f")
                        income = st.number_input("median_income", value=3.0, format="%.2f")

                        if st.button("Predict DT 🔮"):
                            try:
                                # build a single-row input using medians for missing features
                                df_ref = art.get('df')
                                row = {}
                                medians = df_ref[features].median()
                                for f in features:
                                    row[f] = medians.get(f, 0.0)
                                # overwrite provided features if present in feature set
                                if 'longitude' in features:
                                    row['longitude'] = float(lng)
                                if 'latitude' in features:
                                    row['latitude'] = float(lat)
                                if 'median_income' in features:
                                    row['median_income'] = float(income)

                                x_df = _pd.DataFrame([row], columns=features)
                                pred = model.predict(x_df)[0]
                                emoji_map = {'high': '🤑', 'medium': '🏠', 'low': '🪙'}
                                st.success(f"Predicted class: {pred} {emoji_map.get(pred, '')}")
                            except Exception as e:
                                st.error(f"Prediction failed: {e}")

                        # Show example predictions sampled from retrained df
                        with st.expander("Example Predictions", expanded=True):
                            try:
                                df_full = art.get('df')
                                bins = art.get('preprocessing', {}).get('bins', {})
                                # sample one representative from each class if available
                                rows = []
                                for cls in ['high', 'medium', 'low']:
                                    sub = df_full[df_full['price_range'] == cls]
                                    if len(sub) > 0:
                                        r = sub.sample(1).iloc[0]
                                        rows.append((r['longitude'], r['latitude'], r.get('median_income', np.nan), cls))

                                if rows:
                                    ex_df = _pd.DataFrame(rows, columns=['longitude', 'latitude', 'median_income', 'predicted'])
                                    # map emoji for display
                                    ex_df['predicted_display'] = ex_df['predicted'].map({'high': 'high 🤑', 'medium': 'medium 🏠', 'low': 'low 🪙'})
                                    st.table(ex_df[['longitude', 'latitude', 'median_income', 'predicted_display']])
                                else:
                                    st.write("No example rows available in dataset.")
                            except Exception as e:
                                st.write(f"Could not generate examples: {e}")

                    else:
                        st.info("No Decision Tree model available. Retrain the Decision Tree (Model Configuration tab) or use the retrain button above.")

            elif selected_algo == "Support Vector Machine":
                st.markdown("### Predict with Support Vector Machine (SVC)")
                if 'svm_artifacts' not in st.session_state:
                    st.warning("Please train the SVM model on the 'Model Configuration' tab first.")
                else:
                    svm_artifacts = st.session_state['svm_artifacts']
                    # Support both old 'model' and new 'pipeline' keys for backward compatibility
                    model = svm_artifacts.get('pipeline') or svm_artifacts.get('model')
                    scaler = svm_artifacts['scaler']
                    feature_names = svm_artifacts['features']  # Full list of features after one-hot encoding
                    
                    st.info("Provide input values. The model uses engineered features (rooms_per_person, bedrooms_per_room, population_per_household).")
                    
                    # Define base features needed for engineering
                    st.write("##### Base Features (for feature engineering)")
                    user_inputs = {}
                    
                    cols = st.columns(3)
                    with cols[0]:
                        user_inputs['housing_median_age'] = st.number_input(
                            "Housing Median Age (years)", 
                            min_value=1.0, max_value=100.0, value=30.0, step=1.0, format="%.1f"
                        )
                        user_inputs['total_rooms'] = st.number_input(
                            "Total Rooms", 
                            min_value=100.0, max_value=50000.0, value=2000.0, step=100.0, format="%.0f"
                        )
                    with cols[1]:
                        user_inputs['total_bedrooms'] = st.number_input(
                            "Total Bedrooms", 
                            min_value=10.0, max_value=10000.0, value=400.0, step=50.0, format="%.0f"
                        )
                        user_inputs['population'] = st.number_input(
                            "Population", 
                            min_value=50.0, max_value=50000.0, value=1000.0, step=100.0, format="%.0f"
                        )
                    with cols[2]:
                        user_inputs['households'] = st.number_input(
                            "Households", 
                            min_value=10.0, max_value=10000.0, value=400.0, step=50.0, format="%.0f"
                        )
                        user_inputs['median_income'] = st.number_input(
                            "Median Income (x$10k)", 
                            min_value=0.5, max_value=15.0, value=3.5, step=0.5, format="%.2f"
                        )
                    
                    st.write("##### Categorical Feature")
                    ocean_proximity_categories = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
                    user_inputs['ocean_proximity'] = st.selectbox(
                        "Ocean Proximity",
                        options=ocean_proximity_categories,
                        index=0,
                        key="svm_input_ocean"
                    )
                    
                    if st.button("Predict SVM Class 🔮", key="predict_svm"):
                        try:
                            import pandas as _pd
                            
                            # Apply feature engineering (same as in training)
                            df_input = _pd.DataFrame([user_inputs])
                            df_input['rooms_per_person'] = df_input['total_rooms'] / df_input['population']
                            df_input['bedrooms_per_room'] = df_input['total_bedrooms'] / df_input['total_rooms']
                            df_input['population_per_household'] = df_input['population'] / df_input['households']
                            
                            # Drop the original columns (same as in training)
                            df_input = df_input.drop(columns=['total_rooms', 'total_bedrooms', 'population', 'households'])
                            
                            # One-hot encode the categorical feature
                            input_df_encoded = _pd.get_dummies(df_input, columns=['ocean_proximity'], drop_first=False)
                            
                            # Reindex to match the model's training columns
                            input_df_final = input_df_encoded.reindex(columns=feature_names, fill_value=0)
                            
                            # Apply the saved scaler
                            input_scaled = scaler.transform(input_df_final)
                            
                            # 5. Make prediction
                            prediction = model.predict(input_scaled)
                            
                            st.success(f"Predicted Price Category: **{prediction[0]}**")
                            
                        except Exception as e:
                            st.error(f"An error occurred during prediction: {e}")
                            st.exception(e)  # Show full error details

            elif selected_algo == "PCA":
                st.markdown("### PCA — Project a custom input to principal components")
                if 'pca_artifacts' not in st.session_state:
                    st.warning("Please run PCA from Model Configuration or Run Model to enable PCA Predictions.")
                else:
                    artifacts = st.session_state['pca_artifacts']
                    scaler = artifacts.get('scaler')
                    feature_names = artifacts.get('feature_names')  # processed feature columns after get_dummies
                    pca_model = artifacts.get('pca_model')
                    # Load original dataset to get numeric feature defaults and categories
                    import pandas as _pd
                    default_path = os.path.join(os.getcwd(), "housing.csv")
                    if not os.path.exists(default_path):
                        st.error("housing.csv not found in project root. Cannot build input form.")
                    else:
                        df_full = _pd.read_csv(default_path)
                        X_raw = df_full.drop(columns=['median_house_value'], errors='ignore')
                        # numeric columns for user input
                        numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
                        # categorical columns (expect 'ocean_proximity')
                        cat_cols = X_raw.select_dtypes(include=['object','category']).columns.tolist()
                        st.write("Provide values for the base features (others will use median defaults).")
                        user_vals = {}
                        cols = st.columns(3)
                        for i, col in enumerate(numeric_cols):
                            with cols[i % 3]:
                                median_val = float(X_raw[col].median())
                                user_vals[col] = st.number_input(col, value=median_val, format="%.3f", key=f"pca_in_{col}")
                        # categorical inputs
                        for cat in cat_cols:
                            options = sorted(X_raw[cat].dropna().unique().tolist())
                            user_vals[cat] = st.selectbox(cat, options=options, index=0, key=f"pca_in_{cat}")
                        if st.button("Project to PCA 🔮", key="predict_pca"):
                            try:
                                # Build single-row df, apply same preprocessing as training
                                input_df = _pd.DataFrame([user_vals])
                                # Impute total_bedrooms if present
                                if 'total_bedrooms' in input_df.columns:
                                    input_df['total_bedrooms'] = input_df['total_bedrooms'].fillna(X_raw['total_bedrooms'].median())
                                # One-hot encode categorical columns using same approach (drop_first=True used in training)
                                input_encoded = _pd.get_dummies(input_df, columns=['ocean_proximity'], drop_first=True)
                                # Reindex to match feature_names
                                input_aligned = input_encoded.reindex(columns=feature_names, fill_value=0)
                                # Scale if scaler available
                                if scaler is not None:
                                    input_scaled = scaler.transform(input_aligned)
                                else:
                                    input_scaled = input_aligned.values.astype(float)
                                # Project
                                pc = pca_model.transform(input_scaled)
                                pc1 = float(pc[0, 0]) if pc.shape[1] >= 1 else None
                                pc2 = float(pc[0, 1]) if pc.shape[1] >= 2 else None
                                st.success(f"Projected coordinates: PC1={pc1:.3f}" + (f", PC2={pc2:.3f}" if pc2 is not None else ""))
                                # Overlay on PCA scatter if figs exist
                                if st.session_state.get('pca_figs'):
                                    import matplotlib.pyplot as plt
                                    import io
                                    fig = None
                                    try:
                                        # copy original scatter fig (assumed at index 1)
                                        orig_fig = st.session_state['pca_figs'][1] if len(st.session_state['pca_figs']) > 1 else st.session_state['pca_figs'][0]
                                        # create a new fig by plotting the original data points again and overlay
                                        # We will reuse pca_df if available
                                        pca_df = artifacts.get('pca_df')
                                        if pca_df is not None and 'PC1' in pca_df.columns and 'PC2' in pca_df.columns:
                                            fig, ax = plt.subplots(figsize=(10,8))
                                            sc = ax.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df.get('median_house_value', None), cmap='jet', alpha=0.5)
                                            ax.scatter(pc1, pc2, c='red', s=200, marker='*', edgecolors='black')
                                            if 'median_house_value' in pca_df.columns:
                                                plt.colorbar(sc, label='Median House Value')
                                            ax.set_xlabel('PC1')
                                            ax.set_ylabel('PC2')
                                            ax.set_title('PCA projection with your input (red star)')
                                            buf = io.BytesIO()
                                            fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
                                            buf.seek(0)
                                            st.image(buf)
                                            plt.close(fig)
                                        else:
                                            st.info("No PCA scatter data available to overlay the projected point.")
                                    except Exception:
                                        st.info("Could not render overlay plot; showing numeric PC coordinates only.")
                            except Exception as e:
                                st.error(f"PCA prediction failed: {e}")

            elif selected_algo == "Ensemble Learning (Bagging/Boosting)":
                st.markdown("### Predict with Ensemble (Random Forest)")
                if 'ensemble_artifacts' not in st.session_state:
                    st.warning("Please train the Ensemble models on the 'Model Configuration' tab first.")
                else:
                    # We only use the BEST model (Random Forest) for prediction
                    try:
                        ensemble_artifacts = st.session_state['ensemble_artifacts']['random_forest']
                        model = ensemble_artifacts['model']
                        feature_names = ensemble_artifacts['features']
                    except KeyError:
                        st.error("Model artifacts are not in the correct format. Please retrain the model.")
                    else:
                        st.info(f"Using the Random Forest model. Provide values for all {len(feature_names)} features.")
                        
                        # Define original features (same as SVM but with engineered features)
                        numeric_features = [
                            'housing_median_age', 'median_income',
                            'rooms_per_person', 'bedrooms_per_room', 'population_per_household'
                        ]
                        categorical_feature = 'ocean_proximity'
                        ocean_proximity_categories = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
                        
                        user_inputs = {}
                        
                        st.write("##### Numeric Features")
                        cols = st.columns(3)
                        with cols[0]:
                            user_inputs['housing_median_age'] = st.number_input(
                                "Housing Median Age (years)", 
                                min_value=1.0, max_value=100.0, value=30.0, step=1.0, format="%.1f",
                                key="ensemble_input_age"
                            )
                            user_inputs['median_income'] = st.number_input(
                                "Median Income (x$10k)", 
                                min_value=0.5, max_value=15.0, value=3.5, step=0.5, format="%.2f",
                                key="ensemble_input_income"
                            )
                        with cols[1]:
                            user_inputs['rooms_per_person'] = st.number_input(
                                "Rooms per Person", 
                                min_value=0.1, max_value=50.0, value=2.0, step=0.1, format="%.2f",
                                key="ensemble_input_rpp"
                            )
                            user_inputs['bedrooms_per_room'] = st.number_input(
                                "Bedrooms per Room", 
                                min_value=0.01, max_value=1.0, value=0.2, step=0.01, format="%.2f",
                                key="ensemble_input_bpr"
                            )
                        with cols[2]:
                            user_inputs['population_per_household'] = st.number_input(
                                "Population per Household", 
                                min_value=0.5, max_value=20.0, value=2.5, step=0.1, format="%.2f",
                                key="ensemble_input_pph"
                            )
                        
                        st.write("##### Categorical Feature")
                        user_inputs[categorical_feature] = st.selectbox(
                            "Ocean Proximity",
                            options=ocean_proximity_categories,
                            index=0,
                            key="ensemble_input_ocean"
                        )
                        
                        if st.button("Predict Ensemble Class 🔮", key="predict_ensemble"):
                            try:
                                import pandas as _pd
                                
                                # 1. Create a single-row DataFrame from user inputs
                                input_df_single_row = _pd.DataFrame([user_inputs])
                                
                                # 2. One-hot encode the categorical feature
                                input_df_encoded = _pd.get_dummies(input_df_single_row, columns=[categorical_feature], drop_first=False)
                                
                                # 3. Reindex to match the model's training columns
                                input_df_final = input_df_encoded.reindex(columns=feature_names, fill_value=0)
                                
                                # 4. Make prediction (NO SCALER NEEDED)
                                prediction = model.predict(input_df_final)
                                
                                # Display prediction with color coding
                                price_category = prediction[0]
                                if price_category == 'Low':
                                    st.success(f"Predicted Price Category: **{price_category}** 🪙")
                                elif price_category == 'Medium':
                                    st.info(f"Predicted Price Category: **{price_category}** 🏠")
                                else:  # High
                                    st.warning(f"Predicted Price Category: **{price_category}** 🤑")
                                
                            except Exception as e:
                                st.error(f"An error occurred during prediction: {e}")
                                st.exception(e)  # Show full error details

            # DBSCAN Clustering - Analytical Only
            elif selected_algo == "DBSCAN Clustering":
                st.markdown("### DBSCAN Clustering — Analytical (No Prediction)")
                
                st.info("""
                ℹ️ **DBSCAN is an analytical clustering algorithm, not a predictive model.**
                
                **Why DBSCAN doesn't support prediction:**
                - DBSCAN identifies dense regions in **existing data**
                - It doesn't create a model that generalizes to new points
                - Each clustering run is specific to the dataset analyzed
                - New points would require re-running the entire algorithm
                
                **For predictive clustering, use:**
                - **K-Means Clustering** - Assigns new points to nearest centroid
                - **Decision Tree/Random Forest** - Predicts specific categories
                
                **DBSCAN is best for:**
                - Exploratory data analysis
                - Outlier/anomaly detection
                - Identifying dense geographic regions
                - Understanding spatial patterns in your dataset
                """)
                
                if 'dbscan_results' in st.session_state:
                    st.divider()
                    st.write("#### Current Analysis Summary")
                    
                    results = st.session_state['dbscan_results']
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Clusters Found", results['n_clusters'])
                    col2.metric("Noise Points", f"{results['n_noise']:,}")
                    col3.metric("Noise Ratio", f"{results['noise_ratio']:.1f}%")
                    
                    st.write("**Parameters Used:**")
                    st.write(f"- Eps: {results['eps']:.4f}")
                    st.write(f"- Min Samples: {results['min_samples']}")
                    st.write(f"- Features: {', '.join(results['feature_names'])}")
                    
                    st.divider()
                    st.success("💡 To analyze your clusters, go to the 'Overview / Visualization' tab!")
                else:
                    st.warning("No DBSCAN analysis found. Run the analysis from Model Configuration first.")
            
            # K-Means Clustering prediction UI
            elif selected_algo == "K-Means Clustering":
                st.markdown("### K-Means Clustering — Assign to Cluster")
                st.markdown("Provide feature values to predict which cluster the data point belongs to.")
                
                if 'kmeans_results' not in st.session_state:
                    st.info("No K-Means model found. Run the K-Means clustering analysis first from Model Configuration.")
                else:
                    results = st.session_state['kmeans_results']
                    model = results['final_model']
                    feature_names = results['feature_names']
                    
                    st.write(f"**Number of Clusters:** {results['optimal_k']}")
                    st.write(f"**Features Used:** {', '.join(feature_names)}")
                    
                    st.divider()
                    
                    # Create input fields for all features
                    st.markdown("#### Enter Feature Values")
                    st.info("💡 Tip: Use the median values from your dataset as a starting point, then modify as needed.")
                    
                    # Get median values from original data for defaults
                    df_ref = results.get('df_full')
                    
                    cols = st.columns(3)
                    inputs = {}
                    
                    for i, feature in enumerate(feature_names):
                        col = cols[i % 3]
                        if df_ref is not None and feature in df_ref.columns:
                            default_val = float(df_ref[feature].median())
                        else:
                            default_val = 0.0
                        
                        # Format label nicely
                        label = feature.replace('_', ' ').title()
                        inputs[feature] = col.number_input(label, value=default_val, format="%.4f", key=f"kmeans_{feature}")
                    
                    st.divider()
                    
                    if st.button("🔮 Predict Cluster Assignment", type="primary"):
                        try:
                            # Create input array
                            input_values = [inputs[f] for f in feature_names]
                            input_array = np.array([input_values])
                            
                            # Scale the input using the same scaler
                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            scaler.fit(results['X_scaled'])  # Fit on the original scaled data
                            
                            # Actually, the X_scaled is already scaled, so we need to scale our new input
                            # We need to fit scaler on original unscaled data
                            # Let me get the original X from prepare_data
                            import pandas as pd
                            from algorithms.decision_tree import prepare_data
                            
                            data_path = os.path.join(os.getcwd(), "housing.csv")
                            X_original, _, _, _ = prepare_data(data_path)
                            
                            # Handle NaN
                            if X_original.isnull().any().any():
                                mask = ~X_original.isnull().any(axis=1)
                                X_original = X_original[mask]
                            
                            # Fit scaler on original data
                            scaler = StandardScaler()
                            scaler.fit(X_original)
                            
                            # Create input DataFrame
                            input_df = pd.DataFrame([inputs], columns=feature_names)
                            
                            # Scale the input
                            input_scaled = scaler.transform(input_df)
                            
                            # Predict cluster
                            cluster_id = model.predict(input_scaled)[0]
                            
                            # Get distance to all centroids
                            distances = model.transform(input_scaled)[0]
                            
                            st.success(f"### 🎯 Predicted Cluster: **{cluster_id}**")
                            
                            # Show cluster information
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Assigned Cluster", cluster_id)
                            
                            with col2:
                                cluster_size = np.sum(results['cluster_labels'] == cluster_id)
                                total_size = len(results['cluster_labels'])
                                st.metric("Cluster Size", f"{cluster_size:,} ({cluster_size/total_size*100:.1f}%)")
                            
                            with col3:
                                st.metric("Distance to Centroid", f"{distances[cluster_id]:.4f}")
                            
                            st.divider()
                            
                            # Show distances to all centroids
                            st.markdown("#### 📏 Distances to All Cluster Centroids")
                            distance_df = pd.DataFrame({
                                'Cluster ID': range(results['optimal_k']),
                                'Distance': distances,
                                'Is Closest': ['✅ Yes' if i == cluster_id else '' for i in range(results['optimal_k'])]
                            })
                            distance_df = distance_df.sort_values('Distance')
                            st.dataframe(distance_df, use_container_width=True, hide_index=True)
                            
                            # Show cluster characteristics
                            st.divider()
                            st.markdown("#### 🔍 Cluster Characteristics")
                            
                            # Get centroid for this cluster
                            centroid = results['centroids'][cluster_id]
                            
                            # Create comparison
                            comparison_data = {
                                'Feature': feature_names,
                                'Your Input': [inputs[f] for f in feature_names],
                                'Cluster Centroid (scaled)': centroid,
                            }
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                            
                            # Show which price category is dominant in this cluster
                            st.divider()
                            st.markdown("#### 💰 Price Category Analysis for This Cluster")
                            
                            cluster_mask = results['cluster_labels'] == cluster_id
                            cluster_prices = results['y'][cluster_mask]
                            
                            price_counts = cluster_prices.value_counts()
                            price_percentages = (price_counts / len(cluster_prices) * 100).round(1)
                            
                            price_df = pd.DataFrame({
                                'Price Category': price_counts.index,
                                'Count': price_counts.values,
                                'Percentage': [f"{p}%" for p in price_percentages.values]
                            })
                            
                            st.dataframe(price_df, use_container_width=True, hide_index=True)
                            
                            # Dominant category
                            dominant_category = price_counts.index[0]
                            dominant_pct = price_percentages.iloc[0]
                            
                            emoji_map = {'high': '🤑', 'medium': '🏠', 'low': '🪙'}
                            st.info(f"**Dominant Price Category:** {dominant_category.upper()} {emoji_map.get(dominant_category, '')} ({dominant_pct}% of cluster)")
                            
                        except Exception as e:
                            st.error(f"Cluster prediction failed: {e}")
                            import traceback
                            st.code(traceback.format_exc())

            else:
                # keep existing LR prediction UI
                if "latest_artifacts" not in st.session_state:
                    st.info("No trained model found. Run Linear Regression first from Quick actions or Model Configuration.")
                else:
                    art = st.session_state["latest_artifacts"]
                    model = art.get("model")

                    income_input = st.number_input("Enter median_income", value=3.85, format="%.2f")
                    if st.button("Predict 🔮"):
                        try:
                            arr = np.array([[float(income_input)]])
                            pred = model.predict(arr)[0]
                            st.success(f"Predicted median_house_value: ${pred:,.2f}")
                            st.write(f"Using model: intercept={art.get('intercept'):.4f}, slope={art.get('slope'):.4f}")
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")

            # Multivariate prediction UI
            if selected_algo == "Multivariate Linear Regression":
                st.markdown("### Multivariate Non-linear Model — Single prediction")
                if "mvnl_artifacts" not in st.session_state:
                    st.info("No multivariate model found. Run the model from Model Configuration or use Quick-run in Overview.")
                else:
                    mv = st.session_state['mvnl_artifacts']
                    features = mv.get('numeric_features', [])
                    cat_feats = mv.get('categorical_features', [])

                    st.markdown("Provide input values for the features below (missing features will use dataset medians).")
                    cols = st.columns(3)
                    inputs = {}
                    for i, f in enumerate(features):
                        col = cols[i % 3]
                        # reasonable default: median from training df
                        df_ref = mv.get('df')
                        default_val = df_ref[f].median() if f in df_ref.columns else 0.0
                        inputs[f] = col.number_input(f, value=float(default_val))

                    # categorical
                    cat_inputs = {}
                    for cf in cat_feats:
                        df_ref = mv.get('df')
                        choices = df_ref[cf].dropna().unique().tolist() if cf in df_ref.columns else []
                        if choices:
                            cat_inputs[cf] = st.selectbox(cf, options=choices)

                    if st.button("Predict Multivariate 🔮"):
                        try:
                            import pandas as _pd
                            row = {f: inputs[f] for f in features}
                            for cf, val in cat_inputs.items():
                                row[cf] = val
                            x_df = _pd.DataFrame([row])
                            model = mv.get('model_pipeline')
                            pred = model.predict(x_df)[0]
                            st.success(f"Predicted median_house_value: ${pred:,.2f}")
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")

    # Footer
    st.markdown("---")
    st.caption("© ML Mini-Project.")


if __name__ == "__main__":
    main()
