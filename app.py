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
            "This app is a UI skeleton — ML logic will be plugged in later."
        )

        # No upload option — app uses the bundled sample dataset by default
        dataset_mode = "Use sample dataset"
        uploaded_file = None

        st.markdown("---")
        st.markdown("Built with ❤️ • Streamlit UI skeleton")
    return algo, dataset_mode, uploaded_file


def main():
    # Available algorithms
    algorithms = [
        "Linear Regression",
        "Multivariate Linear Regression",
        "Decision Tree Classifier",
        "SVM",
        "Random Forest",
        "K-Means",
        "DBSCAN",
        "PCA/SVD",
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
        show_gallery = st.checkbox("Visualization Gallery (full dashboard)", value=st_session["show_gallery"])
        st_session["show_gallery"] = show_gallery

    # Header area
    st.title("Machine Learning Dashboard 📊")
    st.markdown(f"### You selected: **{selected_algo}**")

    st.markdown(
        "---\n"
        "This interface provides a clean layout for exploring datasets, configuring models, and making predictions."
    )

    # If the Linear Regression module failed to import (usually missing matplotlib or sklearn), show guidance
    if not _LR_AVAILABLE:
        st.error(
            "One or more optional dependencies failed to import (needed for algorithms).\n"
            "Error: {}\n\n"
            "To fix, install required packages: `pip install -r requirements.txt`\n"
            "Then restart the app.".format(str(_LR_IMPORT_ERROR))
        )

    # Quick actions: show a compact control on the main page when specific algorithms are selected
    if selected_algo == "Linear Regression":
        with st.expander("Quick actions — Linear Regression", expanded=False):
            st.write("Run a quick linear regression using `median_income` → `median_house_value`.\n"
                     "You can also run the model from the Model Configuration tab.")

            if not _LR_AVAILABLE:
                st.warning("Linear Regression is unavailable because some dependencies failed to import.\n"
                           "Run: `pip install -r requirements.txt` and restart the app.")
            else:
                if st.button("Run Linear Regression (quick)"):
                    st.info("Training Linear Regression...")
                    # prefer uploaded file if present, otherwise use bundled housing.csv
                    if uploaded_file is not None:
                        data_path = os.path.join(os.getcwd(), "uploaded_housing.csv")
                        with open(data_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    else:
                        data_path = os.path.join(os.getcwd(), "housing.csv")

                    try:
                        metrics, artifacts = linear_regression.train_and_evaluate(data_path)
                        st.success("Training complete")
                        st.subheader("Performance Metrics")
                        # Show intermediate info
                        st.write(f"Records used (feature -> target): {len(artifacts['X_test']) + 0} test records")
                        st.write(f"Train/test split: {len(artifacts['X_test'])} test records")
                        st.write(f"Fitted line: {target_col if 'target_col' in locals() else 'median_house_value'} = {artifacts.get('intercept'):.4f} + ({artifacts.get('slope'):.4f} * median_income)")

                        # Show metrics summary
                        st.subheader("Performance Metrics (summary)")
                        for k, v in metrics.items():
                            if k == "MSE":
                                st.write(f"{k}: {v:,.2f}")
                            else:
                                st.write(f"{k}: {v:.4f}")

                        # Generate and store plot for Overview but do not display here
                        png = linear_regression.plot_results(artifacts)
                        st.session_state["latest_metrics"] = metrics
                        st.session_state["latest_plot_lr"] = png
                        st.session_state["latest_artifacts"] = artifacts
                    except FileNotFoundError:
                        st.error("Dataset not found. Place 'housing.csv' in the project root or upload a CSV.")

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

                        # Suggestions for improving performance
                        with st.expander("Suggested preprocessing steps to improve accuracy", expanded=False):
                            st.markdown("- Handle missing values (imputation)\n- Scale numeric features (StandardScaler/MinMax)\n- Encode categorical variables if present (OneHot/Ordinal)\n- Feature engineering (ratios, interactions)\n- Remove or cap outliers\n- Try class balancing (SMOTE/undersampling) if classes are imbalanced\n- Perform feature selection (recursive or model-based)")
                    except FileNotFoundError:
                        st.error("housing.csv not found in project root.")
    if show_gallery:
        st.header("Visualization Gallery 🎨📊")
        st.markdown("A collection of visualizations for exploratory analysis. Click expanders for details.")

        # Grid of visual placeholders
        for i in range(0, 6, 2):
            c1, c2 = st.columns(2)
            with c1:
                st.subheader(f"Visualization {i+1}")
                ph = st.empty()
                ph.info("Graph Placeholder")
            with c2:
                st.subheader(f"Visualization {i+2}")
                ph2 = st.empty()
                ph2.info("Graph Placeholder")

        st.markdown("---")
        st.info("Gallery mode: use the sidebar to switch back to tabbed view or choose an algorithm.")
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
                graph_placeholder = st.empty()
                # If Decision Tree artifacts exist or current selection is DT, show DT visuals
                if (selected_algo == "Decision Tree Classifier" and "dt_artifacts" in st.session_state) or (
                    "dt_artifacts" in st.session_state
                ):
                    # Always offer heatmap as a visualization option; show guidance if no grid results exist
                    dt_options = ["confusion_matrix", "tree"]

                    dt_choice = st.selectbox("DT visualization", options=dt_options, index=0)
                    if dt_choice == "confusion_matrix":
                        if "dt_confusion_png" in st.session_state:
                            graph_placeholder.image(st.session_state["dt_confusion_png"], use_column_width=True)
                        else:
                            graph_placeholder.info("Confusion matrix not available yet. Run the Decision Tree model.")
                    elif dt_choice == "heatmap":
                        # Prefer stored heatmap bytes
                        if "dt_grid_heatmap_png" in st.session_state:
                            graph_placeholder.image(st.session_state["dt_grid_heatmap_png"], use_column_width=True)
                        elif "dt_grid_info" in st.session_state:
                            # attempt to regenerate from stored cv_results
                            try:
                                cvr = st.session_state["dt_grid_info"].get("cv_results")
                                if cvr and isinstance(cvr.get("params"), list) and len(cvr.get("params")) > 0:
                                    keys = list(cvr.get("params")[0].keys())
                                    if len(keys) >= 2:
                                        heat_png = decision_tree.plot_cv_heatmap(cvr, keys[0], keys[1])
                                        graph_placeholder.image(heat_png, use_column_width=True)
                                    else:
                                        graph_placeholder.info("Heatmap unavailable: GridSearch used fewer than 2 tunable parameters.")
                                else:
                                    graph_placeholder.info("GridSearchCV results not available. Run GridSearch first from Model Configuration.")
                            except Exception as e:
                                graph_placeholder.error(f"Failed to generate heatmap: {e}")
                        else:
                            graph_placeholder.info("Heatmap not available. Run GridSearch first from Model Configuration.")
                    else:
                        if "dt_tree_png" in st.session_state:
                            graph_placeholder.image(st.session_state["dt_tree_png"], use_column_width=True)
                        else:
                            graph_placeholder.info("Decision tree image not available yet. Run the Decision Tree model.")
                else:
                    # Visualization type selector (defaults to scatter) for regression
                    plot_type = st.selectbox("Plot type", options=["scatter", "residual", "hist"], index=0)

                    if "latest_artifacts" in st.session_state and _LR_AVAILABLE:
                        # Render chosen plot type from stored artifacts
                        art = st.session_state["latest_artifacts"]
                        png = linear_regression.plot_results(art, plot_type=plot_type)
                        graph_placeholder.image(png, use_column_width=True)
                    elif "latest_plot_lr" in st.session_state:
                        # fallback to the previously generated default plot
                        graph_placeholder.image(st.session_state["latest_plot_lr"], use_column_width=True)
                    else:
                        # Placeholder Plotly/Matplotlib chart can be rendered here later
                        graph_placeholder.info("Graph Placeholder — Plotly charts will be placed here.")

            with col2:
                st.markdown("#### Metrics & Model Info")
                metrics_placeholder = st.empty()
                if "latest_metrics" in st.session_state:
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
                # If decision tree metrics exist, show them as a table
                if "dt_metrics" in st.session_state:
                    dtm = st.session_state["dt_metrics"]
                    st.markdown("---")
                    st.markdown("**Decision Tree Classifier Metrics**")
                    import pandas as _pd

                    dt_df = _pd.DataFrame.from_dict(dtm, orient="index", columns=["value"]) \
                        .reset_index().rename(columns={"index": "metric"})
                    dt_df["value"] = dt_df["value"].apply(lambda x: round(x, 4) if isinstance(x, float) else x)
                    st.table(dt_df)

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
                                        st.image(heat_png, caption="GridSearchCV heatmap (mean_test_score)", use_column_width=True)
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

            elif selected_algo == "SVM":
                with st.expander("SVM settings", expanded=True):
                    c_val = st.slider("C (regularization)", min_value=0.01, max_value=10.0, value=1.0)
                    kernel = st.selectbox("Kernel", options=["rbf", "linear", "poly", "sigmoid"])  # placeholder
                    st.write("C:", c_val, "kernel:", kernel)

            elif selected_algo == "Random Forest":
                with st.expander("Random Forest settings", expanded=True):
                    n_estimators = st.slider("n_estimators", min_value=10, max_value=1000, value=100, step=10)
                    max_features = st.selectbox("max_features", options=["auto", "sqrt", "log2"])  # placeholder
                    st.write({"n_estimators": n_estimators, "max_features": max_features})

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

            elif selected_algo == "PCA/SVD":
                with st.expander("PCA/SVD settings", expanded=True):
                    n_components = st.slider("n_components", min_value=1, max_value=50, value=5)
                    svd_solver = st.selectbox("svd_solver", options=["auto", "full", "arpack", "randomized"])  # placeholder
                    st.write({"n_components": n_components, "svd_solver": svd_solver})

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

    # Footer
    st.markdown("---")
    st.caption("© ML Mini-Project.")


if __name__ == "__main__":
    main()
