import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.ensemble import IsolationForest

# 1. Page Configuration
st.set_page_config(
    page_title="Financial Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Premium Custom CSS Integration (Glassmorphism & Sleek Dark theme)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Apply modern font globally */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif;
}

/* Glassmorphic Metrics Grid Container */
.metrics-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 1.2rem;
    margin-bottom: 2rem;
    margin-top: 1.5rem;
}

/* Individual Metrics Card */
.metric-card {
    flex: 1;
    min-width: 220px;
    background: rgba(22, 28, 41, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(180deg, #6366F1, #3B82F6);
}

.metric-card.alert::before {
    background: linear-gradient(180deg, #EF4444, #F43F5E);
}

.metric-card.success::before {
    background: linear-gradient(180deg, #10B981, #059669);
}

.metric-card:hover {
    transform: translateY(-6px);
    border-color: rgba(99, 102, 241, 0.35);
    box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.15);
}

.metric-card.alert:hover {
    border-color: rgba(239, 68, 68, 0.35);
    box-shadow: 0 12px 40px 0 rgba(239, 68, 68, 0.15);
}

.metric-title {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #94A3B8;
    margin-bottom: 0.6rem;
    font-weight: 600;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    line-height: 1.1;
    background: linear-gradient(135deg, #E2E8F0, #94A3B8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-value.indigo {
    background: linear-gradient(135deg, #818CF8, #38BDF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-value.red {
    background: linear-gradient(135deg, #F87171, #F43F5E);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-value.green {
    background: linear-gradient(135deg, #34D399, #059669);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-subtitle {
    font-size: 0.8rem;
    color: #64748B;
    margin-top: 0.6rem;
    font-weight: 400;
}

/* Dynamic Sidebar Control Box */
.sidebar-box {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* Style main title nicely */
.main-title {
    font-weight: 800;
    background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.sub-title {
    color: #64748B;
    font-weight: 400;
    margin-bottom: 2rem;
}

/* Beautiful custom divider */
.custom-hr {
    border: 0;
    height: 1px;
    background: linear-gradient(90deg, rgba(99, 102, 241, 0) 0%, rgba(99, 102, 241, 0.4) 50%, rgba(99, 102, 241, 0) 100%);
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)


# 3. High-Fidelity Synthetic Data Generator
def generate_synthetic_transactions(n_samples=1200):
    np.random.seed(42)
    # Generate continuous timestamp range in seconds (24h)
    time = np.sort(np.random.uniform(0, 86400, n_samples))
    
    # Amount following log-normal distribution (mostly small transactions with a few large ones)
    amount = np.random.lognormal(mean=3.8, sigma=0.9, size=n_samples)
    
    # Anonymized features representing latent patterns (PCA-like components V1, V2, V3)
    v1 = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
    v2 = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
    v3 = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
    
    # Ground truth labels (Class: 0 = Legit, 1 = Fraud)
    # Target fraud incidence: ~0.8%
    class_labels = np.zeros(n_samples, dtype=int)
    fraud_count = int(n_samples * 0.008)
    fraud_indices = np.random.choice(n_samples, size=fraud_count, replace=False)
    class_labels[fraud_indices] = 1
    
    # Correlate fraud anomalies
    # Fraud transactions generally involve unusually large amounts and outlier latent components
    amount[fraud_indices] = amount[fraud_indices] * np.random.uniform(4.0, 12.0, size=fraud_count)
    v1[fraud_indices] = v1[fraud_indices] - np.random.uniform(3.0, 7.5, size=fraud_count)
    v2[fraud_indices] = v2[fraud_indices] + np.random.uniform(3.0, 7.5, size=fraud_count)
    v3[fraud_indices] = v3[fraud_indices] - np.random.uniform(2.5, 6.0, size=fraud_count)
    
    df = pd.DataFrame({
        "Time": time.round(0),
        "Amount": amount.round(2),
        "V1": v1.round(4),
        "V2": v2.round(4),
        "V3": v3.round(4),
        "Merchant_Code": np.random.choice(["MC_728", "MC_102", "MC_991", "MC_334", "MC_048"], size=n_samples),
        "Class": class_labels
    })
    
    return df


# 4. Cached Data Handling for High Performance
@st.cache_data
def get_cached_synthetic_data(n_samples):
    return generate_synthetic_transactions(n_samples)

@st.cache_data
def load_uploaded_dataset(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data
def train_and_predict_anomaly(features, contamination, n_estimators):
    # Train Isolation Forest
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=42
    )
    predictions = model.fit_predict(features)
    # Raw anomaly scores (negative values; more negative = highly anomalous)
    raw_scores = model.score_samples(features)
    return predictions, raw_scores


# 5. Sidebar Control Center
st.sidebar.markdown("## 🔧 Analytics Control Panel")

data_source = st.sidebar.radio(
    "Select Transaction Data Source",
    ["Synthetic Data Stream (Recommended)", "Upload Custom CSV File"],
    help="Synthetic stream provides instant, ground-truth-labeled interactive scenarios."
)

# File Upload or Load Synthetic
if data_source == "Upload Custom CSV File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Financial Transaction CSV",
        type=["csv"],
        help="Upload a CSV with transactions. The app will automatically clean it and identify metrics."
    )
    if uploaded_file is not None:
        df_raw = load_uploaded_dataset(uploaded_file)
        data_loaded = True
    else:
        st.sidebar.info("Please upload a CSV file to begin.")
        data_loaded = False
else:
    # Synthetic generator size selector
    sample_size = st.sidebar.slider("Synthetic Stream Sample Size", 500, 3000, 1500, 100)
    df_raw = get_cached_synthetic_data(sample_size)
    data_loaded = True

if data_loaded:
    st.sidebar.markdown("<hr style='margin: 10px 0; opacity: 0.15;' />", unsafe_allow_html=True)
    st.sidebar.markdown("### ⚙️ Isolation Forest Hyperparameters")
    
    contamination = st.sidebar.slider(
        "Expected Contamination Rate",
        0.001, 0.05, 0.006, 0.001,
        format="%.3f",
        help="Proportion of outliers/fraud estimated to exist in the dataset."
    )
    
    n_estimators = st.sidebar.slider(
        "Model Estimators",
        50, 300, 150, 50,
        help="Number of decision trees inside the Isolation Forest forest."
    )


# 6. Main Dashboard Execution Flow
if data_loaded:
    df = df_raw.copy()
    
    # Header Section
    st.markdown('<h1 class="main-title">💳 Financial Fraud Analytics Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Advanced real-time anomaly detection using calibrated Isolation Forest modeling</p>', unsafe_allow_html=True)
    
    # Robust Preprocessing & Column Matching
    # Identify numeric columns for the model
    numeric_columns = list(df.select_dtypes(include=np.number).columns)
    
    # Exclude ground-truth label from fitting if present
    ground_truth_col = None
    for col in df.columns:
        if col.lower() in ["class", "fraud", "is_fraud", "label", "is_anomalous"]:
            ground_truth_col = col
            break
            
    if ground_truth_col and ground_truth_col in numeric_columns:
        numeric_columns.remove(ground_truth_col)
        
    if len(numeric_columns) > 0:
        # Preprocess features robustly: Handle infinite values and fill NaNs safely
        features_df = df[numeric_columns].copy()
        for col in features_df.columns:
            features_df[col] = features_df[col].replace([np.inf, -np.inf], np.nan)
            if features_df[col].isnull().any():
                col_median = features_df[col].median()
                features_df[col] = features_df[col].fillna(col_median if not np.isnan(col_median) else 0.0)
        
        # Fit model and predict (cached)
        predictions, raw_scores = train_and_predict_anomaly(features_df, contamination, n_estimators)
        
        # Format predictions (1 = Safe, -1 = Suspicious)
        df["Prediction"] = predictions
        df["Fraud_Flag"] = np.where(df["Prediction"] == -1, "Suspicious", "Legitimate")
        
        # Calibrate Fraud Risk Score (Scale raw scores to a clean 0% - 100% metric)
        # Isolation Forest score_samples is in range [-1.0, 0.0]. Lower is more anomalous.
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        if (max_score - min_score) > 1e-7:
            # Calibrate: min_score -> 100% Risk, max_score -> 0% Risk
            df["Fraud_Risk_Score"] = (100.0 * (max_score - raw_scores) / (max_score - min_score)).round(2)
        else:
            df["Fraud_Risk_Score"] = 0.0
            
        # Locate amount-like columns for metrics
        amount_col = None
        for col in df.columns:
            if col.lower() == "amount":
                amount_col = col
                break
        if not amount_col:
            for col in df.columns:
                if "amount" in col.lower():
                    amount_col = col
                    break
        if not amount_col and len(numeric_columns) > 0:
            amount_col = numeric_columns[0]
            
        # 7. Dynamic Metrics Grid Rendering
        total_tx = len(df)
        suspicious_tx = (df["Prediction"] == -1).sum()
        anomaly_ratio = (suspicious_tx / total_tx) * 100.0
        
        # Compute Amount metrics if available
        if amount_col:
            total_volume = df[amount_col].sum()
            flagged_volume = df[df["Prediction"] == -1][amount_col].sum()
            volume_text = f"${total_volume:,.2f}"
            flagged_volume_text = f"${flagged_volume:,.2f}"
        else:
            volume_text = "N/A"
            flagged_volume_text = "N/A"
            
        # Ground Truth check
        ground_truth_available = ground_truth_col is not None
        match_rate_html = ""
        
        if ground_truth_available:
            # Ensure ground truth is binary integer
            df[ground_truth_col] = df[ground_truth_col].astype(int)
            # Alignment: Predicted fraud is 1 (where prediction == -1), Safe is 0 (prediction == 1)
            predicted_fraud = np.where(df["Prediction"] == -1, 1, 0)
            actual_fraud = df[ground_truth_col].values
            
            # Confusion Matrix cells
            tp = int(((predicted_fraud == 1) & (actual_fraud == 1)).sum())
            fp = int(((predicted_fraud == 1) & (actual_fraud == 0)).sum())
            fn = int(((predicted_fraud == 0) & (actual_fraud == 1)).sum())
            tn = int(((predicted_fraud == 0) & (actual_fraud == 0)).sum())
            
            # Calculate classification metrics
            precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
            recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = ((tp + tn) / total_tx) * 100.0
            
            # Display Match Rate / Accuracy
            match_rate_html = f"""
            <div class="metric-card success">
                <div class="metric-title">🎯 Model Accuracy</div>
                <div class="metric-value green">{accuracy:.1f}%</div>
                <div class="metric-subtitle">F1-Score: {f1/100:.2f} | Rec: {recall:.1f}%</div>
            </div>
            """
        else:
            # Fallback if no ground truth
            match_rate_html = f"""
            <div class="metric-card success">
                <div class="metric-title">📡 Engine Status</div>
                <div class="metric-value green">ACTIVE</div>
                <div class="metric-subtitle">Unsupervised Model Running</div>
            </div>
            """
            
        # Render the dynamic glassmorphic grid
        st.markdown(f"""
        <div class="metrics-container">
            <div class="metric-card">
                <div class="metric-title">💳 Volume Processed</div>
                <div class="metric-value indigo">{volume_text}</div>
                <div class="metric-subtitle">Across {total_tx:,} transactions</div>
            </div>
            <div class="metric-card alert">
                <div class="metric-title">🚨 Flagged Anomalies</div>
                <div class="metric-value red">{suspicious_tx:,}</div>
                <div class="metric-subtitle">Flagged volume: {flagged_volume_text}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">📉 Anomaly Ratio</div>
                <div class="metric-value">{(anomaly_ratio):.2f}%</div>
                <div class="metric-subtitle">Target rate: {contamination*100:.2f}%</div>
            </div>
            {match_rate_html}
        </div>
        """, unsafe_allow_html=True)
        
        # 8. High-Dimensional Visualizations
        st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
        st.subheader("📊 Dynamic Dimensionality Visualization")
        
        # Check scatter dimensionality options
        st.sidebar.markdown("<hr style='margin: 10px 0; opacity: 0.15;' />", unsafe_allow_html=True)
        st.sidebar.markdown("### 📈 Visualizer Configurations")
        plot_dim = st.sidebar.selectbox("Scatter Plot Dimension", ["2D Scatter Plot", "3D Scatter Plot"])
        
        # Select features to plot
        x_axis = st.sidebar.selectbox("X-Axis Feature", numeric_columns, index=0 if len(numeric_columns) > 0 else 0)
        y_axis = st.sidebar.selectbox("Y-Axis Feature", numeric_columns, index=1 if len(numeric_columns) > 1 else 0)
        
        z_axis = None
        if plot_dim == "3D Scatter Plot":
            z_axis = st.sidebar.selectbox("Z-Axis Feature", numeric_columns, index=2 if len(numeric_columns) > 2 else 0)

        # Plotly chart drawing
        col_chart, col_score = st.columns([2, 1])
        
        with col_chart:
            if plot_dim == "2D Scatter Plot":
                fig = px.scatter(
                    df,
                    x=x_axis,
                    y=y_axis,
                    color="Fraud_Risk_Score",
                    color_continuous_scale="Viridis",
                    hover_data=["Amount", "Fraud_Flag", "Fraud_Risk_Score"],
                    title=f"Transaction Scatter Plot: {x_axis} vs {y_axis}",
                    labels={"Fraud_Risk_Score": "Risk Score (%)"}
                )
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=50, b=30, l=10, r=10)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.scatter_3d(
                    df,
                    x=x_axis,
                    y=y_axis,
                    z=z_axis,
                    color="Fraud_Risk_Score",
                    color_continuous_scale="Viridis",
                    hover_data=["Amount", "Fraud_Flag", "Fraud_Risk_Score"],
                    title=f"3D Space Plot: {x_axis} vs {y_axis} vs {z_axis}",
                    labels={"Fraud_Risk_Score": "Risk (%)"}
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=50, b=10, l=10, r=10)
                )
                st.plotly_chart(fig, use_container_width=True)
                
        with col_score:
            # Anomaly score distribution chart
            fig_hist = px.histogram(
                df,
                x="Fraud_Risk_Score",
                color="Fraud_Flag",
                nbins=30,
                color_discrete_map={"Suspicious": "#EF4444", "Legitimate": "#6366F1"},
                title="Risk Score Distribution & Density",
                labels={"Fraud_Risk_Score": "Calibrated Fraud Risk Score (%)"}
            )
            fig_hist.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=50, b=30, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        # 9. Interactive Anomaly Explorer & Threshold Selector
        st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
        
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.subheader("🔍 Interactive Fraud Risk Explorer")
            st.markdown("Adjust the custom threshold to filter anomalous transactions based on the calibrated model risk score.")
            
            # Interactive threshold filter
            risk_threshold = st.slider("Calibrated Risk Threshold (%)", 0.0, 100.0, 70.0, 1.0)
            
            flagged_above_threshold = df[df["Fraud_Risk_Score"] >= risk_threshold]
            st.info(f"⚡ Currently showing {len(flagged_above_threshold):,} transactions with risk score ≥ {risk_threshold}%")
            
            # Interactive amount distributions
            if amount_col:
                fig_box = px.box(
                    df,
                    x="Fraud_Flag",
                    y=amount_col,
                    color="Fraud_Flag",
                    color_discrete_map={"Suspicious": "#EF4444", "Legitimate": "#34D399"},
                    title="Transaction Value Profiles: Flagged vs Legit"
                )
                fig_box.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_box, use_container_width=True)
                
        with col_right:
            if ground_truth_available:
                st.subheader("🎯 Ground Truth Performance Evaluation")
                st.markdown("Comparison between unsupervised model outputs and actual fraud classifications.")
                
                # Confusion Matrix Heatmap
                z = [[tn, fp],
                     [fn, tp]]
                x_labels = ['Actual Safe (0)', 'Actual Fraud (1)']
                y_labels = ['Predicted Safe', 'Predicted Fraud']
                
                fig_cm = ff.create_annotated_heatmap(
                    z, x=x_labels, y=y_labels,
                    annotation_text=[[f"True Neg<br><b>{tn}</b>", f"False Pos<br><b>{fp}</b>"],
                                     [f"False Neg<br><b>{fn}</b>", f"True Pos<br><b>{tp}</b>"]],
                    colorscale='Viridis',
                    showscale=False
                )
                fig_cm.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=300,
                    margin=dict(t=30, b=20, l=10, r=10)
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Render beautiful analytics bullet points
                st.markdown(f"""
                *   **Precision Rate (Positive Predictive Value)**: `{(precision):.2f}%`
                    *   *How many transactions flagged as fraud are actually fraud.*
                *   **Recall / Sensitivity**: `{(recall):.2f}%`
                    *   *What percentage of actual fraud cases were successfully flagged.*
                *   **F1-Score**: `{f1/100:.3f}`
                    *   *Harmonic mean of precision and recall.*
                """)
            else:
                st.subheader("🛡️ Model Insights & Interpretability")
                st.markdown("""
                *   **Calibrated Risk Scoring**: By fitting the entire numerical subspace, the Isolation Forest marks transaction risk relative to extreme multidimensional feature isolation.
                *   **Self-Learning Framework**: This engine runs in an **unsupervised** configuration. It requires no historic label streaming, making it ideal for zero-day threat detection.
                *   **Real-time Decisioning ready**: Isolation Forest's structural isolation trees evaluate incoming data streams in log-linear time, allowing sub-millisecond classification.
                """)
                
        # 10. Suspicious Transaction Registry Table & Downloader
        st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
        st.subheader("⚠️ Suspicious Transaction Registry")
        st.markdown("Detailed report of transactions ordered by calibrated Fraud Risk Score.")
        
        # Sort and filter table
        flagged_tx_sorted = flagged_above_threshold.sort_values(by="Fraud_Risk_Score", ascending=False)
        st.dataframe(flagged_tx_sorted, use_container_width=True)
        
        # Download marked CSV button
        csv_data = flagged_tx_sorted.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Marked Anomalies CSV",
            data=csv_data,
            file_name="suspicious_fraud_anomalies.csv",
            mime="text/csv",
            help="Extract flagged records with computed Fraud Risk Scores for offline audit."
        )

    else:
        st.error("No numeric columns could be processed for anomaly detection.")
else:
    # Landing instructions
    st.info("👋 Welcome! Please select **Synthetic Data Stream** or upload your transaction dataset CSV in the sidebar control panel to begin.")