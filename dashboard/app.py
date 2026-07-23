import os
import sys
import glob
import pandas as pd
import numpy as np
import streamlit as st

# 1. Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.predict import NIDSPredictor
from src.visualization.plots import (
    plot_attack_distribution,
    plot_confusion_matrix,
    plot_feature_importance
)

# 2. Basic Page Configuration
st.set_page_config(
    page_title="NIDS | AI Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. Custom Dark Cybersecurity CSS
st.markdown("""
<style>
    /* General App Background */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Main Header Styling */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 25px;
        text-align: center;
    }
    .main-title {
        color: #38bdf8;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .main-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
    }

    /* Metric Cards Styling */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-value-safe {
        color: #10b981;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .metric-value-danger {
        color: #ef4444;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Status Alerts */
    .status-alert-safe {
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #34d399;
        padding: 16px;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 20px;
    }
    .status-alert-danger {
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #f87171;
        padding: 16px;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Function to get the latest generated CSV file dynamically
def get_latest_sample_file():
    sample_dir = os.path.join("data", "sample")
    list_of_files = glob.glob(os.path.join(sample_dir, "*.csv"))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file

# 4. Load AI Predictor
#@st.cache_resource
def get_predictor():
    return NIDSPredictor()

try:
    predictor = get_predictor()
except Exception as e:
    st.error("⚠️ Trained model not found! Please run 'python src/model/train.py' in the Terminal first.")
    st.stop()

# 5. Display Main Header
st.markdown("""
<div class="main-header">
    <div class="main-title">🛡️ AI Network Intrusion Detection System</div>
    <div class="main-subtitle">AI-Powered System for Monitoring and Analyzing Network Cyber Threats</div>
</div>
""", unsafe_allow_html=True)

# 6. Sidebar Configuration
st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=70)
st.sidebar.title("🎮 Dashboard")

model_choice = st.sidebar.selectbox(
    "🤖 Select AI Model",
    options=["random_forest", "xgboost"],
    format_func=lambda x: "Random Forest (Highest Accuracy)" if x == "random_forest" else "XGBoost (Fast)"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Data Source")

load_sample = st.sidebar.button("⚡ Load Sample Network Traffic", use_container_width=True)
uploaded_file = st.sidebar.file_uploader("Or upload your own CSV file", type=["csv"])

# 7. Data Loading Logic (Includes the Dynamic File Fetcher)
df_raw = None
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
elif load_sample or "df_raw" not in st.session_state:
    sample_path = get_latest_sample_file()
    if sample_path and os.path.exists(sample_path):
        df_raw = pd.read_csv(sample_path)
        st.session_state["df_raw"] = df_raw
        if load_sample:
            st.sidebar.success(f"Loaded: {os.path.basename(sample_path)}")
    elif load_sample:
        st.sidebar.error("No sample files found in data/sample/")

if "df_raw" in st.session_state and df_raw is None:
    df_raw = st.session_state["df_raw"]

# 8. Processing, Predictions, and Visualization
if df_raw is not None:
    with st.spinner("🔍 Analyzing network traffic with AI..."):
        try:
            df_results = predictor.predict(df_raw, model_name=model_choice)
        except Exception as e:
            st.error(f"An error occurred during data analysis: {str(e)}")
            st.stop()

    # Calculate Statistics
    total_records = len(df_results)
    normal_count = (df_results['prediction'] == 'Normal').sum()
    attack_count = total_records - normal_count
    threat_ratio = (attack_count / total_records) * 100 if total_records > 0 else 0

    # Overall Security Status Alert
    if attack_count == 0:
        st.markdown('<div class="status-alert-safe">✅ Network is entirely safe! No suspicious activities detected in the current data.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-alert-danger">🚨 Security Alert! Detected {attack_count} suspicious activities/cyber attacks out of {total_records} connections.</div>', unsafe_allow_html=True)

    # Display Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value-safe" style="color:#38bdf8;">{total_records:,}</div><div class="metric-label">Total Connections</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value-safe">{normal_count:,}</div><div class="metric-label">Safe Connections (Normal)</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value-danger">{attack_count:,}</div><div class="metric-label">Detected Attacks</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value-danger" style="color:#f59e0b;">{threat_ratio:.1f}%</div><div class="metric-label">Threat Ratio</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Attack Distribution & Charts", "📋 Detailed Data Log", "⚙️ AI Model Details"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Network Traffic Classification")
            fig_dist = plot_attack_distribution(df_results)
            st.plotly_chart(fig_dist, use_container_width=True)
        with col_b:
            st.subheader("Top Features Influencing Detection")
            fig_feat = plot_feature_importance(predictor.models[model_choice], predictor.feature_names)
            st.plotly_chart(fig_feat, use_container_width=True)

    with tab2:
        st.subheader("Current Data Inspection Results")
        filter_attacks = st.checkbox("Show only attacks and suspicious activities 🚨")
        display_df = df_results[df_results['prediction'] != 'Normal'] if filter_attacks else df_results

        st.dataframe(display_df, use_container_width=True, height=350)
        
        csv_data = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Inspection Report (CSV)",
            data=csv_data,
            file_name="nids_detection_report.csv",
            mime="text/csv"
        )

    with tab3:
        st.subheader("Model Accuracy Information")
        st.info(f"Currently Active Model: **{model_choice.upper()}**")
        st.write("This model was trained and evaluated using top algorithms to process complex network traffic data.")

else:
    st.info("👈 Please upload a traffic file (CSV) from the sidebar or click 'Load Sample Network Traffic' to begin.")