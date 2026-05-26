import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Fraud Detection Dashboard")

st.title("💳 Financial Fraud Detection Dashboard")

st.markdown("### Upload Transaction Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    numeric_columns = df.select_dtypes(include=np.number).columns

    if len(numeric_columns) > 0:

        model = IsolationForest(
            contamination=0.002,
            random_state=42
        )

        df["Fraud"] = model.fit_predict(df[numeric_columns])

        fraud_count = (df["Fraud"] == -1).sum()

        st.subheader("🚨 Fraud Detection Summary")

        st.metric(
            "Suspicious Transactions",
            fraud_count
        )

        st.subheader("📊 Transaction Distribution")

        if "Amount" in df.columns:

            fig = px.histogram(
                df,
                x="Amount",
                nbins=30,
                title="Transaction Amount Distribution"
            )

            st.plotly_chart(fig)

        st.subheader("⚠️ Suspicious Transactions")

        fraud_df = df[df["Fraud"] == -1]

        st.dataframe(fraud_df)

    else:

        st.error("No numeric columns found in dataset.")