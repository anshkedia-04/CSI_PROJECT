import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="🚨 Network Anomaly Detector", layout="wide")
st.title("🚨 Network Traffic Anomaly Detection")
st.markdown("Upload a dataset and detect anomalies using Isolation Forest.")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        st.subheader("📄 Uploaded Data")
        st.dataframe(df.head())

        numeric_data = df.select_dtypes(include=[np.number])

        if numeric_data.shape[1] < 2:
            st.warning("⚠️ Upload a CSV with at least 2 numeric columns.")
        else:
            model = IsolationForest(contamination=0.05, random_state=42)
            df['anomaly'] = model.fit_predict(numeric_data)
            df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})  # 1 = anomaly

            # Anomaly stats
            st.subheader("📊 Detection Summary")
            st.metric("Total Records", len(df))
            st.metric("Anomalies Found", df['anomaly'].sum())

            # Scatter plot
            st.subheader("📈 Scatter Plot of Anomalies")
            fig, ax = plt.subplots()
            sns.scatterplot(
                data=df,
                x=numeric_data.columns[0],
                y=numeric_data.columns[1],
                hue="anomaly",
                palette=["green", "red"],
                ax=ax
            )
            ax.set_title("Anomaly Detection (Red = Anomaly)")
            st.pyplot(fig)

            st.subheader("🔍 Data with Anomaly Column")
            st.dataframe(df)

            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Result CSV", csv, "anomaly_results.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
else:
    st.info("📤 Please upload a CSV file to begin.")
