import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Network Traffic Anomaly Detection", layout="wide")
st.title("🚨 Network Traffic Anomaly Detection")
st.markdown("Detect security threats using **Isolation Forest** and **Autoencoders**.")

# Sidebar: Upload CSV
st.sidebar.header("📁 Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload a preprocessed network traffic CSV", type=["csv"])

# Load models
@st.cache_resource
def load_models():
    try:
        isf_model = joblib.load("models/isolation_forest.pkl")
        ae_model = load_model("models/autoencoder_model.h5")
        return isf_model, ae_model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🧾 Raw Data Preview")
    st.dataframe(df.head())

    # Preprocess
    df_clean = df.select_dtypes(include=[np.number]).dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    isf, ae = load_models()

    # Isolation Forest
    df['isf_anomaly'] = isf.predict(X_scaled)
    df['isf_anomaly'] = df['isf_anomaly'].apply(lambda x: 1 if x == -1 else 0)

    # Autoencoder
    reconstructions = ae.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, 95)

    df['ae_mse'] = mse
    df['ae_anomaly'] = (df['ae_mse'] > threshold).astype(int)

    # Threshold Info
    st.sidebar.markdown("### ⚙️ Threshold")
    st.sidebar.write(f"Autoencoder MSE Threshold: `{threshold:.4f}`")

    # Anomaly Summary
    st.subheader("📊 Anomaly Summary")
    st.write("#### Isolation Forest Results")
    st.dataframe(df['isf_anomaly'].value_counts().rename(index={0: 'Normal', 1: 'Anomaly'}).to_frame('Count'))

    st.write("#### Autoencoder Results")
    st.dataframe(df['ae_anomaly'].value_counts().rename(index={0: 'Normal', 1: 'Anomaly'}).to_frame('Count'))

    # MSE Scatter Plot
    st.subheader("📈 Autoencoder MSE Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.scatterplot(x=df.index, y=df['ae_mse'], hue=df['ae_anomaly'], palette="coolwarm", ax=ax)
    plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
    plt.title("Autoencoder Reconstruction Error (MSE)")
    plt.legend()
    st.pyplot(fig)

    # 📉 Isolation Forest Anomaly Scatter Plot
    st.subheader("📈 Isolation Forest Anomalies")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.scatterplot(x=df.index, y=df_clean.iloc[:, 0], hue=df['isf_anomaly'], palette='coolwarm', ax=ax2)
    ax2.set_title("Isolation Forest Anomaly Distribution")
    ax2.set_xlabel("Index")
    ax2.set_ylabel(df_clean.columns[0])
    st.pyplot(fig2)

    # Filter View
    st.subheader("🔍 View Anomalies Only")
    view_option = st.radio("Filter rows by", ["All", "Only Isolation Forest Anomalies", "Only Autoencoder Anomalies"])
    if view_option == "Only Isolation Forest Anomalies":
        st.dataframe(df[df["isf_anomaly"] == 1])
    elif view_option == "Only Autoencoder Anomalies":
        st.dataframe(df[df["ae_anomaly"] == 1])
    else:
        st.dataframe(df)

    # Download button
    st.download_button("📥 Download Results", df.to_csv(index=False), file_name="anomaly_results.csv")

else:
    st.info("📤 Upload a CSV file from your preprocessed dataset to begin anomaly detection.")
