import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Universal Anomaly Detector", layout="wide")

# Sidebar Info
st.sidebar.title("📌 About")
st.sidebar.markdown("""
This app detects **anomalies** in any uploaded CSV dataset using the **Isolation Forest** algorithm.
- Auto-preprocessing: encoding & scaling
- No target labels needed
- Visualizations included
""")

st.title("🔍 Universal Anomaly Detection App")
st.markdown("Upload a CSV file and let the app find **anomalous rows**.")

uploaded_file = st.file_uploader("📁 Upload your CSV file here", type=["csv"])

if uploaded_file:
    st.info("✅ File uploaded.")
    raw_df = pd.read_csv(uploaded_file)
    st.dataframe(raw_df.head())

    # Copy for processing
    df = raw_df.copy()

    # Encode categorical columns
    non_numeric_cols = df.select_dtypes(include=['object', 'category']).columns
    st.markdown("### 🧠 Preprocessing")
    st.write(f"🔠 Encoding columns: `{list(non_numeric_cols)}`")

    for col in non_numeric_cols:
        try:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        except:
            df.drop(columns=[col], inplace=True)
            st.warning(f"Could not encode {col}, dropped.")

    # Drop nulls
    df = df.dropna()
    st.write(f"✅ Shape after dropping nulls: {df.shape}")

    # Save for model
    df_model = df.copy()

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_model)

    # Anomaly Detection
    st.markdown("### 🚨 Running Isolation Forest (5% expected anomalies)")
    isf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    preds = isf.fit_predict(X_scaled)
    scores = isf.decision_function(X_scaled)

    # Add predictions
    df_model['anomaly'] = np.where(preds == -1, 'Anomaly', 'Normal')
    df_model['anomaly_score'] = scores

    st.success("🎯 Anomaly detection complete!")

    # Data preview
    st.markdown("### 📊 Annotated Dataset")
    st.dataframe(df_model)

    # Filter option
    if st.checkbox("🔎 Show only anomalies"):
        st.write(df_model[df_model['anomaly'] == 'Anomaly'])

    # Download
    csv = df_model.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Result", csv, "anomaly_results.csv", "text/csv")

    # ==== 📉 Visualizations ====

    st.markdown("## 📈 Visual Analysis")

    col1, col2 = st.columns(2)

    # Scatter Plot (PCA)
    with col1:
        st.markdown("### 🟢 Scatter Plot (via PCA)")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        df_plot = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
        df_plot['Anomaly'] = df_model['anomaly']

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_plot, x='PC1', y='PC2', hue='Anomaly', palette={'Normal': 'blue', 'Anomaly': 'red'}, alpha=0.6)
        plt.title('PCA Scatter Plot of Anomalies')
        st.pyplot(plt.gcf())
        plt.clf()

    with col2:
        st.markdown("### 📊 Anomaly Statistics")
        total = len(df_model)
        anomaly_count = (df_model['anomaly'] == 'Anomaly').sum()
        normal_count = total - anomaly_count
        anomaly_percent = (anomaly_count / total) * 100

        st.write(f"**Total rows:** {total}")
        st.write(f"**Normal rows:** {normal_count}")
        st.write(f"**Anomalies detected:** {anomaly_count}")
        st.write(f"**Anomaly percentage:** {anomaly_percent:.2f}%")




    # Footer
    st.markdown("---")
    st.markdown("Made by Ansh Kedia during the **Celebal Summer Internship 2025**.")

else:
    st.warning("👆 Please upload a CSV file to get started.")
