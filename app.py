import streamlit as st
import numpy as np
import cv2
import joblib
import pandas as pd
import altair as alt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Breast Cancer Detection App",
    page_icon="🩺",
    layout="centered"
)

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #2c3e50;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #34495e;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Breast Cancer Detection using XGBoost</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a breast ultrasound image for classification</div>', unsafe_allow_html=True)
st.write("")

# ---------------- LOAD MODEL ----------------
model = joblib.load("busi_xgboost_model.pkl")
scaler = joblib.load("busi_scaler.pkl")

IMG_SIZE = 128
categories = ["benign", "malignant", "normal"]

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader("📤 Upload Ultrasound Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_flat = img.flatten().reshape(1, -1)
    img_scaled = scaler.transform(img_flat)

    prediction = model.predict(img_scaled)
    probability = model.predict_proba(img_scaled)

    predicted_class = categories[prediction[0]]
    confidence = np.max(probability) * 100

    st.markdown("## 🧾 Prediction Result")
    st.success(f"**Class:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # ---------------- PROBABILITY BAR CHART ----------------
    st.markdown("### 📊 Class Probability Distribution")

    prob_df = pd.DataFrame({
        "Class": categories,
        "Probability": probability[0] * 100
    })

    chart = alt.Chart(prob_df).mark_bar().encode(
        x="Class",
        y="Probability"
    )

    st.altair_chart(chart, use_container_width=True)

# ---------------- MEDICAL DISCLAIMER ----------------
st.markdown("---")
st.warning("""
⚠️ **Medical Disclaimer:**  
This application is for research and educational purposes only.  
It is NOT a substitute for professional medical diagnosis.  
Please consult a certified medical practitioner for clinical decisions.
""")