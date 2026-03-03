def predictor():

    import streamlit as st
    import numpy as np
    import cv2
    import joblib
    import pandas as pd
    import altair as alt
    import datetime
    import os
    import time

    # PDF libraries
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.platypus import HRFlowable

    # =====================================================
    # SAFE PATH SETUP
    # =====================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    LOG_FILE = os.path.join(DATA_DIR, "prediction_log.csv")

    os.makedirs(DATA_DIR, exist_ok=True)

    # =====================================================
    # SAVE PREDICTION FUNCTION
    # =====================================================
    def save_prediction(result, confidence):

        new_data = pd.DataFrame([{
            "Prediction": result,
            "Confidence": float(confidence),
            "Time": datetime.datetime.now()
        }])

        # retry mechanism (Windows fix)
        for _ in range(3):
            try:
                if os.path.exists(LOG_FILE):
                    new_data.to_csv(
                        LOG_FILE,
                        mode="a",
                        header=False,
                        index=False
                    )
                else:
                    new_data.to_csv(LOG_FILE, index=False)
                break
            except PermissionError:
                time.sleep(0.5)

    # ---------------- TITLE ----------------
    st.title("🩺 Breast Cancer Detection using XGBoost")

    # ---------------- LOAD MODEL ----------------
    model = joblib.load(os.path.join(BASE_DIR, "busi_xgboost_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "busi_scaler.pkl"))

    IMG_SIZE = 128
    categories = ["benign", "malignant", "normal"]

    # ---------------- PATIENT DETAILS ----------------
    st.subheader("👤 Patient Details")

    patient_id = st.text_input("Patient ID")
    scan_date = st.date_input("Scan Date", datetime.date.today())

    # ---------------- FILE UPLOAD ----------------
    uploaded_file = st.file_uploader(
        "📤 Upload Ultrasound Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None and patient_id != "":

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        st.image(img, caption="Uploaded Image", use_container_width=True)

        img_flat = img.flatten().reshape(1, -1)
        img_scaled = scaler.transform(img_flat)

        prediction = model.predict(img_scaled)
        probability = model.predict_proba(img_scaled)

        predicted_class = categories[prediction[0]]
        confidence = float(np.max(probability) * 100)

        # ---------------- SAVE EVERY PREDICTION ----------------
        save_prediction(predicted_class, confidence)

        # ---------------- FLAG FOR DASHBOARD AUTO-REFRESH ----------------
        st.session_state.new_prediction = True

        # ---------------- RESULT ----------------
        st.subheader("🧾 Prediction Result")
        st.success(f"Class: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")

        # ---------------- PROBABILITY CHART ----------------
        prob_df = pd.DataFrame({
            "Class": categories,
            "Probability": probability[0] * 100
        })

        chart = alt.Chart(prob_df).mark_bar().encode(
            x="Class",
            y="Probability"
        )

        st.altair_chart(chart, use_container_width=True)

        # ---------------- DIGITAL ANALYSIS ----------------
        st.subheader("🧬 Digital Analysis Report")

        if predicted_class == "benign":
            suggestion = "Benign lesion detected. Usually non-cancerous."
        elif predicted_class == "malignant":
            suggestion = "Malignant pattern observed. Immediate medical review recommended."
        else:
            suggestion = "Normal tissue pattern observed."

        st.write(suggestion)

        # ---------------- REPORT TEXT ----------------
        report_text = f"""
Patient ID : {patient_id}
Scan Date  : {scan_date}

Predicted Class : {predicted_class}
Confidence Score : {confidence:.2f} %

Clinical Suggestion:
{suggestion}
"""

        st.subheader("📋 AI Diagnostic Report")
        st.text(report_text)

        # ---------------- CREATE PDF ----------------
        pdf_file = os.path.join(BASE_DIR, f"AI_Report_{patient_id}.pdf")

        doc = SimpleDocTemplate(pdf_file, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        logo_path = os.path.join(BASE_DIR, "hospital_logo.png")
        if os.path.exists(logo_path):
            story.append(Image(logo_path, width=120, height=60))

        story.append(Spacer(1, 12))
        story.append(Paragraph(
            "<b>AI BREAST ULTRASOUND DIAGNOSTIC REPORT</b>",
            styles['Title']
        ))

        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        story.append(Spacer(1, 15))

        for line in report_text.split("\n"):
            story.append(Paragraph(line, styles["Normal"]))
            story.append(Spacer(1, 5))

        story.append(Spacer(1, 30))
        story.append(Paragraph("Doctor Signature:", styles["Normal"]))

        sign_path = os.path.join(BASE_DIR, "doctor_sign.png")
        if os.path.exists(sign_path):
            story.append(Image(sign_path, width=150, height=60))

        story.append(Paragraph("<b>Dr. Jerlin</b>", styles["Normal"]))
        story.append(Paragraph("AI Radiology Department", styles["Normal"]))

        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))

        story.append(Paragraph(
            "Disclaimer: This AI-generated report is for research purposes only.",
            styles["Normal"]
        ))

        doc.build(story)

        with open(pdf_file, "rb") as f:
            st.download_button(
                "📄 Download Medical Report (PDF)",
                f,
                file_name=os.path.basename(pdf_file),
                mime="application/pdf"
            )

    # ---------------- ANALYTICS PANEL ----------------
    st.markdown("## 📈 Doctor Analytics Panel")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", "75%")
    col2.metric("Classes", "3")
    col3.metric("Input Features", "16384")

    # ---------------- DISCLAIMER ----------------
    st.markdown("---")
    st.warning("""
⚠️ Medical Disclaimer:
This application is for research and educational purposes only.
Not a substitute for professional diagnosis.
""")