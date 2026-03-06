def predictor():

    import streamlit as st
    import numpy as np
    import shap
    import cv2
    import joblib
    import pandas as pd
    import altair as alt
    import datetime
    import os
    import time
    import matplotlib.pyplot as plt

    # PDF libraries
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.platypus import HRFlowable

    # =====================================================
    # PATH SETUP
    # =====================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    LOG_FILE = os.path.join(DATA_DIR, "prediction_log.csv")

    os.makedirs(DATA_DIR, exist_ok=True)

    # =====================================================
    # SAVE PREDICTION
    # =====================================================
    def save_prediction(result, confidence):

        new_data = pd.DataFrame([{
            "Prediction": result,
            "Confidence": float(confidence),
            "Time": datetime.datetime.now()
        }])

        for _ in range(3):
            try:
                if os.path.exists(LOG_FILE):
                    new_data.to_csv(LOG_FILE, mode="a",
                                    header=False, index=False)
                else:
                    new_data.to_csv(LOG_FILE, index=False)
                break
            except PermissionError:
                time.sleep(0.5)

    # =====================================================
    # UI TITLE
    # =====================================================
    st.title("🩺 Breast Cancer Detection using XGBoost")

    # =====================================================
    # LOAD MODEL
    # =====================================================
    model = joblib.load(os.path.join(BASE_DIR, "busi_xgboost_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "busi_scaler.pkl"))

    explainer = shap.TreeExplainer(model)

    IMG_SIZE = 128
    categories = ["benign", "malignant", "normal"]

    # =====================================================
    # PATIENT DETAILS
    # =====================================================
    st.subheader("👤 Patient Details")

    patient_id = st.text_input("Patient ID")
    scan_date = st.date_input("Scan Date", datetime.date.today())

    # =====================================================
    # IMAGE UPLOAD
    # =====================================================
    st.markdown("### 📤 Upload Ultrasound Image for Analysis")

    uploaded_file = st.file_uploader(
        "Upload Ultrasound Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None and patient_id != "":

        file_bytes = np.asarray(bytearray(uploaded_file.read()),
                                dtype=np.uint8)

        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # save temporary image for PDF
        temp_image_path = os.path.join(BASE_DIR, "temp_ultrasound.png")
        cv2.imwrite(temp_image_path, img)

        img_flat = img.flatten().reshape(1, -1)
        img_scaled = scaler.transform(img_flat)

        # ================= LOADING SPINNER =================
        with st.spinner("AI analyzing ultrasound image..."):
            prediction = model.predict(img_scaled)
            probability = model.predict_proba(img_scaled)

        predicted_class = categories[prediction[0]]
        confidence = float(np.max(probability) * 100)

        save_prediction(predicted_class, confidence)

        # ================= TWO COLUMN LAYOUT =================
        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Uploaded Ultrasound Image",
                     use_container_width=True)

        with col2:

            st.markdown("### 🧾 Prediction Result")

            st.markdown(f"""
            **Predicted Class:** {predicted_class}  
            **Confidence:** {confidence:.2f} %
            """)

            # Result color indicator
            if predicted_class == "malignant":
                st.error("⚠ Malignant pattern detected")
            elif predicted_class == "benign":
                st.success("✅ Benign lesion detected")
            else:
                st.info("Normal tissue pattern detected")

        # Probability chart
        prob_df = pd.DataFrame({
            "Class": categories,
            "Probability": probability[0] * 100
        })

        chart = alt.Chart(prob_df).mark_bar().encode(
            x="Class",
            y="Probability"
        )

        st.altair_chart(chart, use_container_width=True)

        # =====================================================
        # SHAP EXPLANATION
        # =====================================================
        st.markdown("### 🧠 Explainable AI Analysis")
        st.caption(
            "Blue bars increase the prediction probability, while red bars decrease the prediction probability in the model decision."
        )

        try:

            shap_values = explainer.shap_values(img_scaled)
            class_index = int(prediction[0])

            if isinstance(shap_values, list):
                shap_single = shap_values[class_index][0]
                base_value = explainer.expected_value[class_index]
            else:
                shap_single = shap_values[0][:, class_index]
                base_value = explainer.expected_value[class_index]

            explanation = shap.Explanation(
                values=shap_single,
                base_values=base_value,
                data=img_scaled[0]
            )

            fig1 = plt.figure(figsize=(8, 6))
            shap.plots.waterfall(
                explanation,
                max_display=20,
                show=False
            )

            st.pyplot(fig1)
            plt.close()

            st.subheader("📊 Top 20 Important Features")

            fig2 = plt.figure(figsize=(8, 6))
            shap.summary_plot(
                np.array([shap_single]),
                img_scaled,
                plot_type="bar",
                max_display=20,
                show=False
            )

            st.pyplot(fig2)
            plt.close()

        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

        # =====================================================
        # DIGITAL ANALYSIS
        # =====================================================
        st.subheader("🧬 Digital Analysis Report")

        if predicted_class == "benign":
            suggestion = "Benign lesion detected. Usually non-cancerous."
        elif predicted_class == "malignant":
            suggestion = "Malignant pattern observed. Immediate medical review recommended."
        else:
            suggestion = "Normal tissue pattern observed."

        st.write(suggestion)

        # =====================================================
        # PDF GENERATION
        # =====================================================
        pdf_file = os.path.join(BASE_DIR, f"AI_Report_{patient_id}.pdf")

        doc = SimpleDocTemplate(pdf_file, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph(
            "<b>AI BREAST ULTRASOUND DIAGNOSTIC REPORT</b>",
            styles['Title']
        ))

        story.append(Spacer(1, 10))

        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.black))

        story.append(Spacer(1, 15))

        story.append(Paragraph("<b>Ultrasound Image</b>", styles["Heading3"]))
        story.append(Spacer(1, 10))

        story.append(Image(temp_image_path, width=250, height=250))

        story.append(Spacer(1, 20))

        story.append(Paragraph("<b>Patient Details</b>", styles["Heading3"]))
        story.append(Paragraph(f"Patient ID : {patient_id}", styles["Normal"]))
        story.append(Paragraph(f"Scan Date : {scan_date}", styles["Normal"]))

        story.append(Spacer(1, 10))

        story.append(Paragraph("<b>Prediction Result</b>", styles["Heading3"]))
        story.append(Paragraph(
            f"Predicted Class : {predicted_class}", styles["Normal"]))
        story.append(Paragraph(
            f"Confidence Score : {confidence:.2f} %", styles["Normal"]))

        story.append(Spacer(1, 10))

        story.append(Paragraph("<b>Clinical Suggestion</b>", styles["Heading3"]))
        story.append(Paragraph(suggestion, styles["Normal"]))

        story.append(Spacer(1, 15))

        story.append(Paragraph("<b>AI System Details</b>", styles["Heading3"]))
        story.append(Paragraph("Model Used : XGBoost Classifier", styles["Normal"]))
        story.append(Paragraph("Input Type : Breast Ultrasound Image", styles["Normal"]))
        story.append(Paragraph("Image Size : 128 × 128 pixels", styles["Normal"]))
        story.append(Paragraph("Feature Extraction : Pixel Intensity Based", styles["Normal"]))
        story.append(Paragraph(
            "Explainability Method : SHAP (Shapley Additive Explanations)",
            styles["Normal"]))

        story.append(Spacer(1, 15))

        story.append(Paragraph("<b>Report Generated On</b>", styles["Heading3"]))
        story.append(Paragraph(str(datetime.datetime.now()), styles["Normal"]))

        story.append(Spacer(1, 20))

        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.grey))

        story.append(Paragraph(
            "Disclaimer: This report is generated by an AI-based research prototype and is intended for educational purposes only.",
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