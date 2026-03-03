import streamlit as st

def dashboard():

    st.title("🏥 Breast Cancer AI Diagnostic System")

    st.markdown("""
    ### Welcome Doctor 👩‍⚕️

    This AI system assists in:
    - Ultrasound image classification
    - Tumor detection support
    - Explainable AI predictions

    ⚠️ This tool supports clinical decisions only.
    """)

    st.info("Use the sidebar to start prediction.")