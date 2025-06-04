import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import os
from transformers import pipeline

# Ensure assets directory exists
if not os.path.exists("assets"):
    os.makedirs("assets")

# Set up the app configuration
st.set_page_config(page_title="OncoPlan - Radiation Therapy Planner", page_icon="⚕️", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Patient Information", "Treatment Plan", "AI Chatbot"])

# Cancer Types & Regions
cancer_types = {
    "Brain Tumors": ["Left Cerebral Hemisphere", "Right Cerebral Hemisphere", "Frontal Lobe", "Parietal Lobe", "Temporal Lobe", "Occipital Lobe", "Brainstem", "Cerebellum", "Intraventricular Region", "Pineal Region", "Peripheral Region", "Brain Parenchyma"],
    "Breast Cancer": ["Left Breast", "Right Breast", "Ductal Region", "Lobular Region", "Axillary Lymph Nodes"],
    "Lung Cancer": ["Left Lung - Upper Lobe", "Left Lung - Lower Lobe", "Right Lung - Upper Lobe", "Right Lung - Middle Lobe", "Right Lung - Lower Lobe", "Mediastinum", "Pleura"],
    "Other": ["Custom"]
}

# Home Page
if page == "Home":
    st.title("OncoPlan - AI-Powered Radiation Therapy Planner")
    st.image("oncoplan_banner.webp")
    st.markdown("""
    ## Welcome to OncoPlan
    OncoPlan is an advanced AI-powered radiation therapy planning tool designed for oncologists.
    
    **Features:**
    - Accurate Radiation Therapy Planning
    - Dynamic Cancer Type & Region Selection
    - AI Chatbot for Treatment Guidance
    - Tumor Reduction & Recovery Graphs
    """)

# Patient Information Page
if page == "Patient Information":
    st.title("Patient Information")
    name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    gender = st.radio("Gender", ["Male", "Female", "Other"])
    cancer_type = st.selectbox("Type of Cancer", list(cancer_types.keys()))
    region = st.selectbox("Region of Body", cancer_types[cancer_type])

    has_brain_mets = st.checkbox("Brain Metastases (Brain Mets)")
    brain_mets_primary_cancer = st.selectbox("Primary Cancer", ["Breast", "Lung", "Melanoma", "Colon", "Kidney", "Prostate", "Other"]) if has_brain_mets else None
    brain_mets_regions = st.multiselect("Affected Brain Regions", [
            "Frontal Lobe", "Parietal Lobe", "Temporal Lobe", "Occipital Lobe",
            "Cerebellum", "Brainstem", "Multiple Regions", "Meninges (Leptomeningeal Spread)"
        ])

    stage = st.selectbox("Stage of Cancer", ["I", "II", "III", "IV"])
    separation = st.number_input("Separation (cm)")
    machine = st.selectbox("Machine", ["Co-60", "LINAC", "Teletherapy"])

    if st.button("Save Patient Info"):
        st.session_state["patient_data"] = {"name": name, "age": age, "gender": gender, "cancer_type": cancer_type, "region": region, "stage": stage, "separation": separation, "machine": machine}
        st.success("Patient Information Saved!")

# Treatment Plan Page
if page == "Treatment Plan":
    if "patient_data" not in st.session_state:
        st.warning("Please enter patient details first!")
    else:
        data = st.session_state["patient_data"]
        st.title(f"Radiotherapy Treatment Plan for {data['name']}")

        # Treatment Plan Logic
        num_sessions = 15 if data['age'] < 40 else 10 if data['age'] < 65 else 8
        interval_days = 1 if data['age'] < 40 else 3 if data['age'] < 65 else 7
        start_date = datetime.date.today()
        table_data = []
        total_dose = 0
        dose_per_session = 2

        for session in range(1, num_sessions + 1):
            session_date = start_date + datetime.timedelta(days=(session - 1) * interval_days)
            total_dose += dose_per_session
            table_data.append([session, session_date.strftime('%Y-%m-%d'), dose_per_session, total_dose])

        df = pd.DataFrame(table_data, columns=["Session #", "Date", "Dose (Gy)", "Total Dose (Gy)"])
        st.table(df)

        # Tumor Reduction Graph
        sessions = np.arange(1, num_sessions + 1)
        tumor_size = 100 * np.exp(-0.15 * sessions)
        fig, ax = plt.subplots()
        ax.plot(sessions, tumor_size, marker='o', linestyle='-')
        ax.set_xlabel("Sessions")
        ax.set_ylabel("Tumor Size (%)")
        ax.set_title("Tumor Shrinkage Over Treatment")
        st.pyplot(fig)

# AI Chatbot Page
if page == "AI Chatbot":
    st.subheader("💬 AI Chatbot")
    chatbot = pipeline("question-answering", model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    user_input = st.text_input("Ask about cancer, radiotherapy, or treatments:")

    if user_input:
        response = chatbot(question=user_input, context="Cancer is a disease involving abnormal cell growth...")
        st.write(f"**OncoBot:** {response['answer']}")

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed for Medical Use Only. Consult a Specialist for Professional Advice.")
