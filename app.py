import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Page Configuration
st.set_page_config(
    page_title="ApexHealth | Clinical Decision Support System",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Theme-Adaptive High-End Design System (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Brand Hero Banner */
    .hero-container {
        padding: 1.8rem 2.2rem;
        background: linear-gradient(135deg, #091e3a 0%, #17375e 50%, #1e4d7b 100%);
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(9, 30, 58, 0.25);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .hero-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        color: #ffffff !important;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #cbd5e1 !important;
        margin-top: 0.3rem;
        margin-bottom: 0;
    }

    /* Theme-Adaptive UI Surface Cards */
    .card-surface {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 14px;
        padding: 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-color);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Adaptive Status Badges */
    .badge-status {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }
    .badge-verified {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981 !important;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-high-risk {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444 !important;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-optimal {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981 !important;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* Primary Action Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(2, 132, 199, 0.35);
    }
</style>
""", unsafe_allow_html=True)

# Training & Caching Core Engine
@st.cache_resource
def initialize_clinical_engine():
    try:
        data = pd.read_csv("dataset.csv")
    except FileNotFoundError:
        data = pd.read_csv("diabetes.csv")
    
    feature_columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 
        'SkinThickness', 'Insulin', 'BMI', 
        'DiabetesPedigreeFunction', 'Age'
    ]
    X = data[feature_columns]
    y = data['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return data, model, acc, prec, rec, cm

data, model, accuracy, precision, recall, cm = initialize_clinical_engine()

# Hero Header Component
st.markdown("""
<div class="hero-container">
    <div>
        <h1 class="hero-title">ApexHealth Diagnostics • CDSS</h1>
        <p class="hero-subtitle">Machine Learning-Driven Diabetes Risk Stratification & Metabolic Screening Portal</
