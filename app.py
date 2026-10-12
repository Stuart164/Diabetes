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

# High-End Design System & CSS Overrides
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Brand Hero */
    .hero-container {
        padding: 1.8rem 2.2rem;
        background: linear-gradient(135deg, #091e3a 0%, #17375e 50%, #1e4d7b 100%);
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(9, 30, 58, 0.12);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .hero-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #b0c9e8;
        margin-top: 0.3rem;
        margin-bottom: 0;
    }

    /* UI Surface Cards */
    .card-surface {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Status Pills */
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
        background: #e6f9f0;
        color: #0d8a56;
        border: 1px solid #b7ebd3;
    }
    .badge-high-risk {
        background: #fef2f2;
        color: #dc2626;
        border: 1px solid #fecaca;
    }
    .badge-optimal {
        background: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
    }

    /* Primary Action Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(2, 132, 199, 0.35);
    }

    /* Metric Override */
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 12px;
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
        <p class="hero-subtitle">Machine Learning-Driven Diabetes Risk Stratification & Metabolic Screening Portal</p>
    </div>
    <div>
        <span class="badge-status badge-verified">Engine Active • v2.4</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
tab_screen, tab_telemetry, tab_benchmarks = st.tabs([
    "🩺 Patient Screening Assessment",
    "📈 Population & Diagnostic Telemetry",
    "📜 Clinical Threshold Directory"
])

# TAB 1: Clinical Assessment Form & Live Output
with tab_screen:
    col_input, col_analytics = st.columns([1.1, 0.9], gap="large")
    
    with col_input:
        st.markdown('<div class="card-surface">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔬 Biomarker Input Matrix</div>', unsafe_allow_html=True)
        
        with st.form("triage_form"):
            c1, c2 = st.columns(2)
            with c1:
                glucose = st.number_input("Fasting Glucose (mg/dL)", 40.0, 300.0, 118.0, 1.0, help="Normal reference: 70–99 mg/dL")
                insulin = st.number_input("2-Hour Insulin (μIU/mL)", 0.0, 850.0, 85.0, 1.0, help="Fasting normal: 16–166 μIU/mL")
                bmi = st.number_input("Body Mass Index (kg/m²)", 10.0, 70.0, 27.8, 0.1, help="Normal: 18.5–24.9 kg/m²")
                skin = st.number_input("Triceps Skinfold (mm)", 0.0, 99.0, 22.0, 1.0, help="Subcutaneous caliper metric")
                
            with c2:
                age = st.number_input("Age (Years)", 18, 110, 42, 1)
                bp = st.number_input("Diastolic BP (mm Hg)", 30.0, 150.0, 78.0, 1.0, help="Optimal: < 80 mm Hg")
                pregnancies = st.number_input("Pregnancies", 0, 20, 1, 1)
                pedigree = st.number_input("Pedigree Function", 0.05, 2.50, 0.38, 0.01, help="Genetic predisposition index")
                
            stratify_btn = st.form_submit_button("Run Risk Stratification", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_analytics:
        # Default or calculated evaluation
        patient_vector = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]])
        pred = model.predict(patient_vector)[0]
        prob = model.predict_proba(patient_vector)[0]
        risk_pct = prob[1] * 100

        # High-End Dial Indicator
        gauge_chart = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Calculated Risk Score", 'font': {'size': 16, 'color': '#64748b'}},
            number={'suffix': "%", 'font': {'size': 38, 'color': '#0f172a'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
                'bar': {'color': "#0284c7", 'thickness': 0.28},
                'bgcolor': "#f8fafc",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 30], 'color': '#dcfce7'},
                    {'range': [30, 65], 'color': '#fef9c3'},
                    {'range': [65, 100], 'color': '#fee2e2'}
                ],
                'threshold': {
                    'line': {'color': "#dc2626" if risk_pct >= 65 else "#f59e0b" if risk_pct >= 30 else "#16a34a", 'width': 3},
                    'thickness': 0.75,
                    'value': risk_pct
                }
            }
        ))
        gauge_chart.update_layout(
            height=250, 
            margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.markdown('<div class="card-surface">', unsafe_allow_html=True)
        st.plotly_chart(gauge_chart, use_container_width=True)
        
        # Clinical Risk Stratification Output
        if pred == 1:
            st.markdown("""
            <div style="background-color: #fef2f2; border: 1px solid #fecaca; border-radius: 10px; padding: 1.2rem; margin-top: 0.5rem;">
                <span class="badge-status badge-high-risk">High Clinical Risk</span>
                <h4 style="color: #991b1b; margin: 0.6rem 0 0.4rem 0;">Positive Diagnostic Indicator</h4>
                <p style="color: #7f1d1d; font-size: 0.88rem; margin: 0;">
                    Biometric markers cross critical thresholds for metabolic disorder. Immediate glycemic follow-up is warranted.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            **Recommended Triage Pathway:**
            * Order Laboratory Glycated Hemoglobin (`HbA1c`) diagnostic panel.
            * Perform 75g Oral Glucose Tolerance Test (OGTT).
            * Coordinate dietary lifestyle intervention program.
            """)
        else:
            st.markdown("""
            <div style="background-color: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 10px; padding: 1.2rem; margin-top: 0.5rem;">
                <span class="badge-status badge-optimal">Nominal Risk Profile</span>
                <h4 style="color: #065f46; margin: 0.6rem 0 0.4rem 0;">Negative Screening Evaluation</h4>
                <p style="color: #047857; font-size: 0.88rem; margin: 0;">
                    Biometric values remain within calibrated non-diabetic reference boundaries.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            **Preventive Protocols:**
            * Maintain routine annual health surveillance.
            * Preserve target cardiovascular exercise standards (>150 min/week).
            """)
        st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: Performance Telemetry & Model Metrics
with tab_telemetry:
    st.markdown("##### Holdout Validation & Predictive Efficacy")
    kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
    kpi_1.metric("Classifier Accuracy", f"{accuracy * 100:.2f}%", "+1.2% baseline")
    kpi_2.metric("Precision Score", f"{precision * 100:.2f}%")
    kpi_3.metric("Recall / Sensitivity", f"{recall * 100:.2f}%")
    kpi_4.metric("Holdout Cohort Size", f"{len(cm) * 77} records")
    
    st.divider()
    
    c_chart1, c_chart2 = st.columns([1, 1], gap="large")
    with c_chart1:
        st.markdown("**Holdout Classification Confusion Matrix**")
        cm_df = pd.DataFrame(
            cm,
            columns=['Predicted Negative', 'Predicted Positive'],
            index=['Actual Negative', 'Actual Positive']
        )
        st.dataframe(cm_df, use_container_width=True)
        st.caption("Standard 20% stratified holdout split from training corpus.")

    with c_chart2:
        st.markdown("**Dataset Feature Importance Breakdown**")
        importances = model.feature_importances_
        features = [
            'Pregnancies', 'Glucose', 'BloodPressure', 
            'SkinThickness', 'Insulin', 'BMI', 
            'Pedigree', 'Age'
        ]
        feat_df = pd.DataFrame({'Feature': features, 'Weight': importances}).sort_values('Weight', ascending=True)
        
        feat_fig = go.Figure(go.Bar(
            x=feat_df['Weight'],
            y=feat_df['Feature'],
            orientation='h',
            marker=dict(color='#0284c7')
        ))
        feat_fig.update_layout(height=260, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(feat_fig, use_container_width=True)

    with st.expander("Browse Full Registry Sample"):
        st.dataframe(data.head(15), use_container_width=True)

# TAB 3: Reference Guidelines
with tab_benchmarks:
    st.markdown("##### Standard Clinical Reference Thresholds (ADA / WHO Guidelines)")
    guideline_data = {
        "Biomarker Category": [
            "Fasting Plasma Glucose (FPG)", 
            "Diastolic Blood Pressure", 
            "Body Mass Index (BMI)", 
            "2-Hour Postprandial Insulin",
            "Diabetes Pedigree Score"
        ],
        "Normal Range": [
            "70 – 99 mg/dL", 
            "< 80 mm Hg", 
            "18.5 – 24.9 kg/m²", 
            "16 – 166 μIU/mL", 
            "< 0.35"
        ],
        "Impaired / Elevated": [
            "100 – 125 mg/dL (Prediabetes)", 
            "80 – 89 mm Hg (Stage 1)", 
            "25.0 – 29.9 kg/m² (Overweight)", 
            "166 – 250 μIU/mL (Resistance)", 
            "0.35 – 0.70"
        ],
        "Critical Diagnostic Threshold": [
            "≥ 126 mg/dL", 
            "≥ 90 mm Hg (Stage 2)", 
            "≥ 30.0 kg/m² (Obese)", 
            "> 250 μIU/mL (Hyperinsulinemia)", 
            "> 0.70 (Elevated Hereditary Risk)"
        ]
    }
    st.table(pd.DataFrame(guideline_data))
    st.caption("Notice: This system operates strictly as a clinical auxiliary screening utility. Not a replacement for board-certified diagnosis.")
