import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Page Configuration
st.set_page_config(
    page_title="ApexHealth | Clinical Decision Support",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Adaptive CSS (Single-line safe injection)
css_code = (
    "<style>\n"
    "@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');\n"
    "html, body, [class*='css'] { font-family: 'Plus Jakarta Sans', sans-serif; }\n"
    ".hero-box {\n"
    "    padding: 1.5rem 2rem;\n"
    "    background: linear-gradient(135deg, #091e3a 0%, #1e4d7b 100%);\n"
    "    border-radius: 14px;\n"
    "    color: #ffffff;\n"
    "    margin-bottom: 1.5rem;\n"
    "}\n"
    ".hero-box h1 { margin: 0; font-size: 1.8rem; color: #ffffff !important; }\n"
    ".hero-box p { margin: 0.3rem 0 0 0; color: #94a3b8 !important; font-size: 0.95rem; }\n"
    "div.stButton > button:first-child {\n"
    "    background: #0284c7;\n"
    "    color: #ffffff !important;\n"
    "    border-radius: 8px;\n"
    "    font-weight: 600;\n"
    "    border: none;\n"
    "    padding: 0.6rem 1.2rem;\n"
    "}\n"
    "</style>"
)
st.markdown(css_code, unsafe_allow_html=True)

# Training & Caching Core Engine
@st.cache_resource
def load_and_train():
    try:
        data = pd.read_csv("dataset.csv")
    except FileNotFoundError:
        data = pd.read_csv("diabetes.csv")
    
    feature_cols = [
        'Pregnancies', 'Glucose', 'BloodPressure', 
        'SkinThickness', 'Insulin', 'BMI', 
        'DiabetesPedigreeFunction', 'Age'
    ]
    X = data[feature_cols]
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

data, model, accuracy, precision, recall, cm = load_and_train()

# Header Component
st.markdown(
    "<div class='hero-box'>"
    "<h1>ApexHealth Diagnostics • CDSS</h1>"
    "<p>Machine Learning Diabetes Risk Stratification & Clinical Screening Portal</p>"
    "</div>",
    unsafe_allow_html=True
)

# Tabs Navigation
tab_screen, tab_telemetry, tab_benchmarks = st.tabs([
    "🩺 Patient Screening Assessment",
    "📈 Performance & Telemetry",
    "📜 Clinical Reference Directory"
])

# TAB 1: Clinical Assessment Form & Output
with tab_screen:
    col_input, col_analytics = st.columns([1.1, 0.9], gap="large")
    
    with col_input:
        st.subheader("🔬 Biomarker Input Matrix")
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

    with col_analytics:
        st.subheader("Diagnostic Risk Stratification")
        
        patient_vector = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]])
        pred = model.predict(patient_vector)[0]
        prob = model.predict_proba(patient_vector)[0]
        risk_pct = prob[1] * 100

        # High-End Theme-Safe Dial
        gauge_chart = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Calculated Risk Score", 'font': {'size': 16}},
            number={'suffix': "%", 'font': {'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#0284c7", 'thickness': 0.28},
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(34, 197, 94, 0.25)'},
                    {'range': [30, 65], 'color': 'rgba(234, 179, 8, 0.25)'},
                    {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.25)'}
                ],
                'threshold': {
                    'line': {'color': "#ef4444" if risk_pct >= 65 else "#eab308" if risk_pct >= 30 else "#22c55e", 'width': 3},
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
        st.plotly_chart(gauge_chart, use_container_width=True)
        
        # Clinical Risk Stratification Output
        if pred == 1:
            st.error("🔴 **High Clinical Risk — Positive Diagnostic Indicator**")
            st.write("Biometric markers cross critical thresholds for metabolic disorder. Immediate glycemic follow-up is warranted.")
            st.markdown(
                "**Recommended Clinical Pathway:**\n"
                "- Order formal Laboratory HbA1c diagnostic panel.\n"
                "- Schedule oral glucose tolerance testing (OGTT).\n"
                "- Initiate personalized dietary lifestyle guidance."
            )
        else:
            st.success("🟢 **Nominal Risk Profile — Negative Screening Evaluation**")
            st.write("Biometric values remain within calibrated non-diabetic reference boundaries.")
            st.markdown(
                "**Preventive Protocols:**\n"
                "- Maintain routine annual metabolic surveillance.\n"
                "- Sustain regular aerobic exercise standards (>150 min/week)."
            )

# TAB 2: Performance Telemetry
with tab_telemetry:
    st.markdown("##### Holdout Validation & Predictive Efficacy")
    kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
    kpi_1.metric("Classifier Accuracy", f"{accuracy * 100:.2f}%")
    kpi_2.metric("Precision Score", f"{precision * 100:.2f}%")
    kpi_3.metric("Recall / Sensitivity", f"{recall * 100:.2f}%")
    kpi_4.metric("Dataset Total Records", f"{len(data)} entries")
    
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
        feat_fig.update_layout(
            height=260, 
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
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
