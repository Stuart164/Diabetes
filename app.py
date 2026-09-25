import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Diabetes Risk Assessment", layout="centered")

st.title("🩺 AI Diabetes Risk Assessment")
st.markdown("Predict the likelihood of diabetes based on standard health metrics.")

# 1. Load and prepare dataset
@st.cache_resource
def train_model():
    data = pd.read_csv("diabetes.csv")
    
    # Matching the 4 features required by the project specifications
    features = ['Glucose', 'BloodPressure', 'BMI', 'Age']
    X = data[features]
    y = data['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

try:
    model, accuracy = train_model()
    st.sidebar.success(f"Model trained! Test Accuracy: {accuracy * 100:.2f}%")
except Exception as e:
    st.error(f"Error loading diabetes.csv: {e}")
    st.stop()

# 2. Input Fields
st.subheader("Patient Health Metrics")
col1, col2 = st.columns(2)

with col1:
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
    bp = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)

with col2:
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    age = st.number_input("Age (Years)", min_value=1, max_value=120, value=30, step=1)

# 3. Prediction Action
if st.button("Predict Risk", type="primary"):
    input_data = np.array([[glucose, bp, bmi, age]])
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    st.divider()
    if prediction == 1:
        st.error(f"⚠️ **Result:** The patient is likely to have diabetes.")
    else:
        st.success(f"✅ **Result:** The patient is not likely to have diabetes.")
        
    st.write(f"- **Probability of Diabetes:** {probabilities[1]*100:.2f}%")
    st.write(f"- **Probability of Non-Diabetes:** {probabilities[0]*100:.2f}%")
