import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# --- Page Config ---
st.set_page_config(page_title="AI Predictive Maintenance", page_icon="🏭", layout="wide")

# --- Title and Header ---
st.title("🏭 AI-Powered Predictive Maintenance System")
st.markdown("This interactive dashboard monitors simulated IoT sensor telemetry and uses a Machine Learning model to predict if an industrial machine is going to fail before it happens.")

# --- Data Loading and Model Training (Cached for speed) ---
@st.cache_resource
def load_and_train_model():
    """Builds and trains the model strictly once upon page load"""
    data_path = 'data/sensor_data.csv'
    if not os.path.exists(data_path):
        st.error("Error: Sensor dataset not found. Please run main.py first to generate the simulation.")
        return None, None
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Machine_ID', 'Failure'])
    y = df['Failure']
    
    # Train Model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X, y)
    
    return model, df

model, df = load_and_train_model()

if model:
    # --- Sidebar for Interactive Simulation ---
    st.sidebar.header("🎛️ Live Sensor Simulation")
    st.sidebar.markdown("Adjust the sliders below to simulate live incoming IoT telemetry. The AI will instantly process the values predict the machine's health status.")
    
    temp = st.sidebar.slider("Temperature (°C)", min_value=50.0, max_value=120.0, value=75.0, step=0.5)
    vib = st.sidebar.slider("Vibration (mm/s)", min_value=1.0, max_value=15.0, value=5.0, step=0.1)
    pressure = st.sidebar.slider("Pressure (psi)", min_value=60.0, max_value=150.0, value=100.0, step=1.0)
    rpm = st.sidebar.slider("Rotational Speed (RPM)", min_value=1000.0, max_value=3000.0, value=2000.0, step=10.0)
    
    # Real-time Prediction
    input_data = pd.DataFrame({
        'Temperature_C': [temp],
        'Vibration_mm_s': [vib],
        'Pressure_psi': [pressure],
        'Rotational_Speed_rpm': [rpm]
    })
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] # Probability of failure
    
    # --- Main Dashboard ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Live AI Prediction Status")
        if prediction == 1:
            st.error(f"🚨 **CRITICAL: FAILURE PREDICTED** 🚨\n\nThe AI detects an incoming breakdown! \n\n**Confidence / Failure Probability:** {probability*100:.1f}%")
        else:
            st.success(f"✅ **SYSTEM HEALTHY**\n\nThe machine is operating safely. \n\n**Failure Probability:** {probability*100:.1f}%")
            
        st.markdown("### 📊 Dataset Overview (Latest 5 Readings)")
        st.dataframe(df.tail(5), use_container_width=True)

    with col2:
        st.subheader("📈 System Analytics")
        # Display the pre-generated plots from main.py
        try:
            st.image('images/feature_importance.png', caption='AI Feature Importance', use_container_width=True)
            st.image('images/confusion_matrix.png', caption='Model Accuracy Metrics', use_container_width=True)
        except Exception as e:
            st.warning("Analytics plots not found. Ensure you ran main.py first.")
            
else:
    st.stop()
