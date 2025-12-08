"""
Web application for Cardiovascular Disease Prediction
Includes both FastAPI (backend) and Streamlit (frontend)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.inference import CardiovascularPredictor
from src import config

# Page config
st.set_page_config(
    page_title="Cardiovascular Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load predictor (cached)"""
    return CardiovascularPredictor()


def main():
    # Load predictor
    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
    
    # Header
    st.title("❤️ Cardiovascular Disease Risk Prediction")
    st.markdown("""
    This application predicts the risk of cardiovascular disease based on patient data.
    Please enter the patient information below to get a risk assessment.
    """)
    
    # Sidebar - Model Info
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.metric("Model Type", "XGBoost")
        st.metric("ROC-AUC Score", f"{predictor.metadata['test_roc_auc']:.4f}")
        st.metric("Accuracy", f"{predictor.metadata['test_accuracy']:.4f}")
        st.metric("Features", len(predictor.features))
        
        st.markdown("---")
        st.markdown("""
        **Risk Categories:**
        - 🟢 Low Risk: < 40%
        - 🟡 Medium Risk: 40-70%
        - 🔴 High Risk: > 70%
        """)
        
        st.markdown("---")
        st.markdown("""
        **Model Version:** 1.0  
        **Last Updated:** 2024
        """)
    
    # Input Form
    st.header("📋 Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        age_years = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
        age_days = age_years * 365
        
        gender = st.selectbox("Gender", 
                             options=[1, 2],
                             format_func=lambda x: "Female" if x == 1 else "Male")
        
        height = st.number_input("Height (cm)", min_value=120, max_value=220, value=170)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    
    with col2:
        st.subheader("Clinical Measurements")
        ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=80, max_value=250, value=120)
        ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=200, value=80)
        
        cholesterol = st.selectbox("Cholesterol", 
                                   options=[1, 2, 3],
                                   format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x])
        
        gluc = st.selectbox("Glucose", 
                           options=[1, 2, 3],
                           format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x])
    
    with col3:
        st.subheader("Lifestyle Factors")
        smoke = st.selectbox("Smoking", options=[0, 1], 
                            format_func=lambda x: "No" if x == 0 else "Yes")
        alco = st.selectbox("Alcohol", options=[0, 1], 
                           format_func=lambda x: "No" if x == 0 else "Yes")
        active = st.selectbox("Physical Activity", options=[0, 1], 
                             format_func=lambda x: "No" if x == 0 else "Yes")
    
    # Predict button
    st.markdown("---")
    if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
        # Prepare input
        input_data = {
            "age": age_days,
            "gender": gender,
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": cholesterol,
            "gluc": gluc,
            "smoke": smoke,
            "alco": alco,
            "active": active
        }
        
        # Make prediction
        with st.spinner("Analyzing..."):
            try:
                result = predictor.predict(input_data)
                
                # Display results
                st.markdown("---")
                st.header("📊 Prediction Results")
                
                # Risk probability gauge
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=result['probability'] * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Risk Probability", 'font': {'size': 24}},
                        delta={'reference': 50, 'increasing': {'color': "red"}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkblue"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 40], 'color': '#90EE90'},
                                {'range': [40, 70], 'color': '#FFD700'},
                                {'range': [70, 100], 'color': '#FF6B6B'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': result['probability'] * 100
                            }
                        }
                    ))
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.metric("Prediction", result['prediction_label'])
                    st.metric("Risk Category", result['risk_category'])
                
                with col3:
                    st.metric("Probability", f"{result['probability']:.2%}")
                    st.metric("Confidence", f"{result['confidence']:.2%}")
                
                # Risk interpretation
                st.markdown("---")
                st.subheader("🎯 Risk Interpretation")
                
                if result['probability'] < 0.4:
                    st.success("""
                    **Low Risk**: The patient has a low probability of cardiovascular disease.
                    - Continue healthy lifestyle
                    - Regular checkups recommended
                    - Maintain current health practices
                    """)
                elif result['probability'] < 0.7:
                    st.warning("""
                    **Medium Risk**: The patient has a moderate probability of cardiovascular disease.
                    - Regular medical monitoring recommended
                    - Consider lifestyle modifications
                    - Discuss prevention strategies with doctor
                    """)
                else:
                    st.error("""
                    **High Risk**: The patient has a high probability of cardiovascular disease.
                    - Immediate medical consultation recommended
                    - Comprehensive health assessment needed
                    - Discuss treatment options with cardiologist
                    """)
                
                # Additional metrics
                st.markdown("---")
                st.subheader("📈 Calculated Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                bmi = weight / ((height / 100) ** 2)
                pulse_pressure = ap_hi - ap_lo
                
                with col1:
                    st.metric("BMI", f"{bmi:.1f}")
                    if bmi < 18.5:
                        st.caption("Underweight")
                    elif bmi < 25:
                        st.caption("Normal")
                    elif bmi < 30:
                        st.caption("Overweight")
                    else:
                        st.caption("Obese")
                
                with col2:
                    st.metric("Pulse Pressure", f"{pulse_pressure}")
                
                with col3:
                    map_val = (ap_hi + 2 * ap_lo) / 3
                    st.metric("MAP", f"{map_val:.0f}")
                
                with col4:
                    health_risk = (cholesterol - 1) * 2 + (gluc - 1) * 2 + smoke * 3 + alco * 2 - active * 2
                    st.metric("Health Risk Score", f"{health_risk}")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>⚠️ <b>Disclaimer:</b> This tool is for educational and screening purposes only. 
    It should not replace professional medical advice, diagnosis, or treatment.</p>
    <p>Built with ❤️ using Streamlit and XGBoost | Model ROC-AUC: 0.7912</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()