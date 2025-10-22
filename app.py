import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main background - light and professional */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Remove extra padding and blank space */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        margin-top: 0 !important;
    }

    /* Remove white blocks around widgets */
    div[data-testid="stVerticalBlock"] > div:has(.stMarkdown):empty {
        display: none !important;
    }

    /* Header styling */
    .header-container {
        text-align: center;
        margin-bottom: 2rem;
        padding: 1.5rem;
        background: #D3D3D3;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2F2D72;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        color: #64748b;
        font-size: 1.7rem;
        font-weight: 400;
    }

    /* Input section cards */
    .section-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        margin-bottom: 1.5rem;
    }

    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 4px solid #3b82f6;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: #1e40af;
        font-weight: 500;
    }

    /* Streamlit input styling */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.6rem;
        font-size: 1rem;
        background: white;
        transition: all 0.3s ease;
    }

    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }

    /* Labels */
    label {
        font-weight: 600 !important;
        color: #334155 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
        margin-top: 1rem;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, #2563eb 0%, #1e3a8a 100%);
    }

    /* Result cards */
    .result-card {
        padding: 2.5rem;
        border-radius: 15px;
        margin-top: 2rem;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }

    .result-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }

    .result-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }

    .result-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .result-subtitle {
        font-size: 1.15rem;
        opacity: 0.95;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Column spacing */
    [data-testid="column"] {
        padding: 0 0.5rem;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        font-weight: 600;
        color: #1e3a8a;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 2rem;
        margin-top: 3rem;
        font-size: 0.9rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load model and encoders
@st.cache_resource
def load_model_and_encoders():
    model = joblib.load("extra_trees_credit_model.pkl")
    encoders = {
        col: joblib.load(f"{col}_encoder.pkl") 
        for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
    }
    return model, encoders

model, encoders = load_model_and_encoders()

# Header
st.markdown("""
<div class="header-container">
   <div>
    <h1 class="main-title">Credit Risk Assessment System</h1>
    <p class="subtitle">AI-powered creditworthiness evaluation</p>
   </div>
</div>
""", unsafe_allow_html=True)

# Info box
st.markdown("""
<div class="info-box">
    <strong>ℹ️ How it works:</strong> Enter the applicant's information below. 
    Our machine learning model will analyze the data and provide an instant credit risk assessment.
</div>
""", unsafe_allow_html=True)

# Personal Information Section

st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=80, value=30, help="Applicant's age in years")

with col2:
    sex = st.selectbox("Gender", ["male", "female"], help="Applicant's gender")

with col3:
    job = st.number_input("Job Level", min_value=0, max_value=3, value=1, 
                         help="Job classification (0: unskilled, 1: skilled, 2: highly skilled, 3: management)")


# Housing & Accounts Section

st.markdown('<div class="section-header">🏠 Housing & Accounts</div>', unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    housing = st.selectbox("Housing Status", ["own", "rent", "free"], help="Current housing situation")

with col5:
    saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich"], help="Savings account balance status")

with col6:
    checking_account = st.selectbox("Checking Account", ["little", "moderate", "quite rich"], help="Checking account balance status")



# Credit Information Section

st.markdown('<div class="section-header">💰 Credit Information</div>', unsafe_allow_html=True)

col7, col8 = st.columns(2)

with col7:
    credit_amount = st.number_input("Credit Amount ($)", min_value=0, value=1000, step=100, help="Requested credit amount in dollars")

with col8:
    duration = st.number_input("Duration (months)", min_value=1, max_value=72, value=12, help="Loan duration in months")



# Prediction button
if st.button("🔍 Analyze Credit Risk"):
    input_df = pd.DataFrame({
        "Age": [age],
        "Sex": [encoders["Sex"].transform([sex])[0]],
        "Job": [job],
        "Housing": [encoders["Housing"].transform([housing])[0]],
        "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
        "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
        "Credit amount": [credit_amount],
        "Duration": [duration]
    })

    pred = model.predict(input_df)[0]

    if pred == 1:
        st.markdown("""
        <div class="result-card result-low">
            <div class="result-title">✅ LOW RISK</div>
            <div class="result-subtitle">This applicant demonstrates good creditworthiness</div>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown("""
        <div class="result-card result-high">
            <div class="result-title">⚠️ HIGH RISK</div>
            <div class="result-subtitle">This applicant requires additional review</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📊 View Application Summary"):
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.write("**Personal Details:**")
            st.write(f"- Age: {age} years")
            st.write(f"- Gender: {sex.capitalize()}")
            st.write(f"- Job Level: {job}")
        with summary_col2:
            st.write("**Financial Details:**")
            st.write(f"- Credit Amount: ${credit_amount:,.2f}")
            st.write(f"- Duration: {duration} months")
            st.write(f"- Housing: {housing.capitalize()}")

# Footer
st.markdown("""
<div class="footer">
    Powered by Machine Learning | © 2025 Credit Risk Assessment System
</div>
""", unsafe_allow_html=True)
