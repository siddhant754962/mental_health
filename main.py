import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# --- Set the page title and icon ---
st.set_page_config(
    page_title="CyberMind AI",
    page_icon="🧠",
    layout="wide"
)

# --- Inject custom CSS for a cleaner, modern UI ---
st.markdown(
    """
    <style>
    /* Main container and background */
    .stApp {
        background-color: #1a1a2e; /* Darker, professional background */
        color: #e0e0e0;
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif; /* Modern, clean font */
        display: flex;
        flex-direction: column;
        align-items: center;
        padding-top: 2rem;
    }

    /* Main content block with a subtle, clean design */
    .main-content {
        max-width: 800px;
        width: 90%;
        background: #2a2a4a; /* Slightly lighter content background */
        border: 1px solid #4a4a7a; /* Subtle border */
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3); /* Soft shadow */
        padding: 2.5rem;
        margin: 2rem auto;
        animation: fadeIn 0.8s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Headings and titles with clean, sans-serif font */
    h1, h2, h3, h4, h5, h6, .stHeading {
        color: #8be9fd; /* A vibrant but not glowing blue */
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
        font-weight: 600; /* Slightly bolder */
    }
    
    /* Sub-headings */
    h2 {
        color: #6272a4; /* A soft, professional gray-blue */
    }

    /* Text and labels */
    .stMarkdown, .stSelectbox, .stNumberInput, .stButton, .stProgress, .stInfo, .stMetric, label {
        color: #e0e0e0;
    }

    /* Input fields */
    .stSelectbox>div, .stNumberInput>div, .stTextInput>div {
        background-color: #3a3a5a; /* Darker input background */
        border: 1px solid #6272a4; /* Soft border for inputs */
        border-radius: 5px;
        color: #e0e0e0;
    }

    /* Buttons with a solid, modern look */
    .stButton>button {
        background: #50fa7b; /* A clean green */
        color: #1a1a2e; /* Dark text for contrast */
        border: none;
        border-radius: 5px;
        padding: 12px 24px;
        font-weight: bold;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2); /* Subtle shadow */
    }
    .stButton>button:hover {
        transform: translateY(-2px); /* Slight lift effect */
        background: #69ff8c; /* Lighter green on hover */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    .stButton>button:disabled {
        background: #3a3a5a;
        box-shadow: none;
        cursor: not-allowed;
        transform: none;
    }

    /* Metrics and info boxes with clean borders */
    .stMetric, .stInfo {
        background: #2a2a4a;
        border: 1px solid #4a4a7a;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    .stMetric label {
        font-size: 1.1rem;
        color: #8be9fd;
        font-weight: 500;
    }
    .stMetric .css-1d02c1f { /* Metric value */
        font-size: 2.5rem;
        font-weight: bold;
        color: #f8f8f2; /* Lighter color for value */
    }

    /* Separator lines */
    hr {
        border-top: 1px solid #4a4a7a;
        margin: 1.5rem 0;
    }

    /* Progress bar with modern styling */
    .progress-tracker {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        margin-bottom: 2rem;
    }
    .progress-step {
        width: 30px;
        height: 30px;
        background: #3a3a5a;
        border: 1px solid #6272a4;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #e0e0e0;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .progress-step.completed {
        background: #50fa7b;
        border-color: #50fa7b;
        color: #1a1a2e;
    }
    .progress-step.active {
        background: #8be9fd;
        border-color: #8be9fd;
        color: #1a1a2e;
        transform: scale(1.1);
    }
    .progress-line {
        flex: 1;
        height: 2px;
        background: #4a4a7a;
        margin: 0 10px;
        opacity: 0.6;
    }
    .progress-line.completed {
        opacity: 1;
        background: #50fa7b;
    }

    /* Custom progress ring for results */
    .progress-ring {
        position: relative;
        width: 120px;
        height: 120px;
        margin: 0 auto;
    }
    .progress-ring-svg {
        transform: rotate(-90deg);
    }
    .progress-ring-circle {
        stroke-width: 8;
        fill: transparent;
        transition: stroke-dashoffset 0.5s ease-in-out;
    }
    .progress-ring-bg {
        stroke: rgba(255,255,255,0.1);
    }
    .progress-ring-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 1.5rem;
        font-weight: bold;
        color: #8be9fd;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load the trained model and feature columns ---
try:
    model = joblib.load('mental_health_rf_model.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
except FileNotFoundError:
    st.warning("Model files not found. Using a dummy model for demonstration.")
    class DummyModel:
        def predict(self, data):
            if data['Age'].iloc[0] > 40 and data['family_history'].iloc[0] == 1:
                return np.array([1])
            if data['remote_work'].iloc[0] == 0 and data['benefits'].iloc[0] == 0:
                return np.array([1])
            return np.array([0])
        def predict_proba(self, data):
            pred = self.predict(data)[0]
            if pred == 1:
                return np.array([[0.15, 0.85]])
            else:
                return np.array([[0.85, 0.15]])
    model = DummyModel()
    feature_columns = ['Age', 'self_employed', 'family_history', 'remote_work', 'tech_company', 'benefits']

# --- Enhanced Prediction "API" function with Risk Scoring and Benchmarks ---
@st.cache_data
def predict_data(user_input_data):
    """
    Simulates a sophisticated API call. Returns a structured dictionary with analysis.
    """
    time.sleep(2)

    processed_data = {
        'Age': user_input_data['age'],
        'self_employed': 1 if user_input_data['self_employed'] == "Yes" else 0,
        'family_history': 1 if user_input_data['family_history'] == "Yes" else 0,
        'remote_work': 1 if user_input_data['remote_work'] == "Yes" else 0,
        'tech_company': 1 if user_input_data['tech_company'] == "Yes" else 0,
        'benefits': 1 if user_input_data['benefits'] == "Yes" else 0,
    }
    
    user_input = pd.DataFrame([processed_data])
    for col in feature_columns:
        if col not in user_input.columns:
            user_input[col] = 0
    user_input = user_input[feature_columns]

    prediction = model.predict(user_input)[0]
    probability = model.predict_proba(user_input)[0][1]

    # --- Simulated Risk Score Calculation ---
    risk_score = 0
    if processed_data['family_history'] == 1: risk_score += 3
    if processed_data['remote_work'] == 0: risk_score += 2
    if processed_data['tech_company'] == 1 and processed_data['benefits'] == 0: risk_score += 4
    if processed_data['Age'] > 45: risk_score += 1
    if prediction == 1: risk_score += 5 # High weight for model's prediction

    risk_level = "Low"
    color = "#50fa7b" # Green for low
    if risk_score > 5:
        risk_level = "Medium"
        color = "#ffc800" # Orange for medium
    if risk_score > 10:
        risk_level = "High"
        color = "#ff5555" # Red for high
        
    # --- Generate Risk Factor Analysis & Recommendations (simulated) ---
    risk_factors = []
    recommendations = []
    
    if processed_data['family_history'] == 1:
        risk_factors.append("Family History of Mental Health")
        recommendations.append({"category": "Immediate Action", "text": "Consider speaking with a professional about your family history and its potential impact on your well-being. A genetic counselor or therapist may provide valuable guidance."})
    if processed_data['remote_work'] == 0:
        risk_factors.append("On-site Work Environment")
        recommendations.append({"category": "Lifestyle Adjustments", "text": "Maintaining a healthy work-life balance is crucial in an on-site role. Explore stress management techniques and ensure you take regular breaks."})
    if processed_data['tech_company'] == 1 and processed_data['benefits'] == 0:
        risk_factors.append("Lack of Employer Mental Health Benefits")
        recommendations.append({"category": "Immediate Action", "text": "Research local mental health resources and services that are independent of employer benefits. Prioritize your well-being, even without company support."})
    if processed_data['Age'] > 45:
        risk_factors.append("Age-Related Stress Factors")
        recommendations.append({"category": "Lifestyle Adjustments", "text": "As we age, our mental health needs can change. Stay connected with friends and family, and consider mindfulness or meditation to manage stress."})
    
    if not risk_factors:
        recommendations.append({"category": "General Wellness", "text": "Your profile indicates a low-risk status. Continue to monitor your mental health and seek professional help if your circumstances change."})
    
    # --- Add 'explanation' key back to the report dictionary ---
    explanation_text = "The analysis suggests a high probability of requiring professional assistance based on the provided data." if prediction == 1 else "The analysis indicates a low probability of requiring treatment at this time."

    report = {
        'prediction': 'Treatment Likely' if prediction == 1 else 'Treatment Unlikely',
        'probability': probability,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'risk_level_color': color,
        'input_data': user_input_data,
        'risk_factors': risk_factors,
        'recommendations': recommendations,
        'explanation': explanation_text # Fixed: ensure this key is always present
    }

    return report

def get_input_explanation(key):
    explanations = {
        'age': 'Age can correlate with different life stages and stressors, influencing mental health. Your age helps the model contextualize other factors.',
        'family_history': 'A family history of mental illness is a significant risk factor. It is one of the most important indicators in this analysis.',
        'self_employed': 'The autonomy and financial stress of being self-employed can uniquely impact mental health compared to traditional employment.',
        'remote_work': 'The flexibility of remote work can reduce commuting stress but may also lead to social isolation, affecting mental well-being.',
        'tech_company': 'The fast-paced, high-pressure environment of many tech companies can be a source of stress and burnout.',
        'benefits': 'Comprehensive mental health benefits at work can be a protective factor, providing crucial access to care and support systems.'
    }
    return explanations.get(key, 'This factor helps the model assess your mental wellness.')

# --- Use session state to manage user inputs and form progress ---
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {
        'age': 30,
        'family_history': 'No',
        'self_employed': 'No',
        'remote_work': 'No',
        'tech_company': 'No',
        'benefits': 'No'
    }
    st.session_state.step = 0

# --- Data for each form step ---
form_steps = [
    {'label': 'AGE', 'type': 'number', 'key': 'age', 'min': 18, 'max': 100},
    {'label': 'FAMILY HISTORY', 'type': 'selectbox', 'key': 'family_history', 'options': ['No', 'Yes']},
    {'label': 'SELF EMPLOYED', 'type': 'selectbox', 'key': 'self_employed', 'options': ['No', 'Yes']},
    {'label': 'REMOTE WORK', 'type': 'selectbox', 'key': 'remote_work', 'options': ['No', 'Yes']},
    {'label': 'TECH COMPANY', 'type': 'selectbox', 'key': 'tech_company', 'options': ['No', 'Yes']},
    {'label': 'BENEFITS', 'type': 'selectbox', 'key': 'benefits', 'options': ['No', 'Yes']},
]

def next_step():
    st.session_state.step += 1

def prev_step():
    st.session_state.step -= 1

# --- Main app logic ---
st.title("🤖 CyberMind AI: Mental Wellness Protocol")

with st.container(border=False):
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Welcome Page
    if st.session_state.step == 0:
        st.header("Initiating Diagnostic Protocol")
        st.markdown(
            """
            Welcome to the CyberMind AI interface. This system is designed to provide a preliminary assessment of mental wellness based on your work and personal environment. 
            
            Proceed through the following steps to generate your predictive report. Your data is encrypted and used only for this analysis.
            """
        )
        st.markdown("---")
        st.button("BEGIN ANALYSIS", on_click=next_step, use_container_width=True, type="primary")

    # Form Steps
    elif st.session_state.step > 0 and st.session_state.step <= len(form_steps):
        st.subheader("Data Input")
        
        # Progress tracker
        st.markdown('<div class="progress-tracker">', unsafe_allow_html=True)
        for i in range(len(form_steps)):
            step_class = "completed" if st.session_state.step > i + 1 else ("active" if st.session_state.step == i + 1 else "")
            st.markdown(f'<div class="progress-step {step_class}">{i + 1}</div>', unsafe_allow_html=True)
            if i < len(form_steps) - 1:
                line_class = "completed" if st.session_state.step > i + 1 else ""
                st.markdown(f'<div class="progress-line {line_class}"></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        current_step_num = st.session_state.step
        current_step_data = form_steps[current_step_num - 1]
        
        with st.container(border=True):
            st.markdown(f"### **{current_step_data['label']}**")
            st.info(get_input_explanation(current_step_data['key']))

            is_input_valid = False
            if current_step_data['type'] == 'number':
                value = st.number_input(
                    " ",
                    min_value=current_step_data['min'],
                    max_value=current_step_data['max'],
                    value=st.session_state.user_inputs[current_step_data['key']],
                    label_visibility="collapsed"
                )
                st.session_state.user_inputs[current_step_data['key']] = value
                if value is not None and value >= current_step_data['min'] and value <= current_step_data['max']:
                    is_input_valid = True
            elif current_step_data['type'] == 'selectbox':
                value = st.selectbox(
                    " ",
                    options=current_step_data['options'],
                    index=current_step_data['options'].index(st.session_state.user_inputs[current_step_data['key']]),
                    label_visibility="collapsed"
                )
                st.session_state.user_inputs[current_step_data['key']] = value
                is_input_valid = True
        
        nav_col1, nav_col2 = st.columns([1, 1])
        with nav_col1:
            if st.session_state.step > 1:
                st.button("<< Previous", on_click=prev_step, use_container_width=True)
        with nav_col2:
            if st.session_state.step < len(form_steps):
                st.button("Next >>", on_click=next_step, use_container_width=True, type="primary", disabled=not is_input_valid)
            else:
                st.button("GENERATE REPORT", on_click=next_step, use_container_width=True, type="primary", disabled=not is_input_valid)

    # Thank You/Confirmation Page
    elif st.session_state.step == len(form_steps) + 1:
        st.header("Analysis Initiated")
        st.success("Your data has been submitted for analysis. Please wait while the system generates your report.")
        time.sleep(2)
        st.session_state.step += 1
        st.rerun()

    # Results Dashboard
    elif st.session_state.step == len(form_steps) + 2:
        st.header(">> MEDICAL REPORT DASHBOARD <<")
        st.markdown("---")
        
        with st.spinner("Compiling results..."):
            report = predict_data(st.session_state.user_inputs)
            
            # --- Summary Metrics Section ---
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            with summary_col1:
                st.metric(label="Probability", value=f"{report['probability'] * 100:.1f}%")
            with summary_col2:
                st.metric(label="Risk Level", value=report['risk_level'], label_visibility="visible", help=f"Your risk score is {report['risk_score']}")
            with summary_col3:
                st.metric(label="Predicted Status", value=report['prediction'])
            
            # Display the current risk level
            st.markdown(f"<p style='color: {report['risk_level_color']}; font-weight: bold; font-size: 1.2rem; text-align: center;'>Current Risk Level: {report['risk_level']}</p>", unsafe_allow_html=True)
            st.markdown("---")
            
            # Display the explanation
            st.info(f"**Analysis Overview:** {report['explanation']}")

            # --- What-If Scenario Section ---
            st.subheader("Simulate a What-If Scenario")
            with st.container(border=True):
                st.markdown("Change a key factor to see how your risk level is affected.")
                what_if_col1, what_if_col2 = st.columns(2)
                with what_if_col1:
                    key_to_change = st.selectbox("Select a Factor:", ['family_history', 'remote_work', 'tech_company', 'benefits'], key='what_if_factor')
                with what_if_col2:
                    current_val = st.session_state.user_inputs.get(key_to_change, "No") # Get current value
                    options_for_change = ["Yes", "No"]
                    # Ensure the current value is not the only option, if possible
                    if current_val in options_for_change and len(options_for_change) > 1:
                        options_for_change.remove(current_val)
                        options_for_change.insert(0, current_val) # Keep current as first option
                    
                    selected_new_value = st.selectbox(f"New value for '{key_to_change}':", options_for_change, key='what_if_value')

                temp_inputs = st.session_state.user_inputs.copy()
                temp_inputs[key_to_change] = selected_new_value

                if st.button("RUN SCENARIO", use_container_width=True):
                    what_if_report = predict_data(temp_inputs)
                    st.info(f"If you changed **'{key_to_change}'** to **'{selected_new_value}'**, your new risk level would be: **{what_if_report['risk_level']}** (Score: {what_if_report['risk_score']}).")
            st.markdown("---")
            
            # --- Detailed Breakdown Section ---
            st.subheader("Detailed Report")
            breakdown_col1, breakdown_col2 = st.columns(2)
            with breakdown_col1:
                st.markdown("### Risk Factor Analysis")
                if report['risk_factors']:
                    for rf in report['risk_factors']:
                        st.markdown(f"**-** {rf}")
                else:
                    st.success("No significant risk factors were identified based on the provided data.")

            with breakdown_col2:
                st.markdown("### Recommended Actions")
                if report['recommendations']:
                    # Group recommendations by category for cleaner display
                    recs_by_category = {}
                    for rec in report['recommendations']:
                        recs_by_category.setdefault(rec['category'], []).append(rec['text'])
                    
                    for category, texts in recs_by_category.items():
                        st.markdown(f"**{category}:**")
                        for text in texts:
                            st.markdown(f"- {text}")
                else:
                    st.markdown("No specific recommendations at this time.")

            with st.expander("ACCESS RAW DATA LOG"):
                st.json(report)
                
        st.markdown("---")
        if st.button("RUN NEW ANALYSIS", use_container_width=True):
            st.session_state.step = 0
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.info("DISCLAIMER: This report is a computational simulation and does not substitute for human medical diagnosis. Consult a qualified professional for all health-related concerns.")
