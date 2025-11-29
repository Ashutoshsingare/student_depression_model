import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image

# ----------------------------
# Configure Streamlit App
# ----------------------------
st.set_page_config(
    page_title="Student Depression Predictor", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
.main-header {
    text-align: center;
    padding: 2rem 0;
    background-color: #E3F2FD;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.main-header h1 {
    color: #000000;
    font-size: 3rem;
    font-weight: 700;
    margin: 0;
}

.main-header p {
    color: #000000;
    font-size: 1.2rem;
    margin-top: 0.5rem;
}
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 2rem 0 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .section-header h3 {
        color: white;
        margin: 0;
        font-size: 1.5rem;
    }
    
    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Input containers */
    .stNumberInput, .stTextInput, .stSelectbox {
        margin-bottom: 1rem;
    }
    
    /* Predict button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        font-weight: 600;
        height: 3.5em;
        width: 100%;
        border-radius: 12px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        margin-top: 2rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Result boxes */
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        animation: fadeIn 0.5s;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .result-very-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .result-low {
        background: linear-gradient(135deg, #5ee7df 0%, #b490ca 100%);
        color: white;
    }
    
    .result-moderate {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .result-high {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    
    .result-very-high {
        background: linear-gradient(135deg, #fc6076 0%, #ff9a44 100%);
        color: white;
    }
    
    /* Sidebar styling - Simple black background */
    [data-testid="stSidebar"] {
        background-color: #000000;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background-color: #000000;
    }
    
    /* Sidebar content visibility */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] a {
        color: #ffffff !important;
        text-decoration: underline;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    /* Hide default streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Banner image container */
    .banner-container {
        max-height: 300px;
        overflow: hidden;
        border-radius: 15px;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .banner-container img {
        width: 100%;
        height: 300px;
        object-fit: cover;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("AdaBoost_model.pkl")

model = load_model()

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='color: white; margin: 0;'>🧠</h1>
            <h2 style='color: white; margin: 0.5rem 0;'>Depression Predictor</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <div style='color: white; padding: 1rem;'>
            <p style='font-size: 1rem; line-height: 1.6;'>
                This tool uses machine learning to assess depression likelihood in students based on lifestyle factors and personal circumstances.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <div style='color: white;'>
            <h3 style='color: white;'>📊 About</h3>
            <p><strong>Model:</strong> AdaBoost Classifier</p>
            <p><strong>Developer:</strong> Ashutosh</p>
            <br>
            <h3 style='color: white;'>🔗 Connect</h3>
            <p>GitHub: <a href='https://github.com/ashutosh637' style='color: #4facfe;'>Ashutosh singare</a></p>
            <p>LinkedIn: <a href='https://www.linkedin.com/in/ashutosh-singare-6ba2bb327/' style='color: #4facfe;'>Ashutosh Singare</a></p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<p style='color: #ccc; font-size: 0.8rem; text-align: center;'>⚠️ This tool is for educational purposes only. Please consult healthcare professionals for medical advice.</p>", unsafe_allow_html=True)

# ----------------------------
# Main Header
# ----------------------------
st.markdown("""
    <div class='main-header'>
        <h1>🧠 Student Depression Predictor</h1>
        <p>AI-powered mental health risk assessment for students</p>
    </div>
""", unsafe_allow_html=True)

# Banner image - smaller and contained
st.markdown("<div class='banner-container'>", unsafe_allow_html=True)
try:
    image = Image.open('banner.png')
    st.image(image, use_container_width=True)
except:
    st.info("📸 Banner image not found - continuing without it")
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Instructions Card
# ----------------------------
st.markdown("""
    <div class='info-card'>
        <h4>📋 How to use this tool:</h4>
        <ol>
            <li>Fill in all the student information fields below</li>
            <li>Be as accurate as possible with the ratings and values</li>
            <li>Click the "Predict Depression Risk" button to get results</li>
            <li>Review the risk assessment and recommendations</li>
        </ol>
    </div>
""", unsafe_allow_html=True)

# ----------------------------
# Personal Information Section
# ----------------------------
st.markdown("""
    <div class='section-header'>
        <h3>👤 Personal Information</h3>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    id_val = st.number_input("🆔 Student ID", min_value=0, step=1, help="Enter any numeric identifier")
    gender = st.radio("⚧ Gender", ["Male", "Female"])

with col2:
    age = st.number_input("🎂 Age", min_value=1, max_value=120, step=1, value=20)
    city = st.text_input("🏙️ City", placeholder="Enter city name")

with col3:
    profession = st.text_input("💼 Profession", placeholder="e.g., Student, Part-time worker")
    degree = st.text_input("🎓 Degree", placeholder="e.g., B.Tech, MBA")

# ----------------------------
# Academic Factors Section
# ----------------------------
st.markdown("""
    <div class='section-header'>
        <h3>📚 Academic Factors</h3>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    academic_pressure = st.slider("📖 Academic Pressure", 1.0, 5.0, 3.0, 0.5, 
                                  help="1 = Very Low, 5 = Very High")
    cgpa = st.number_input("📊 CGPA", min_value=0.0, max_value=10.0, step=0.01, value=7.5)

with col2:
    study_satisfaction = st.slider("😊 Study Satisfaction", 1.0, 5.0, 3.0, 0.5,
                                   help="1 = Very Unsatisfied, 5 = Very Satisfied")
    study_pressure_hours = st.number_input("⏰ Work/Study Hours per Week", 
                                          min_value=0, max_value=168, step=1, value=40)

with col3:
    work_pressure = st.slider("💼 Work Pressure", 0.0, 5.0, 0.0, 0.5,
                             help="1 = Very Low, 5 = Very High")
    job_satisfaction = st.slider("😌 Job Satisfaction", 0.0, 5.0, 0.0, 0.5,
                                help="0 = Not Applicable, 5 = Very Satisfied")

# ----------------------------
# Lifestyle & Health Section
# ----------------------------
st.markdown("""
    <div class='section-header'>
        <h3>🏃 Lifestyle & Health</h3>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    sleep_duration = st.selectbox("😴 Sleep Duration", 
                                  ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"],
                                  index=2)
    dietary_habits = st.radio("🥗 Dietary Habits", ["Healthy", "Unhealthy"])

with col2:
    financial_stress = st.slider("💰 Financial Stress", 1, 5, 3,
                                help="1 = Very Low, 5 = Very High")
    family_history = st.radio("🧬 Family History of Mental Illness", ["Yes", "No"])

with col3:
    suicidal_thoughts = st.radio("⚠️ Ever Had Suicidal Thoughts?", ["Yes", "No"])
    st.markdown("<p style='color: #666; font-size: 0.85rem; margin-top: -0.5rem;'>Your privacy is protected</p>", unsafe_allow_html=True)

# ----------------------------
# Mapping categorical to numeric
# ----------------------------
gender = 1 if gender == 'Male' else 0
dietary_habits = 1 if dietary_habits == 'Healthy' else 0
suicidal_thoughts = 1 if suicidal_thoughts == 'Yes' else 0
family_history = 1 if family_history == 'Yes' else 0

sleep_mapping = {
    'Less than 5 hours': 4,
    '5-6 hours': 5.5,
    '7-8 hours': 7.5,
    'More than 8 hours': 9
}
sleep_duration = sleep_mapping.get(sleep_duration, 7.5)

# ----------------------------
# Create Input DataFrame
# ----------------------------
columns = ['id', 'Gender', 'Age', 'City', 'Profession', 'Academic Pressure',
           'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction',
           'Sleep Duration', 'Dietary Habits', 'Degree',
           'Have you ever had suicidal thoughts ?', 'Work/Study Hours',
           'Financial Stress', 'Family History of Mental Illness']

input_df = pd.DataFrame([[id_val, gender, age, city, profession, academic_pressure,
                          work_pressure, cgpa, study_satisfaction, job_satisfaction,
                          sleep_duration, dietary_habits, degree, suicidal_thoughts,
                          study_pressure_hours, financial_stress, family_history]],
                        columns=columns)

# ----------------------------
# Prediction Button
# ----------------------------
if st.button("🔮 Predict Depression Risk"):
    with st.spinner("🔄 Analyzing data..."):
        try:
            prediction_proba = model.predict_proba(input_df)
            depression_prob = prediction_proba[0][1] * 100

            # Determine risk level and styling
            if depression_prob < 20:
                risk_level = "Very Low Risk"
                risk_class = "result-very-low"
                emoji = "✅"
                message = "The assessment indicates a very low likelihood of depression. Continue maintaining healthy habits!"
                recommendations = [
                    "Keep up your healthy lifestyle choices",
                    "Maintain good sleep hygiene",
                    "Stay connected with friends and family",
                    "Continue balanced study-life routine"
                ]
            elif depression_prob < 40:
                risk_level = "Low Risk"
                risk_class = "result-low"
                emoji = "🟢"
                message = "The assessment shows a low likelihood of depression. Stay mindful of your mental health."
                recommendations = [
                    "Monitor stress levels regularly",
                    "Maintain work-life balance",
                    "Practice relaxation techniques",
                    "Keep communication channels open"
                ]
            elif depression_prob < 60:
                risk_level = "Moderate Risk"
                risk_class = "result-moderate"
                emoji = "🟡"
                message = "The assessment indicates moderate risk. Consider taking preventive measures."
                recommendations = [
                    "Talk to a counselor or trusted person",
                    "Develop stress management strategies",
                    "Improve sleep schedule and diet",
                    "Engage in regular physical activity"
                ]
            elif depression_prob < 80:
                risk_level = "High Risk"
                risk_class = "result-high"
                emoji = "🟠"
                message = "The assessment shows concerning signs. Professional support is strongly recommended."
                recommendations = [
                    "Seek professional counseling services",
                    "Reach out to campus mental health resources",
                    "Connect with trusted friends or family",
                    "Consider reducing workload if possible"
                ]
            else:
                risk_level = "Very High Risk"
                risk_class = "result-very-high"
                emoji = "🔴"
                message = "The assessment indicates serious concerns. Please seek professional help immediately."
                recommendations = [
                    "Contact a mental health professional urgently",
                    "Reach out to crisis hotlines if needed",
                    "Inform trusted individuals about your situation",
                    "Don't hesitate to visit student counseling services"
                ]

            # Display results
            st.markdown(f"""
                <div class='result-box {risk_class}'>
                    <h1 style='margin: 0; font-size: 3rem;'>{emoji}</h1>
                    <h2 style='margin: 1rem 0;'>{risk_level}</h2>
                    <p style='font-size: 1.5rem; font-weight: 600;'>Risk Score: {depression_prob:.1f}%</p>
                    <p style='font-size: 1.1rem; margin-top: 1rem;'>{message}</p>
                </div>
            """, unsafe_allow_html=True)

            # Recommendations
            st.markdown("""
                <div class='section-header'>
                    <h3>💡 Recommendations</h3>
                </div>
            """, unsafe_allow_html=True)

            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                    <div class='info-card'>
                        <p style='margin: 0; font-size: 1rem;'><strong>{i}.</strong> {rec}</p>
                    </div>
                """, unsafe_allow_html=True)

            # Resources
            st.markdown("""
                <div class='info-card' style='border-left-color: #667eea; margin-top: 2rem;'>
                    <h4>📞 Mental Health Resources</h4>
                    <p><strong>National Crisis Helpline:</strong> Available 24/7 for immediate support</p>
                    <p><strong>Campus Counseling:</strong> Most universities offer free counseling services</p>
                    <p><strong>Online Support:</strong> Many organizations provide confidential online chat services</p>
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.info("Please ensure all fields are filled correctly and the model file is available.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p><strong>Disclaimer:</strong> This is an AI-based screening tool for educational purposes only.</p>
    </div>
""", unsafe_allow_html=True)