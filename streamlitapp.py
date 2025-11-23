import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="💊",
    layout="centered"
)

# Title and description
st.title("Insurance Cost Prediction")
st.markdown("""
### Predict your insurance charges based on your profile

Fill in the form below to get an estimate of your insurance costs.
All fields are **required**.
""")

# Load the saved model and model info
@st.cache_resource
def load_model():
    """Load the saved model and model info"""
    try:
        with open('insurance_cost_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('model_info.pkl', 'rb') as file:
            model_info = pickle.load(file)
        return model, model_info
    except FileNotFoundError:
        st.error("Model files not found! Please make sure 'insurance_cost_model.pkl' and 'model_info.pkl' are in the same directory.")
        st.stop()

model, model_info = load_model()

# Sidebar with model information
with st.sidebar:
    st.header("Model Information")
    st.write(f"**Model Type:** {model_info['model_type'].replace('_', ' ').title()}")
    st.write(f"**R² Score:** {model_info['r2_score']:.2%}")
    st.write(f"**Features Used:** {len(model_info['feature_names'])}")
    st.write(f"**Feature Engineering:** {'Yes' if model_info['uses_feature_engineering'] else 'No'}")
    st.write(f"**Log Transform:** {'Yes' if model_info['uses_log_transform'] else 'No'}")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app predicts insurance charges using a Linear Regression model trained on demographic and health factors.
    
    **Key Factors:**
    - Age
    - BMI (Body Mass Index)
    - Smoking Status
    - Number of Children
    - Region
    - Gender
    """)

# Create input form
st.header("Enter Your Information")

col1, col2 = st.columns(2)

with col1:
    # Age input
    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=100,
        value=30,
        step=1,
        help="Enter your age"
    )
    
    # BMI input
    bmi = st.number_input(
        "BMI (Body Mass Index)",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1,
        format="%.1f",
        help="BMI = weight (kg) / height (m)². Normal range: 18.5-24.9"
    )
    
    # Children input
    children = st.number_input(
        "Number of Children/Dependents",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        help="Number of children covered by insurance"
    )

with col2:
    # Sex input
    sex = st.selectbox(
        "Gender",
        options=["male", "female"],
        help="Select your gender"
    )
    
    # Smoker input
    smoker = st.selectbox(
        "Smoking Status",
        options=["no", "yes"],
        help="Do you smoke?"
    )
    
    # Region input
    region = st.selectbox(
        "Region",
        options=["northeast", "northwest", "southeast", "southwest"],
        help="Select your geographic region"
    )

# Display BMI category helper
bmi_category = ""
if bmi < 18.5:
    bmi_category = "Underweight"
elif bmi < 25:
    bmi_category = "Normal"
elif bmi < 30:
    bmi_category = "Overweight"
else:
    bmi_category = "Obese"

st.info(f"Your BMI category: **{bmi_category}** (BMI: {bmi:.1f})")

# Function to preprocess input data (same as training)
def preprocess_input(age, sex, bmi, children, smoker, region, model_info):
    """Preprocess user input to match training data format"""
    
    # Create base dataframe
    data = {
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    }
    df = pd.DataFrame(data)
    
    # One-hot encoding (same as training)
    df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
    
    # If model uses feature engineering, add those features
    if model_info['uses_feature_engineering']:
        # Ensure we have the required columns (handle missing ones)
        required_cols = ['age', 'bmi', 'children']
        for col in required_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = data[col][0]
        
        # Add interaction terms
        if 'smoker_yes' in df_encoded.columns:
            df_encoded['smoker_age'] = df_encoded['age'] * df_encoded['smoker_yes']
            df_encoded['smoker_bmi'] = df_encoded['bmi'] * df_encoded['smoker_yes']
        else:
            df_encoded['smoker_age'] = 0
            df_encoded['smoker_bmi'] = 0
        
        df_encoded['age_bmi'] = df_encoded['age'] * df_encoded['bmi']
        df_encoded['age_children'] = df_encoded['age'] * df_encoded['children']
        
        # Polynomial features
        df_encoded['age_squared'] = df_encoded['age'] ** 2
        df_encoded['bmi_squared'] = df_encoded['bmi'] ** 2
        
        # Categorical features
        df_encoded['bmi_obese'] = (df_encoded['bmi'] > 30).astype(int)
        df_encoded['bmi_overweight'] = ((df_encoded['bmi'] > 25) & (df_encoded['bmi'] <= 30)).astype(int)
        df_encoded['age_old'] = (df_encoded['age'] > 50).astype(int)
        df_encoded['age_middle'] = ((df_encoded['age'] > 30) & (df_encoded['age'] <= 50)).astype(int)
    
    # Ensure all required features are present (fill missing with 0)
    feature_names = model_info['feature_names']
    for feature in feature_names:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0
    
    # Reorder columns to match training data
    df_encoded = df_encoded[feature_names]
    
    return df_encoded

# Prediction button
st.markdown("---")
predict_button = st.button("Predict Insurance Cost", type="primary", use_container_width=True)

if predict_button:
    # Preprocess input
    try:
        input_data = preprocess_input(age, sex, bmi, children, smoker, region, model_info)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # If model uses log transform, convert back
        if model_info['uses_log_transform']:
            prediction = np.expm1(prediction)
        
        # Display result
        st.markdown("---")
        st.success("Prediction Complete!")
        
        # Main result display
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                <h2 style='color: #1f77b4; margin-bottom: 10px;'>Estimated Insurance Cost</h2>
                <h1 style='color: #2e7d32; font-size: 48px; margin: 20px 0;'>${prediction:,.2f}</h1>
                <p style='color: #666; font-size: 14px;'>per year</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional insights
        st.markdown("### Insights")
        
        insights = []
        
        # Age insight
        if age > 50:
            insights.append(f"**Age Factor**: Being over 50 may increase your insurance costs.")
        elif age > 30:
            insights.append(f"**Age Factor**: Your age ({age}) is in the middle range.")
        else:
            insights.append(f"**Age Factor**: Your age ({age}) is relatively young, which may help keep costs lower.")
        
        # BMI insight
        if bmi > 30:
            insights.append(f"**BMI Factor**: Your BMI ({bmi:.1f}) is in the obese range, which significantly increases insurance costs.")
        elif bmi > 25:
            insights.append(f"**BMI Factor**: Your BMI ({bmi:.1f}) is in the overweight range, which may increase costs.")
        else:
            insights.append(f"**BMI Factor**: Your BMI ({bmi:.1f}) is in a healthy range.")
        
        # Smoking insight
        if smoker == "yes":
            insights.append(f"**Smoking Factor**: Smoking significantly increases insurance costs (often 2-3x higher).")
        else:
            insights.append(f"**Smoking Factor**: Being a non-smoker helps keep your insurance costs lower.")
        
        # Children insight
        if children > 0:
            insights.append(f"**Dependents**: Having {children} dependent(s) is factored into your insurance cost.")
        
        for insight in insights:
            st.write(insight)
        
        # Disclaimer
        st.markdown("---")
        st.caption("""
        **Disclaimer**: This is a predictive estimate based on machine learning models. 
        Actual insurance costs may vary based on additional factors not included in this model, 
        such as medical history, pre-existing conditions, and specific insurance provider policies.
        """)
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Please check that all fields are filled correctly.")

