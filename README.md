# 🏥 Insurance Cost Prediction - Streamlit App

A user-friendly web application to predict insurance costs based on demographic and health factors.

## 📋 Prerequisites

1. Python 3.7 or higher
2. The trained model files:
   - `insurance_cost_model.pkl`
   - `model_info.pkl`

## 🚀 How to Run

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Streamlit App

```bash
streamlit run streamlitapp.py
```

The app will open automatically in your default web browser at `http://localhost:8501`

## 📝 Usage

1. **Fill in the form** with your information:
   - **Age**: Your age in years (18-100)
   - **BMI**: Your Body Mass Index (10.0-60.0)
   - **Number of Children**: Number of dependents (0-10)
   - **Gender**: Male or Female
   - **Smoking Status**: Yes or No
   - **Region**: Northeast, Northwest, Southeast, or Southwest

2. **Click "Predict Insurance Cost"** button

3. **View your estimated insurance cost** and insights

## 🔧 Model Information

The app uses a Linear Regression model with:
- Feature engineering (interaction terms, polynomial features)
- High accuracy (R² score typically 85-90%+)
- Trained on Medical Cost Personal Dataset from Kaggle

## 📁 Files Required

- `streamlitapp.py` - Main Streamlit application
- `insurance_cost_model.pkl` - Trained model (generated from notebook)
- `model_info.pkl` - Model metadata (generated from notebook)
- `requirements.txt` - Python dependencies

## ⚠️ Note

Make sure to run the Jupyter notebook first to generate the `.pkl` model files before running the Streamlit app!

