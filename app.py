import streamlit as st
import pandas as pd
import joblib

# Load the trained model
# This model was saved from your 'employee salary prediction (1).ipynb' notebook
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Prediction")
st.write("This app predicts whether an employee's income is more or less than $50K based on their details.")

# --- User Inputs ---
st.sidebar.header("Enter Employee Details")

# Create inputs in the sidebar for the most important features
age = st.sidebar.slider("Age", 17, 75, 30)
workclass = st.sidebar.selectbox("Work Class", ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Others'])
educational_num = st.sidebar.slider("Education Level (Numeric)", 5, 16, 10)
occupation = st.sidebar.selectbox("Occupation", ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces', 'Others'])
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0)
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])

# --- Preprocessing the Inputs ---
# The model was trained on numbers, so we must convert the text inputs to numbers
# These mappings are based on the LabelEncoder from your notebook
workclass_map = {'Private': 3, 'Self-emp-not-inc': 5, 'Local-gov': 1, 'State-gov': 6, 'Self-emp-inc': 4, 'Federal-gov': 0, 'Others': 2}
occupation_map = {'Prof-specialty': 10, 'Craft-repair': 2, 'Exec-managerial': 3, 'Adm-clerical': 0, 'Sales': 12, 'Other-service': 8, 'Machine-op-inspct': 6, 'Transport-moving': 14, 'Handlers-cleaners': 5, 'Farming-fishing': 4, 'Tech-support': 13, 'Protective-serv': 11, 'Priv-house-serv': 9, 'Armed-Forces': 1, 'Others': 7}
gender_map = {'Male': 1, 'Female': 0}

# Create a dictionary with all the features the model expects
input_data = {
    'age': age,
    'workclass': workclass_map[workclass],
    'fnlwgt': 189778,  # Using a common (mean) value for features we don't ask for
    'educational-num': educational_num,
    'marital-status': 2, # Common value
    'occupation': occupation_map[occupation],
    'relationship': 0, # Common value
    'race': 4, # Common value
    'gender': gender_map[gender],
    'capital-gain': capital_gain,
    'capital-loss': 0, # Common value
    'hours-per-week': hours_per_week,
    'native-country': 39 # Common value
}

# Create a DataFrame from the dictionary
input_df = pd.DataFrame([input_data])

st.write("### 🔎 Input Features (After Processing)")
st.write(input_df)

# --- Prediction ---
if st.button("Predict Salary Class"):
    # The model expects the data in a specific order, which this DataFrame provides
    prediction = model.predict(input_df)
    
    st.markdown("---")
    if prediction[0] == '<=50K':
        st.success(f"**Prediction: The employee's income is likely less than or equal to $50K.**")
    else:
        st.success(f"**Prediction: The employee's income is likely greater than $50K.**")
