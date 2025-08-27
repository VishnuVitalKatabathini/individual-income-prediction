import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and encoders
model = joblib.load("random forest model\income_classifier_model.joblib")
scaler = joblib.load("random forest model\scaler.joblib")
label_encoders = joblib.load("random forest model\label_encoders.joblib")

st.title("Income Prediction App")
st.write("Fill in the details below to predict income level.")

# Input fields
age = st.slider("Age", 17, 90, 30)
workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov'])
education = st.selectbox("Education", ['Bachelors', 'HS-grad', 'Some-college', 'Masters'])
maritial_status = st.selectbox("Maritial Status", ['Never-married', 'Married-civ-spouse', 'Divorced'])
occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Exec-managerial'])
relationship = st.selectbox("Relationship", ['Husband', 'Not-in-family', 'Own-child', 'Unmarried'])
race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander'])
sex = st.selectbox("Sex", ['Male', 'Female'])
capital_gain = st.number_input("Capital Gain", 0, 99999, 0)
capital_loss = st.number_input("Capital Loss", 0, 99999, 0)
education_num = st.slider("Education Num", 1, 16, 9)
fnlwgt = st.number_input("Final Weight (fnlwgt)", 10000, 1000000, 300000)
hours_per_week = st.slider("Hours per week", 1, 99, 40)
native_country = st.selectbox("Native Country", ['United-States', 'India', 'Mexico'])

# Input dictionary
if st.button("Predict Income"):
    input_dict = {
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education-num": education_num,
        "maritial-status": maritial_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country,
   
    
    }

# Create DataFrame
    input_df = pd.DataFrame([input_dict])

# Encode categorical columns
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col].astype(str))

# Scale features
    input_scaled = scaler.transform(input_df)

# Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    result = ">50K" if prediction == 1 else "<=50K"

# Output
    st.success(f"Predicted Income: **{result}**")
    st.info(f"Probability of >50K income: {prob:.2%}")
    