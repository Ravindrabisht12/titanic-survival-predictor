# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

import os
model_path = os.path.join(os.path.dirname(__file__), '../models', 'titanic_model.pkl')
model = joblib.load(model_path)

st.title("🛳️ Titanic Survival Predictor")

# User inputs
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare Paid", 0.0, 500.0, 32.0)
sex = st.radio("Sex", ["Male", "Female"])
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode inputs
sex_male = 1 if sex == "Male" else 0
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0

# Build input DataFrame
input_data = pd.DataFrame([{
    'Pclass': pclass,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Sex_male': sex_male,
    'Embarked_Q': embarked_q,
    'Embarked_S': embarked_s
}])

# Predict
if st.button("Predict Survival"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.markdown(f"### 🧠 Predicted: {'Survived ✅' if pred==1 else 'Did Not Survive ❌'}")
    st.progress(int(prob * 100))
    st.caption(f"Probability of survival: **{prob:.2%}**")
