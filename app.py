import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Sample dataset (same as your example)
data = {
    'WBC': [12000, 4500, 18000, 3000, 11000, 5000, 25000, 6000, 27000, 4000],
    'RBC': [4.5, 5.0, 3.2, 4.9, 4.2, 5.1, 2.8, 5.0, 2.5, 5.3],
    'Platelets': [150000, 250000, 90000, 270000, 120000, 230000, 70000, 210000, 60000, 260000],
    'Hemoglobin': [12.5, 14.0, 10.0, 14.5, 11.2, 13.9, 9.0, 14.2, 8.5, 15.0],
    'Age': [50, 25, 60, 30, 45, 28, 65, 33, 70, 24],
    'Diagnosis': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Prepare features and labels
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

st.title("Blood cancer Prediction App")

st.write("Enter patient blood test results:")

wbc = st.number_input("WBC", min_value=0, value=5000)
rbc = st.number_input("RBC", min_value=0.0, value=4.5, format="%.2f")
platelets = st.number_input("Platelets", min_value=0, value=150000)
hemoglobin = st.number_input("Hemoglobin", min_value=0.0, value=12.5, format="%.2f")
age = st.number_input("Age", min_value=0, value=30)

if st.button("Predict"):
    input_data = [[wbc, rbc, platelets, hemoglobin, age]]
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.error(f"⚠️ Likely diagnosis: LEUKEMIA ({round(prob[1]*100, 2)}% confidence)")
    else:
        st.success(f"✅ Likely diagnosis: HEALTHY ({round(prob[0]*100, 2)}% confidence)")
