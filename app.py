import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Sample dataset
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

# ---------- 🔴 RED BACKGROUND ----------
st.markdown(
    """
    <style>
        body {
            background-color: #FFCCCC;
        }
        .main > div {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- 🧬 Title ----------
st.markdown("<h1 style='text-align: center; color: #8B0000;'>🧬 Leukemia Prediction App</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---------- 📋 Input Form ----------
st.markdown("<h4 style='color: #333;'>🔬 Enter Patient Blood Test Results:</h4>", unsafe_allow_html=True)

wbc = st.number_input("WBC", min_value=0, value=5000)
rbc = st.number_input("RBC", min_value=0.0, value=4.5, format="%.2f")
platelets = st.number_input("Platelets", min_value=0, value=150000)
hemoglobin = st.number_input("Hemoglobin", min_value=0.0, value=12.5, format="%.2f")
age = st.number_input("Age", min_value=0, value=30)

# ---------- 🔍 Prediction ----------
if st.button("🔎 Predict"):
    input_data = [[wbc, rbc, platelets, hemoglobin, age]]
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.markdown(
            f"""
            <div style='background-color:#ff6666; padding:20px; border-radius:10px;'>
                <h3 style='color:white;'>⚠️ Likely Diagnosis: <b>LEUKEMIA</b></h3>
                <p style='color:white;'>Confidence: <b>{round(prob[1]*100, 2)}%</b></p>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='background-color:#d4edda; padding:20px; border-radius:10px;'>
                <h3 style='color:#155724;'>✅ Likely Diagnosis: <b>HEALTHY</b></h3>
                <p style='color:#155724;'>Confidence: <b>{round(prob[0]*100, 2)}%</b></p>
            </div>
            """, unsafe_allow_html=True
        )

# ---------- 👣 Footer ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#660000;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
