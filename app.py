import streamlit as st
import joblib
import numpy as np

# Load trained model and label encoder
model = joblib.load('model/ipl_model.pkl')
le = joblib.load('model/label_encoder.pkl')

# Sample values (update based on your data)
teams = sorted(list(set([
    "Chennai Super Kings", "Mumbai Indians", "Kolkata Knight Riders", "Royal Challengers Bangalore",
    "Delhi Capitals", "Sunrisers Hyderabad", "Rajasthan Royals", "Punjab Kings",
    "Gujarat Titans", "Lucknow Super Giants"
])))
venues = sorted([
    "Mumbai", "Chennai", "Kolkata", "Delhi", "Bangalore", "Ahmedabad", "Lucknow", "Hyderabad",
    "Jaipur", "Pune", "Raipur", "Visakhapatnam"
])
decisions = ['BAT FIRST', 'BOWL FIRST']

st.title("🏏 IPL Match Winner Predictor (Pre-Match)")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Select Home Team", teams)
with col2:
    away_team = st.selectbox("Select Away Team", teams)

venue = st.selectbox("Select Venue", venues)

col3, col4 = st.columns(2)
with col3:
    toss_winner = st.selectbox("Who won the toss?", teams)
with col4:
    toss_decision = st.selectbox("Toss Decision", decisions)

# Prediction Button
if st.button("Predict Winner"):
    try:
        input_data = np.array([
            le.transform([home_team])[0],
            le.transform([away_team])[0],
            le.transform([venue])[0],
            le.transform([toss_decision])[0],
            le.transform([toss_winner])[0]
        ]).reshape(1, -1)

        prediction = model.predict(input_data)
        winner = le.inverse_transform(prediction)[0]

        st.success(f"🎉 Predicted Winner: **{winner}**")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
