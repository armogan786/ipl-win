# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load('xgb_model.pkl')

st.title("\U0001F3CF IPL Win Predictor (Chasing Team)")

# User Inputs
runs_left = st.number_input("Runs Left", min_value=0, max_value=300, value=50)
balls_left = st.number_input("Balls Left", min_value=1, max_value=120, value=60)
wickets_fallen = st.number_input("Wickets Fallen", min_value=0, max_value=10, value=4)
current_score = st.number_input("Current Score", min_value=0, value=100)
ball_number = 120 - balls_left
target = current_score + runs_left

# Derived Features
run_rate = current_score / (ball_number / 6) if ball_number != 0 else 0
required_run_rate = (runs_left / balls_left) * 6 if balls_left != 0 else 0

# Prediction input
features = pd.DataFrame({
    'runs_left': [runs_left],
    'balls_left': [balls_left],
    'wickets_fallen': [wickets_fallen],
    'run_rate': [run_rate],
    'required_run_rate': [required_run_rate]
})

# Prediction
prediction = model.predict_proba(features)[0][1]  # Probability of winning
st.metric("Win Probability (Batting Team)", f"{prediction * 100:.2f}%")
