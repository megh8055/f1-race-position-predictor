import streamlit as st
import joblib
import numpy as np

model = joblib.load("f1_position_predictor.pkl")

st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️")
st.title("🏁 F1 Race Position Predictor")
st.markdown("Predict a driver's finishing position based on race performance stats.")

avg_lap_time_ms = st.number_input("Average Lap Time (in milliseconds)", min_value=50000, max_value=150000, value=90000)
lap_time_std_dev = st.number_input("Lap Time Std Deviation (in ms)", min_value=0, max_value=10000, value=2500)
avg_pit_stop_sec = st.number_input("Average Pit Stop Duration (in seconds)", min_value=1.0, max_value=10.0, value=2.5)

if st.button("Predict Finishing Position"):
    input_features = np.array([[avg_lap_time_ms, lap_time_std_dev, avg_pit_stop_sec]])
    prediction = model.predict(input_features)[0]
    st.success(f"🎯 Predicted Finishing Position: **{round(prediction)}**")

st.markdown("---")
st.caption("Model trained on historical F1 race data. Predictions are approximate.")
