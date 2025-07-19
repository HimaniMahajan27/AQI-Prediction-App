
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved artifacts
model = joblib.load('station_aqi_model.pkl')
scaler = joblib.load('station_scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')
unique_cities = joblib.load('unique_cities.pkl')

st.title("🌍 AQI Prediction App (Station-Level)")
st.markdown("Predict Air Quality Index based on pollutants and date features.")

# --- User Inputs ---
st.sidebar.header("📥 Input Parameters")

# City selection
selected_city = st.sidebar.selectbox("Select City", unique_cities)

# Pollutant inputs
pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
              'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

input_data = {}
for pollutant in pollutants:
    input_data[pollutant] = st.sidebar.slider(f"{pollutant}", 0.0, 1000.0, 50.0)

# Date input
date_input = st.sidebar.date_input("Measurement Date")

# Date features
date_features = {
    'Month': date_input.month,
    'Day': date_input.day,
    'Year': date_input.year,
    'DayOfWeek': date_input.weekday(),
    'DayOfYear': date_input.timetuple().tm_yday,
    'WeekOfYear': date_input.isocalendar().week,
    'IsWeekend': int(date_input.weekday() >= 5)
}

# One-hot encoding for city
city_encoded = {f'City_{city}': 0 for city in unique_cities if f'City_{city}' in feature_columns}
if f'City_{selected_city}' in city_encoded:
    city_encoded[f'City_{selected_city}'] = 1

# Combine all features
input_features = {**input_data, **date_features, **city_encoded}
final_input = [input_features.get(col, 0) for col in feature_columns]

# Scale input
X_input_scaled = scaler.transform([final_input])

# Predict AQI
if st.button("🚀 Predict AQI"):
    pred_log = model.predict(X_input_scaled)[0]
    predicted_aqi = round(np.expm1(pred_log), 2)

    def get_aqi_category(aqi):
        if aqi <= 50:
            return 'Good'
        elif aqi <= 100:
            return 'Satisfactory'
        elif aqi <= 200:
            return 'Moderate'
        elif aqi <= 300:
            return 'Poor'
        elif aqi <= 400:
            return 'Very Poor'
        else:
            return 'Severe'

    category = get_aqi_category(predicted_aqi)

    st.success(f"✅ Predicted AQI: **{predicted_aqi}**")
    st.info(f"🟢 AQI Category: **{category}**")
