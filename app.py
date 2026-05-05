import streamlit as st
import numpy as np
import pickle

# =========================
# LOAD MODELS
# =========================
with open("accident_model.pkl", "rb") as f:
    accident_model = pickle.load(f)

with open("severity_model.pkl", "rb") as f:
    severity_model = pickle.load(f)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Traffic Accident Risk System",
    layout="wide"
)

st.title("🚦 Traffic Accident Prediction & Severity System")
st.markdown("### AI-powered smart road safety dashboard")

# =========================
# SIDEBAR INPUTS (12 FEATURES)
# =========================
st.sidebar.header("📊 Input Features")

weather = st.sidebar.selectbox("Weather", ["Clear", "Rainy", "Foggy", "Hot", "Cloudy"])
road_type = st.sidebar.selectbox("Road Type", ["Highway", "City", "Rural"])
time_of_day = st.sidebar.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
traffic = st.sidebar.selectbox("Traffic Density", ["Low", "Medium", "High"])

speed = st.sidebar.slider("Speed (km/h)", 0, 120, 60)
vehicles = st.sidebar.slider("Number of Vehicles", 1, 50, 10)
alcohol = st.sidebar.selectbox("Alcohol Influence", ["No", "Yes"])
road_condition = st.sidebar.selectbox("Road Condition", ["Good", "Wet", "Damaged", "Slippery"])
vehicle_type = st.sidebar.selectbox("Vehicle Type", ["Car", "Bike", "Truck", "Bus"])

driver_age = st.sidebar.slider("Driver Age", 18, 80, 30)
driver_experience = st.sidebar.slider("Driver Experience (years)", 0, 50, 5)
road_light = st.sidebar.selectbox("Road Light Condition", ["Daylight", "Night", "Streetlight"])

# =========================
# ENCODING (MUST MATCH TRAINING)
# =========================
weather_map = {"Clear":0, "Rainy":1, "Foggy":2, "Hot":3, "Cloudy":4}
road_type_map = {"Highway":0, "City":1, "Rural":2}
time_map = {"Morning":0, "Afternoon":1, "Evening":2, "Night":3}
traffic_map = {"Low":0, "Medium":1, "High":2}
alcohol_map = {"No":0, "Yes":1}
road_condition_map = {"Good":0, "Wet":1, "Damaged":2, "Slippery":3}
vehicle_type_map = {"Car":0, "Bike":1, "Truck":2, "Bus":3}
road_light_map = {"Daylight":0, "Night":1, "Streetlight":2}

# =========================
# INPUT VECTOR (12 FEATURES ORDER MUST MATCH TRAINING)
# =========================
input_data = np.array([[

    weather_map[weather],
    road_type_map[road_type],
    time_map[time_of_day],
    traffic_map[traffic],

    speed,
    vehicles,
    alcohol_map[alcohol],
    road_condition_map[road_condition],
    vehicle_type_map[vehicle_type],

    driver_age,
    driver_experience,
    road_light_map[road_light]

]])

# =========================
# PREDICTION
# =========================
if st.button("🚨 Predict Accident Risk"):

    accident_prob = accident_model.predict_proba(input_data)[0][1]

    accident_pred = 1 if accident_prob >= 0.35 else 0

    severity_pred = severity_model.predict(input_data)[0]

    # =========================
    # RESULTS
    # =========================
    st.subheader("📌 Prediction Results")

    if accident_pred == 1:
        st.error("🚨 Accident Likely")
    else:
        st.success("🟢 No Accident Risk")

    st.write(f"**Accident Probability:** {accident_prob:.2f}")

    # Severity
    if severity_pred == 0:
        st.success("🟢 Low Risk")
        severity_text = "Low Risk"
    elif severity_pred == 1:
        st.warning("🟡 Medium Risk")
        severity_text = "Medium Risk"
    else:
        st.error("🔴 High Risk")
        severity_text = "High Risk"

    # Risk score
    risk_score = int(accident_prob * 100)

    st.metric("🚦 Risk Score", f"{risk_score}/100")
    st.progress(risk_score)

    # Summary
    st.markdown("### 📊 Summary")
    st.write(f"- Accident: {'Yes' if accident_pred else 'No'}")
    st.write(f"- Severity: {severity_text}")
    st.write(f"- Risk Score: {risk_score}/100")