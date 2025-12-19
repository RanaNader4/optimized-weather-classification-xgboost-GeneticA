import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ============================
# Load Model
# ============================
MODEL_PATH = r"D:\AI Engineering\Level 3\AI-Based Programming\Project\best_xgb_optimized.pkl"
DATA_PATH  = r"D:\AI Engineering\Level 3\AI-Based Programming\Project\df_clean.csv"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_columns():
    df = pd.read_csv(DATA_PATH)
    return df.drop("Weather Type", axis=1).columns.tolist()

model = load_model()
feature_names = load_columns()

# ============================
# Streamlit UI
# ============================
st.title("🌤️ Weather Type Prediction App")
st.write("Predict weather type using the optimized XGBoost model.")

st.subheader("🔧 Input Features")

# Create input fields dynamically
inputs = {}
for feature in feature_names:
    inputs[feature] = st.number_input(
        f"Enter {feature}:", 
        value=0.0,
        format="%.4f"
    )

# Convert to DataFrame for the model
input_df = pd.DataFrame([inputs])

# ============================
# Predict Button
# ============================
if st.button("Predict Weather Type"):
    # Mapping from encoded label → actual weather name
    label_map = {
        0: "Cloudy",
        1: "Rainy",
        2: "Snowy",
        3: "Sunny"
    }

    prediction = model.predict(input_df)[0]
    weather_name = label_map.get(prediction, "Unknown")

    st.success(f"🌤️ **Predicted Weather Type: {weather_name}**")


    # If probabilities needed:
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(input_df)[0]

        st.subheader("📊 Class Probabilities")
        proba_df = pd.DataFrame({
            "Weather Type": model.classes_,
            "Probability": probas
        })
        st.dataframe(proba_df)

# ============================
# Footer
# ============================
st.markdown("---")
st.caption("AI Based Programming Course — NINU — Streamlit + XGBoost")
