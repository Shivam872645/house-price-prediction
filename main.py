import streamlit as st
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")

# ------------------ LOAD CONFIG ------------------
with open("config.json", "r") as f:
    config = json.load(f)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

df = load_data()

st.subheader("📊 Dataset Overview")
st.write(df.head())

# ------------------ DATA PREPARATION ------------------
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config["test_size"],
    random_state=config["random_state"],
    stratify=y
)

# ------------------ MODEL TRAINING ------------------
@st.cache_resource
def train_model():
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        random_state=config["random_state"],
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

model = train_model()

# ------------------ MODEL EVALUATION ------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("✅ Model Performance")
st.write(f"**Accuracy:** {accuracy:.4f}")

# ------------------ TEST DATA UPLOAD ------------------
st.subheader("📤 Upload Test Data (CSV)")

uploaded_file = st.file_uploader(
    "Upload a CSV file WITHOUT the Class column",
    type=["csv"]
)

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    st.write("🔍 Test Data Preview")
    st.write(test_df.head())

    # Prediction
    predictions = model.predict(test_df)
    probabilities = model.predict_proba(test_df)[:, 1]

    result_df = test_df.copy()
    result_df["Fraud_Prediction"] = predictions
    result_df["Fraud_Probability"] = probabilities

    st.subheader("🧾 Prediction Results")
    st.write(result_df)

    st.download_button(
        label="⬇ Download Results",
        data=result_df.to_csv(index=False),
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )
