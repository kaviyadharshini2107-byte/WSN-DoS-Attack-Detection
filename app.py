
import streamlit as st
import pandas as pd
import joblib
import sys
import os

# Make src modules accessible
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from preprocessing import clean_wsn_data, validate_features
from predict import predict_attack

# Page configuration
st.set_page_config(
    page_title="WSN DoS Attack Detection",
    page_icon="🛡️",
    layout="wide"
)

# Load saved model and preprocessing objects
@st.cache_resource
def load_project():
    model_data = joblib.load(
        os.path.join(BASE_DIR, "results", "final_balanced_goa_ga_cart.pkl")
    )

    prepared_data = joblib.load(
        os.path.join(BASE_DIR, "dataset", "processed", "prepared_data.pkl")
    )

    return (
        model_data["model"],
        model_data["selected_features"],
        prepared_data["feature_names"],
        prepared_data["group_scaler"],
        prepared_data["label_encoder"],
        model_data["test_accuracy"]
    )

model, selected_features, feature_names, scaler, label_encoder, accuracy = load_project()

# Application heading
st.title("🛡️ WSN Denial-of-Service Attack Detection")
st.write(
    "Machine Learning-Based Detection of Denial-of-Service "
    "Attacks in Wireless Sensor Networks."
)

st.metric("Previously Evaluated Test Accuracy", f"{accuracy * 100:.4f}%")

st.divider()

# Upload dataset
st.subheader("Upload WSN Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)
        data = clean_wsn_data(data)

        validate_features(data, feature_names)

        st.success("Dataset uploaded and validated successfully!")

        st.write("Dataset Preview")
        st.dataframe(data.head())

        if st.button("Detect Attacks"):

            X = data[feature_names]

            X_scaled = scaler.transform(X)

            selected_indices = [
                feature_names.index(feature)
                    for feature in selected_features
            ]

            X_selected = X_scaled[:, selected_indices]
            predictions = model.predict(X_selected)  
            predictions = label_encoder.inverse_transform(predictions) 
            data["Predicted Attack"] = predictions 

            st.success("Attack detection completed!")

            st.subheader("Prediction Results")
            st.dataframe(data)

            st.subheader("Prediction Summary")
            st.bar_chart(data["Predicted Attack"].value_counts())

            csv_data = data.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Results",
                data=csv_data,
                file_name="wsn_attack_predictions.csv",
                mime="text/csv"
            )

    except Exception as error:
        st.error(f"Error: {error}")

else:
    st.info("Upload a WSN CSV dataset to begin attack detection.")

st.divider()
st.caption("WSN DoS Detection | GOA-GA Feature Selection + CART Classifier")
