import requests
import streamlit as st

st.title("Sentiment Analysis App")

# Input for user
user_input = st.text_area("Enter text for sentiment analysis:", "")

if st.button("Predict"):
    # Backend API URL
    ml_container_url = "https://fast-api-service-828705663866.europe-north1.run.app/predict/"  # Your backend URL

    payload = {"text": user_input}
    headers = {
        "Content-Type": "application/json",
    }

    # Send request to the backend
    response = requests.post(ml_container_url, headers=headers, json=payload)

    # Display the result
    if response.status_code == 200:
        prediction = response.json()
        st.write(f"Text: {prediction['text']}")
        st.write(f"Predicted Sentiment Class: {prediction['predicted_class']}")
        st.write(f"Probabilities: {prediction['probabilities']}")
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
