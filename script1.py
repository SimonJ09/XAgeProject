import streamlit as st
import joblib
import numpy as np
from PIL import Image
import cv2

# Load the gender model
gender_model = joblib.load('gender_model.pkl')

# Load the age model
age_model = joblib.load('age_model.pkl')

# Load the race model
race_model = joblib.load('race_model.pkl')

# Function to preprocess the input image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (48, 48))
    image = image / 255.0
    image = image.reshape(1, -1)
    return image

# Function to predict gender
def predict_gender(image):
    preprocessed_image = preprocess_image(image)
    gender_prediction = gender_model.predict(preprocessed_image)
    return gender_prediction[0]

# Function to predict age
def predict_age(image):
    preprocessed_image = preprocess_image(image)
    age_prediction = age_model.predict(preprocessed_image)
    return int(round(age_prediction[0]))

# Function to predict race
def predict_race(image):
    preprocessed_image = preprocess_image(image)
    race_prediction = race_model.predict(preprocessed_image)
    return race_prediction[0]

# Streamlit app
def main():
    st.title("Face Attributes Prediction")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, channels="RGB", caption="Uploaded Image")

        gender = predict_gender(image)
        st.write("Gender Prediction:", "Male" if gender == 0 else "Female")

        age = predict_age(image)
        st.write("Age Prediction:", age)

        race = predict_race(image)
        race_mapping = {
            0: "White",
            1: "Black",
            2: "Asian",
            3: "Indian",
            4: "Other"
        }
        st.write("Race Prediction:", race_mapping.get(race, "Unknown"))

if __name__ == '__main__':
    main()
