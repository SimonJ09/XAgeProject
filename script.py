import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the model
best_model = tf.keras.models.load_model('Model.h5')

# Define the labels for gender and race
gender_labels = ['Male', 'Female']
race_labels = ['White', 'Black', 'Asian', 'Indian', 'Others']

# Function to make predictions
def predict(image):
    # Preprocess the image
    # Add your image preprocessing steps here
    processed_image = preprocess_image(image)  # Replace 'preprocess_image' with your actual preprocessing function
    
    # Reshape the image
    processed_image = np.expand_dims(processed_image, axis=0)
    
    # Make predictions
    predictions = best_model.predict(processed_image)
    
    # Get the gender, age, and race predictions
    gender_predictions = np.argmax(predictions[0], axis=1)
    age_predictions = predictions[1].flatten().astype(int)
    race_predictions = np.argmax(predictions[2], axis=1)
    
    # Get the predicted gender, age, and race
    gender = gender_labels[gender_predictions[0]]
    age = age_predictions[0]
    race = race_labels[race_predictions[0]]
    
    return gender, age, race

# Streamlit app
def main():
    st.title("Image Prediction")
    st.write("Upload an image and get gender, age, and race predictions.")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = plt.imread(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Make predictions
        gender, age, race = predict(image)
        
        # Display the predictions
        st.write(f"Gender: {gender}")
        st.write(f"Age: {age}")
        st.write(f"Race: {race}")

if __name__ == "__main__":
    main()
