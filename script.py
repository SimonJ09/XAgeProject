# Importer les bibliothèques nécessaires

import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input

# Charger le modèle
best_model = tf.keras.models.load_model('Model.h5')

# Définir les étiquettes pour le genre et la race
gender_labels = ['Male', 'Female']
race_labels = ['White', 'Black', 'Asian', 'Indian', 'Others']

# Fonction pour effectuer les prédictions
def predict(image):
    # Prétraiter l'image
    processed_image = preprocess_input(image)
    
    # Remodeler l'image
    processed_image = np.expand_dims(processed_image, axis=0)
    
    # Effectuer les prédictions
    predictions = best_model.predict(processed_image)
    
    # Récupérer les prédictions de genre, d'âge et de race
    gender_predictions = np.argmax(predictions[0], axis=1)
    age_predictions = predictions[1].flatten().astype(int)
    race_predictions = np.argmax(predictions[2], axis=1)
    
    # Récupérer le genre, l'âge et la race prédits
    gender = gender_labels[gender_predictions[0]]
    age = age_predictions[0]
    race = race_labels[race_predictions[0]]
    
    return gender, age, race

# Application Streamlit
def main():
    st.title("Image Prediction")
    st.write("Upload an image and get gender, age, and race predictions.")

    # Uploader de fichier
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Afficher l'image téléchargée
        image = plt.imread(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Effectuer les prédictions
        gender, age, race = predict(image)
        
        # Afficher les prédictions
        st.write(f"Gender: {gender}")
        st.write(f"Age: {age}")
        st.write(f"Race: {race}")

if __name__ == "__main__":
    main()
