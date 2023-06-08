import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Chargement du modèle pré-entraîné
model = load_model('model.h5')

# Définition des classes
gender_classes = ['Male', 'Female']
race_classes = ['White', 'Black', 'Asian', 'Indian', 'Others']
age_classes = list(range(0, 117))

# Fonction de prétraitement de l'image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (48, 48))
    image = image / 255.0
    return image

# Fonction de prédiction
def predict(image):
    # Prétraitement de l'image
    processed_image = preprocess_image(image)

    # Ajouter une dimension supplémentaire pour l'entrée du modèle
    processed_image = np.expand_dims(processed_image, axis=0)

    # Prédiction avec le modèle chargé
    gender_prob, race_prob, age_prob = model.predict(processed_image)

    # Obtenir les indices des classes prédites
    gender_index = np.argmax(gender_prob)
    race_index = np.argmax(race_prob)
    age_index = np.argmax(age_prob)

    # Obtenir les prédictions réelles
    gender_prediction = gender_classes[gender_index]
    race_prediction = race_classes[race_index]
    age_prediction = age_classes[age_index]

    return gender_prediction, race_prediction, age_prediction

# Interface utilisateur Streamlit
st.title("Prédiction d'âge, de sexe et de race à partir d'une photo")
st.write("Veuillez charger une photo pour obtenir les prédictions.")

# Chargement de l'image
uploaded_image = st.file_uploader("Sélectionnez une image", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    # Lecture de l'image
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)

    # Affichage de l'image
    st.image(image, channels="RGB")

    # Prédiction
    gender, race, age = predict(image)

    # Affichage des prédictions
    st.write("Prédictions :")
    st.write("Sexe :", gender)
    st.write("Race :", race)
    st.write("Âge :", age)
