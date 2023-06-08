#!/bin/bash

pip install --upgrade protobuf


# Mettre à jour pip
/opt/render/project/src/.venv/bin/python -m pip install --upgrade pip

# Installer les dépendances
/opt/render/project/src/.venv/bin/python -m pip install streamlit pandas tensorflow numpy opencv-python

# Exécuter l'application Streamlit
/opt/render/project/src/.venv/bin/streamlit run script.py
