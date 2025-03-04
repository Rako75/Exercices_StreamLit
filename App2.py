import streamlit as st
from PIL import Image  # Import Pillow to handle images

st.title("Bienvenue sur mon application Streamlit !")

# Champ de texte pour entrer le nom
nom = st.text_input("Entrez votre nom:")

# Affichage du message personnalisé
if nom:
    st.write(f"Bienvenue, {nom} ! 😊")

# Ouvrir et afficher l'image avec Pillow pour éviter les erreurs
try:
    image = Image.open("wallpapersden.com_black-sphere-4k_3584x2048.jpg")
    st.image(image, caption="Bienvenue sur mon application !", use_container_width=True)
except Exception as e:
    st.error(f"Erreur lors du chargement de l'image : {e}")
