import streamlit as st

st.title("Bienvenue sur mon application Streamlit !")

# Champ de texte pour entrer le nom
nom = st.text_input("Entrez votre nom:")

# Affichage du message personnalisé
if nom:
    st.write(f"Bienvenue, {nom} ! 😊")

# Affichage d'une image
st.image("intelligence-artificielle-school-logo.png", caption="Bienvenue sur mon application !", use_column_width=True)
