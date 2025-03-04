import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Génération des données
data = np.random.normal(size=1000)
data = pd.DataFrame(data, columns=["Distribution_Normale"])

# Affichage du jeu de données
st.write("### Jeu de Données")
st.write(data.head())

# Histogramme avec couleur personnalisée
st.write("### Histogramme")
fig, ax = plt.subplots()
ax.hist(data["Distribution_Normale"], bins=30, color="skyblue", edgecolor="black")
st.pyplot(fig)

# Ajout d'un boxplot avec un titre
st.write("### Boîte à Moustaches")
fig, ax = plt.subplots()
ax.boxplot(data["Distribution_Normale"])
ax.set_title("Boîte à Moustaches")
st.pyplot(fig)
