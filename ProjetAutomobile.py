import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from PIL import Image

# === Configuration de la page ===
st.set_page_config(page_title="Prédiction du prix des voitures", layout="wide")

# === Chargement du logo ===
st.image("28804.jpg", use_container_width=True)

# === Titre de l'application ===
st.title("🚗 Prédiction du Prix des Voitures")

# === 1️⃣ Importer le jeu de données ===
st.subheader("📂 Chargement des données")
df = pd.read_csv("automobile_data.csv")
st.write("Aperçu des données :", df.head())

# === 2️⃣ Nettoyage des données ===
st.subheader("🧹 Nettoyage des données")
df.dropna(inplace=True)  # Supprime les valeurs manquantes
df.drop_duplicates(inplace=True)  # Supprime les doublons
st.write(f"Nombre total de lignes après nettoyage : {df.shape[0]}")

# === 3️⃣ Visualisation des données ===
st.subheader("📊 Visualisation des données")

# Distribution des prix des voitures
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["price"], bins=30, kde=True, ax=ax)
ax.set_title("Distribution des prix des voitures")
st.pyplot(fig)

# === 4️⃣ Préparation des données ===
st.subheader("🛠️ Préparation des données")
X = df.drop(columns=["price"])  # Variables indépendantes
y = df["price"]  # Variable cible

# Encodage des variables catégorielles
X = pd.get_dummies(X, drop_first=True)

# Division en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5️⃣ Modélisation (Régression Linéaire & Random Forest) ===
st.subheader("🤖 Entraînement des modèles")

# Modèle 1 : Régression Linéaire
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Modèle 2 : Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# === 6️⃣ Évaluation des modèles ===
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

st.write(f"📉 **MAE Régression Linéaire :** {mae_lr:.2f}")
st.write(f"🌲 **MAE Random Forest :** {mae_rf:.2f}")

# Sélection du meilleur modèle
best_model = rf if mae_rf < mae_lr else lr
st.success(f"✅ **Modèle sélectionné : {'Random Forest' if best_model == rf else 'Régression Linéaire'}**")

# === 7️⃣ Application Streamlit pour prédictions ===
st.sidebar.header("🎯 Prédiction du prix d'une voiture")

# Sélection des caractéristiques utilisateur
year = st.sidebar.number_input("Année", min_value=1980, max_value=2025, value=2015)
mileage = st.sidebar.number_input("Kilométrage", min_value=0, value=50000)

# Transformer l'entrée utilisateur en DataFrame
input_data = pd.DataFrame([[year, mileage]], columns=["year", "mileage"])

# Prédiction du prix
if st.sidebar.button("🔍 Prédire le prix"):
    prediction = best_model.predict(input_data)
    st.sidebar.success(f"💰 Prix estimé : {prediction[0]:,.2f} €")

# === Footer ===
st.write("---")
st.write("🚀 **Projet Machine Learning - Streamlit** | Développé avec ❤️ par [Votre Nom]")
