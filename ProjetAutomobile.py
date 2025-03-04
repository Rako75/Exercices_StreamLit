import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# === Configuration de la page Streamlit ===
st.set_page_config(page_title="Prédiction du Prix des Voitures", layout="wide")

st.title("🚗 Prédiction du Prix des Voitures")

# === 1️⃣ Importation et Nettoyage des données ===
st.subheader("📂 Chargement et nettoyage des données")

# Charger les données
df = pd.read_csv("automobile_data.csv", sep=";")

# Supprimer les doublons
df = df.drop_duplicates()

# Convertir certaines colonnes en numérique
numeric_cols = ["bore", "stroke", "horsepower", "peak-rpm"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# Suppression des valeurs manquantes après conversion
df = df.dropna()

st.write(f"✅ Données nettoyées ({df.shape[0]} lignes, {df.shape[1]} colonnes)")

# === 2️⃣ Visualisation des données ===
st.subheader("📊 Distribution des prix")

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["price"], bins=30, kde=True, ax=ax)
ax.set_title("Distribution des prix des voitures")
st.pyplot(fig)

# === 3️⃣ Préparation des données pour le Machine Learning ===
st.subheader("🛠️ Préparation des données")

# Encodage des variables catégorielles
df = pd.get_dummies(df, columns=["body-style", "drive-wheels", "engine-location", 
                                 "engine-type", "fuel-system", "num-of-cylinders"], drop_first=True)

# Vérification et suppression des valeurs NaN après encodage
df = df.dropna()

# Séparer la cible et les features
X = df.drop(columns=["price"])
y = df["price"]

# Vérification finale des NaN avant le split
X = X.dropna()

# Division en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write("✅ Données prêtes pour l'entraînement")

# === 4️⃣ Entraînement des modèles ===
st.subheader("🤖 Entraînement des modèles")

# Régression Linéaire
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# === 5️⃣ Évaluation des modèles ===
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

st.write(f"📉 **MAE Régression Linéaire :** {mae_lr:.2f}")
st.write(f"🌲 **MAE Random Forest :** {mae_rf:.2f}")

# Sélection du meilleur modèle
best_model = rf if mae_rf < mae_lr else lr
st.success(f"✅ **Modèle sélectionné : {'Random Forest' if best_model == rf else 'Régression Linéaire'}**")

# === 6️⃣ Interface Streamlit pour prédictions ===
st.sidebar.header("🎯 Prédiction du prix d'une voiture")

# Formulaire utilisateur
wheel_base = st.sidebar.number_input("Empattement (wheel-base)", min_value=80.0, max_value=150.0, value=100.0)
length = st.sidebar.number_input("Longueur (length)", min_value=140.0, max_value=220.0, value=180.0)
width = st.sidebar.number_input("Largeur (width)", min_value=50.0, max_value=100.0, value=60.0)
height = st.sidebar.number_input("Hauteur (height)", min_value=40.0, max_value=80.0, value=55.0)
curb_weight = st.sidebar.number_input("Poids à vide (curb-weight)", min_value=500, max_value=5000, value=2500)
engine_size = st.sidebar.number_input("Taille moteur (engine-size)", min_value=50, max_value=500, value=150)

# Création du DataFrame pour la prédiction
input_data = pd.DataFrame([[wheel_base, length, width, height, curb_weight, engine_size]],
                          columns=["wheel-base", "length", "width", "height", "curb-weight", "engine-size"])

# Prédiction du prix
if st.sidebar.button("🔍 Prédire le prix"):
    prediction = best_model.predict(input_data)
    st.sidebar.success(f"💰 Prix estimé : {prediction[0]:,.2f} €")

# === Footer ===
st.write("---")
st.write("🚀 **Projet Machine Learning - Streamlit** | Développé par [Alex Rakotomalala]")
