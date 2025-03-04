import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# === Configuration de la page Streamlit ===
st.set_page_config(page_title="🚗 Prédiction du Prix des Voitures", layout="wide")

st.title("🚗 Prédiction du Prix des Voitures")

# === 1️⃣ Importation et Nettoyage des données ===
st.subheader("📂 Chargement et nettoyage des données")

# Charger les données
df = pd.read_csv("automobile_data.csv", sep=";")

# Supprimer les doublons
df = df.drop_duplicates()

# Convertir les colonnes numériques
numeric_cols = ["bore", "stroke", "horsepower", "peak-rpm", "compression-ratio", "city-mpg", "highway-mpg", "price"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")  # Convertir en nombre, gérer erreurs

# Supprimer les lignes où le prix est manquant
df = df.dropna(subset=["price"])

# Remplacer les autres valeurs NaN par la médiane de chaque colonne
df.fillna(df.median(numeric_only=True), inplace=True)

# Vérifier s'il y a encore des valeurs infinies
df.replace([float("inf"), float("-inf")], df.median(numeric_only=True), inplace=True)

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

# Séparer la cible et les features
X = df.drop(columns=["price"])
y = df["price"]

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division en train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

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
wheel_base = st.sidebar.slider("Empattement (wheel-base)", min_value=86.6, max_value=120.9, value=98.8)
length = st.sidebar.slider("Longueur (length)", min_value=141.1, max_value=208.1, value=174.3)
width = st.sidebar.slider("Largeur (width)", min_value=60.3, max_value=72.0, value=65.9)
height = st.sidebar.slider("Hauteur (height)", min_value=47.8, max_value=59.8, value=53.8)
curb_weight = st.sidebar.slider("Poids à vide (curb-weight)", min_value=1488, max_value=4066, value=2559)
engine_size = st.sidebar.slider("Taille moteur (engine-size)", min_value=61, max_value=326, value=127)
compression_ratio = st.sidebar.slider("Ratio de compression", min_value=7.0, max_value=23.0, value=10.2)
city_mpg = st.sidebar.slider("Consommation en ville (city-mpg)", min_value=13, max_value=49, value=25)
highway_mpg = st.sidebar.slider("Consommation sur autoroute (highway-mpg)", min_value=16, max_value=54, value=30)

# Création du DataFrame pour la prédiction
input_data = pd.DataFrame([[wheel_base, length, width, height, curb_weight, engine_size, 
                            compression_ratio, city_mpg, highway_mpg]],
                          columns=["wheel-base", "length", "width", "height", "curb-weight", "engine-size", 
                                   "compression-ratio", "city-mpg", "highway-mpg"])

# Normalisation des données d'entrée
input_data_scaled = scaler.transform(input_data)

# Prédiction du prix
if st.sidebar.button("🔍 Prédire le prix"):
    prediction = best_model.predict(input_data_scaled)
    st.sidebar.success(f"💰 Prix estimé : {prediction[0]:,.2f} €")

# === Footer ===
st.write("---")
st.write("🚀 **Projet Machine Learning - Streamlit** | Développé avec ❤️ par Alex Rakotomalala")
