import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Prédiction du Prix des Voitures", layout="wide")
st.title("🚗 Prédiction du Prix des Voitures")

# Chargement et nettoyage des données
df = pd.read_csv("automobile_data.csv", sep=";").drop_duplicates()

# Traduction des colonnes en français
column_translation = {
    "body-style": "Style de carrosserie",
    "drive-wheels": "Roues motrices",
    "engine-location": "Emplacement moteur",
    "engine-type": "Type de moteur",
    "num-of-cylinders": "Nombre de cylindres",
    "fuel-system": "Système de carburant",
    "wheel-base": "Empattement",
    "length": "Longueur",
    "width": "Largeur",
    "height": "Hauteur",
    "curb-weight": "Poids à vide",
    "engine-size": "Taille moteur",
    "bore": "Alésage",
    "stroke": "Course",
    "compression-ratio": "Taux de compression",
    "horsepower": "Puissance",
    "peak-rpm": "Régime max",
    "city-mpg": "Consommation ville",
    "highway-mpg": "Consommation autoroute",
    "price": "Prix"
}
df.rename(columns=column_translation, inplace=True)

# Conversion en numérique
df[list(column_translation.values())] = df[list(column_translation.values())].apply(pd.to_numeric, errors="coerce")
df.dropna(inplace=True)
st.write(f"✅ Données nettoyées : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# Visualisation des prix
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["Prix"], bins=30, kde=True, ax=ax)
st.pyplot(fig)

# Préparation des données
df = pd.get_dummies(df, drop_first=True)
X, y = df.drop(columns=["Prix"]), df["Prix"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement des modèles
models = {"Régression Linéaire": LinearRegression(), "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    results[name] = mae
    st.write(f"📉 **MAE {name} :** {mae:.2f}")

best_model = min(results, key=results.get)
st.success(f"✅ Modèle sélectionné : {best_model}")

# Interface Streamlit pour prédiction
st.sidebar.header("🎯 Prédiction du prix")
inputs = {col: st.sidebar.number_input(col, value=df[col].mean()) for col in X.columns}
if st.sidebar.button("🔍 Prédire"):
    model = models[best_model]
    prediction = model.predict(pd.DataFrame([inputs]))[0]
    st.sidebar.success(f"💰 Prix estimé : {prediction:,.2f} €")
