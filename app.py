# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Titre
st.title("📊 Visualisation & Prédiction des prix immobiliers au Maroc")

# Chargement des données
df = pd.read_csv("immobilier_maroc.csv")

# Nettoyage (au cas où certaines colonnes ont des valeurs manquantes)
df = df.dropna(subset=["Prix", "Surface", "Chambres", "Quartier", "Vue_mer"])

# --- 📌 BAR PLOT Quartier ---
st.subheader("1️⃣ Répartition des annonces par quartier")
quartier_counts = df["Quartier"].value_counts()
st.bar_chart(quartier_counts)

# --- 📌 BAR PLOT Chambres ---
st.subheader("2️⃣ Répartition des annonces par nombre de chambres")
chambre_counts = df["Chambres"].value_counts().sort_index()
st.bar_chart(chambre_counts)

# --- 📌 SCATTER PLOT Prix vs Surface ---
st.subheader("3️⃣ Relation entre Surface et Prix")
fig1, ax1 = plt.subplots()
sns.scatterplot(data=df, x="Surface", y="Prix", hue="Chambres", palette="viridis", ax=ax1)
st.pyplot(fig1)

# --- 📌 MODÈLE DE RÉGRESSION ---
st.subheader("4️⃣ Prédiction du prix avec régression linéaire")

# Sélection des variables
X = df[["Surface", "Chambres", "Vue_mer"]]
y = df["Prix"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Affichage des coefficients
coeffs = pd.DataFrame({
    "Variable": X.columns,
    "Coefficient": model.coef_
})
st.write("📌 Coefficients du modèle :")
st.dataframe(coeffs)

# --- 📌 Prédiction personnalisée ---
st.markdown("### 🎯 Estimation de prix")

surface_input = st.number_input("Surface (m²)", min_value=20.0, max_value=1000.0, step=1.0)
chambres_input = st.slider("Nombre de chambres", 1, 10, 2)
vue_mer_input = st.selectbox("Vue sur mer", ["Oui", "Non"])
vue_mer_value = 1 if vue_mer_input == "Oui" else 0

# Prédiction
if st.button("Prédire le prix estimé"):
    pred = model.predict([[surface_input, chambres_input, vue_mer_value]])
    st.success(f"💰 Prix estimé : {pred[0]:,.0f} MAD")
