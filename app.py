import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- Config de la page ---
st.set_page_config(page_title="🏠 Analyse Immobilier", layout="wide")

# --- Chargement des données ---
df = pd.read_csv("immobilier_maroc.csv")
df = df.dropna(subset=["Prix", "Surface", "Chambres", "Quartier", "Vue_mer"])

# ✅ Définition du modèle juste ici, AVANT la navigation
X = df[["Surface", "Chambres", "Vue_mer"]]
y = df["Prix"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
coeffs = pd.DataFrame({
    "Variable": X.columns,
    "Coefficient": model.coef_
})

# --- Menu de navigation ---
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Aller vers :", ["🏠 Accueil", "📊 Visualisation", "📈 Prédiction", "🎯 Estimation"])

# 🏠 Page d’accueil
if page == "🏠 Accueil":
    st.markdown("<h1 style='text-align: center;'>🏡 Dashboard Immobilier - Maroc</h1>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1600585154340-be6161a56a0c", use_column_width=True)
    st.markdown("""
    Bienvenue sur le tableau de bord immobilier 🏘️.  
    Explorez les données, visualisez les tendances et prédisez les prix.
    """)
#visualisation
elif page == "📊 Visualisation":
    st.header("📊 Visualisation des données")

    # Nettoyage des valeurs de la colonne "Ville"
    ville_list = df["Ville"].dropna().astype(str).unique()
    ville_filter = st.sidebar.selectbox("Filtrer par ville", ["Toutes"] + sorted(ville_list))

    if ville_filter != "Toutes":
        df_filtered = df[df["Ville"] == ville_filter]
    else:
        df_filtered = df

    # Quartier
    st.subheader("📍 Répartition des annonces par Quartier")
    quartier_counts = df_filtered["Quartier"].value_counts().head(10)
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=quartier_counts.values, y=quartier_counts.index, ax=ax1, palette="magma")
    ax1.set_xlabel("Nombre d'annonces")
    st.pyplot(fig1)

    # Chambres
    st.subheader("🛏️ Répartition par nombre de chambres")
    chambre_counts = df_filtered["Chambres"].value_counts().sort_index()
    fig2, ax2 = plt.subplots()
    sns.barplot(x=chambre_counts.index, y=chambre_counts.values, ax=ax2, palette="crest")
    ax2.set_xlabel("Chambres")
    ax2.set_ylabel("Nombre d'annonces")
    st.pyplot(fig2)

    # Scatter plot
    st.subheader("📉 Surface vs Prix")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df_filtered, x="Surface", y="Prix", hue="Chambres", palette="viridis", ax=ax3)
    ax3.set_title("Relation entre surface et prix")
    st.pyplot(fig3)

# 📈 Prédiction
elif page == "📈 Prédiction":
    st.header("📈 Modèle de régression linéaire")
    st.dataframe(coeffs)
    st.markdown(f"**R² score du modèle :** `{score:.2f}`")

# 🎯 Estimation
elif page == "🎯 Estimation":
    st.header("🎯 Estimation de prix")
    surface_input = st.number_input("Surface (m²)", min_value=20.0, max_value=1000.0, value=100.0)
    chambres_input = st.slider("Nombre de chambres", 1, 10, 2)
    vue_mer_input = st.radio("Vue sur mer ?", ["Oui", "Non"])
    vue_mer_value = 1 if vue_mer_input == "Oui" else 0

    if st.button("🔍 Prédire le prix"):
        pred = model.predict([[surface_input, chambres_input, vue_mer_value]])
        st.success(f"💰 Prix estimé : `{pred[0]:,.0f}` MAD")
