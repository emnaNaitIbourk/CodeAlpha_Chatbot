import streamlit as st
import pandas as pd
import joblib

# =========================
# Configuration de la page
# =========================
st.set_page_config(
    page_title="Chatbot S26 Ultra",
    page_icon="📱",
    layout="centered",  # ou "wide"
    initial_sidebar_state="auto",
)

# =========================
# Chargement des fichiers avec cache
# =========================

@st.cache_data
def load_data():
    faq = pd.read_csv("data/Samsung_S26_Ultra_questions.csv", encoding="latin1")
    faq.columns = faq.columns.str.strip()
    
    product = pd.read_csv("data/Samsung_S26_Ultra_final_clean.csv", encoding="utf-8-sig")
    product.columns = product.columns.str.strip()
    
    product.columns = [c.lower().strip() for c in product.columns]
    for col in product.columns:
        if "caract" in col:
            product = product.rename(columns={col: "caractéristiques"})
            break
    return faq, product

@st.cache_resource
def load_models():
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    faq_vectors = joblib.load("models/faq_vectors.pkl")
    return vectorizer, faq_vectors

faq, product = load_data()
vectorizer, faq_vectors = load_models()

# =========================
# Import de la fonction chatbot
# =========================
from chatbot import get_answer

# =========================
# Interface Streamlit avec Sidebar
# =========================

# Affichage de l'image originale dans la barre latérale (propre et non déformée)
st.sidebar.image("background.jpg", caption="Samsung Galaxy S26 Ultra", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.info("Ce chatbot répond à toutes vos questions concernant les caractéristiques, le prix et les fonctionnalités du S26 Ultra.")

# Contenu principal
st.title("📱 Samsung Galaxy S26 Ultra Chatbot")
st.write("Posez votre question sur le Samsung Galaxy S26 Ultra")

# Utilisation d'un formulaire Streamlit (st.form)
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Votre question :")
    submit_button = st.form_submit_button("Envoyer")

if submit_button:
    if user_input.strip() != "":
        answer, score = get_answer(user_input)
        st.success(answer)
        st.info(f"Score de similarité : {score:.3f}")
    else:
        st.warning("Please enter a question.")