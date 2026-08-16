faq_mapping = {
    "prix": "price",
    "coûte": "price",
    "coute": "price",
    "vendu": "price",
    "tarif": "price",
    "coût": "price",
    "cout": "price",
    "garantie": "specifications_1",
    "garanti": "specifications_1",
    "garantie constructeur": "specifications_1",
    "processeur": "specifications_2",
    "cpu": "specifications_2",
    "puce": "specifications_2",
    "snapdragon": "specifications_2",
    "performance": "specifications_2",
    "performant": "specifications_2",
    "écran": "specifications_3",
    "ecran": "specifications_3",
    "amoled": "specifications_3",
    "résolution écran": "specifications_3",
    "resolution ecran": "specifications_3",
    "dynamic amoled": "specifications_3",
    "taille": "specifications_3",
    "120hz": "specifications_3",
    "rafraîchissement": "specifications_3",
    "rafraichissement": "specifications_3",
    "caméra": "specifications_4",
    "camera": "specifications_4",
    "photo": "specifications_4",
    "photos": "specifications_4",
    "appareil photo": "specifications_4",
    "objectif": "specifications_4",
    "zoom": "caractéristiques",
    "space zoom": "caractéristiques",
    "nuit": "caractéristiques",
    "photo de nuit": "caractéristiques",
    "photos de nuit": "caractéristiques",
    "mode nuit": "caractéristiques",
    "5g": "specifications_5",
    "sim": "specifications_5",
    "double sim": "specifications_5",
    "réseau": "specifications_5",
    "reseau": "specifications_5",
    "connectivité": "specifications_5",
    "connectivite": "specifications_5",
    "wifi": "specifications_5",
    "wi-fi": "specifications_5",
    "bluetooth": "specifications_5",
    "batterie": "caractéristiques",
    "autonomie": "caractéristiques",
    "mah": "caractéristiques",
    "charge": "caractéristiques",
    "charger": "caractéristiques",
    "wireless": "caractéristiques",
    "powershare": "caractéristiques",
    "now brief": "caractéristiques",
    "now nudge": "caractéristiques",
    "couleur": "caractéristiques",
    "couleurs": "caractéristiques",
    "noir": "caractéristiques",
    "blanc": "caractéristiques",
    "bleu": "caractéristiques",
    "violet": "caractéristiques",
    "design": "caractéristiques",
    "aluminium": "caractéristiques",
    "gorilla": "caractéristiques",
    "ip68": "caractéristiques",
    "eau": "caractéristiques",
    "poussière": "caractéristiques",
    "poussiére": "caractéristiques",
    "matériaux": "caractéristiques",
    "materiaux": "caractéristiques",
    "matériau": "caractéristiques",
    "construction": "caractéristiques",
    "précommande": "caractéristiques",
    "precommande": "caractéristiques",
    "commander": "caractéristiques",
    "résolution caméra": "specifications_4",
    "resolution camera": "specifications_4",
    "200 mp": "caractéristiques",
    "mégapixel": "specifications_4",
    "mégapixels": "specifications_4",
    "megapixels": "specifications_4",
    "capteur": "specifications_4",
    "capteur principal": "specifications_4",
    "caméra arrière": "specifications_4",
    "camera arriere": "specifications_4",
    "arrière": "specifications_4",
    "arriere": "specifications_4",
    "source": "source",
    "lien": "source",
    "site": "source",
    "officiel": "source",
    "officielle": "source",
}

import re
import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

stop_words = set(stopwords.words("french"))

faq = pd.read_csv(
    "data/Samsung_S26_Ultra_questions.csv",
    encoding="latin1"
)

product = pd.read_csv(
    "data/Samsung_S26_Ultra_final_clean.csv",
    encoding="utf-8-sig"
)
product.columns = product.columns.str.strip()

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

faq_vectors = joblib.load(
    "models/faq_vectors.pkl"
)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zàâçéèêëîïôùûü0-9 ]',' ',text)
    tokens = word_tokenize(text, language="french")
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

def search_question(user_question):
    processed = preprocess(user_question)
    vector = vectorizer.transform([processed])
    similarity = cosine_similarity(
        vector,
        faq_vectors
    )
    index = similarity.argmax()
    score = similarity[0][index]
    return faq.iloc[index]["question"], score

def extract_best_sentence(user_question, text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    question = user_question.lower()
    # Cas spécial : photos / vidéo de nuit
    # Cas spécial : photos de nuit
    if any(word in question for word in [
    "nuit",
    "photo de nuit",
    "photos de nuit",
    "mode nuit"]):
        for sentence in sentences:
            sentence_lower = sentence.lower()

            if (
            "vidéo de nuit" in sentence_lower
            or "video de nuit" in sentence_lower
            or "photo de nuit" in sentence_lower
            or "photos de nuit" in sentence_lower
            or "faible luminosité" in sentence_lower
            or "faible lumière" in sentence_lower
            or "moins de bruit" in sentence_lower):
                return sentence
    # Cas spécial : résolution caméra
    # Cas spécial : résolution caméra
    camera_pattern = r'\bmp\b|mégapixel|mégapixels|capteur'
    if re.search(camera_pattern, question):
        for sentence in sentences:
            if "200 mp" in sentence.lower():
               return sentence
   
    # 1. Gestion spécifique des matériaux
    # 1. Gestion spécifique des matériaux
    if any(word in question for word in [
    "matériau",
    "matériaux",
    "materiau",
    "materiaux",
    "aluminium",
    "construction"
]):
        strong_materials = ["aluminium", "titane", "gorilla", "verre", "armor"]
        weak_materials = ["cadre", "dos"]
        # Priorité aux phrases contenant un vrai nom de matériau
        for s in sentences:
            if any(m in s.lower() for m in strong_materials):
                return s
        for s in sentences:
            if any(m in s.lower() for m in weak_materials):
                return s

    # 2. Gestion de la batterie / autonomie / charge
    if any(word in question for word in ["batterie", "autonomie", "mah", "charge", "charger"]):
        for s in sentences:
            if any(m in s.lower() for m in ["batterie", "5000", "5 000", "mah", "charge", "autonomie", "heures"]):
                return s

    # Par défaut, similarité TF-IDF sur les phrases du texte
    processed_sentences = [preprocess(s) for s in sentences]
    processed_question = preprocess(user_question)

    tfidf = TfidfVectorizer()
    sentence_vectors = tfidf.fit_transform(processed_sentences)
    question_vector = tfidf.transform([processed_question])
    similarities = cosine_similarity(question_vector, sentence_vectors)

    best_index = similarities.argmax()
    return sentences[best_index]

def get_answer(user_question):
    question = user_question.lower()

    # 1. Gestion des messages d'au revoir
    goodbye_words = ["au revoir", "bye", "salut", "à bientôt", "a bientot", "a+", "ciao", "merci"]
    if any(word in question for word in goodbye_words):
        return "Au revoir ! J'espère avoir pu t'aider. N'hésite pas si tu as d'autres questions sur le Samsung Galaxy S26 Ultra.", 1.0

    # 2. Filtre des produits concurrents hors sujet (UNIQUEMENT les autres marques de téléphones)
    other_products = [
        "iphone", "apple", "xiaomi", "redmi", "oppo",
        "huawei", "honor", "realme", "vivo", "google pixel",
        "nokia", "motorola"
    ]
    if any(prod in question for prod in other_products):
        return (
            "Désolé, je ne peux pas répondre à cette question. Je me spécialise uniquement dans les caractéristiques du Samsung Galaxy S26 Ultra.",
            0
        )

    # 3. SÉCURITÉ : Liste des mots autorisés (avec gestion propre de l'IA par regex \b)
    allowed_words = [
        "samsung", "galaxy", "s26", "ultra", "téléphone", "portable", "prix",
        "coûte", "cout", "tarif", "écran", "camera", "caméra", "photo", "zoom",
        "batterie", "charge", "garantie", "couleur", "couleurs", "noir", "blanc",
        "bleu", "violet", "design", "matériau", "materiau", "matériaux", "materiaux",
        "aluminium", "galaxy ai", "intelligence", "artificielle", 
        "now brief", "now nudge", "processeur", "fiche", "rafraîchissement",
        "rafraichissement", "hz", "macro", "mégapixels", "megapixels", "mégapixel",
        "megapixel", "mega pixel", "mega pixels", "ip68", "capteur", "arrière", 
        "poussiére", "poussière", "arriere", "connectivité", "connectivite",
        "wifi", "wi-fi", "bluetooth", "5g", "double sim", "sim", "source", "officiel",
        "officielle", "lien", "site", "performance", "performant", "snapdragon",
        "puce", "cpu", "nuit", "photo de nuit", "mode nuit", "autonomie", 
        "précommande", "precommander", "commander" 
    ]

    ai_pattern = r'\b(ai|ia|intelligence artificielle|galaxy ai)\b'

    has_allowed = any(word in question for word in allowed_words) or bool(re.search(ai_pattern, question))
    
    if not has_allowed:
        return (
            "Désolé, je ne peux pas répondre à cette question. Je me spécialise uniquement dans les caractéristiques du Samsung Galaxy S26 Ultra.",
            0
        )

    paragraph = product["caractéristiques"].iloc[0]

    # 4. Traitement spécifique des questions sur l'IA / CPU
    if re.search(ai_pattern, question):
        if any(w in question for w in ["npu", "cpu", "performance", "performant", "processeur"]):
            for s in re.split(r'(?<=[.!?])\s+', paragraph):
                if "npu" in s.lower() or "processeur" in s.lower():
                    return s, 1.0
        for s in re.split(r'(?<=[.!?])\s+', paragraph):
            if "galaxy ai" in s.lower() or "intelligence" in s.lower():
                return s, 1.0
    # Cas spécial : photos de nuit
    # Cas spécial : photos de nuit
    if any(word in question for word in [
    "nuit",
    "photo de nuit",
    "photos de nuit",
    "mode nuit"]):
        for s in re.split(r'(?<=[.!?])\s+', paragraph):
            s_lower = s.lower()

            if (
            "vidéo de nuit" in s_lower
            or "video de nuit" in s_lower
            or "photo de nuit" in s_lower
            or "photos de nuit" in s_lower
            or "faible luminosité" in s_lower
            or "faible lumière" in s_lower
            or "moins de bruit" in s_lower):
               return s, 1.0

    # 5. Utiliser le mapping FAQ standard (gère le Wi-Fi, Bluetooth, etc.)
    for keyword, column in faq_mapping.items():
        if keyword in question:
            if keyword in ["wifi", "wi-fi"]:
                paragraph = product["caractéristiques"].iloc[0]
                for s in re.split(r'(?<=[.!?])\s+', paragraph):
                    if "wi-fi 7" in s.lower() or "wifi 7" in s.lower():
                        return "Le téléphone prend en charge le Wi-Fi 7.", 1.0
                return "Aucune information précise sur le Wi-Fi n'a été trouvée.", 0.3

            if keyword == "bluetooth":
                paragraph = product["caractéristiques"].iloc[0]
                for s in re.split(r'(?<=[.!?])\s+', paragraph):
                    if "bluetooth 6.0" in s.lower():
                        return "Le téléphone utilise le Bluetooth 6.0.", 1.0
                return "Aucune information précise sur le Bluetooth n'a été trouvée.", 0.3
            if column == "caractéristiques":
                answer = extract_best_sentence(user_question, paragraph)
            elif column == "specifications_4":
                camera_text = product[column].iloc[0]
                if "200 MP" in camera_text:
                    answer = "La caméra principale arrière possède une résolution de 200 MP."
                else:
                    answer = camera_text
            else:
                # Pour specifications_5 (Wi-Fi, Bluetooth, 5G, SIM) ou specifications_2, etc.
                answer = product[column].iloc[0]
            return answer, 1.0

    # 6. Recherche par similarité (TF-IDF) en dernier recours
    best_question, score = search_question(user_question)
    
    if score < 0.35:
        return (
            "Je n'ai pas trouvé d'information précise concernant votre demande sur le Samsung Galaxy S26 Ultra.",
            score
        )
        
    return extract_best_sentence(user_question, paragraph), score