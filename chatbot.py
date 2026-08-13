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
    "bluetooth": "specifications_5",
    "batterie": "caractéristiques",
    "autonomie": "caractéristiques",
    "mah": "caractéristiques",
    "charge": "caractéristiques",
    "charger": "caractéristiques",
    "wireless": "caractéristiques",
    "powershare": "caractéristiques",
    "galaxy ai": "caractéristiques",
    "intelligence artificielle": "caractéristiques",
   
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
product.columns = product.columns.str.strip()  # Removes invisible spaces around headers

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
    sentences = re.split(r'(?<=[.!?])\s+|(?<=\bleu)\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    question = user_question.lower()
   
    if "nuit" in question or "photo de nuit" in question or "photos de nuit" in question:
        for sentence in sentences:
            if "vidéo de nuit" in sentence.lower() or "video de nuit" in sentence.lower():
                return sentence

    if any(word in question for word in ["couleur", "couleurs", "noir", "blanc", "bleu", "violet"]):
        for sentence in sentences:
            if "il est proposé en violet" in sentence.lower():
               return sentence

    if (
    "mégapixel" in question
    or "mégapixels" in question
    or "capteur" in question
    or re.search(r"\bmp\b", question)):
        for sentence in sentences:
            if "200 mp" in sentence.lower():
                return sentence

    keyword_mapping = {
        "charger": ["charge", "30 minutes", "wireless powershare"],
        "charge": ["charge", "30 minutes", "wireless powershare"],
        "batterie": ["batterie", "5000 mah", "31 heures"],
        "autonomie": ["batterie", "31 heures"],
        "galaxy ai": ["galaxy ai", "now brief", "now nudge"],
        "design": ["design", "aluminium", "armor"],
        "aluminium": ["aluminium", "armor"],
        "ip68": ["ip68"],
        "zoom": ["100x", "space zoom"],
        "photo": ["200 mp", "caméra", "photo"],
        "caméra": ["200 mp", "caméra"],
        "camera": ["200 mp", "camera"],
        "rafraîchissement": ["120hz", "120 hz", "taux de rafraîchissement"],
        "rafraichissement": ["120hz", "120 hz", "taux de rafraîchissement"],
        "hz": ["120hz", "120 hz"],
        "matériaux": ["aluminium", "armor", "gorilla"],
        "materiaux": ["aluminium", "armor", "gorilla"],
        "matériau": ["aluminium", "armor", "gorilla"],
        "construction": ["aluminium", "armor", "gorilla"],
    }

    if any(word in question for word in ["intelligence artificielle", "ia", "ai", "galaxy ai"]):
        for sentence in sentences:
            if "galaxy ai" in sentence.lower() or "intelligence" in sentence.lower():
                return sentence

    if "zoom" in question:
        for sentence in sentences:
            if "space zoom" in sentence.lower():
                return sentence

    if "précommande" in question or "precommande" in question or "commander" in question:
       for sentence in sentences:
           if "précommander" in sentence.lower() or "precommander" in sentence.lower():
             return sentence

    if "autonomie" in question:
       for sentence in sentences:
           if "batterie" in sentence.lower() or "5 000 mah" in sentence.lower():
              return sentence

    for key, words in keyword_mapping.items():
        if key in question:
            for sentence in sentences:
                if any(word.lower() in sentence.lower() for word in words):
                    return sentence

    processed_sentences = [preprocess(s) for s in sentences]
    processed_question = preprocess(user_question)

    if any(word in question for word in ["matériau", "matériaux", "materiau", "materiaux", "aluminium", "construction"]):
        for sentence in sentences:
            if any(m in sentence.lower() for m in ["aluminium", "gorilla", "verre", "titane", "cadre", "dos"]):
                return sentence

    if "résolution" in question and ("caméra" in question or "camera" in question):
        for sentence in sentences:
             if "200 mp" in sentence.lower():
                return sentence

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

    # 2. Filtre des produits concurrents hors sujet (Bloqué en premier pour éviter les confusions)
    other_products = [
        "iphone", "apple", "xiaomi", "redmi", "oppo",
        "huawei", "honor", "realme", "vivo", "google pixel",
        "nokia", "motorola"
    ]
    if any(product_name in question for product_name in other_products):
        return (
            "Je réponds uniquement aux questions concernant le Samsung Galaxy S26 Ultra.",
            0
        )

    # 3. Vérification des mots-clés autorisés (Sécurité globale)
    # 3. Vérification des mots-clés autorisés (on remet "ia" et "ai" ici)
    allowed_words = [
        "samsung", "galaxy", "s26", "ultra", "téléphone", "portable", "prix",
        "coûte", "cout", "tarif", "écran", "camera", "caméra", "photo", "zoom",
        "batterie", "charge", "garantie", "couleur", "couleurs", "noir", "blanc",
        "bleu", "violet", "design", "matériau", "materiau", "matériaux", "materiaux",
        "aluminium", "galaxy ai", "intelligence", "artificielle", "ia", "ai", 
        "now brief", "now nudge", "processeur", "fiche", "rafraîchissement",
        "rafraichissement", "hz", "macro", "mégapixels", "megapixels", "mégapixel",
        "megapixel", "mega pixel", "mega pixels", "ip68", "capteur", "arrière", 
        "poussiére", "poussière", "arriere", "connectivité", "connectivite",
        "wifi", "bluetooth", "5g", "double sim", "sim", "source", "officiel",
        "officielle", "lien", "site", "performance", "performant", "snapdragon",
        "puce", "cpu", "nuit", "photo de nuit", "mode nuit", "autonomie", 
        "précommande", "precommander", "commander"
    ]

    ai_pattern = r'\b(ai|ia|intelligence artificielle|galaxy ai)\b'

    # SÉCURITÉ : On bloque si aucun mot autorisé n'est présent ET que ce n'est pas non plus un "ia"/"ai" isolé
    has_allowed = any(word in question for word in allowed_words) or bool(re.search(ai_pattern, question))
    
    if not has_allowed:
        return (
            "Désolé, je ne peux pas répondre à cette question. Je me spécialise uniquement dans les caractéristiques du Samsung Galaxy S26 Ultra.",
            0
        )

    paragraph = product["caractéristiques"].iloc[0]
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', paragraph) if s.strip()]

    # 4. PRIORITÉS ABSOLUES (Traitement direct sans passer par le mapping)
    
    # A. Photo / Vidéo de nuit (expressions ciblées)
    if any(w in question for w in ["photo de nuit", "photos de nuit", "mode nuit", "video de nuit", "vidéo de nuit"]):
        for s in sentences:
            if "nuit" in s.lower():
                return s, 1.0

    # B. Now Brief & Now Nudge
    if "now brief" in question or "brief" in question:
        for s in sentences:
            if "now brief" in s.lower():
                return s, 1.0

    if "now nudge" in question or "nudge" in question:
        for s in sentences:
            if "now nudge" in s.lower():
                return s, 1.0

    # C. Galaxy AI (Mot entier avec \b pour éviter le piège de Cristiano)
    # GESTION INTELLIGENTE DE GALAXY AI
    ai_pattern = r'\b(ai|ia|intelligence artificielle|galaxy ai)\b'
    if re.search(ai_pattern, question):
        paragraph = product["caractéristiques"].iloc[0]
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', paragraph) if s.strip()]
        
        # 1. Si l'utilisateur demande spécifiquement les performances, le NPU ou le CPU
        if any(w in question for w in ["npu", "cpu", "performance", "performant", "processeur"]):
            for s in sentences:
                if "npu" in s.lower():
                    return s, 1.0
        
        # 2. Par défaut pour l'IA, on priorise les fonctions pratiques (retouche, creative studio, etc.)
        for s in sentences:
            if "retouche photo" in s.lower() or "creative studio" in s.lower():
                return s, 1.0
                
        # 3. Sécurité de secours
        for s in sentences:
            if "galaxy ai" in s.lower():
                return s, 1.0

    # D. Cas spécial : matériaux
    if any(word in question for word in ["matériau", "materiau", "matériaux", "materiaux", "aluminium", "construction"]):
        answer = extract_best_sentence(user_question, paragraph)
        return answer, 1.0

    # 5. Utiliser le mapping FAQ standard
    for keyword, column in faq_mapping.items():
        if keyword in question:
            if column == "caractéristiques":
                answer = extract_best_sentence(user_question, paragraph)
            elif column == "specifications_4":
                camera_text = product[column].iloc[0]
                if "200 MP" in camera_text:
                    answer = "La caméra principale arrière possède une résolution de 200 MP."
                else:
                    answer = camera_text
            else:
                answer = product[column].iloc[0]
            return answer, 1.0

    # 6. Recherche par similarité (TF-IDF) en dernier recours
    best_question, score = search_question(user_question)
    
    if score < 0.2:
        return (
            "Je n'ai pas trouvé d'information précise concernant votre demande sur le Samsung Galaxy S26 Ultra.",
            score
        )
        
    return product["caractéristiques"].iloc[0], score