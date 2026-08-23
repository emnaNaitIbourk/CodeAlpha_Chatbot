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
    "power share": "caractéristiques",
    "résistant à l'eau": "caractéristiques",
    "resistant a l'eau": "caractéristiques",
    "résistant à l’eau": "caractéristiques",
    "resistant a l’eau": "caractéristiques",
    "étanche": "caractéristiques",
    "etanche": "caractéristiques",
    "acheter": "caractéristiques",
    "achat": "caractéristiques",
    "où acheter": "caractéristiques",
    "ou acheter": "caractéristiques",
    "où peut-on acheter": "caractéristiques",
    "ou peut on acheter": "caractéristiques",
    "où trouver": "caractéristiques",
    "ou trouver": "caractéristiques",
    "disponible à l'achat": "caractéristiques",
}

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
    
    # Gestion des questions générales sur la qualité des photos
    if any(word in question for word in [
        "bon pour prendre les photos",
        "bon pour les photos",
        "bon en photo",
        "qualité des photos",
        "qualité photo",
        "qualite des photos",
        "qualite photo"
    ]):
        for s in sentences:
            s_lower = s.lower()
            if (
                "caméra" in s_lower
                or "camera" in s_lower
                or "photo" in s_lower
                or "photographie" in s_lower
            ):
                return s
                
    # Cas spécial : résolution caméra
    camera_pattern = r'\bmp\b|mégapixel|mégapixels|capteur'
    if re.search(camera_pattern, question):
        for sentence in sentences:
            if "200 mp" in sentence.lower():
                return sentence

    # 1. Gestion spécifique des matériaux
    if any(word in question for word in [
        "matériau", "matériaux", "materiau", "materiaux", "aluminium", "construction", "titane", "gorilla", "verre"
    ]):
        strong_materials = ["aluminium", "titane", "gorilla", "verre", "armor"]
        weak_materials = ["cadre", "dos"]
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

    # Questions sur la précommande
    if any(word in question for word in [
        "précommande",
        "precommande",
        "précommander",
        "precommander",
        "disponible en précommande"]):
        for s in sentences:
            s_lower = s.lower()
            if (
                "précommande" in s_lower
                or "precommande" in s_lower
                or "précommander" in s_lower
                or "precommander" in s_lower
            ):
                return s
        return "Aucune information précise sur la précommande n'a été trouvée."
    
    # Gestion des questions sur la commande / achat
    if any(word in question for word in [
    "commander",
    "comment commander",
    "comment peut-on le commander",
    "comment le commander",
    "où commander",
    "ou commander",
    "acheter",
    "où acheter",
    "ou acheter",
    "acheter le téléphone",
    "acheter ce téléphone",
    "où peut-on acheter",
    "ou peut on acheter",
    "où trouver le téléphone",
    "ou trouver le téléphone"]):
        for s in sentences:
            s_lower = s.lower()
            if any(word in s_lower for word in [
                "commander",
                "commande",
                "acheter",
                "achat",
                "précommande",
                "précommander"
            ]):
                return s.strip()

    # Si aucune phrase spécifique n'est trouvée par règles, similarité TF-IDF sur les phrases du texte
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

    # 2. Filtre des produits concurrents hors sujet
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

    # 3. GESTION PRIORITAIRE ET STRICTE DE LA CONNECTIVITÉ (ANTI-MÉLANGE)
    # Placée ici pour s'arrêter immédiatement sans déclencher le reste du code
    paragraph = product["caractéristiques"].iloc[0]
    connectivite_answers = []
    
    if any(k in question for k in ["wifi", "wi-fi", "bluetooth", "powershare", "power share"]):
        if any(k in question for k in ["wifi", "wi-fi"]):
            for s in re.split(r'(?<=[.!?])\s+', paragraph):
                if "wi-fi 7" in s.lower() or "wifi 7" in s.lower():
                    connectivite_answers.append("Le téléphone prend en charge le Wi-Fi 7.")
                    break
                    
        if "bluetooth" in question:
            for s in re.split(r'(?<=[.!?])\s+', paragraph):
                if "bluetooth 6.0" in s.lower():
                    connectivite_answers.append("Le téléphone utilise le Bluetooth 6.0.")
                    break
                    
        if any(k in question for k in ["powershare", "power share"]):
            for s in re.split(r'(?<=[.!?])\s+', paragraph):
                if "powershare" in s.lower() or "power share" in s.lower():
                    connectivite_answers.append(s.strip())
                    break
                    
        if connectivite_answers:
            return " ".join(connectivite_answers), 1.0

    # 4. SÉCURITÉ : Liste des mots autorisés
    allowed_words = [
        "samsung", "galaxy", "s26", "ultra", "téléphone", "portable", "prix",
        "coûte", "cout", "tarif", "écran", "camera", "caméra", "photo", "zoom",
        "batterie", "charge", "garantie", "couleur", "couleurs", "noir", "blanc",
        "bleu", "violet","powershare", "power share", "eau", "étanche", "etanche",
        "résistant", "resistant", "design", "matériau", "materiau", "matériaux", "materiaux",
        "aluminium", "galaxy ai", "intelligence", "artificielle", 
        "now brief", "now nudge", "processeur", "fiche", "rafraîchissement",
        "rafraichissement", "hz", "macro", "mégapixels", "megapixels", "mégapixel",
        "megapixel", "mega pixel", "mega pixels", "ip68", "capteur", "arrière", 
        "poussiére", "poussière", "arriere", "connectivité", "connectivite",
        "wifi", "wi-fi", "bluetooth", "5g", "double sim", "sim", "source", "officiel",
        "officielle", "lien", "site", "performance", "performant", "snapdragon",
        "puce", "cpu", "nuit", "photo de nuit", "mode nocturne", "mode nuit", "autonomie", 
        "où peut-on acheter", "précommande", "precommande", "précommander", "precommander", 
        "commander", "acheter", "achat", "acheter le téléphone", "où acheter", "ou acheter", 
        "disponible à l'achat", "puissant", "puissance"
    ]

    ai_pattern = r'\b(ai|ia|intelligence artificielle|galaxy ai)\b'

    has_allowed = any(word in question for word in allowed_words) or bool(re.search(ai_pattern, question))
    
    if not has_allowed:
        return (
            "Désolé, je ne peux pas répondre à cette question. Je me spécialise uniquement dans les caractéristiques du Samsung Galaxy S26 Ultra.",
            0
        )

    # ==========================================
    # === PRIORITÉ ABSOLUE : CAS SPÉCIFIQUES ===
    # ==========================================

    # A. Photos de nuit / mode nocturne
    if any(word in question for word in [
        "nuit", "photo de nuit", "photos de nuit", "comment sont les photos de nuit",
        "qualité des photos de nuit", "qualite des photos de nuit", "photos nocturnes",
        "photographie de nuit", "basse lumière", "faible luminosité", "faible lumière",
        "mode nuit", "mode nocturne"
    ]):
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        for s in sentences:
            s_lower = s.lower()
            if any(word in s_lower for word in [
                "photo de nuit", "photos de nuit", "mode nuit", "faible luminosité",
                "faible lumière", "moins de bruit", "nuit"
            ]):
                return s.strip(), 1.0
        return "Aucune information précise sur les photos de nuit n'a été trouvée.", 0.3

    # B. Résistance à l'eau / IP68
    if any(word in question for word in [
        "eau", "étanche", "etanche", "ip68", "résistant à l'eau", "resistant a l'eau",
        "résistant à l’eau", "resistant a l’eau"
    ]):
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        for s in sentences:
            s_lower = s.lower()
            if (
                "ip68" in s_lower
                or "eau" in s_lower
                or "résistant à l'eau" in s_lower
                or "resistant a l'eau" in s_lower
            ):
                return s.strip(), 1.0
        return "Aucune information précise sur la résistance à l'eau n'a été trouvée.", 0.3

    # C. Matériaux / Design / Aluminium
    if any(word in question for word in [
        "matériau", "matériaux", "materiau", "materiaux", "aluminium", "construction", "titane", "gorilla", "verre"
    ]):
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        for s in sentences:
            if any(m in s.lower() for m in ["aluminium", "titane", "gorilla", "verre", "cadre", "dos", "armor"]):
                return s.strip(), 1.0
        return "Aucune information précise sur les matériaux n'a été trouvée.", 0.3

    # D. Processeur et performances
    performance_keywords = [
        "processeur", "cpu", "puce", "snapdragon", "performance", "performant", "puissant", "puissance", "npu"
    ]
    if any(keyword in question for keyword in performance_keywords):
        processor_text = str(product["specifications_2"].iloc[0])
        return processor_text, 1.0

    # E. Questions générales sur la qualité photo
    if any(word in question for word in [
        "bon pour prendre les photos", "bon pour les photos", "bon en photo",
        "bon pour prendre des photos", "est il bon pour prendre les photos",
        "est-il bon pour prendre les photos", "est il bon en photo", "est-il bon en photo",
        "est ce qu'il est bon en photo", "est-ce qu'il est bon en photo", "bonne qualité photo",
        "bonne qualité des photos", "qualité des photos", "qualite des photos",
        "qualité photo", "qualite photo", "appareil photo performant", "caméra performante"
    ]):
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        for s in sentences:
            s_lower = s.lower()
            if any(word in s_lower for word in [
                "qualité photo", "qualité des photos", "photographie",
                "photo détaillée", "photos détaillées", "détails",
                "stabilisation", "amélioration ia", "intelligence artificielle"
            ]):
                return s.strip(), 1.0
        for s in sentences:
            s_lower = s.lower()
            if "caméra" in s_lower or "camera" in s_lower:
                return s.strip(), 1.0
        return "Aucune information précise sur la qualité photo n'a été trouvée.", 0.3

    # F. Questions générales sur Galaxy AI
    if re.search(ai_pattern, question):
        for s in re.split(r'(?<=[.!?])\s+', paragraph):
            s_lower = s.lower()
            if (
                "galaxy ai facilite" in s_lower
                or "retouche photo" in s_lower
                or "creative studio" in s_lower
                or "now nudge" in s_lower
                or "now brief" in s_lower
            ):
                return s, 1.0

    # ==========================================
    # === UTILISER LE MAPPING FAQ GÉNÉRAL ===
    # ==========================================
    collected_answers = []
    checked_columns = set()

    for keyword, column in faq_mapping.items():
        if keyword in question:
            if column in checked_columns and column != "caractéristiques":
                continue
            
            answer = None

            if column == "caractéristiques":
                battery_terms = ["batterie", "autonomie", "mah", "charge", "charger"]
                is_battery_query = any(bt in question for bt in battery_terms)
                
                if keyword in battery_terms or is_battery_query:
                    answer = extract_best_sentence(user_question, paragraph)
                else:
                    continue
            
            elif column == "specifications_4":
                camera_text = product[column].iloc[0]
                if "200 MP" in camera_text:
                    answer = "La caméra principale arrière possède une résolution de 200 MP."
                else:
                    answer = camera_text
            else:
                answer = product[column].iloc[0]

            if answer and answer not in collected_answers:
                collected_answers.append(answer)
                checked_columns.add(column)

    if collected_answers:
        return " ".join(collected_answers), 1.0

    # ==========================================
    # === RECHERCHE PAR SIMILARITÉ (TF-IDF) ====
    # ==========================================
    best_question, score = search_question(user_question)
    
    if score < 0.35:
        return (
            "Je n'ai pas trouvé d'information précise concernant votre demande sur le Samsung Galaxy S26 Ultra.",
            score
        )
        
    return extract_best_sentence(user_question, paragraph), score