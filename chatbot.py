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
    "disponible à l'achat": "caractéristiques"
}

faq = pd.read_csv("data/Samsung_S26_Ultra_questions.csv", encoding="latin1")

product = pd.read_csv("data/Samsung_S26_Ultra_final_clean.csv", encoding="utf-8-sig")
product.columns = product.columns.str.strip()

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
faq_vectors = joblib.load("models/faq_vectors.pkl")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zàâçéèêëîïôùûü0-9 ]', ' ', text)
    tokens = word_tokenize(text, language="french")
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

def search_question(user_question):
    processed = preprocess(user_question)
    vector = vectorizer.transform([processed])
    similarity = cosine_similarity(vector, faq_vectors)
    index = similarity.argmax()
    score = similarity[0][index]
    return faq.iloc[index]["question"], score

def extract_best_sentence(user_question, text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    question = user_question.lower()
    
    # Gestion des questions générales sur la qualité des photos
    if any(word in question for word in [
        "bon pour prendre les photos", "bon pour les photos", "bon en photo",
        "qualité des photos", "qualité photo", "qualite des photos", "qualite photo"
    ]):
        for s in sentences:
            s_lower = s.lower()
            if "caméra" in s_lower or "camera" in s_lower or "photo" in s_lower or "photographie" in s_lower:
                return s
            
    # Cas spécial : résolution caméra
    camera_pattern = r'\bmp\b|mégapixel|mégapixels|capteur'
    if re.search(camera_pattern, question):
        for sentence in sentences:
            if "200 mp" in sentence.lower():
                return sentence

    # 1. Gestion spécifique des matériaux
    if any(word in question for word in ["matériau", "matériaux", "materiau", "materiaux", "aluminium", "construction", "titane", "gorilla", "verre"]):
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
    if any(word in question for word in ["précommande", "precommande", "précommander", "precommander", "disponible en précommande"]):
        for s in sentences:
            s_lower = s.lower()
            if "précommande" in s_lower or "precommande" in s_lower or "précommander" in s_lower or "precommander" in s_lower:
                return s
        return "Aucune information précise sur la précommande n'a été trouvée."
    
    # Gestion des questions sur la commande / achat
    if any(word in question for word in [
        "commander", "comment commander", "comment peut-on le commander", "comment le commander",
        "où commander", "ou commander", "acheter", "où acheter", "ou acheter",
        "acheter le téléphone", "acheter ce téléphone", "où peut-on acheter", "ou peut on acheter",
        "où trouver le téléphone", "ou trouver le téléphone"
    ]):
        for s in sentences:
            s_lower = s.lower()
            if any(word in s_lower for word in ["commander", "commande", "acheter", "achat", "précommande", "précommander"]):
                return s.strip()

    # Si aucune phrase spécifique n'est trouvée par règles, similarité TF-IDF sur les phrases
    processed_sentences = [preprocess(s) for s in sentences]
    processed_question = preprocess(user_question)

    tfidf = TfidfVectorizer()
    sentence_vectors = tfidf.fit_transform(processed_sentences)
    question_vector = tfidf.transform([processed_question])
    similarities = cosine_similarity(question_vector, sentence_vectors)

    best_index = similarities.argmax()
    return sentences[best_index]

def get_answer(user_question):
    if not user_question or not user_question.strip():
        return ("Veuillez poser une question sur le Samsung Galaxy S26 Ultra.", 0.0)

    question = user_question.lower().strip()

    # 1. Gestion des messages d'au revoir
    goodbye_words = ["au revoir", "bye", "salut", "à bientôt", "a bientot", "a+", "ciao", "merci"]
    if any(word in question for word in goodbye_words):
        return ("Au revoir ! J'espère avoir pu t'aider. N'hésite pas si tu as d'autres questions sur le Samsung Galaxy S26 Ultra.", 1.0)

    # 2. Filtre des produits concurrents hors sujet
    other_products = ["iphone", "apple", "xiaomi", "redmi", "oppo", "huawei", "honor", "realme", "vivo", "google pixel", "nokia", "motorola"]
    if any(prod in question for prod in other_products):
        return ("Désolé, je ne peux pas répondre à cette question. Je me spécialise uniquement dans les caractéristiques du Samsung Galaxy S26 Ultra.", 0)

    paragraph = str(product["caractéristiques"].iloc[0])

    # === PRIORITÉ ABSOLUE 1 : PROCESSEUR, PUCE & GAMING ===
    performance_keywords = ["processeur", "cpu", "puce", "snapdragon", "performance", "performant", "puissant", "puissance", "npu"]
    if any(keyword in question for keyword in performance_keywords):
        proc_text = str(product["specifications_2"].iloc[0])
        return proc_text, 1.0

    # === PRIORITÉ ABSOLUE 2 : GALAXY AI / INTELLIGENCE ARTIFICIELLE ===
    ai_pattern_check = r"\b(ai|ia|galaxy ai|intelligence artificielle)\b"
    if re.search(ai_pattern_check, question):
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        for s in sentences:
            s_lower = s.lower()
            if any(kw in s_lower for kw in ["retouche", "creative studio", "stickers", "now nudge", "now brief"]):
                return s.strip(), 1.0
        for s in sentences:
            s_lower = s.lower()
            if "galaxy ai" in s_lower and ("facilite" in s_lower or "fonction" in s_lower or "assistant" in s_lower):
                return s.strip(), 1.0
        for s in sentences:
            s_lower = s.lower()
            if "npu" in s_lower or "traitement" in s_lower:
                return s.strip(), 1.0
        return ("Galaxy AI intègre des outils de retouche photo et des assistants intelligents.", 1.0)

    # === 3. GESTION PRIORITAIRE ET STRICTE DE LA CONNECTIVITÉ ===
    connectivite_answers = []
    if any(k in question for k in ["wifi", "wi-fi", "bluetooth", "powershare", "power share"]):
        if any(k in question for k in ["wifi", "wi-fi"]):
            for s in re.split(r"(?<=[.!?])\s+", paragraph):
                if "wi-fi 7" in s.lower() or "wifi 7" in s.lower():
                    connectivite_answers.append("Le téléphone prend en charge le Wi-Fi 7.")
                    break
        if "bluetooth" in question:
            for s in re.split(r"(?<=[.!?])\s+", paragraph):
                if "bluetooth 6.0" in s.lower():
                    connectivite_answers.append("Le téléphone utilise le Bluetooth 6.0.")
                    break
        if any(k in question for k in ["powershare", "power share"]):
            for s in re.split(r"(?<=[.!?])\s+", paragraph):
                if "powershare" in s.lower() or "power share" in s.lower():
                    connectivite_answers.append(s.strip())
                    break

    if connectivite_answers:
        return " ".join(connectivite_answers), 1.0

    # 4. SÉCURITÉ : Liste des mots autorisés
    allowed_words = [
        "samsung", "galaxy", "s26", "ultra", "téléphone", "portable", "prix", "coûte", "cout", "tarif",
        "écran", "camera", "caméra", "photo", "zoom", "batterie", "charge", "garantie", "couleur", "couleurs",
        "noir", "blanc", "bleu", "violet", "powershare", "power share", "eau", "étanche", "etanche",
        "résistant", "resistant", "design", "matériau", "materiau", "matériaux", "materiaux", "aluminium",
        "galaxy ai", "intelligence", "artificielle", "now brief", "now nudge", "processeur", "fiche",
        "rafraîchissement", "rafraichissement", "hz", "macro", "mégapixels", "megapixels", "mégapixel",
        "megapixel", "mega pixel", "mega pixels", "ip68", "capteur", "arrière", "poussiére", "poussière",
        "arriere", "connectivité", "connectivite", "wifi", "wi-fi", "bluetooth", "5g", "double sim",
        "sim", "source", "officiel", "officielle", "lien", "site", "performance", "performant",
        "snapdragon", "puce", "cpu", "nuit", "photo de nuit", "mode nocturne", "mode nuit", "autonomie",
        "où peut-on acheter", "précommande", "precommande", "précommander", "precommander", "commander",
        "acheter", "achat", "acheter le téléphone", "où acheter", "ou acheter", "disponible à l'achat",
        "puissant", "puissance", "nouveau", "nouvelle", "modèle", "modele", "qualité", "qualite",
        "selfie", "frontale", "avant", "caméra avant"
    ]
    
    ai_pattern = r"\b(ai|ia|intelligence artificielle|galaxy ai)\b"
    has_allowed = any(word in question for word in allowed_words) or bool(re.search(ai_pattern, question))
    if not has_allowed:
        return ("Désolé, je ne peux pas répondre à cette question. Je me spécialise uniquement dans les caractéristiques du Samsung Galaxy S26 Ultra.", 0)

    # === A. GESTION INTELLIGENTE DES PHOTOS ===
    if any(word in question for word in ["photo", "photos", "qualité", "qualite", "caméra", "camera", "megapixel", "mégapixels", "capteur", "nuit", "nocturne", "zoom", "selfie", "frontale", "avant"]):
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        # 1. SI LA QUESTION CONCERNE LA NUIT (Priorité haute pour la nuit)
        if any(w in question for w in ["nuit", "nocturne", "basse lumière", "sombre", "lumière"]):
            for s in sentences:
                s_lower = s.lower()
                if any(term in s_lower for term in ["nuit", "optique", "traitement", "bruit", "lumineux", "sombre"]):
                    return s.strip(), 1.0
            return ("L'optique et le traitement logiciel du Galaxy S26 Ultra sont spécialement pensés pour la vidéo et les photos de nuit.", 1.0)

        # 2. SI LA QUESTION CONCERNE LE SELFIE / CAMÉRA FRONTALE
        if any(w in question for w in ["selfie", "frontale", "avant"]):
            for s in sentences:
                if any(w in s.lower() for w in ["selfie", "frontale", "12 mp"]):
                    return s.strip(), 1.0
            return ("La caméra frontale (selfie) du Samsung Galaxy S26 Ultra possède une résolution de 12.0 MP.", 1.0)

        # 3. SI LA QUESTION CONCERNE LE ZOOM
        if any(w in question for w in ["zoom", "loin", "éloigné", "eloigne"]):
            for s in sentences:
                if "zoom" in s.lower():
                    return s.strip(), 1.0

        # 4. SI LA QUESTION CONCERNE LA QUALITÉ GÉNÉRALE OU LES CAPTEURS ARRIÈRE
        if "specifications_4" in product.columns:
            camera_text = str(product["specifications_4"].iloc[0])
            if camera_text and camera_text.lower() != "nan":
                return camera_text, 1.0

        return ("Le Samsung Galaxy S26 Ultra dispose d'un système photo avancé ultra performant.", 1.0)

    # === B. Résistance à l'eau / IP68 ===
    if any(word in question for word in ["eau", "étanche", "etanche", "ip68", "résistant à l'eau", "resistant a l'eau", "résistant à l’eau", "resistant a l’eau"]):
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        for s in sentences:
            s_lower = s.lower()
            if "ip68" in s_lower or "eau" in s_lower or "résistant à l'eau" in s_lower or "resistant a l'eau" in s_lower:
                return s.strip(), 1.0
        return ("Aucune information précise sur la résistance à l'eau n'a été trouvée.", 0.3)

    # === C. Matériaux / Design / Aluminium ===
    if any(word in question for word in ["matériau", "matériaux", "materiau", "materiaux", "aluminium", "construction", "titane", "gorilla", "verre"]):
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        for s in sentences:
            if any(m in s.lower() for m in ["aluminium", "titane", "gorilla", "verre", "cadre", "dos", "armor"]):
                return s.strip(), 1.0
        return ("Aucune information précise sur les matériaux n'a été trouvée.", 0.3)

    # === UTILISER LE MAPPING FAQ GÉNÉRAL ===
    collected_answers = []
    checked_columns = set()

    for keyword, column in faq_mapping.items():
        if keyword not in question:
            continue

        if column in checked_columns and column != "caractéristiques":
            continue

        answer = None

        if column == "caractéristiques":
            battery_terms = ["batterie", "autonomie", "mah", "charge", "charger"]
            is_battery_query = any(bt in question for bt in battery_terms)

            if keyword in battery_terms or is_battery_query:
                answer = extract_best_sentence(user_question, paragraph)
            elif keyword in ["now brief", "now nudge"]:
                for s in re.split(r"(?<=[.!?])\s+", paragraph):
                    if keyword in s.lower():
                        answer = s.strip()
                        break
                if not answer:
                    answer = extract_best_sentence(user_question, paragraph)
            elif keyword in ["commander", "acheter", "achat", "précommande", "precommande"]:
                for s in re.split(r"(?<=[.!?])\s+", paragraph):
                    if any(k in s.lower() for k in ["commander", "acheter", "achat", "précommande", "precommande"]):
                        answer = s.strip()
                        break
                if not answer:
                    answer = extract_best_sentence(user_question, paragraph)
            else:
                continue
        elif column == "specifications_4":
            camera_text = str(product[column].iloc[0])
            if "200 MP" in camera_text:
                answer = "La caméra principale arrière possède une résolution de 200 MP."
            else:
                answer = camera_text
        else:
            answer = str(product[column].iloc[0])

        if answer and answer not in collected_answers:
            collected_answers.append(answer)
            checked_columns.add(column)

    if collected_answers:
        return " ".join(collected_answers), 1.0

    # === RECHERCHE PAR SIMILARITÉ (TF-IDF) ====
    best_question, score = search_question(user_question)

    if score < 0.35:
        return ("Je n'ai pas trouvé d'information précise concernant votre demande sur le Samsung Galaxy S26 Ultra.", score)

    return extract_best_sentence(user_question, paragraph), score