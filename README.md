# 🤖 Samsung Galaxy S26 Ultra FAQ Chatbot

## 📌 Description

Ce projet consiste à développer un chatbot intelligent capable de répondre aux questions fréquentes concernant le **Samsung Galaxy S26 Ultra**.

Le chatbot utilise un système de recherche basé sur **TF-IDF et la similarité cosinus** afin d'identifier la question la plus proche parmi une base de questions fréquentes (FAQ).

Il peut répondre à des questions concernant notamment :

* 💰 Prix
* 🛡️ Garantie
* 📱 Écran
* ⚡ Processeur et performances
* 📸 Caméra et mégapixels
* 🌙 Photos de nuit
* 🔋 Batterie et autonomie
* 🤖 Galaxy AI
* 🔭 Zoom
* 🧱 Matériaux et résistance
* 🌐 Connectivité
* 📶 5G, Wi-Fi et Bluetooth
* 🔗 Source et site officiel

Le chatbot refuse également les questions qui ne concernent pas le Samsung Galaxy S26 Ultra.

---

## 🎯 Objectif

L'objectif du projet est de créer un chatbot capable de fournir rapidement des informations pertinentes sur le Samsung Galaxy S26 Ultra à partir d'une base de données préparée à partir des informations du produit.

Le système utilise :

* une base de questions FAQ ;
* les données nettoyées du produit ;
* un modèle TF-IDF ;
* la similarité cosinus ;
* des règles de filtrage et de mots-clés ;
* une extraction de la phrase la plus pertinente pour certaines informations.

---

## 🛠️ Technologies utilisées

* **Python**
* **Streamlit** — interface web
* **Pandas** — manipulation des données
* **NLTK** — prétraitement du texte
* **Scikit-learn** — TF-IDF et similarité cosinus
* **Joblib** — sauvegarde et chargement des modèles
* **Regex (re)** — traitement du texte

---

## 📂 Structure du projet

```text
Task2_Ai/
│
├── data/
│   ├── Samsung_S26_Ultra_questions.csv
│   └── Samsung_S26_Ultra_final_clean.csv
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   └── faq_vectors.pkl
│
├── chatbot.py
├── app.py
├── requirements.txt
└── README.md
```

### 📄 Description des fichiers

**`chatbot.py`**

Contient la logique principale du chatbot :

* chargement des données ;
* chargement des modèles `.pkl` ;
* prétraitement des questions ;
* recherche de la question FAQ la plus proche ;
* calcul de la similarité ;
* filtrage des questions hors sujet ;
* génération de la réponse.

**`app.py`**

Contient l'interface utilisateur développée avec Streamlit.

**`Samsung_S26_Ultra_questions.csv`**

Contient les questions FAQ utilisées par le chatbot.

**`Samsung_S26_Ultra_final_clean.csv`**

Contient les informations nettoyées du Samsung Galaxy S26 Ultra utilisées pour générer les réponses.

**`tfidf_vectorizer.pkl`**

Modèle TF-IDF sauvegardé depuis Google Colab.

**`faq_vectors.pkl`**

Vecteurs TF-IDF des questions FAQ sauvegardés depuis Google Colab.

---

## ⚙️ Installation

### 1. Cloner le projet

```bash
git clone <URL_DU_REPOSITORY>
cd Task2_Ai
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
```

### 3. Activer l'environnement virtuel sous Windows

```bash
.venv\Scripts\activate
```

### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## ▶️ Exécution du chatbot

Pour tester le chatbot directement dans le terminal :

```bash
python chatbot.py
```

Le programme affiche :

```text
🤖 Chatbot du Samsung S26 Ultra prêt — Tapez 'exit' pour quitter
```

Vous pouvez ensuite poser une question, par exemple :

```text
Quelle est la résolution de la caméra ?
```

Le chatbot retourne la réponse ainsi que le score de similarité.

Pour quitter :

```text
exit
```

ou :

```text
bye
```

---

## 🌐 Lancer l'application Streamlit

Pour lancer l'interface graphique :

```bash
streamlit run app.py
```

Streamlit ouvre ensuite l'application dans le navigateur.

---

## 🧠 Fonctionnement

Le fonctionnement général du chatbot est le suivant :

```text
Question utilisateur
        ↓
Prétraitement du texte
        ↓
Vérification de la question
        ↓
Filtrage des questions hors sujet
        ↓
Recherche dans les FAQ
        ↓
TF-IDF
        ↓
Similarité cosinus
        ↓
Identification de la question la plus proche
        ↓
Recherche de l'information correspondante
        ↓
Réponse du chatbot
```

Pour certaines catégories, des règles spécifiques permettent de sélectionner directement l'information pertinente, par exemple pour :

* caméra ;
* mégapixels ;
* autonomie ;
* batterie ;
* Galaxy AI ;
* zoom ;
* matériaux ;
* IP68 ;
* photos de nuit.

---

## 🧪 Exemples de questions

### Caméra

```text
Quelle est la résolution de la caméra ?
Combien de mégapixels possède la caméra arrière ?
Quel est le capteur principal ?
```

### Batterie

```text
Quelle est l'autonomie ?
Combien d'heures tient la batterie ?
Quelle est la capacité de la batterie ?
```

### Processeur

```text
Quel est le processeur ?
Quel Snapdragon utilise-t-il ?
Le téléphone est-il performant ?
```

### Galaxy AI

```text
Qu'est-ce que Galaxy AI ?
Que fait Galaxy AI ?
```

### Résistance

```text
Le téléphone est-il en aluminium ?
Le téléphone possède-t-il IP68 ?
Le téléphone est-il résistant à l'eau ?
```

### Questions hors sujet

```text
Quel temps fait-il aujourd'hui ?
Combien coûte l'iPhone 17 ?
Qui est Cristiano Ronaldo ?
```

Pour ces questions, le chatbot indique qu'il répond uniquement aux questions concernant le Samsung Galaxy S26 Ultra.

---

## 📊 Résultat

Le chatbot est capable d'identifier différentes formulations d'une même question et de retourner l'information correspondante à partir de la base de connaissances.

Le score de similarité permet également d'évaluer la proximité entre la question de l'utilisateur et les questions présentes dans la base FAQ.

---

## 👩‍💻 Projet

Projet réalisé dans le cadre de la tâche **Chatbot for FAQs**.

**Technologie principale : Python + Streamlit + NLP**
