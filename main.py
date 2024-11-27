from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# Télécharger les stopwords français (uniquement si ce n'est pas déjà fait)
nltk.download('stopwords')
stop_words = set(stopwords.words('french'))
tokenizer = RegexpTokenizer(r"[a-zA-ZÀ-ÿ']+")

# Charger le modèle et le vectoriseur sauvegardés
try:
    model = joblib.load('model.pkl')  # Logistic Regression Model
    vectorizer = joblib.load('vectorizer.pkl')  # TF-IDF Vectorizer
    print("Modèle et vectoriseur chargés avec succès.")
except FileNotFoundError as e:
    print(f"Erreur : {e}")
    exit()

# Initialiser l'application FastAPI
app = FastAPI()

# Modèle pour valider les requêtes
class TextRequest(BaseModel):
    texte: str

# **1. Fonction de nettoyage**
def clean_text(text):
    text = str(text).lower()  # Minuscule
    text = re.sub(r"[\u2018\u2019\u201A\u201B\u2032\u2035]", "'", text)  # Unifier les apostrophes
    text = re.sub(r"<[^>]+>", "", text)  # Supprimer les balises HTML
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s']", " ", text)  # Garder lettres, accents et apostrophes
    text = re.sub(r"\s+", " ", text).strip()  # Supprimer espaces multiples
    return text

# **2. Supprimer les stopwords**
def remove_stopwords(text):
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# **3. Endpoint principal pour prédire la classe**
@app.post("/predict/")
def predict_class(request: TextRequest):
    try:
        # Nettoyer et vectoriser le texte
        cleaned_text = clean_text(request.texte)
        cleaned_text = remove_stopwords(cleaned_text)
        vectorized_text = vectorizer.transform([cleaned_text])
        
        # Prédire la classe
        prediction = model.predict(vectorized_text)[0]
        return {"classe_predite": int(prediction)}
    except Exception as e:
        return {"erreur": str(e)}

# **4. Endpoint pour afficher un message de base**
@app.get("/")
def root():
    return {"message": "API de classification textuelle est en ligne et fonctionnelle !"}
