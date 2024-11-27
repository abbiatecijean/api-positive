import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# Télécharger les stopwords français (uniquement si ce n'est pas déjà fait)
nltk.download('stopwords')

# Stopwords et tokenizer
stop_words = set(stopwords.words('french'))
tokenizer = RegexpTokenizer(r"[a-zA-ZÀ-ÿ']+")

# **1. Fonction de nettoyage**
def clean_text(text):
    text = str(text).lower()  # Tout en minuscules
    text = re.sub(r"[\u2018\u2019\u201A\u201B\u2032\u2035]", "'", text)  # Unifier les apostrophes
    text = re.sub(r"<[^>]+>", "", text)  # Supprimer les balises HTML
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s']", " ", text)  # Conserver seulement les lettres, accents, apostrophes
    text = re.sub(r"\s+", " ", text).strip()  # Supprimer les espaces multiples
    return text

# **2. Fonction pour supprimer les stopwords**
def remove_stopwords(text):
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)



# **3. Charger les données**
file_path = 'solok.csv'

try:
    data = pd.read_csv(file_path)
    print("Fichier chargé avec succès.")
except FileNotFoundError:
    print(f"Erreur : Le fichier '{file_path}' est introuvable. Vérifiez son emplacement.")
    exit()

# Vérifier les colonnes nécessaires
if 'texte' not in data.columns or 'label' not in data.columns:
    print("Erreur : Colonnes nécessaires 'texte' et/ou 'label' absentes.")
    exit()

# Nettoyer les textes
print("\nNettoyage des textes en cours...")
data['texte_cleaned'] = data['texte'].apply(clean_text)
data['texte_cleaned'] = data['texte_cleaned'].apply(remove_stopwords)

# **4. Vectorisation avec TF-IDF**
print("\nVectorisation des textes avec TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=1000,   # Limite à 2000 mots les plus fréquents
    min_df=3,            # Exclut les mots apparaissant dans moins de 2 documents
    max_df=0.8,          # Exclut les mots apparaissant dans plus de 80% des documents
    ngram_range=(1, 2),  # Inclut des unigrams et bigrammes

    
)
X = vectorizer.fit_transform(data['texte_cleaned'])
y = data['label']

# **5. Division des données**
print("\nDivision des données en ensembles d'entraînement et de test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **6. Entraîner le modèle**
print("\nEntraînement du modèle...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Modèle entraîné avec succès.")

# **7. Évaluer le modèle**
print("\nÉvaluation du modèle sur l'ensemble de test...")
y_pred = model.predict(X_test)
print("\nPrécision :", accuracy_score(y_test, y_pred))
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))
print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

# **8. Sauvegarder le modèle et le vectoriseur**
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("\nModèle et vectoriseur sauvegardés.")

# **9. Fonction de prédiction pour un nouveau texte**
def predict_new_text(text):
    cleaned_text = clean_text(text)
    cleaned_text = remove_stopwords(cleaned_text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]

# Exemple d'utilisation
nouveau_texte = "tVainqueur de la troisième partie, Gukesh revient à hauteur de Ding au Championnat du monde d'échecs"
classe_predite = predict_new_text(nouveau_texte)
print(f"\nTexte : {nouveau_texte}")
print(f"Classe prédite : {classe_predite}")

feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]


# Identifier les faux positifs avec leurs indices
faux_positifs = (y_test == 0) & (y_pred == 1)

# Récupérer les indices des faux positifs dans le DataFrame d'origine
faux_positifs_indices = y_test[faux_positifs].index

# Extraire les lignes correspondantes dans le DataFrame d'origine
faux_positifs_df = data.loc[faux_positifs_indices]

# Limiter à 10 exemples de faux positifs
faux_positifs_sample = faux_positifs_df.head(4)  # Affiche les 10 premiers faux positifs

# Afficher les faux positifs (texte brut et nettoyé)
print("\nExemples de faux positifs :")
for index, row in faux_positifs_sample.iterrows():
    print(f"Texte brut : {row['texte']}")
    print(f"Texte nettoyé : {row['texte_cleaned']}")
    print("-" * 50)


# Récupérer les noms des mots et les coefficients
feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]  # Coefficients pour Logistic Regression

# Trier les mots par importance
top_positive_indices = np.argsort(coefs)[-10:]  # Top 10 mots pour la classe positive
top_negative_indices = np.argsort(coefs)[:10]  # Top 10 mots pour la classe négative

# Mots influents
top_positive_words = [(feature_names[i], coefs[i]) for i in top_positive_indices]
top_negative_words = [(feature_names[i], coefs[i]) for i in top_negative_indices]

# Afficher les mots influents
print("\nMots influents pour la classe positive :")
for word, coef in reversed(top_positive_words):
    print(f"{word}: {coef:.4f}")

print("\nMots influents pour la classe négative :")
for word, coef in top_negative_words:
    print(f"{word}: {coef:.4f}")
