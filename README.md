# Projet_DL : Détection Automatique de Sentiment dans des Appels Vocaux à l’aide de Wav2Vec 2.0 et BERT

# 🎙️ Analyse de Sentiment Vocal avec Transcription Automatique

Ce projet utilise deux modèles puissants :
- **Wav2Vec2** pour la transcription vocale en texte (speech-to-text)
- **RoBERTa** pour l’analyse de sentiment à partir du texte transcrit

> **Objectif :** Permettre à un utilisateur d'uploader un fichier audio `.wav` pour obtenir automatiquement :
> - sa transcription
> - l'analyse de sentiment
> - l'export des résultats en JSON et CSV

---

## Fonctionnalités

- **Transcription automatique** d'audio `.wav` (modèle : `jonatasgrosman/wav2vec2-large-xlsr-53-french`lien : )
- **Analyse de sentiment** (modèle : `cardiffnlp/twitter-xlm-roberta-base-sentiment` lien : )
- **Export des résultats** en :
  - `results.json` (historique des prédictions)
  - `results.csv` (version tabulaire exploitable)
- **Interface web** simple avec **Gradio** pour tester notre solution sur **colab**
- Nous créé aussi une **API** avec le FrameWork **FastAPI** dans **VSCode**

---

## Les étapes de réalisation du projet :

### 1. **Création (`python -m venv venv`) et activation (`wandb_env\Scripts\activate`) d'un environnement virtuel** 

### 2. **Création du repo GitHub** 
Nous créé ce repo `https://github.com/malcogb/Projet_DL.git`

### 3. Clonage du repo
Nous avons cloné le repo pour l'avoir en local avce la commande : `git clone httpps://github.com/malcogb/Projet_DL.git`

### 4. **Installation des dépendances** 
Nous les avons directement installées dans une cellule sur Google Colab et avec `ip install -r requirements.txt` dans le terminal de VSCode

### 5. Développement de la solution "Détection Automatique de Sentiment dans les audios"
Après avoir fini le déloppement et test avec **Gradio** sur Colab, nous avons créé son API afin de rendre notre solution utilisable par d'autres développeurs.

### 6. Utilisation
- Sur Colab : Nous lançons l’interface Gradio avec `demo.launch(share=True)` en fin de NoteBook Colab. accessible sur URL: `https://f9556ddf0a8961a72b.gradio.live`
- Dans VSCode, notre API (FastAPI) a été développée en Python et nommée `app.py`. Nous lançons l'API avec la commande `uvicorn app:app --reload`. Une fois que le serveur **FastAPI** de notre **API** a démarré, nous nous connectons à l'interface web (`http://127.0.0.1:8000`) ou directement sur `http://127.0.0.1:8000/docs`.

### 7. Endpoints de l'API
Nous avons utilisé 3 endpoints à savoir :
- **@app.post("/audio_transcription and sentiment_analyze/")** pour permettre à l'utilisateur d'envoyer à l'API le fichier audio
- **@app.get("/download_json/")** pour permettre à l'utilisateur de télécharger le résultat du traitement et de la prédiction en fichier JSON
- **@app.get("/download_csv/")** pour permettre à l'utilisateur de télécharger le résultat du traitement et de la prédiction en fichier CSV

### 8. Exemple de sortie après traitement du fichier audio
voir dossier capture

### 9. Export
**Après chaque analyse :**
- les résultats sont ajoutés dans results.json
- puis convertis automatiquement en results.csv

**10. Contenu des fichiers JSON et CSV :**
- Transcription
- Sentiment
- Confiance
- Nombre de chunks
- Score moyen
- Date (horodatage)
- Nom du fichier audio

### 11. À faire / améliorations possibles
- Authentification des utilisateurs
- Support multilingue étendu
- Interface front-end pour visualiser les résultats
- Déploiement sur HuggingFace Spaces ou Docker

### 12. Auteur
Projet développé par **Malco GBAKPA** étudiant en **Master 2 IA** Spécialité **Data Science**
