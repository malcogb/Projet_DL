# -*- coding: utf-8 -*-
"""Examen_Deep Learning 2_Malco GBAKPA.ipynb

# **Les étapes concrètes pour mettre en œuvre le projet :** Détection automatique de Sentiment dans des Appels Vocaux à l’aide de Wav2Vec 2.0 et BERT

## 1. INSTALLATION DES DÉPENDANCES
"""

# !pip install torch torchaudio transformers gradio datasets --quiet
# !pip install kenlm pyctcdecode --quiet
# !apt install ffmpeg -y
#!pip install gradio

#!pip uninstall -y numpy
#!pip install numpy --upgrade --force-reinstall

"""## 2. IMPORTS"""

import subprocess
import json, os
import torch
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr
from collections import Counter
import csv
from datetime import datetime

"""## 3. PIPELINE DE TRANSCRIPTION : Transcription vocale avec Wav2Vec2 (nous avons utilisé le model **"jonatasgrosman/wav2vec2-large-xlsr-53-french"** spécialisé pour les audios en français)"""

asr = pipeline("automatic-speech-recognition", model="jonatasgrosman/wav2vec2-large-xlsr-53-french")

# test  de la transcription sur un fichier audio
# transcription = asr("//content/audio_5.wav")["text"]
# print(transcription)

"""## 4. PIPELINE DE SENTIMENT : Analyse de sentiment avec BERT (nous avons utilisé RoBERTa qui est une variante de BERT avec une meilleure performance)"""

model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

"""## 5. --- UTILITAIRES : FONCTIONS ---"""

# Conversion de n’importe quel format audio (ex. .mp3, .m4a, etc.) en .wav mono 16kHz, adapté à Wav2Vec2.
def convert_to_wav(input_path, output_path="converted.wav"):
    """Convertit un fichier audio en wav 16kHz mono"""
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000",  # Resample à 16 kHz
        "-ac", "1",      # Mono
        output_path
    ]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        raise RuntimeError("Erreur de conversion audio avec ffmpeg.")
    return output_path

# Division de texte long en morceaux de taille inférieure à 512 tokens (limite max de BERT/roBERTa).
# Cela permet de traiter des textes trop longs pour un seul passage dans le modèl
def split_text(text, max_tokens=512):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_ids = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

# Application du modèle de classification des sentiments (roBERTa) à un texte court.
# Retourne :le label prédit (ex. "Positive", "Neutral", "Negative") et le score (probabilité associée)
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        score, label_id = torch.max(probs, dim=1)
        label = model.config.id2label[label_id.item()]
        return {"label": label, "score": round(score.item(), 3)}


# Application de predict_sentiment() à chaque chunk du texte découpé.
# Puis :
# compte le nombre de chaque type de sentiment (majoritaire)
# calcule la confiance globale (fréquence du sentiment dominant)
# retourne aussi le détail chunk par chunk.
def analyze_long_text(text):
    chunks = split_text(text)
    results = [predict_sentiment(chunk) for chunk in chunks]
    labels = [r['label'] for r in results]
    counter = Counter(labels)
    final, count = counter.most_common(1)[0]
    return {
        "final_sentiment": final,
        "confidence": round(count / len(results), 2),
        "chunks": results
    }

# Enregistrement des résultats dans un fichier JSON
# Permet de garder une trace historique des analyses faites
def save_to_json(result, file_path="results_gradio.json"):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    data.append(result)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# Exportation des résultats dans un fichier CSV à partir du fichier JSON
def json_to_csv(json_path="results_gradio.json", csv_path="results_gradio.csv"):
    try:
        with open(json_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        # Vérifier que la liste n'est pas vide
        if not data:
            print("Le fichier JSON est vide.")
            return

        # Extraire les clés du premier élément pour l'en-tête
        keys = data[0].keys()

        with open(csv_path, "w", newline='', encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)

        print(f"Export terminé : {csv_path}")

    except Exception as e:
        print(f"Erreur lors de l'export : {e}")

"""## 6. TRAITEMENT PRINCIPAL  
Ici, **la fonction analyse_audio** :
- Convertit l’audio en .wav si besoin.
- Transcrit le fichier avec Wav2Vec2.
- Analyse le sentiment avec RoBERTa.
- Construit un résumé texte de l’analyse.
- Sauvegarde les données dans un .json et un .csv.
"""

def analyse_audio(audio_file):
    if not audio_file:
        return "Aucun fichier audio fourni."

    # Conversion si nécessaire
    wav_file = convert_to_wav(audio_file)

    # Transcription
    transcription = asr(wav_file)["text"]
    if len(transcription.strip()) < 10:
        return "Transcription trop courte ou inaudible."

    # Analyse de sentiment
    result = analyze_long_text(transcription)

    # Sauvegarde
   
    result_entry = {
        "transcription": transcription,
        "sentiment": result['final_sentiment'],
        "confiance": result['confidence'],
        "nb_chunks": len(result["chunks"]),
        "score_moyen": round(sum([c["score"] for c in result["chunks"]]) / len(result["chunks"]), 3),
        "horodatage": datetime.now().isoformat(),
        "nom_du_fichier": os.path.basename(audio_file)
    }



    save_to_json(result_entry)
    json_to_csv()  # utilise les valeurs par défaut



    # Format de sortie
    output = f"""📜 **Transcription :**\n{transcription}\n\n
🧠 **Analyse de sentiment :**
- Sentiment majoritaire : {result['final_sentiment']}
- Confiance : {result['confidence']:.2f}
- Détail :
"""
    for i, r in enumerate(result["chunks"]):
        output += f"  - Chunk {i+1} : {r['label']} (score {r['score']})\n"

    if result["confidence"] < 0.6:
        output += "\n Résultat peu fiable. L'audio peut être mal transcrit."

    return output

"""## 7. INTERFACE GRADIO :
- Crée une interface utilisateur simple pour :
- Permet d'uploader un fichier audio
- Permet de lancer automatiquement toutes les étapes : conversion → transcription → analyse
- Permet d'afficher le résumé de l’analyse dans un bloc texte
"""

demo = gr.Interface(
    fn=analyse_audio,
    inputs=gr.Audio(type="filepath", label="🎤 Téléversez un fichier .wav"),
    outputs="text",
    title="🎙️ Analyse de Sentiment Vocal (Wav2Vec2 + BERT (roBERTa))",
    description="Upload un fichier audio → Transcription automatique → Analyse de sentiment",
)

"""## 8. LANCEMENT"""

demo.launch(share=True)