# LE SCRIPT DE CREATION DE L'API AVEC FastAPI

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from transformers import pipeline, XLMRobertaTokenizer, AutoModelForSequenceClassification
import torch
import torchaudio
import tempfile
import os
import subprocess
import torch.nn.functional as F
from collections import Counter
import json
import csv
from fastapi.responses import FileResponse
from datetime import datetime


app = FastAPI()

# ----------- Modèles -----------

# ASR - Reconnaissance vocale
asr_pipeline = pipeline("automatic-speech-recognition", model="jonatasgrosman/wav2vec2-large-xlsr-53-french")

# Analyse de sentiment
sentiment_model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = XLMRobertaTokenizer.from_pretrained(sentiment_model_name)
model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
model.eval()

# ----------- Fonctions -----------

def convert_to_wav(input_path, output_path):
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
        raise HTTPException(status_code=500, detail="Erreur de conversion audio avec ffmpeg.")

def split_text(text, max_tokens=512):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_ids = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        score, label_id = torch.max(probs, dim=1)
        label = model.config.id2label[label_id.item()]
        return {"label": label, "score": round(score.item(), 3)}

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


def save_to_json(result, json_path="results.json"):
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as jf:
            data = json.load(jf)
    else:
        data = []

    data.append(result)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(data, jf, ensure_ascii=False, indent=4)



def export_to_csv(json_path="results.json", csv_path="results.csv"):
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Aucun fichier JSON trouvé pour l’export.")

    try:
        with open(json_path, "r", encoding="utf-8") as jf:
            data = json.load(jf)

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



# ----------- API Endpoint -----------

@app.post("/audio_transcription_and_sentiment_analyze/")
async def analyze_audio(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Aucun fichier audio fourni.")
    
    try:
        # Enregistrer fichier temporairement
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_input:
            tmp_input.write(await file.read())
            tmp_input_path = tmp_input.name

        # Préparer le fichier de sortie en .wav
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            tmp_wav_path = tmp_wav.name

        # Conversion si besoin
        convert_to_wav(tmp_input_path, tmp_wav_path)

        print(f" Fichier temporaire créé : {tmp_wav_path}")

        # 🎙️ Transcription vocale
        transcription = asr_pipeline(tmp_wav_path)["text"]
        print(" Transcription :", transcription)

        # Vérification de la transcription
        if len(transcription.strip()) < 10:
            raise HTTPException(status_code=400, detail=" Transcription trop courte ou inaudible. Essayez un autre fichier.")


        # Analyse de sentiment
        sentiment_result = analyze_long_text(transcription)

        result_entry = {
        "transcription": transcription,
        "sentiment": sentiment_result['final_sentiment'],
        "confiance": sentiment_result['confidence'],
        "nb_chunks": len(sentiment_result["chunks"]),
        "score_moyen": round(sum([c["score"] for c in sentiment_result["chunks"]]) / len(sentiment_result["chunks"]), 3),
        "horodatage": datetime.now().isoformat(),
        "nom_du_fichier": os.path.basename(file.filename) # file.filename donne le nom du fichier audio
    }
        save_to_json(result_entry)  # ← Enregistre dans results.json
        export_to_csv()

        return result_entry
        
    except Exception as e:
        print(" Erreur rencontrée :", str(e))
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")
    


@app.get("/download_json/")
def download_results():
    json_path = "results.json"  # Chemin vers le fichier sauvegardé
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Aucun fichier JSON trouvé.")
    return FileResponse(path=json_path, media_type="application/json", filename="resultats.json")



@app.get("/download_csv/")
def export_csv():
    json_path = "results.json"
    csv_path = "results.csv"

    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Fichier JSON introuvable.")

    with open(json_path, "r", encoding="utf-8") as f_json:
        data = json.load(f_json)

    # On extrait les colonnes principales uniquement
    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["transcription", "sentiment_majoritaire", "confiance"])
        writer.writeheader()
        for entry in data:
            writer.writerow({
                "transcription": entry.get("transcription", ""),
                "sentiment_majoritaire": entry.get("sentiment_majoritaire", ""),
                "confiance": entry.get("confiance", "")
            })

    return FileResponse(path=csv_path, media_type="text/csv", filename="resultats.csv")

