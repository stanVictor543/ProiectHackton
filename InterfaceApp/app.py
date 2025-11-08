import os
import sqlite3
import datetime
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import io

# --- Importuri specifice pentru Model (PyTorch) ---
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# from torchvision.models import ... # Exemplu: Aici ai importa arhitectura modelului dacă e nevoie

# =====================================================================
# 1. CONFIGURARE FLASK & BAZĂ DE DATE
# =====================================================================
app = Flask(__name__)
DB_NAME = "predictions.db"
MODEL_PATH = "model.pt" # Numele modelului de la Membrul 2
IMG_SIZE = 128          # Dimensiunea imaginii cerută

# Funcție pentru a crea tabela în baza de date
def init_db():
    print("Inițializare bază de date...")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        predicted_class TEXT NOT NULL,
        confidence REAL NOT NULL,
        timestamp TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()
    print("Baza de date inițializată.")

# =====================================================================
# 2. ÎNCĂRCARE MODEL & PREPROCESARE
# (Aici vei integra codul de la colegii tăi)
# =====================================================================

# --- DEFINIREA TRANSFORMĂRII PENTRU PREPROCESARE ---
# Folosim logica de 'test' (validare/inferență) pe care ai furnizat-o
data_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Încărcăm modelul o singură dată la pornire
# --- EXEMPLU PYTORCH (de adaptat) ---
# Pas 1: Definește arhitectura modelului (dacă nu e salvată complet)
# class SimpleCNN(nn.Module):
#     ...
# model = SimpleCNN()
#
# Pas 2: Încarcă 'state_dict' (greutățile salvate)
# model.load_state_dict(torch.load(MODEL_PATH))
# model.eval() # !! IMPORTANT: Setează modelul în modul de evaluare
# print(f"Model PyTorch încărcat din {MODEL_PATH}")
# --- Sfârșit Exemplu PyTorch ---

print(f"Placeholder: Modelul ar fi încărcat din {MODEL_PATH}")


# Definirea claselor (IMPORTANT: trebuie să fie în ordinea corectă)
CLASS_NAMES = ["human", "robot"] # Sau invers, verifică cu Membrul 2

def preprocess_image(image_bytes):
    """
    Funcție de preprocesare care folosește transformările PyTorch.
    """
    # Deschide imaginea din bytes
    img = Image.open(io.BytesIO(image_bytes))
    
    # --- IMPORTANT: Asigură-te că imaginea e RGB ---
    # Modelul așteaptă 3 canale (conform normalizării)
    img = img.convert("RGB")
    
    # Aplică transformările definite global
    # Acesta va returna un Tensor
    img_tensor = data_transform(img)
    
    # Adaugă dimensiunea 'batch' (de la [C, H, W] la [1, C, H, W])
    # Modelul se așteaptă la un batch de imagini, chiar dacă e doar una
    img_tensor = img_tensor.unsqueeze(0) 
    
    print(f"Imagine preprocesată cu shape: {img_tensor.shape}")
    return img_tensor

# =====================================================================
# 3. LOGICA DE SALVARE ÎN BAZA DE DATE
# =====================================================================
def save_prediction_to_db(filename, pred_class, confidence):
    """
    Funcția care salvează rezultatul în SQLite.
    Este apelată automat de endpoint-ul /predict.
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        timestamp = datetime.datetime.now().isoformat()
        
        cursor.execute("""
        INSERT INTO predictions (filename, predicted_class, confidence, timestamp)
        VALUES (?, ?, ?, ?)
        """, (filename, pred_class, confidence, timestamp))
        
        conn.commit()
        conn.close()
        print(f"Rezultat salvat în DB: {filename}, {pred_class}")
        
    except Exception as e:
        print(f"*** Eroare la salvarea în DB: {e}")

# =====================================================================
# 4. ENDPOINT-URILE APLICAȚIEI
# =====================================================================

@app.route('/')
def index():
    """Servește pagina HTML principală."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Primește imaginea, o prelucrează, face predicția 
    și SALVEAZĂ AUTOMAT rezultatul.
    """
    if 'file' not in request.files:
        return jsonify({"error": "Niciun fișier trimis"}), 400
    
    file = request.files['file']
    filename = file.filename
    
    if filename == '':
        return jsonify({"error": "Niciun fișier selectat"}), 400

    try:
        # Citim imaginea
        img_bytes = file.read()
        
        # 1. Preprocesare (folosind noua funcție cu PyTorch)
        processed_image_tensor = preprocess_image(img_bytes)

        # 2. Predicție (folosind modelul de la Membrul 2)
        
        # --- EXEMPLU PYTORCH (de adaptat și decomentat) ---
        # with torch.no_grad(): # Dezactivează calculul gradientului pentru viteză
        #     output_logits = model(processed_image_tensor)
        #     
        #     # Aplică Softmax pentru a obține probabilități
        #     probabilities = torch.nn.functional.softmax(output_logits, dim=1)[0]
        #     
        #     # Găsește clasa cu probabilitatea cea mai mare
        #     confidence_tensor = torch.max(probabilities)
        #     predicted_index_tensor = torch.argmax(probabilities)
        #     
        #     # Convertește din tensori în numere simple
        #     confidence = float(confidence_tensor.item())
        #     predicted_index = int(predicted_index_tensor.item())
        #     
        #     predicted_class = CLASS_NAMES[predicted_index]
        # --- Sfârșit Exemplu PyTorch ---
        
        # --- PLACEHOLDER (de șters când ai modelul) ---
        print("!!! ATENȚIE: Se folosește un rezultat placeholder !!!")
        import random
        confidence = random.uniform(0.9, 0.99)
        predicted_class = random.choice(CLASS_NAMES)
        # --- Sfârșit Placeholder ---


        # 3. SALVARE AUTOMATĂ ÎN BAZA DE DATE
        save_prediction_to_db(filename, predicted_class, confidence)

        # 4. Returnare rezultat către Frontend
        return jsonify({
            "filename": filename,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "timestamp": datetime.datetime.now().isoformat()
        })

    except Exception as e:
        print(f"*** Eroare la predicție: {e}")
        return jsonify({"error": f"Eroare server: {e}"}), 500

# =====================================================================
# 5. PORNIREA APLICAȚIEI
# =====================================================================
if __name__ == '__main__':
    init_db() # Creează baza de date la prima rulare
    app.run(debug=True) # debug=True te ajută să vezi erorile