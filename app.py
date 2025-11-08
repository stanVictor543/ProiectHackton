import os
import sqlite3
import datetime
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import io

# --- Importuri specifice pentru Model (Exemplu cu Keras/TensorFlow) ---
# Vei decomenta și adapta în funcție de ce primești de la Membrul 2
# from tensorflow.keras.models import load_model 
# from tensorflow.keras.preprocessing.image import img_to_array

# =====================================================================
# 1. CONFIGURARE FLASK & BAZĂ DE DATE
# =====================================================================
app = Flask(__name__)
DB_NAME = "predictions.db"
MODEL_PATH = "model.pt" # Numele modelului de la Membrul 2

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

# Încărcăm modelul o singură dată la pornire
# model = load_model(MODEL_PATH) # Decomentează când ai modelul
print(f"Placeholder: Modelul ar fi încărcat din {MODEL_PATH}")

# Definirea claselor (IMPORTANT: trebuie să fie în ordinea corectă)
CLASS_NAMES = ["human", "robot"] # Sau invers, verifică cu Membrul 2

def preprocess_image(image_bytes):
    """
    Funcție de preprocesare. 
    Vei primi logica exactă de la Membrul 1.
    """
    # Exemplu de preprocesare (TREBUIE ADAPTAT!)
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224)) # Dimensiunea așteptată de model
    img_array = np.array(img) / 255.0 # Normalizare
    img_array = np.expand_dims(img_array, axis=0) # Adaugă dimensiunea 'batch'
    
    print(f"Imagine preprocesată cu shape: {img_array.shape}")
    return img_array

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
        
        # 1. Preprocesare (folosind funcția de la Membrul 1)
        processed_image = preprocess_image(img_bytes)

        # 2. Predicție (folosind modelul de la Membrul 2)
        # --- EXEMPLU (de adaptat) ---
        # prediction_scores = model.predict(processed_image)[0]
        # confidence = float(np.max(prediction_scores))
        # predicted_index = np.argmax(prediction_scores)
        # predicted_class = CLASS_NAMES[predicted_index]
        # --- Sfârșit Exemplu ---
        
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