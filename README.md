# 🤖 Clasificator Binar Imagini: Roboti vs. Oameni

## 📖 Prezentare Generală

Acest proiect prezintă dezvoltarea unui clasificator binar de imagini capabil să facă distincția între imagini cu oameni și imagini cu roboți. Soluția utilizează o arhitectură de tip **Transfer Learning**, bazată pe modelul **ResNet18** pre-antrenat.

Proiectul a fost dezvoltat în patru etape principale:
1.  **Colectarea și Pregătirea Datelor:** Agregarea și procesarea seturilor de date.
2.  **Antrenarea Modelului:** Dezvoltarea și antrenarea modelului folosind PyTorch.
3.  **Aplicație FullStack:** Crearea unei interfețe web pentru interacțiunea cu modelul.
4.  **Evaluare:** Analiza rezultatelor și identificarea pașilor următori.

---

## 📊 Pregătirea Datelor

### Seturi de Date Utilizate

Am fost folosite două seturi de date publice pentru antrenarea modelului:

* **Roboți:** [Humanoid Robot Pose Estimation](https://github.com/AIS-Bonn/HumanoidRobotPoseEstimation?tab=readme-ov-file)
* **Oameni:**
    * [Leeds-Sport pose (LSP)](https://www.kaggle.com/datasets/dkrivosic/leeds-sports-pose-lsp)
    * [Human Images Dataset - Men and Women](https://www.kaggle.com/datasets/snmahsa/human-images-dataset-men-and-women)

Script-ul `PhotoScripts/organize.py` a fost utilizat pentru a structura imaginile în directoarele necesare.

### Augmentarea Datelor

Deoarece setul de date pentru roboți a fost insuficient din punct de vedere numeric, am aplicat tehnici de augmentare pentru a mări volumul de date de antrenare.

Tehnicile aplicate (folosind `PhotoScripts/editing.py`):
* Image Cropping (Decupare)
* Image Flipping (Oglindire)
* Grayscale (Conversie alb-negru)

---

## 🧠 Antrenarea Modelului

Modelul a fost dezvoltat folosind **PyTorch** împreună cu biblioteca **Torchvision**.

### Arhitectură și Hiperparametri

* **Model:** Se folosește **Transfer Learning** cu modelul **ResNet18**, pre-antrenat pe setul de date ImageNet.
* **Strategie:** Straturile convoluționale (de extragere a caracteristicilor) ale ResNet18 au fost "înghețate" (frozen). S-a antrenat *doar* ultimul strat de clasificare (fully-connected layer), care a fost adaptat pentru cele 2 clase ale noastre (oameni vs. roboți).
* **Funcție de Pierdere (Loss):** `CrossEntropyLoss` cu **ponderare automată** (calculată pe baza distribuției claselor) pentru a contracara dezechilibrul setului de date.
* **Optimizator:** `Adam` (aplicat doar pe parametrii stratului final).
* **Rata de învățare (Learning Rate):** 0.001
* **LR Scheduler:** `StepLR` (rata de învățare este redusă automat în timpul antrenamentului).
* **Număr Epoci:** 25

---

## 🖥️ Aplicația Web

Pentru a demonstra funcționalitatea modelului, a fost creată o aplicație web FullStack care permite utilizatorilor să încarce o imagine și să primească o clasificare.

### Tehnologii Utilizate

* **Frontend:** HTML, CSS și JavaScript (Vanilla)
* **Backend:** Python (Flask)
* **Bază de date:** SQLite
* **Particularități:** Folosirea live a camerei

---

## 📈 Rezultate și Îmbunătățiri Viitoare

Proiectul a atins cu succes stadiul de **Minimum Viable Product (MVP)**, oferind un clasificator funcțional.

> **Context:** Întregul prototip a fost finalizat într-un interval de timp limitat de **10 ore**. Acest constrângere a influențat alegerea tehnologiilor și volumul de date utilizat.

### Direcții Viitoare

Având la dispoziție mai mult timp, următoarele îmbunătățiri ar putea fi implementate:

* **Model și Date:**
    * Mărirea considerabilă a setului de date de antrenare.
    * Colectarea de imagini suplimentare din surse variate.
    * Explorarea altor framework-uri (de exemplu, Keras/TensorFlow).
    * **Fine-tuning:** "Dezghețarea" mai multor straturi din ResNet18 pentru a antrena o parte mai mare a rețelei (acum că validarea inițială a funcționat).
* **Stack Tehnologic:**
    * **Frontend:** Adoptarea unui framework JavaScript modern (React, Vue.js sau Angular) pentru o interfață mai interactivă.
    * **Backend:** Migrarea către o soluție mai robustă și scalabilă (de exemplu, .NET sau Django).
    * **Bază de date:** Înlocuirea SQLite (serverless) cu o soluție client-server (de exemplu, PostgreSQL sau MySQL).