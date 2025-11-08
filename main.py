#Etapa 2 CV_Lock - Adaptat pentru ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import hashlib
import os

# --- 1. MODIFICARE: Transformări pentru ImageFolder (3 canale) ---
# Poți schimba această valoare (ex: 64, 128, 224)
# Rețeaua de mai jos se va adapta automat.
IMG_SIZE = 128 # Am setat 128 ca exemplu, care probabil a cauzat eroarea

transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), # Redimensionează orice imagine
    transforms.ToTensor(),
    transforms.RandomRotation(10),
    # Normalizare pentru 3 canale (RGB)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- 2. MODIFICARE: Încărcare date din ImageFolder ---
# Folosirea r'...' previne erorile de cale pe Windows
data_dir = r'D:\data' 

try:
    trainset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    
    testset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    
    # Obținem numărul de clase (va fi 2: 'oameni', 'roboti')
    num_classes = len(trainset.classes)
    print(f"Clase găsite: {trainset.classes}")

except FileNotFoundError:
    print(f"EROARE: Nu am găsit folderele 'train' sau 'test' în interiorul '{data_dir}'")
    exit()

# --- 3. MODIFICARE: Arhitectura CNN (Varianta Robustă) ---
class CVNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # --- Partea de "extragere a caracteristicilor" (convoluție) ---
        # Definim straturile care procesează imaginea
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3), # 3 canale RGB, 32 filtre, kernel 3x3
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Înjumătățește dimensiunea
            nn.Dropout(0.5)     # Dropout-ul original
        )
        
        # --- Calcul automat al dimensiunii (Soluția la eroarea ta) ---
        # 1. Creăm un tensor "fals" cu dimensiunile inputului
        #    (1 imagine, 3 canale, IMG_SIZE, IMG_SIZE)
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        
        # 2. Trecem tensorul fals prin straturile de convoluție
        dummy_output = self.features(dummy_input)
        
        # 3. Aflăm dimensiunea aplatizată (flattened)
        #    ex: (1, 32, 63, 63) -> (1, 127008) -> 127008
        #    .view(1, -1) îl aplatizează, .shape[1] ia mărimea
        flattened_size = dummy_output.view(1, -1).shape[1]
        
        print(f"Dimensiunea de intrare pentru FC calculată automat: {flattened_size}")
        
        # --- Partea de "clasificare" (complet conectată) ---
        self.classifier = nn.Sequential(
            # 4. Acum folosim dimensiunea corectă, calculată automat
            nn.Linear(flattened_size, num_classes)
        )

    def forward(self, x):
        # 1. Trece prin convoluții
        x = self.features(x)
        
        # 2. Aplatizează (flatten) pentru stratul Linear
        #    x.size(0) este batch_size. Acesta este modul robust de a aplatiza.
        x = x.view(x.size(0), -1) 
        
        # 3. Trece prin clasificator
        x = self.classifier(x)
        return x

# --- Inițializare model și antrenament ---
net = CVNet(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
# MODIFICAT (CORECTURĂ): 0.01 este prea mare pentru Adam, 0.001 e mai stabil
optimizer = optim.Adam(net.parameters(), lr=0.001) 

num_epochs = 5 # 3 epoci e posibil să fie prea puțin

print("--- Începe antrenamentul ---")
# MODIFICAT (CORECTURĂ): Trecem modelul în modul de antrenare
net.train() 
i = 0
for epoch in range(num_epochs):
    i = i + 1
    print(i)
    running_loss = 0.0
    for images, labels in trainloader:
        out = net(images)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoca [{epoch+1}/{num_epochs}], Pierdere: {running_loss/len(trainloader):.4f}")

print("--- Antrenament finalizat ---")

# --- Evaluare ---
correct = 0
total = 0
# MODIFICAT (CORECTURĂ): Trecem modelul în modul evaluare (dezactivează dropout)
net.eval() 
with torch.no_grad():
    for images, labels in testloader:
        out = net(images)
        pred = out.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

acc = 100 * correct / total
print(f"Custom Acc: {acc:.2f}%")

# --- 5. MODIFICARE: Salvarea modelului ---
if acc > 10:
    model_save_path = "model.pt"
    torch.save(net.state_dict(), model_save_path)
    print(f"--- Model salvat cu succes ca '{model_save_path}' ---")
    
else:
    # Am corectat comentariul să reflecte codul (10%)
    print(f"Acuratețea de {acc:.2f}% este sub pragul de 10%.")
    print(" Overfit Lock - Încearcă să rulezi din nou sau să ajustezi modelul.")