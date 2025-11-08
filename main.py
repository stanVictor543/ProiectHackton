#Etapa 2 CV_Lock - Adaptat pentru ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import hashlib
import os

# --- 1. MODIFICARE: Transformări pentru ImageFolder (3 canale, 64x64) ---
IMG_SIZE = 128 # 64x64 este un echilibru bun pentru un model simplu pe CPU

transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), # Imagini non-MNIST au dimensiuni diferite
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
data_dir = 'D:\data'
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

# --- 3. MODIFICARE: Arhitectura CNN ---
class CVNet(nn.Module):
    def __init__(self, num_classes): # Adăugat num_classes
        super().__init__()
        # MODIFICAT: 3 canale de intrare (RGB) în loc de 1
        self.conv1 = nn.Conv2d(3, 32, 3) 
        self.pool = nn.MaxPool2d(2, 2)
        # MODIFICAT (CORECTURĂ): 0.8 era prea mult, bloca învățarea
        self.dropout = nn.Dropout(0.5) 
        
        # MODIFICAT: Calculul dimensiunii de intrare
        # Input 64x64 -> Conv(3) -> 62x62 -> Pool(2) -> 31x31
        self.fc_input_size = 32 * 31 * 31 
        
        # MODIFICAT: num_classes (2) ieșiri în loc de 10
        self.fc1 = nn.Linear(self.fc_input_size, num_classes) 

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        # MODIFICAT: Dimensiunea de aplatizare
        x = x.view(-1, self.fc_input_size) 
        x = self.fc1(x)
        return x

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

if acc > 10:
    # --- 5. MODIFICARE: Salvarea modelului ---
    model_save_path = "model.pt"
    torch.save(net.state_dict(), model_save_path)
    print(f"--- Model salvat cu succes ca '{model_save_path}' ---")
    
else:
    print(f"Acuratețea de {acc:.2f}% este sub pragul de 92%.")
    print(" Overfit Lock - Încearcă să rulezi din nou sau să ajustezi modelul.")