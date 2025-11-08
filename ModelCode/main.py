#Etapa 2 CV_Lock - Adaptat pentru ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import hashlib
import os

# --- 0. MODIFICARE: Verificare GPU (Best Practice) ---
# Selectează automat GPU (cuda) dacă e disponibil, altfel CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Se va folosi dispozitivul: {device} ---")


# --- 1. Transformări pentru ImageFolder (Neschimbat) ---
IMG_SIZE = 128 

transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), 
    transforms.RandomRotation(10),
    # --- MODIFICARE: Adăugăm mai multă augmentare ---
    transforms.RandomHorizontalFlip(), # O augmentare foarte comună
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Ajută la variații de lumină
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- 2. Încărcare date din ImageFolder (Neschimbat) ---
data_dir = r'D:\data' 

try:
    trainset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    
    testset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    
    num_classes = len(trainset.classes)
    print(f"Clase găsite: {trainset.classes}") # ex: ['humans', 'robots']

except FileNotFoundError:
    print(f"EROARE: Nu am găsit folderele 'train' sau 'test' în interiorul '{data_dir}'")
    exit()


# --- 3. MODIFICARE: Arhitectura CNN (Varianta Robustă și Adâncă) ---
class CVNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # --- Partea de "extragere a caracteristicilor" (convoluție) ---
        self.features = nn.Sequential(
            # Bloc 1: 128x128 -> 64x64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Dimensiune: (32, 64, 64)
            
            # Bloc 2: 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Dimensiune: (64, 32, 32)
            
            # Bloc 3: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # Dimensiune: (128, 16, 16)
        )
        
        # --- Calcul automat al dimensiunii (Neschimbat) ---
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        dummy_output = self.features(dummy_input)
        
        # Dimensiunea aplatizată va fi 128 * 16 * 16 = 32768
        flattened_size = dummy_output.view(1, -1).shape[1]
        
        print(f"Dimensiunea de intrare pentru FC calculată automat: {flattened_size}")
        
        # --- Partea de "clasificare" (complet conectată) ---
        self.classifier = nn.Sequential(
            # Strat ascuns (hidden layer) pentru a învăța combinații
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            # Dropout-ul este mult mai eficient aici, între straturile Linear
            nn.Dropout(0.5), 
            
            # Stratul final de ieșire
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 1. Trece prin convoluții
        x = self.features(x)
        
        # 2. Aplatizează (flatten)
        x = x.view(x.size(0), -1) 
        
        # 3. Trece prin clasificator
        x = self.classifier(x)
        return x

# --- Inițializare model și antrenament ---

# --- MODIFICARE 4: Ponderarea pierderii (Loss Weighting) ---
# Presupunând că 'humans' e clasa 0 și 'robots' e clasa 1 (ImageFolder sortează alfabetic)
# Dăm o pondere de 1.6 clasei 'robots' (clasa 1) pentru a compensa dezechilibrul
# Asigură-te că ordinea corespunde cu print(trainset.classes)
# Dacă clasele tale sunt ['oameni', 'roboti'], ordinea e corectă.
weights = torch.tensor([1.0, 1.6]).to(device)

net = CVNet(num_classes=num_classes).to(device) # Mutăm modelul pe device
criterion = nn.CrossEntropyLoss(weight=weights) # Folosim ponderile
optimizer = optim.Adam(net.parameters(), lr=0.001) 

# --- MODIFICARE 5: Mai multe epoci ---
num_epochs = 30 # 5 era mult prea puțin

print("--- Începe antrenamentul ---")


for epoch in range(num_epochs):
    i = 0
    # --- MODIFICARE 6: Trecem modelul în modul de antrenare ---
    net.train() 
    running_loss = 0.0
    
    for images, labels in trainloader:
        i = i + 1
        print(i)
        # --- MODIFICARE 7: Mutăm datele pe device ---
        images, labels = images.to(device), labels.to(device)
        
        out = net(images)
        loss = criterion(out, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Calculăm și afișăm acuratețea de antrenament (opțional, dar util)
    train_loss = running_loss / len(trainloader)
    
    # --- Evaluare pe setul de testare LA FIECARE EPOCĂ ---
    # E util să vezi cum progresează modelul pe date noi
    correct = 0
    total = 0
    test_loss = 0.0
    
    # --- MODIFICARE 8: Trecem modelul în modul evaluare ---
    net.eval() 
    with torch.no_grad():
        for images, labels in testloader:
            # --- MODIFICARE 9: Mutăm datele pe device ---
            images, labels = images.to(device), labels.to(device)
            
            out = net(images)
            loss = criterion(out, labels) # Putem calcula și pierderea pe test
            test_loss += loss.item()
            
            pred = out.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    
    print(f"Epoca [{epoch+1}/{num_epochs}], "
          f"Loss Antrenare: {train_loss:.4f}, "
          f"Loss Test: {test_loss/len(testloader):.4f}, "
          f"Acuratețe Test: {acc:.2f}%")

print("--- Antrenament finalizat ---")

# --- MODIFICARE 10: Prag de salvare mai realist ---
if acc > 65: # Am crescut pragul de la 10% la 65%
    model_save_path = "model_imbunatatit.pt"
    torch.save(net.state_dict(), model_save_path)
    print(f"--- Model salvat cu succes ca '{model_save_path}' (Acc: {acc:.2f}%) ---")
    
else:
    print(f"Acuratețea de {acc:.2f}% este sub pragul de 65%.")
    print("Modelul nu a învățat suficient. Încearcă mai multe epoci sau mai multe date.")