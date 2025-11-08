# Etapa 2 CV_Lock - Adaptat pentru ImageFolder cu TRANSFER LEARNING și LOGGING
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets, models
from torch.optim import lr_scheduler
import hashlib
import os
import collections
import time
from multiprocessing import freeze_support # <-- MODIFICARE: Import necesar

# --- MODIFICARE: Începem blocul de protecție ---
if __name__ == '__main__':
    # --- MODIFICARE: Adăugăm freeze_support() ---
    # Necesar pentru Windows
    freeze_support() 

    # --- 0. Verificare GPU ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Se va folosi dispozitivul: {device} ---") 

    # --- 1. Transformări ---
    IMG_SIZE = 128 

    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), 
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

    # --- 2. Încărcare date ---
    data_dir = r'D:\data' 

    try:
        trainset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
        # Folosim num_workers pentru viteză, acum că avem protecția __main__
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4) 
        
        testset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=4)
        
        num_classes = len(trainset.classes)
        print(f"Clase găsite: {trainset.classes} (Total {num_classes} clase)")
        print(f"Dimensiune Train: {len(trainset)}, Dimensiune Test: {len(testset)}")

    except FileNotFoundError:
        print(f"EROARE: Nu am găsit folderele 'train' sau 'test' în interiorul '{data_dir}'")
        exit()

    
    # --- 3. Arhitectura (TRANSFER LEARNING cu ResNet18) ---
    
    # --- MODIFICARE: Folosim noua sintaxă 'weights' pentru a evita Warning-ul ---
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    for param in net.parameters():
        param.requires_grad = False
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, num_classes)
    net = net.to(device)
    print("--- Modelul ResNet18 a fost încărcat și modificat pentru Transfer Learning ---")


    # --- 4. Ponderarea Automată a Pierderii (Loss Weighting) ---
    class_counts = collections.Counter(trainset.targets)
    class_counts = [class_counts[i] for i in range(num_classes)]
    print(f"Număr imagini pe clase (Train): {dict(zip(trainset.classes, class_counts))}")
    total_samples = len(trainset)
    weights = [total_samples / (num_classes * count) for count in class_counts]
    weights = torch.tensor(weights).float().to(device)
    print(f"Ponderi calculate pentru Loss: {weights}")


    # --- 5. Inițializare ---
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(net.fc.parameters(), lr=0.001) 
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    num_epochs = 20

    print("--- Începe antrenamentul (se antrenează doar ultimul strat) ---")

    # --- 6. Variabile pentru antrenament ---
    best_acc = 0.0

    # --- Bucla de Antrenament ---
    for epoch in range(num_epochs):
        
        start_time = time.time()
        
        # --- FAZA DE ANTRENAMENT ---
        net.train()
        running_loss = 0.0
        
        log_interval = 10 
        total_batches_train = len(trainloader)
        
        for i, (images, labels) in enumerate(trainloader, 1):
            
            images, labels = images.to(device), labels.to(device)
            
            out = net(images)
            loss = criterion(out, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if i % log_interval == 0 or i == total_batches_train:
                print(f"  Epoca [{epoch+1}/{num_epochs}]... Batch [{i}/{total_batches_train}]  | Loss Batch: {loss.item():.4f}")
        
        train_loss = running_loss / total_batches_train
        
        scheduler.step()

        # --- FAZA DE EVALUARE (TESTARE) ---
        correct = 0
        total = 0
        test_loss = 0.0
        
        net.eval()
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                out = net(images)
                loss = criterion(out, labels)
                test_loss += loss.item()
                pred = out.argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        
        epoch_duration = time.time() - start_time
        
        print("-" * 40) # Separator
        print(f"REZUMAT EPOCA [{epoch+1}/{num_epochs}] (Timp: {epoch_duration:.2f}s)")
        print(f"  Loss Antrenare: {train_loss:.4f}")
        print(f"  Loss Test: {test_loss/len(testloader):.4f}")
        print(f"  Acuratețe Test: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            model_save_path = "model_resnet18_best.pt"
            torch.save(net.state_dict(), model_save_path)
            print(f"  -> Model îmbunătățit salvat! (Acc: {best_acc:.2f}%)")
        print("-" * 40 + "\n") 


    print("--- Antrenament finalizat ---")

    if best_acc > 90:
        print(f"Performanță excelentă! Cea mai bună acuratețe: {best_acc:.2f}%.")
    else:
        print(f"Acuratețea finală de {best_acc:.2f}% e sub pragul de 90%.")
        print("Încearcă mai multe epoci sau mai multă augmentare a datelor.")

# --- SFÂRȘITUL blocului 'if __name__ == '__main__':' ---