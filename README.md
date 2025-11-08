# Clasificator binar de imagini roboti-oameni


## Prezentare generala
Am primit task-ul de a clasifica imagini cu oameni si roboti folosind o arhitectura de tipul Convolutional Neural Network (CNN).

Primul pas a fost colectarea seturilor de date pentru antrenare.

Apoi a urmat antrenarea modelului pe datele obtinute folosind libraria pytorch.

Ultimul pas a fost conectarea modelului la o interfata a unei aplicatii FullStack.

## Pregatirea datelor
Am folosit doua set-uri de date publice:

### Data Set pentru Roboti
[Humanoid Robot Pose Estimation](https://github.com/AIS-Bonn/HumanoidRobotPoseEstimation?tab=readme-ov-file)

### Data Set pentru Oameni
[Leeds-Sport pose)](https://www.kaggle.com/datasets/dkrivosic/leeds-sports-pose-lsp)

Pentru organizarea imaginilor in fisiere am folosit un script python (PhotoScripts/organize.py)

Numarul de imagini cu roboti nu a fost suficient pentru model, asa ca am creat imagini noi prelucrand imaginile originale.

Pentru prelucrare am aplicat filtre de image cropping, image flipping, grayscale.

Script-ul folosit pentru asta a fost PhotoScripts/editing.py.

## Antrenarea modelului 

Am ales sa folosim libraria pytorch impreuna cu torchvision.

Modelul se antreneaza cu un learning rate de 0.01, folosind optimizatorul "Adam".

Antrenarea dureaza 10 epoci.

## Aplicatia

Pentru interactiunea cu modelul, am construit o aplicatie web fullstack.

Am folosit urmatoarele tehnologii:

Frontend vanilla cu html si CSS.
 
Backend in Python (flask), baza de date SQLite.

## Rezultate si posibile imbunatatiri

Am obtinut un MVP care clasifica binar oamenii si robotii.

Tehnologiile pe care le-am folosit si volumul de date au fost influentate de perioada mica de timp pe care am avut-o la dispozitie pentru a finaliza prototipul (doar 10 ore).

Daca am fi avut mai mult timp:

-am fi marit volumul de antrenare al modelului

-am fi folosit libraria Keras in loc de pytorch

-adaugare si colectare de imagini extra

-alegerea unui framework de javascript (React, VueJS, Angular)

-un backend mai robust (.NET)

-o baza de date care sa nu foloseasca arhitectura serverless

 


