# Classification de texte multi-classe
## Introduction
Ce programme est une classification de texte multi-classe sur un jeu de données en anglais (contenu), écrit par des personnes de nationalités différentes (tag). Chaque échantillon de texte contient le tag au début de cet échantillon. Certains ont l'anglais comme de la langue maternelle, et d'autres non.   
Concrètement, on veut prédire de quelle nationalité s’agit un texte nouvellement écrit par une personne de nationalité inconnue.
Pour ce dernier, j'ai réalisé un pré-processing des données, puis du feature engineering. Et par la suite, j'ai construit un modèle et le entraîné avec nos jeux de données d'entraînement, puis sorti les résultats de la prédiction dans le fichier.

## Install and usage
1. Configuration de l'environnement.

Option 1 : Activer l'environnement virtuel configuré (venv)
```sh
    source bin/activate
```
Option 2 : Installer les bibliothèques dépendantes à partir de requirements.txt sur votre environnement exsitant
```sh
    pip install -r requirements.txt
```
2. Exécuter le programme
```sh
    python main.py data/input/train.txt data/input/test.txt
```
Le temps d'exécution est d'environ 3-4 minutes.

**Un exemple**

<div align="center">

<img align="center" hight="600" width="600" src="https://github.com/JuexiaoZhang/ClassificationTexteMulticlasse/blob/main/data/capture.png">

</div>

## In Details

Folder structure
--------------

```
├── data
│   ├── input           - Ce dossier contient les données à traiter
│   ├── output          - Ce dossier contient la sortie du programme
│   └── working         - Ce dossier contient des données temporaires et des modèles générés lors de l'exécution du programme
│
├── models              - Ce dossier contient tous les modèles de ce projet.
│   └── model.py
│   
├──  main.py            - fichier main     
│  
├──  loaders            - Ce dossier contient tout le traitement des données
│    └── data_generator.py  - Importing, preprocessing, feature engineering...
│ 
├──  utils
│    └── utils.py       - Il contient un seul outil pour l'instant: confirmer le nombre d'arguments
│
└──  venv               - Environnement virtuel
```
