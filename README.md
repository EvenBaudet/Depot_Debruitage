# Optimisation de Filtres de Débruitage - Détermination des Hyperparamètres

Ce script teste et optimise plusieurs filtres (spatiaux et fréquentiels) face à 3 types de bruits distincts qui peuvent être modifiés en choisissant à la ligne 94 du main.py la fonction d'ajout de bruit d'intéret (celles-ci sont présentes dans fonctions_annexe.py). Les performances sont mesurées via PSNR et SSIM.

## 🛠️ Installation & Lancement

1. Téléchargez et extrayez le dossier du projet.

```bash
# 2. Installer les dépendances
python -m pip install -r requirements.txt

# 3. Lancer les tests et générer les graphiques
python main.py