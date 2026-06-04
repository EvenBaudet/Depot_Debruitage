# Optimisation de Filtres de Débruitage - Détermination des Hyperparamètres

Ce script teste et optimise plusieurs filtres (spatiaux et fréquentiels) face à 3 types de bruits distincts qui peuvent être modifiés en choisissant à la ligne 94 du main.py la fonction d'ajout de bruit d'intéret (celles-ci sont présentes dans fonctions_annexe.py). Les performances sont mesurées via PSNR et SSIM.

## 🛠️ Installation & Lancement

1. Téléchargez et extrayez le dossier du projet.

```bash
# 2. Installer les dépendances
python -m pip install -r requirements.txt

# 3. Lancer les tests et générer les graphiques
python main.py
```

graphiques_gaussien_0.1/ : Contient tous les résultats et courbes de performance face au bruit Gaussien classique ($\sigma = 0.1$).
graphiques_poissonien_100/ : Contient tous les résultats et courbes de performance face au bruit de Poisson (Flux à peak = 100).
graphiques_sel_et_poivre_0.05/ : Contient tous les résultats et courbes de performance face au bruit impulsionnel (5% de pixels détruits).
