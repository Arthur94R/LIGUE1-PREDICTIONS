# 🏆 Prédiction de matchs de Ligue 1

Projet de Machine Learning pour prédire les résultats de matchs de football (Home/Draw/Away).

## 📊 Le projet en bref

- **Dataset** : 2791 matchs sur 8 saisons (2017-2025)
- **Features** : 14 stats basées sur les 5 derniers matchs de chaque équipe
- **Modèles** : 4 modèles ML (RandomForest, XGBoost, versions balanced)
- **Meilleure accuracy** : 47% (RandomForest)

## 🚀 Installation et lancement

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Lancer le pipeline complet
```bash
python main.py
```
⏱️ Durée : 2-3 minutes

Ça va :
- Nettoyer les données (8 CSV → 1 fichier clean)
- Calculer les features (stats sur 5 matchs)
- Entraîner 4 modèles ML
- Sauvegarder tout dans `models/`

### 3. Faire une prédiction
```bash
python demo_prediction.py
```

Exemple de résultat :
```
🎯 Match : PSG (domicile) vs Marseille (extérieur)
Prédiction : Victoire PSG (67% de confiance)
```

## 📁 Structure du projet

```
ligue1-prediction/
├── data/
│   ├── raw/           # 8 CSV originaux
│   ├── interim/       # Données nettoyées
│   └── processed/     # Features finales
├── src/
│   ├── data/          # Nettoyage des données
│   ├── features/      # Calcul des features
│   └── models/        # Entraînement et prédiction
├── models/            # 4 modèles .pkl sauvegardés
├── reports/figures/   # 7 graphiques PNG
└── main.py            # Lance tout le pipeline
```

## 🎯 Les 4 modèles

| Modèle | Accuracy |
|--------|----------|
| **RandomForest** | **47.41%** ⭐ |
| RandomForest Balanced | 44.72% |
| XGBoost | 42.75% |
| XGBoost Balanced | 37.75% |

## 💡 Pourquoi 47% seulement ?

Prédire du foot est **très difficile** :
- Beaucoup de facteurs non mesurables (blessures, météo, motivation...)
- Les pros atteignent max 55-60%
- 47% > 43% (toujours prédire "Home win") → **c'est bon !**

## 🔧 Technologies utilisées

- Python 3.8+
- pandas, numpy (manipulation de données)
- scikit-learn (RandomForest)
- XGBoost (gradient boosting)
- matplotlib, seaborn (visualisations)

## 📈 Graphiques

7 visualisations dans `reports/figures/` :
- Distribution des résultats
- Buts domicile vs extérieur
- Top 15 équipes
- Matrices de confusion
- Feature importance
- Comparaison des modèles

## 👨‍💻 Auteur

Arthur - Master 1 AI & Big Data  
Décembre 2024
