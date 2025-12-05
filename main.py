"""
Script principal pour le projet de prédiction de matchs de Ligue 1

Ce script exécute l'ensemble du pipeline :
1. Chargement et nettoyage des données brutes
2. Feature engineering
3. Entraînement des modèles
4. Évaluation et sauvegarde

Usage:
    python main.py
"""

import sys
import os

# Ajouter le dossier src au path pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.make_dataset import make_dataset
from features.build_features import make_features
from models.train_model import train_all_models


def main():
    """
    Exécute le pipeline complet
    """
    print("\n" + "="*60)
    print("🏆 PRÉDICTION DE MATCHS DE LIGUE 1")
    print("="*60 + "\n")
    
    try:
        # Étape 1 : Préparation des données
        print("🔄 Démarrage du pipeline...\n")
        df_clean = make_dataset(
            raw_data_dir='data/raw',
            interim_data_dir='data/interim'
        )
        
        # Étape 2 : Feature engineering
        df_features = make_features(
            interim_data_path='data/interim/clean_data.csv',
            output_dir='data/processed',
            n_matches=5
        )
        
        # Étape 3 : Entraînement des modèles
        results = train_all_models(
            features_path='data/processed/features.csv',
            output_dir='models',
            test_size=0.2,
            random_state=42
        )
        
        # Résumé final
        print("="*60)
        print("✅ PIPELINE TERMINÉ AVEC SUCCÈS !")
        print("="*60)
        print(f"\n📁 Fichiers générés :")
        print(f"   - data/interim/clean_data.csv")
        print(f"   - data/processed/features.csv")
        print(f"   - models/*.pkl (4 modèles)")
        print(f"   - models/feature_names.txt")
        print(f"\n💡 Pour faire des prédictions :")
        print(f"   python -c 'from src.models.predict_model import example_prediction; example_prediction()'")
        print()
        
    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()