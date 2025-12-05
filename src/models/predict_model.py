"""
Module pour faire des prédictions avec les modèles entraînés
"""
import joblib
import pandas as pd
import numpy as np


def load_model(model_path):
    """
    Charge un modèle sauvegardé
    
    Args:
        model_path (str): Chemin vers le fichier .pkl
        
    Returns:
        Model: Modèle chargé
    """
    print(f"📂 Chargement du modèle : {model_path}")
    model = joblib.load(model_path)
    print(f"   ✓ Modèle chargé : {type(model).__name__}\n")
    return model


def load_feature_names(feature_names_path='models/feature_names.txt'):
    """
    Charge les noms des features
    
    Args:
        feature_names_path (str): Chemin vers le fichier contenant les noms
        
    Returns:
        list: Liste des noms de features
    """
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    return feature_names


def predict_match(model, home_stats, away_stats, feature_names):
    """
    Prédit le résultat d'un match
    
    Args:
        model: Modèle entraîné
        home_stats (dict): Stats de l'équipe à domicile
        away_stats (dict): Stats de l'équipe à l'extérieur
        feature_names (list): Noms des features dans l'ordre
        
    Returns:
        tuple: (prediction, probabilities)
    """
    # Créer le vecteur de features
    features = {}
    for key in home_stats.keys():
        features[f'Home_{key}'] = home_stats[key]
        features[f'Away_{key}'] = away_stats[key]
    
    # Créer DataFrame avec l'ordre correct des colonnes
    X = pd.DataFrame([features])[feature_names]
    
    # Prédire
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    return prediction, probabilities


def interpret_prediction(prediction, probabilities):
    """
    Interprète la prédiction en langage humain
    
    Args:
        prediction (int): 0=Away win, 1=Draw, 2=Home win
        probabilities (array): Probabilités pour chaque classe
        
    Returns:
        str: Résultat formaté
    """
    labels = {0: 'Victoire extérieur', 1: 'Match nul', 2: 'Victoire domicile'}
    result = labels[prediction]
    
    print("🎯 PRÉDICTION")
    print("="*40)
    print(f"Résultat prédit : {result}")
    print("\nProbabilités :")
    print(f"  Victoire domicile : {probabilities[2]:.1%}")
    print(f"  Match nul         : {probabilities[1]:.1%}")
    print(f"  Victoire extérieur: {probabilities[0]:.1%}")
    print("="*40 + "\n")
    
    return result


def predict_from_csv(model_path, features_path, output_path=None):
    """
    Fait des prédictions sur un fichier CSV de features
    
    Args:
        model_path (str): Chemin vers le modèle
        features_path (str): Chemin vers le CSV de features
        output_path (str): Chemin pour sauvegarder les prédictions (optionnel)
        
    Returns:
        pd.DataFrame: DataFrame avec prédictions
    """
    # Charger modèle et données
    model = load_model(model_path)
    feature_names = load_feature_names()
    
    print(f"📂 Chargement des données : {features_path}")
    df = pd.read_csv(features_path)
    print(f"   {len(df)} matchs à prédire\n")
    
    # Faire les prédictions
    X = df[feature_names]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Ajouter les prédictions au DataFrame
    df['Prediction'] = predictions
    df['Proba_Away'] = probabilities[:, 0]
    df['Proba_Draw'] = probabilities[:, 1]
    df['Proba_Home'] = probabilities[:, 2]
    
    # Mapper les labels
    label_map = {0: 'A', 1: 'D', 2: 'H'}
    df['Prediction_Label'] = df['Prediction'].map(label_map)
    
    print(f"✅ Prédictions effectuées\n")
    
    # Sauvegarder si demandé
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"💾 Prédictions sauvegardées : {output_path}\n")
    
    return df


def example_prediction():
    """
    Exemple de prédiction pour un match
    """
    print("="*60)
    print("EXEMPLE DE PRÉDICTION")
    print("="*60 + "\n")
    
    # Charger le meilleur modèle (XGBoost Balanced)
    model = load_model('models/xgboost_balanced.pkl')
    feature_names = load_feature_names()
    
    # Stats d'exemple (moyennes sur 5 matchs)
    home_stats = {
        'buts_marques': 1.6,
        'buts_concedes': 1.0,
        'victoires': 3,
        'nuls': 1,
        'defaites': 1,
        'tirs': 12.4,
        'tirs_cadres': 5.2
    }
    
    away_stats = {
        'buts_marques': 1.2,
        'buts_concedes': 1.4,
        'victoires': 2,
        'nuls': 2,
        'defaites': 1,
        'tirs': 10.8,
        'tirs_cadres': 4.6
    }
    
    print("📊 Statistiques des équipes (5 derniers matchs) :")
    print("\nÉquipe à domicile :")
    for key, val in home_stats.items():
        print(f"  {key:15s}: {val}")
    
    print("\nÉquipe à l'extérieur :")
    for key, val in away_stats.items():
        print(f"  {key:15s}: {val}")
    print()
    
    # Prédire
    prediction, probabilities = predict_match(model, home_stats, away_stats, feature_names)
    interpret_prediction(prediction, probabilities)


if __name__ == '__main__':
    # Exemple de prédiction
    example_prediction()