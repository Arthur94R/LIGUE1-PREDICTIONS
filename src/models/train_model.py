"""
Module pour entraîner les modèles de prédiction de matchs
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_features(features_path='data/processed/features.csv'):
    """
    Charge les features préparées
    
    Args:
        features_path (str): Chemin vers le fichier features
        
    Returns:
        tuple: (X, y, feature_columns)
    """
    print("="*60)
    print("ÉTAPE 3 : ENTRAÎNEMENT DES MODÈLES")
    print("="*60 + "\n")
    
    print(f"📂 Chargement des features : {features_path}")
    df = pd.read_csv(features_path)
    print(f"   {len(df)} matchs chargés\n")
    
    # Sélectionner les colonnes features
    feature_columns = [col for col in df.columns if col.startswith('Home_') or col.startswith('Away_')]
    
    X = df[feature_columns]
    y = df['FTR_encoded']
    
    print(f"📊 Features pour l'entraînement :")
    print(f"   Nombre de features : {len(feature_columns)}")
    print(f"   Shape de X : {X.shape}")
    print(f"   Distribution des classes (y) :")
    print(y.value_counts().sort_index())
    print()
    
    return X, y, feature_columns


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Sépare les données en train/test
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion de test
        random_state (int): Seed pour reproductibilité
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"✂️  Séparation train/test ({int((1-test_size)*100)}% / {int(test_size*100)}%)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"   Train : {len(X_train)} matchs")
    print(f"   Test  : {len(X_test)} matchs")
    print()
    
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, balanced=False):
    """
    Entraîne un RandomForest
    
    Args:
        X_train: Features d'entraînement
        y_train: Target d'entraînement
        balanced (bool): Utiliser class_weight='balanced'
        
    Returns:
        RandomForestClassifier: Modèle entraîné
    """
    model_name = "RandomForest" + (" Balanced" if balanced else "")
    print(f"🌳 Entraînement : {model_name}...")
    
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
    
    if balanced:
        params['class_weight'] = 'balanced'
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    print(f"   ✓ {model_name} entraîné\n")
    return model


def train_xgboost(X_train, y_train, balanced=False):
    """
    Entraîne un XGBoost
    
    Args:
        X_train: Features d'entraînement
        y_train: Target d'entraînement
        balanced (bool): Utiliser sample_weight
        
    Returns:
        XGBClassifier: Modèle entraîné
    """
    model_name = "XGBoost" + (" Balanced" if balanced else "")
    print(f"🚀 Entraînement : {model_name}...")
    
    params = {
        'n_estimators': 100,
        'max_depth': 6 if balanced else 10,
        'learning_rate': 0.1,
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'use_label_encoder': False
    }
    
    model = XGBClassifier(**params)
    
    if balanced:
        sample_weights = compute_sample_weight('balanced', y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)
    
    print(f"   ✓ {model_name} entraîné\n")
    return model


def evaluate_model(model, X_test, y_test, model_name):
    """
    Évalue un modèle sur le test set
    
    Args:
        model: Modèle entraîné
        X_test: Features de test
        y_test: Target de test
        model_name (str): Nom du modèle
        
    Returns:
        dict: Métriques d'évaluation
    """
    print("="*60)
    print(f"📊 ÉVALUATION : {model_name}")
    print("="*60)
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy : {accuracy:.2%}\n")
    
    # Classification report
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred, 
                                target_names=['Away win', 'Draw', 'Home win']))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print("Matrice de confusion :")
    print(cm)
    print("(lignes = vrai résultat, colonnes = prédiction)\n")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'predictions': y_pred
    }


def save_model(model, model_name, output_dir='models'):
    """
    Sauvegarde un modèle entraîné
    
    Args:
        model: Modèle à sauvegarder
        model_name (str): Nom du modèle
        output_dir (str): Dossier de sortie
        
    Returns:
        str: Chemin du fichier sauvegardé
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name.lower().replace(' ', '_')}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    joblib.dump(model, filepath)
    print(f"💾 Modèle sauvegardé : {filepath}\n")
    
    return filepath


def train_all_models(features_path='data/processed/features.csv',
                     output_dir='models',
                     test_size=0.2,
                     random_state=42):
    """
    Pipeline complet : charge, entraîne et évalue tous les modèles
    
    Args:
        features_path (str): Chemin vers features
        output_dir (str): Dossier pour sauvegarder les modèles
        test_size (float): Proportion de test
        random_state (int): Seed
        
    Returns:
        dict: Dictionnaire avec tous les modèles et leurs résultats
    """
    # Charger et séparer les données
    X, y, feature_columns = load_features(features_path)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    
    results = {}
    
    # 1. RandomForest
    rf_model = train_random_forest(X_train, y_train, balanced=False)
    rf_results = evaluate_model(rf_model, X_test, y_test, "RandomForest")
    save_model(rf_model, "RandomForest", output_dir)
    results['rf'] = {'model': rf_model, 'metrics': rf_results}
    
    # 2. RandomForest Balanced
    rf_bal_model = train_random_forest(X_train, y_train, balanced=True)
    rf_bal_results = evaluate_model(rf_bal_model, X_test, y_test, "RandomForest Balanced")
    save_model(rf_bal_model, "RandomForest Balanced", output_dir)
    results['rf_balanced'] = {'model': rf_bal_model, 'metrics': rf_bal_results}
    
    # 3. XGBoost
    xgb_model = train_xgboost(X_train, y_train, balanced=False)
    xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    save_model(xgb_model, "XGBoost", output_dir)
    results['xgb'] = {'model': xgb_model, 'metrics': xgb_results}
    
    # 4. XGBoost Balanced
    xgb_bal_model = train_xgboost(X_train, y_train, balanced=True)
    xgb_bal_results = evaluate_model(xgb_bal_model, X_test, y_test, "XGBoost Balanced")
    save_model(xgb_bal_model, "XGBoost Balanced", output_dir)
    results['xgb_balanced'] = {'model': xgb_bal_model, 'metrics': xgb_bal_results}
    
    # Sauvegarder les noms des features
    feature_names_path = os.path.join(output_dir, 'feature_names.txt')
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(feature_columns))
    print(f"💾 Noms des features sauvegardés : {feature_names_path}\n")
    
    # Résumé final
    print("="*60)
    print("📈 RÉSUMÉ DES PERFORMANCES")
    print("="*60)
    for name, data in results.items():
        acc = data['metrics']['accuracy']
        print(f"{name:20s} : {acc:.2%}")
    print()
    
    return results


if __name__ == '__main__':
    # Exécution standalone
    results = train_all_models()
    print("✅ Tous les modèles ont été entraînés et sauvegardés !")