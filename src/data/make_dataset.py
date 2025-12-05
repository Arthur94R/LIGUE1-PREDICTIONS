"""
Module pour charger et nettoyer les données de matchs de Ligue 1
"""
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_raw_data(data_dir='data/raw'):
    """
    Charge tous les fichiers CSV de matchs et les combine en un seul DataFrame
    
    Args:
        data_dir (str): Chemin vers le dossier contenant les CSV
        
    Returns:
        pd.DataFrame: DataFrame combiné avec tous les matchs
    """
    print(f"📂 Chargement des données depuis {data_dir}...")
    
    fichiers = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    print(f"   Fichiers trouvés : {fichiers}")
    
    df_list = []
    for fichier in fichiers:
        df_temp = pd.read_csv(os.path.join(data_dir, fichier))
        df_list.append(df_temp)
        print(f"   ✓ {fichier}: {len(df_temp)} matchs")
    
    df = pd.concat(df_list, ignore_index=True)
    print(f"✅ Total : {len(df)} matchs chargés\n")
    
    return df


def select_columns(df):
    """
    Sélectionne uniquement les colonnes nécessaires pour l'analyse
    
    Args:
        df (pd.DataFrame): DataFrame complet
        
    Returns:
        pd.DataFrame: DataFrame avec colonnes sélectionnées
    """
    colonnes_necessaires = [
        'Date', 'HomeTeam', 'AwayTeam',
        'FTHG', 'FTAG', 'FTR',
        'HTHG', 'HTAG',
        'HS', 'AS', 'HST', 'AST',
        'HF', 'AF', 'HC', 'AC',
        'HY', 'AY', 'HR', 'AR',
    ]
    
    print(f"📋 Sélection de {len(colonnes_necessaires)} colonnes pertinentes...")
    df = df[colonnes_necessaires]
    print(f"✅ Shape après sélection : {df.shape}\n")
    
    return df


def clean_data(df):
    """
    Nettoie les données : gestion des valeurs manquantes
    
    Args:
        df (pd.DataFrame): DataFrame à nettoyer
        
    Returns:
        pd.DataFrame: DataFrame nettoyé
    """
    print("🧹 Nettoyage des données...")
    
    # Afficher les valeurs manquantes
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"   Valeurs manquantes trouvées :")
        print(missing[missing > 0])
        df = df.dropna()
        print(f"   → Lignes supprimées, reste : {len(df)} matchs")
    else:
        print("   ✓ Aucune valeur manquante")
    
    print(f"✅ Nettoyage terminé\n")
    return df


def process_dates(df):
    """
    Convertit les dates et crée une colonne 'Saison'
    
    Args:
        df (pd.DataFrame): DataFrame avec colonne 'Date'
        
    Returns:
        pd.DataFrame: DataFrame avec dates formatées et saison
    """
    print("📅 Traitement des dates...")
    
    # Convertir en datetime
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Créer colonne Saison
    def get_saison(date):
        if date.month >= 8:
            return f"{date.year}-{date.year+1}"
        else:
            return f"{date.year-1}-{date.year}"
    
    df['Saison'] = df['Date'].apply(get_saison)
    
    saisons = df['Saison'].unique()
    print(f"   Saisons identifiées : {sorted(saisons)}")
    print(f"✅ Dates traitées\n")
    
    return df


def encode_target(df):
    """
    Encode la variable cible FTR (Full Time Result)
    
    Args:
        df (pd.DataFrame): DataFrame avec colonne 'FTR'
        
    Returns:
        pd.DataFrame: DataFrame avec colonne 'FTR_encoded'
    """
    print("🔢 Encodage de la variable cible...")
    
    df['FTR_encoded'] = LabelEncoder().fit_transform(df['FTR'])
    
    # Afficher le mapping
    mapping = df[['FTR', 'FTR_encoded']].drop_duplicates().sort_values('FTR_encoded')
    print("   Mapping :")
    for _, row in mapping.iterrows():
        label = {'A': 'Away win', 'D': 'Draw', 'H': 'Home win'}[row['FTR']]
        print(f"   {row['FTR']} ({label}) → {row['FTR_encoded']}")
    
    print(f"✅ Encodage terminé\n")
    return df


def make_dataset(raw_data_dir='data/raw', interim_data_dir='data/interim'):
    """
    Pipeline complet : charge, nettoie et prépare les données
    
    Args:
        raw_data_dir (str): Chemin vers données brutes
        interim_data_dir (str): Chemin pour sauvegarder données nettoyées
        
    Returns:
        pd.DataFrame: DataFrame nettoyé et prêt pour feature engineering
    """
    print("="*60)
    print("ÉTAPE 1 : PRÉPARATION DES DONNÉES")
    print("="*60 + "\n")
    
    # Pipeline de nettoyage
    df = load_raw_data(raw_data_dir)
    df = select_columns(df)
    df = clean_data(df)
    df = process_dates(df)
    df = encode_target(df)
    
    # Sauvegarder
    os.makedirs(interim_data_dir, exist_ok=True)
    output_path = os.path.join(interim_data_dir, 'clean_data.csv')
    df.to_csv(output_path, index=False)
    print(f"💾 Données sauvegardées : {output_path}")
    print(f"   Shape finale : {df.shape}\n")
    
    return df


if __name__ == '__main__':
    # Exécution standalone
    df = make_dataset()
    print(f"\n✅ Pipeline terminé ! {len(df)} matchs prêts.")