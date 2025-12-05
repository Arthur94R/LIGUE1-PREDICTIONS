"""
Module pour créer les features à partir des statistiques des équipes
"""
import os
import pandas as pd


def get_team_stats(df, team, date, saison, n_matches=5):
    """
    Calcule les statistiques d'une équipe sur ses N derniers matchs
    
    Args:
        df (pd.DataFrame): DataFrame complet des matchs
        team (str): Nom de l'équipe
        date (datetime): Date du match actuel
        saison (str): Saison du match (ex: '2022-2023')
        n_matches (int): Nombre de matchs à considérer
        
    Returns:
        dict: Statistiques calculées
    """
    # Filtrer les matchs précédents de l'équipe dans la même saison
    mask = (
        (df['Saison'] == saison) & 
        (df['Date'] < date) & 
        ((df['HomeTeam'] == team) | (df['AwayTeam'] == team))
    )
    previous_matches = df[mask].tail(n_matches)
    
    # Si pas d'historique : retourner des valeurs par défaut
    if len(previous_matches) == 0:
        return {
            'buts_marques': 0,
            'buts_concedes': 0,
            'victoires': 0,
            'nuls': 0,
            'defaites': 0,
            'tirs': 0,
            'tirs_cadres': 0
        }
    
    # Initialiser les compteurs
    buts_marques = 0
    buts_concedes = 0
    victoires = 0
    nuls = 0
    defaites = 0
    tirs = 0
    tirs_cadres = 0
    
    # Parcourir chaque match précédent
    for _, match in previous_matches.iterrows():
        if match['HomeTeam'] == team:
            # L'équipe jouait à domicile
            buts_marques += match['FTHG']
            buts_concedes += match['FTAG']
            tirs += match['HS']
            tirs_cadres += match['HST']
            
            if match['FTR'] == 'H':
                victoires += 1
            elif match['FTR'] == 'D':
                nuls += 1
            else:
                defaites += 1
        else:
            # L'équipe jouait à l'extérieur
            buts_marques += match['FTAG']
            buts_concedes += match['FTHG']
            tirs += match['AS']
            tirs_cadres += match['AST']
            
            if match['FTR'] == 'A':
                victoires += 1
            elif match['FTR'] == 'D':
                nuls += 1
            else:
                defaites += 1
    
    # Calculer les moyennes
    n = len(previous_matches)
    return {
        'buts_marques': buts_marques / n,
        'buts_concedes': buts_concedes / n,
        'victoires': victoires,
        'nuls': nuls,
        'defaites': defaites,
        'tirs': tirs / n,
        'tirs_cadres': tirs_cadres / n
    }


def build_features(df, n_matches=5):
    """
    Construit toutes les features pour chaque match
    
    Args:
        df (pd.DataFrame): DataFrame avec données nettoyées
        n_matches (int): Nombre de matchs à considérer pour les stats
        
    Returns:
        pd.DataFrame: DataFrame avec features ajoutées
    """
    print("="*60)
    print("ÉTAPE 2 : FEATURE ENGINEERING")
    print("="*60 + "\n")
    
    print(f"🔧 Calcul des features (stats sur {n_matches} derniers matchs)...")
    print(f"   Total de matchs à traiter : {len(df)}")
    
    home_stats = []
    away_stats = []
    
    # Calculer les stats pour chaque match
    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"   Progression : {idx}/{len(df)} matchs...")
        
        h_stats = get_team_stats(df, row['HomeTeam'], row['Date'], row['Saison'], n_matches)
        a_stats = get_team_stats(df, row['AwayTeam'], row['Date'], row['Saison'], n_matches)
        
        home_stats.append(h_stats)
        away_stats.append(a_stats)
    
    print(f"   Progression : {len(df)}/{len(df)} matchs... ✓")
    
    # Convertir en colonnes
    print("\n📊 Création des colonnes de features...")
    for key in home_stats[0].keys():
        df[f'Home_{key}'] = [s[key] for s in home_stats]
        df[f'Away_{key}'] = [s[key] for s in away_stats]
    
    # Afficher les features créées
    feature_cols = [col for col in df.columns if col.startswith('Home_') or col.startswith('Away_')]
    print(f"   Features créées : {len(feature_cols)}")
    print(f"   Liste : {feature_cols}")
    
    print(f"\n✅ Feature engineering terminé !")
    print(f"   Shape finale : {df.shape}\n")
    
    return df


def save_processed_data(df, output_dir='data/processed'):
    """
    Sauvegarde le DataFrame avec features dans le dossier processed
    
    Args:
        df (pd.DataFrame): DataFrame avec features
        output_dir (str): Dossier de sortie
        
    Returns:
        str: Chemin du fichier sauvegardé
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'features.csv')
    
    df.to_csv(output_path, index=False)
    print(f"💾 Features sauvegardées : {output_path}\n")
    
    return output_path


def make_features(interim_data_path='data/interim/clean_data.csv', 
                  output_dir='data/processed',
                  n_matches=5):
    """
    Pipeline complet de feature engineering
    
    Args:
        interim_data_path (str): Chemin vers données nettoyées
        output_dir (str): Dossier de sortie
        n_matches (int): Nombre de matchs pour les stats
        
    Returns:
        pd.DataFrame: DataFrame avec features
    """
    # Charger les données nettoyées
    print(f"📂 Chargement : {interim_data_path}")
    df = pd.read_csv(interim_data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"   {len(df)} matchs chargés\n")
    
    # Construire les features
    df = build_features(df, n_matches)
    
    # Sauvegarder
    save_processed_data(df, output_dir)
    
    return df


if __name__ == '__main__':
    # Exécution standalone
    df = make_features()
    print(f"✅ Pipeline terminé ! Features prêtes pour le ML.")