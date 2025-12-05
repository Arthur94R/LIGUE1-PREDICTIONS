"""
Script pour générer les visualisations du projet
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix
import numpy as np

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

def create_output_dir(output_dir='reports/figures'):
    """Créer le dossier de sortie s'il n'existe pas"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_class_distribution(df, output_dir):
    """Distribution des résultats"""
    print("📊 Génération : Distribution des résultats...")
    
    plt.figure(figsize=(10, 6))
    counts = df['FTR'].value_counts()
    colors = {'H': '#2ecc71', 'D': '#f39c12', 'A': '#e74c3c'}
    
    bars = plt.bar(counts.index, counts.values, 
                   color=[colors[x] for x in counts.index])
    
    plt.title('Distribution des résultats (toutes saisons)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Résultat', fontsize=12)
    plt.ylabel('Nombre de matchs', fontsize=12)
    plt.xticks([0, 1, 2], ['Home win (H)', 'Draw (D)', 'Away win (A)'])
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_class_distribution.png'), dpi=300)
    plt.close()
    print("   ✓ Sauvegardé : 01_class_distribution.png")


def plot_goals_distribution(df, output_dir):
    """Distribution des buts"""
    print("📊 Génération : Distribution des buts...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Buts à domicile
    axes[0].hist(df['FTHG'], bins=range(0, df['FTHG'].max()+2), 
                 color='#3498db', edgecolor='black', alpha=0.7)
    axes[0].set_title('Buts marqués à domicile', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Nombre de buts', fontsize=11)
    axes[0].set_ylabel('Fréquence', fontsize=11)
    axes[0].axvline(df['FTHG'].mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Moyenne: {df["FTHG"].mean():.2f}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Buts à l'extérieur
    axes[1].hist(df['FTAG'], bins=range(0, df['FTAG'].max()+2), 
                 color='#e74c3c', edgecolor='black', alpha=0.7)
    axes[1].set_title("Buts marqués à l'extérieur", fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Nombre de buts', fontsize=11)
    axes[1].set_ylabel('Fréquence', fontsize=11)
    axes[1].axvline(df['FTAG'].mean(), color='blue', linestyle='--', 
                    linewidth=2, label=f'Moyenne: {df["FTAG"].mean():.2f}')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_goals_distribution.png'), dpi=300)
    plt.close()
    print("   ✓ Sauvegardé : 02_goals_distribution.png")


def plot_results_by_season(df, output_dir):
    """Résultats par saison"""
    print("📊 Génération : Résultats par saison...")
    
    results_by_season = df.groupby(['Saison', 'FTR']).size().unstack(fill_value=0)
    
    plt.figure(figsize=(12, 6))
    results_by_season.plot(kind='bar', stacked=True, 
                          color=['#e74c3c', '#f39c12', '#2ecc71'])
    plt.title('Distribution des résultats par saison', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Saison', fontsize=12)
    plt.ylabel('Nombre de matchs', fontsize=12)
    plt.legend(['Away win', 'Draw', 'Home win'], loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_results_by_season.png'), dpi=300)
    plt.close()
    print("   ✓ Sauvegardé : 03_results_by_season.png")


def plot_top_teams(df, output_dir):
    """Top équipes avec le plus de victoires"""
    print("📊 Génération : Top équipes...")
    
    home_wins = df[df['FTR'] == 'H']['HomeTeam'].value_counts()
    away_wins = df[df['FTR'] == 'A']['AwayTeam'].value_counts()
    total_wins = home_wins.add(away_wins, fill_value=0).sort_values(ascending=True)
    
    plt.figure(figsize=(12, 8))
    top_teams = total_wins.tail(15)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_teams)))
    plt.barh(range(len(top_teams)), top_teams.values, color=colors)
    plt.yticks(range(len(top_teams)), top_teams.index)
    plt.title('Top 15 des équipes avec le plus de victoires', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Nombre de victoires', fontsize=12)
    plt.ylabel('Équipe', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_top_teams.png'), dpi=300)
    plt.close()
    print("   ✓ Sauvegardé : 04_top_teams.png")


def plot_confusion_matrices(output_dir):
    """Matrices de confusion pour tous les modèles"""
    print("📊 Génération : Matrices de confusion...")
    
    # Charger les données de test
    df = pd.read_csv('data/processed/features.csv')
    feature_columns = [col for col in df.columns if col.startswith('Home_') or col.startswith('Away_')]
    X = df[feature_columns]
    y = df['FTR_encoded']
    
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'RandomForest': 'models/randomforest.pkl',
        'RandomForest Balanced': 'models/randomforest_balanced.pkl',
        'XGBoost': 'models/xgboost.pkl',
        'XGBoost Balanced': 'models/xgboost_balanced.pkl'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, (name, path) in enumerate(models.items()):
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Away win', 'Draw', 'Home win'],
                   yticklabels=['Away win', 'Draw', 'Home win'],
                   cbar_kws={'label': 'Nombre de matchs'})
        axes[idx].set_title(f'Matrice de confusion - {name}', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Vraie classe', fontsize=10)
        axes[idx].set_xlabel('Classe prédite', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_confusion_matrices.png'), dpi=300)
    plt.close()
    print("   ✓ Sauvegardé : 05_confusion_matrices.png")


def plot_feature_importance(output_dir):
    """Importance des features"""
    print("📊 Génération : Importance des features...")
    
    # Charger le meilleur modèle
    model = joblib.load('models/xgboost_balanced.pkl')
    
    with open('models/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    feature_importance = model.feature_importances_
    features_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(features_df)))
    plt.barh(features_df['Feature'], features_df['Importance'], color=colors)
    plt.xlabel('Importance', fontsize=12)
    plt.title('Importance des features (XGBoost Balanced)', 
              fontsize=16, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_feature_importance.png'), dpi=300)
    plt.close()
    print("   ✓ Sauvegardé : 06_feature_importance.png")


def plot_model_comparison(output_dir):
    """Comparaison des performances des modèles"""
    print("📊 Génération : Comparaison des modèles...")
    
    # Charger les données de test
    df = pd.read_csv('data/processed/features.csv')
    feature_columns = [col for col in df.columns if col.startswith('Home_') or col.startswith('Away_')]
    X = df[feature_columns]
    y = df['FTR_encoded']
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'RandomForest': 'models/randomforest.pkl',
        'RF Balanced': 'models/randomforest_balanced.pkl',
        'XGBoost': 'models/xgboost.pkl',
        'XGB Balanced': 'models/xgboost_balanced.pkl'
    }
    
    results = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}
    
    for name, path in models.items():
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        
        results['Model'].append(name)
        results['Accuracy'].append(accuracy_score(y_test, y_pred))
        results['Precision'].append(precision_score(y_test, y_pred, average='weighted'))
        results['Recall'].append(recall_score(y_test, y_pred, average='weighted'))
        results['F1-Score'].append(f1_score(y_test, y_pred, average='weighted'))
    
    df_results = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df_results['Model']))
    width = 0.2
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, df_results[metric], width, 
               label=metric, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Modèle', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comparaison des performances des modèles', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df_results['Model'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_model_comparison.png'), dpi=300)
    plt.close()
    print("   ✓ Sauvegardé : 07_model_comparison.png")


def generate_all_visualizations():
    """Générer toutes les visualisations"""
    print("="*60)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("="*60 + "\n")
    
    # Créer le dossier de sortie
    output_dir = create_output_dir()
    
    # Charger les données
    print("📂 Chargement des données...")
    df = pd.read_csv('data/interim/clean_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"   {len(df)} matchs chargés\n")
    
    # Générer les graphiques
    plot_class_distribution(df, output_dir)
    plot_goals_distribution(df, output_dir)
    plot_results_by_season(df, output_dir)
    plot_top_teams(df, output_dir)
    plot_confusion_matrices(output_dir)
    plot_feature_importance(output_dir)
    plot_model_comparison(output_dir)
    
    print(f"\n✅ Toutes les visualisations ont été générées dans '{output_dir}/'")
    print(f"   Total : 7 graphiques")


if __name__ == '__main__':
    generate_all_visualizations()