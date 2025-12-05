"""
Prédiction sur un VRAI match du dataset
Ce script prend un match réel et montre ce que le modèle aurait prédit
"""
import pandas as pd
from src.models.predict_model import load_model, load_feature_names
from src.features.build_features import get_team_stats

print("="*70)
print("🎯 PRÉDICTION SUR UN VRAI MATCH DU DATASET")
print("="*70)
print()

# Charger les données
df = pd.read_csv('data/interim/clean_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Charger le modèle
model = load_model('models/randomforest.pkl')
feature_names = load_feature_names()

# ===================================================================
# Choisir un match spécifique (tu peux changer l'index)
# ===================================================================
match_index = 2500  # Un match récent

match = df.iloc[match_index]

print(f"📅 Date        : {match['Date'].strftime('%d/%m/%Y')}")
print(f"🏠 Domicile    : {match['HomeTeam']}")
print(f"✈️  Extérieur   : {match['AwayTeam']}")
print(f"⚽ Score réel  : {int(match['FTHG'])} - {int(match['FTAG'])}")
print(f"🏆 Résultat    : ", end="")
if match['FTR'] == 'H':
    print(f"Victoire {match['HomeTeam']} ✅")
elif match['FTR'] == 'A':
    print(f"Victoire {match['AwayTeam']} ✅")
else:
    print("Match nul ⚖️")

print()
print("-"*70)
print()

# Calculer les stats des équipes AVANT ce match (comme le modèle l'aurait fait)
home_stats_raw = get_team_stats(df, match['HomeTeam'], match['Date'], match['Saison'], n_matches=5)
away_stats_raw = get_team_stats(df, match['AwayTeam'], match['Date'], match['Saison'], n_matches=5)

print(f"📊 {match['HomeTeam']} - Stats des 5 matchs précédents :")
for key, val in home_stats_raw.items():
    print(f"   {key:15s} : {val:.2f}" if isinstance(val, float) else f"   {key:15s} : {val}")

print()

print(f"📊 {match['AwayTeam']} - Stats des 5 matchs précédents :")
for key, val in away_stats_raw.items():
    print(f"   {key:15s} : {val:.2f}" if isinstance(val, float) else f"   {key:15s} : {val}")

print()
print("-"*70)
print()

# Faire la prédiction
# Créer le vecteur de features
features = {}
for key in home_stats_raw.keys():
    features[f'Home_{key}'] = home_stats_raw[key]
    features[f'Away_{key}'] = away_stats_raw[key]

X = pd.DataFrame([features])[feature_names]

prediction = model.predict(X)[0]
probabilities = model.predict_proba(X)[0]

# Afficher la prédiction
labels = {0: f'Victoire {match["AwayTeam"]} (extérieur)', 
          1: 'Match nul', 
          2: f'Victoire {match["HomeTeam"]} (domicile)'}

print("🤖 CE QUE LE MODÈLE AURAIT PRÉDIT :")
print("="*70)
print(f"Prédiction : {labels[prediction]}")
print()
print("Probabilités :")
print(f"  Victoire {match['HomeTeam']:15s} (domicile) : {probabilities[2]:5.1%}  {'🎯' if prediction == 2 else ''}")
print(f"  Match nul                       : {probabilities[1]:5.1%}  {'🎯' if prediction == 1 else ''}")
print(f"  Victoire {match['AwayTeam']:15s} (extérieur): {probabilities[0]:5.1%}  {'🎯' if prediction == 0 else ''}")
print("="*70)
print()

# Vérifier si la prédiction était correcte
actual_result = match['FTR_encoded']
if prediction == actual_result:
    print("✅ PRÉDICTION CORRECTE ! Le modèle avait raison ! 🎉")
else:
    print("❌ PRÉDICTION INCORRECTE. Le modèle s'est trompé.")
    print(f"   Prédit : {labels[prediction]}")
    if actual_result == 0:
        print(f"   Réel   : Victoire {match['AwayTeam']} (extérieur)")
    elif actual_result == 1:
        print(f"   Réel   : Match nul")
    else:
        print(f"   Réel   : Victoire {match['HomeTeam']} (domicile)")

print()
print("="*70)
print("💡 NOTE : Le modèle utilise SEULEMENT les stats des 5 matchs")
print("         précédents, sans connaître le résultat final.")
print("="*70)