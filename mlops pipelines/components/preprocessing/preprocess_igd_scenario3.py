import os
import pandas as pd

# ============================================================
# preprocess_igd_scenario3.py
# Preprocessing IGD — Scénario 3 : Prédiction Proactive
# Input  : datasets/raw/igd_realistic.csv
# Output : datasets/processed/igd_scenario3_processed.csv
# ============================================================

BASE_DIR   = os.path.dirname(__file__)
INPUT_PATH = os.path.join(BASE_DIR, "../../datasets/raw/igd_realistic.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "../../datasets/processed/igd_scenario3_processed.csv")

print("="*60)
print("🔧 PREPROCESSING — IGD Scénario 3 (Prédiction Proactive)")
print("="*60)

# ----------------------------------------
# Étape 1 : Charger le dataset brut
# ----------------------------------------
print("\n📥 Chargement du dataset brut...")
df = pd.read_csv(INPUT_PATH)
print(f"✅ Dataset chargé : {len(df)} lignes, {df['device_id'].nunique()} équipements")

# ----------------------------------------
# Étape 2 : Convertir timestamp en datetime
# ----------------------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ----------------------------------------
# Étape 3 : Trier par device_id + timestamp
# ----------------------------------------
print("\n🔃 Tri par device_id et timestamp...")
df = df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)

# ----------------------------------------
# Étape 4 : Garder uniquement les colonnes nécessaires
# ----------------------------------------
cols_needed = ["device_id", "timestamp", "snr_margin_down_db", "health_score", "root_cause"]
df = df[cols_needed]

# ----------------------------------------
# Étape 5 : Numéroter les mesures T1 → T10
# ----------------------------------------
print("🔢 Numérotation des mesures T1 → T10...")
df["t_index"] = df.groupby("device_id").cumcount() + 1

# ----------------------------------------
# Étape 6 : Vérifier que chaque équipement a bien 10 mesures
# ----------------------------------------
counts = df.groupby("device_id").size()
valid  = counts[counts == 10]
invalid = counts[counts != 10]

print(f"\n📊 Équipements avec 10 mesures : {len(valid)}")
if len(invalid) > 0:
    print(f"⚠️  Équipements avec mesures incomplètes : {len(invalid)}")
    print(invalid)
    # Garder uniquement les équipements avec exactement 10 mesures
    df = df[df["device_id"].isin(valid.index)]
    print(f"✅ Après filtrage : {df['device_id'].nunique()} équipements valides")

# ----------------------------------------
# Étape 7 : Pivoter → une ligne par équipement
# ----------------------------------------
print("\n🔄 Pivotage : une ligne par équipement...")

# SNR margin à chaque instant
snr_pivot = df.pivot(index="device_id", columns="t_index", values="snr_margin_down_db")
snr_pivot.columns = [f"snr_T{i}" for i in snr_pivot.columns]

# Timestamps à chaque instant
ts_pivot = df.pivot(index="device_id", columns="t_index", values="timestamp")
ts_pivot.columns = [f"timestamp_T{i}" for i in ts_pivot.columns]

# Health score et root cause (dernière mesure = T10)
last_measure = df[df["t_index"] == 10][["device_id", "health_score", "root_cause"]].set_index("device_id")

# Combiner tout
df_final = pd.concat([snr_pivot, ts_pivot, last_measure], axis=1).reset_index()

print(f"✅ Dataset pivoté : {len(df_final)} équipements")
print(f"   Colonnes : {list(df_final.columns)}")

# ----------------------------------------
# Étape 8 : Calculer slope et intervalle confiance
# ----------------------------------------
print("\n📈 Calcul des features temporelles...")

# Slope : (SNR_T10 - SNR_T1) / 9 intervalles
df_final["slope"] = (df_final["snr_T10"] - df_final["snr_T1"]) / 9

# SNR moyen sur T1-T10
snr_cols = [f"snr_T{i}" for i in range(1, 11)]
df_final["snr_mean"]   = df_final[snr_cols].mean(axis=1)
df_final["snr_std"]    = df_final[snr_cols].std(axis=1)
df_final["snr_min"]    = df_final[snr_cols].min(axis=1)
df_final["snr_max"]    = df_final[snr_cols].max(axis=1)
df_final["snr_last"]   = df_final["snr_T10"]   # dernière valeur connue
df_final["snr_first"]  = df_final["snr_T1"]    # première valeur

# Intervalle de confiance (95%) sur la dernière valeur
df_final["ci_lower"] = df_final["snr_last"] - 1.96 * df_final["snr_std"]
df_final["ci_upper"] = df_final["snr_last"] + 1.96 * df_final["snr_std"]

# Seuil critique DSL
SNR_THRESHOLD = 6.0
df_final["snr_threshold"] = SNR_THRESHOLD

# Jours avant dépassement du seuil
# Si slope négatif → dégradation → on calcule le temps avant seuil
def jours_avant_seuil(row):
    if row["slope"] >= 0:
        return None  # SNR stable ou en amélioration
    return round((SNR_THRESHOLD - row["snr_last"]) / row["slope"], 2)

df_final["jours_avant_seuil"] = df_final.apply(jours_avant_seuil, axis=1)

# Niveau de risque
def niveau_risque(row):
    if row["slope"] >= 0:
        return "stable"
    j = row["jours_avant_seuil"]
    if j is None:
        return "stable"
    if j <= 2:
        return "critique"
    elif j <= 7:
        return "eleve"
    elif j <= 30:
        return "modere"
    else:
        return "faible"

df_final["niveau_risque"] = df_final.apply(niveau_risque, axis=1)

# ----------------------------------------
# Étape 9 : Résumé statistique
# ----------------------------------------
print(f"\n📊 RÉSUMÉ :")
print(f"{'─'*50}")
print(f"  Équipements total        : {len(df_final)}")
print(f"  Slope moyen              : {df_final['slope'].mean():.3f} dB/mesure")
print(f"  En dégradation (slope<0) : {(df_final['slope'] < 0).sum()}")
print(f"  En amélioration (slope>0): {(df_final['slope'] > 0).sum()}")
print(f"\n  Niveaux de risque :")
for niveau, count in df_final["niveau_risque"].value_counts().items():
    print(f"    {niveau:10s} : {count} équipements")

# ----------------------------------------
# Étape 10 : Sauvegarder
# ----------------------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_final.to_csv(OUTPUT_PATH, index=False)

print(f"\n{'='*60}")
print(f"✅ PREPROCESSING TERMINÉ")
print(f"{'='*60}")
print(f"  Input  : {INPUT_PATH}")
print(f"  Output : {OUTPUT_PATH}")
print(f"  Lignes : {len(df_final)} équipements")
print(f"\n💡 Prochaine étape : exécuter scenario3_igd.py")