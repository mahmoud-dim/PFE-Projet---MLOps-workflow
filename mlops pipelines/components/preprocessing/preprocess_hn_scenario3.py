import os
import pandas as pd

# ============================================================
# preprocess_hn_scenario3.py
# Preprocessing Huawei/Nokia — Scénario 3 : Prédiction Proactive
# Métrique principale : rx_power_dbm (seuil critique = -28 dBm)
# Input  : datasets/raw/fibre_combined_realistic.csv
# Output : datasets/processed/hn_scenario3_processed.csv
# ============================================================

BASE_DIR    = os.path.dirname(__file__)
INPUT_PATH  = os.path.join(BASE_DIR, "../../datasets/raw/fibre_combined_realistic.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "../../datasets/processed/hn_scenario3_processed.csv")

# Seuil critique RxPower GPON
RX_THRESHOLD = -28.0

print("="*60)
print("🔧 PREPROCESSING — Huawei/Nokia Scénario 3 (Prédiction Proactive)")
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
# Étape 3 : Garder uniquement les colonnes nécessaires
# ----------------------------------------
cols_needed = ["device_id", "timestamp", "rx_power_dbm", "health_score", "root_cause", "vendor"]
df = df[cols_needed]

# ----------------------------------------
# Étape 4 : Trier par device_id + timestamp
# ----------------------------------------
print("\n🔃 Tri par device_id et timestamp...")
df = df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)

# ----------------------------------------
# Étape 5 : Numéroter les mesures T1 → T10
# ----------------------------------------
print("🔢 Numérotation des mesures T1 → T10...")
df["t_index"] = df.groupby("device_id").cumcount() + 1

# ----------------------------------------
# Étape 6 : Vérifier que chaque équipement a bien 10 mesures
# ----------------------------------------
counts  = df.groupby("device_id").size()
valid   = counts[counts == 10]
invalid = counts[counts != 10]

print(f"\n📊 Équipements avec 10 mesures : {len(valid)}")
if len(invalid) > 0:
    print(f"⚠️  Équipements avec mesures incomplètes : {len(invalid)}")
    df = df[df["device_id"].isin(valid.index)]
    print(f"✅ Après filtrage : {df['device_id'].nunique()} équipements valides")

# ----------------------------------------
# Étape 7 : Pivoter → une ligne par équipement
# ----------------------------------------
print("\n🔄 Pivotage : une ligne par équipement...")

# RxPower à chaque instant
rx_pivot = df.pivot(index="device_id", columns="t_index", values="rx_power_dbm")
rx_pivot.columns = [f"rx_T{i}" for i in rx_pivot.columns]

# Timestamps à chaque instant
ts_pivot = df.pivot(index="device_id", columns="t_index", values="timestamp")
ts_pivot.columns = [f"timestamp_T{i}" for i in ts_pivot.columns]

# Health score, root cause, vendor (dernière mesure = T10)
last_measure = df[df["t_index"] == 10][["device_id", "health_score", "root_cause", "vendor"]].set_index("device_id")

# Combiner tout
df_final = pd.concat([rx_pivot, ts_pivot, last_measure], axis=1).reset_index()

print(f"✅ Dataset pivoté : {len(df_final)} équipements")

# ----------------------------------------
# Étape 8 : Calculer slope et features temporelles
# ----------------------------------------
print("\n📈 Calcul des features temporelles...")

rx_cols = [f"rx_T{i}" for i in range(1, 11)]

# Slope : (RxPower_T10 - RxPower_T1) / 9 intervalles
# Note : RxPower est négatif, une baisse = dégradation (plus négatif)
df_final["slope"]      = (df_final["rx_T10"] - df_final["rx_T1"]) / 9
df_final["rx_mean"]    = df_final[rx_cols].mean(axis=1)
df_final["rx_std"]     = df_final[rx_cols].std(axis=1)
df_final["rx_min"]     = df_final[rx_cols].min(axis=1)
df_final["rx_max"]     = df_final[rx_cols].max(axis=1)
df_final["rx_last"]    = df_final["rx_T10"]
df_final["rx_first"]   = df_final["rx_T1"]

# Intervalle de confiance (95%)
df_final["ci_lower"]   = df_final["rx_last"] - 1.96 * df_final["rx_std"]
df_final["ci_upper"]   = df_final["rx_last"] + 1.96 * df_final["rx_std"]

# Seuil critique RxPower GPON
df_final["rx_threshold"] = RX_THRESHOLD

# Jours avant dépassement du seuil
# RxPower descend (devient plus négatif) → slope négatif = dégradation
def jours_avant_seuil(row):
    if row["slope"] >= 0:
        return None  # RxPower stable ou en amélioration
    return round((RX_THRESHOLD - row["rx_last"]) / row["slope"], 2)

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
print(f"  Slope moyen              : {df_final['slope'].mean():.3f} dBm/mesure")
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
print(f"✅ PREPROCESSING HUAWEI/NOKIA TERMINÉ")
print(f"{'='*60}")
print(f"  Input  : {INPUT_PATH}")
print(f"  Output : {OUTPUT_PATH}")
print(f"  Lignes : {len(df_final)} équipements")
print(f"\n💡 Prochaine étape : exécuter scenario3_hn.py")