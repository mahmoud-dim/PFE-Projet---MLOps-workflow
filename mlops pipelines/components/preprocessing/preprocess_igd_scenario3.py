import os
import pandas as pd
import boto3
from io import StringIO

# ============================================================
# preprocess_igd_scenario3.py
# Preprocessing IGD — Scénario 3 : Prédiction Proactive
# Input  : MinIO → datasets/raw/igd_realistic.csv
# Output : MinIO → datasets/processed/igd_scenario3_processed.csv
# ============================================================

MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "http://10.98.20.211:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
BUCKET_DATASETS  = "datasets"

INPUT_KEY     = "raw/igd_realistic.csv"
OUTPUT_KEY    = "processed/igd_scenario3_processed.csv"
SNR_THRESHOLD = 6.0

s3 = boto3.client(
    "s3",
    endpoint_url          = MINIO_ENDPOINT,
    aws_access_key_id     = MINIO_ACCESS_KEY,
    aws_secret_access_key = MINIO_SECRET_KEY
)

print("="*60)
print("🔧 PREPROCESSING — IGD Scénario 3 (Prédiction Proactive)")
print("="*60)

# Étape 1 : Charger depuis MinIO
print("\n📥 Chargement depuis MinIO...")
response = s3.get_object(Bucket=BUCKET_DATASETS, Key=INPUT_KEY)
df       = pd.read_csv(StringIO(response["Body"].read().decode("utf-8")))
print(f"✅ {len(df)} lignes, {df['device_id'].nunique()} équipements")

# Étape 2 : Timestamp + colonnes nécessaires
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df[["device_id", "timestamp", "snr_margin_down_db", "health_score", "root_cause", "city"]]

# Étape 3-4 : Trier + numéroter T1→T10
print("🔃 Tri et numérotation...")
df = df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)
df["t_index"] = df.groupby("device_id").cumcount() + 1

# Étape 5 : Filtrer équipements avec 10 mesures
counts = df.groupby("device_id").size()
valid  = counts[counts == 10]
df     = df[df["device_id"].isin(valid.index)]
print(f"✅ Équipements valides (10 mesures) : {df['device_id'].nunique()}")

# Étape 6 : Pivoter → une ligne par équipement
print("🔄 Pivotage...")
snr_pivot    = df.pivot(index="device_id", columns="t_index", values="snr_margin_down_db")
snr_pivot.columns = [f"snr_T{i}" for i in snr_pivot.columns]
ts_pivot     = df.pivot(index="device_id", columns="t_index", values="timestamp")
ts_pivot.columns = [f"timestamp_T{i}" for i in ts_pivot.columns]
last_measure = df[df["t_index"] == 10][["device_id", "health_score", "root_cause", "city"]].set_index("device_id") # added city column for scenario 3
df_final     = pd.concat([snr_pivot, ts_pivot, last_measure], axis=1).reset_index()
print(f"✅ {len(df_final)} équipements pivotés")

# Étape 7 : Features temporelles
snr_cols = [f"snr_T{i}" for i in range(1, 11)]
df_final["slope"]         = (df_final["snr_T10"] - df_final["snr_T1"]) / 9
df_final["snr_mean"]      = df_final[snr_cols].mean(axis=1)
df_final["snr_std"]       = df_final[snr_cols].std(axis=1)
df_final["snr_min"]       = df_final[snr_cols].min(axis=1)
df_final["snr_max"]       = df_final[snr_cols].max(axis=1)
df_final["snr_last"]      = df_final["snr_T10"]
df_final["snr_first"]     = df_final["snr_T1"]
df_final["ci_lower"]      = df_final["snr_last"] - 1.96 * df_final["snr_std"]
df_final["ci_upper"]      = df_final["snr_last"] + 1.96 * df_final["snr_std"]
df_final["snr_threshold"] = SNR_THRESHOLD

def jours_avant_seuil(row):
    if row["slope"] >= 0:
        return None
    return round((SNR_THRESHOLD - row["snr_last"]) / row["slope"], 2)

def niveau_risque(row):
    if row["slope"] >= 0:
        return "stable"
    j = row["jours_avant_seuil"]
    if j is None:    return "stable"
    if j <= 2:       return "critique"
    elif j <= 7:     return "eleve"
    elif j <= 30:    return "modere"
    else:            return "faible"

df_final["jours_avant_seuil"] = df_final.apply(jours_avant_seuil, axis=1)
df_final["niveau_risque"]     = df_final.apply(niveau_risque, axis=1)

# Résumé
print("\n📊 RÉSUMÉ :")
print(f"  Équipements     : {len(df_final)}")
print(f"  Slope moyen     : {df_final['slope'].mean():.3f} dB/mesure")
for niveau, count in df_final["niveau_risque"].value_counts().items():
    print(f"  {niveau:10s}  : {count}")

# Sauvegarder dans MinIO
print(f"\n💾 Sauvegarde dans MinIO → {OUTPUT_KEY}")
csv_buffer = StringIO()
df_final.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)
s3.put_object(Bucket=BUCKET_DATASETS, Key=OUTPUT_KEY, Body=csv_buffer.getvalue().encode("utf-8"))

print("\n✅ PREPROCESSING IGD SCÉNARIO 3 TERMINÉ")
print(f"  Output : {BUCKET_DATASETS}/{OUTPUT_KEY}")
print("💡 Prochaine étape : exécuter scenario3_igd.py")