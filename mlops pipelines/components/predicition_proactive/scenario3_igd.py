import os
import json
import boto3
import pandas as pd
import numpy as np
from io import StringIO

# ============================================================
# scenario3_igd.py
# Scénario 3 : Prédiction Proactive IGD (Time Series)
# Input  : MinIO → datasets/processed/igd_scenario3_processed.csv
# Output : MinIO → results/scenario3/diagnostic_results_scenario3_igd.json
# ============================================================

MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "http://10.98.20.211:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
BUCKET_DATASETS  = "datasets"
BUCKET_RESULTS   = "results"

INPUT_KEY     = "processed/igd_scenario3_processed.csv"
OUTPUT_KEY    = "scenario3/diagnostic_results_scenario3_igd.json"
SNR_THRESHOLD = 6.0

s3 = boto3.client(
    "s3",
    endpoint_url          = MINIO_ENDPOINT,
    aws_access_key_id     = MINIO_ACCESS_KEY,
    aws_secret_access_key = MINIO_SECRET_KEY
)

print("="*60)
print("🚀 SCÉNARIO 3 — PRÉDICTION PROACTIVE IGD (Time Series)")
print("="*60)

# Chargement depuis MinIO
print("\n📥 Chargement du dataset depuis MinIO...")
response = s3.get_object(Bucket=BUCKET_DATASETS, Key=INPUT_KEY)
df       = pd.read_csv(StringIO(response["Body"].read().decode("utf-8")))
print(f"✅ Dataset chargé : {len(df)} équipements")

# ============================================================
# FONCTIONS
# ============================================================

def get_tendance(slope: float) -> str:
    if slope >= 0.5:    return "amélioration rapide"
    elif slope >= 0.1:  return "amélioration légère"
    elif slope >= -0.1: return "stable"
    elif slope >= -0.5: return "dégradation légère"
    elif slope >= -1.0: return "dégradation modérée"
    else:               return "dégradation rapide"


def get_action(niveau_risque: str, jours) -> str:
    if niveau_risque == "stable":    return "Aucune action requise — SNR stable ou en amélioration"
    elif niveau_risque == "faible":  return "Surveillance périodique recommandée"
    elif niveau_risque == "modere":  return "Surveillance proactive — planifier une vérification"
    elif niveau_risque == "eleve":   return "Intervention préventive recommandée dans les 7 jours"
    elif niveau_risque == "critique": return "Intervention urgente requise — seuil critique imminent"
    return "Surveillance recommandée"


def get_message(niveau_risque: str, jours, snr_last: float) -> str:
    if niveau_risque == "stable":
        return f"SNR stable à {snr_last:.2f} dB — aucun dépassement prévu"
    elif jours is not None and not pd.isna(jours):
        if jours <= 0:
            return f"⚠️  Seuil critique déjà dépassé (SNR={snr_last:.2f} dB < {SNR_THRESHOLD} dB)"
        else:
            return f"Dépassement du seuil critique prévu dans {abs(jours):.2f} mesures"
    return "Évolution SNR indéterminée"


def get_risque_cause(root_cause: str, niveau_risque: str) -> str:
    if niveau_risque == "stable":
        return "Aucun risque détecté"
    cause_map = {
        "Line degradation":           "Risque élevé de dégradation de ligne DSL",
        "High Interference / Noise":  "Risque élevé d'interférences ou bruit électrique",
        "Crosstalk / Saturation":     "Risque de saturation de zone ou crosstalk",
        "Configuration / Ports overload": "Risque de surcharge configuration ou port DSLAM",
        "NORMAL":                     "Dégradation SNR sans cause identifiée"
    }
    for key, val in cause_map.items():
        if key.lower() in str(root_cause).lower():
            return val
    return "Risque de perturbation environnementale ou dégradation matérielle"


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
print(f"\n🔍 Analyse de {len(df)} équipements IGD...")

results  = []
critique = eleve = modere = faible = stable = 0

for _, row in df.iterrows():
    device_id     = row["device_id"]
    slope         = row["slope"]
    snr_last      = row["snr_last"]
    snr_first     = row["snr_first"]
    snr_mean      = row["snr_mean"]
    snr_std       = row["snr_std"]
    snr_min       = row["snr_min"]
    snr_max       = row["snr_max"]
    ci_lower      = row["ci_lower"]
    ci_upper      = row["ci_upper"]
    niveau_risque = row["niveau_risque"]
    root_cause    = row["root_cause"]
    health_score  = row["health_score"]

    jours = row["jours_avant_seuil"]
    if pd.isna(jours):
        jours = None

    snr_evolution = {f"snr_T{i}": round(row[f"snr_T{i}"], 2) for i in range(1, 11)}

    result = {
        "device_id":    device_id,
        "health_score": int(health_score),
        "root_cause":   root_cause,
        "snr_evolution": {
            **snr_evolution,
            "snr_first": round(snr_first, 2), "snr_last":  round(snr_last, 2),
            "snr_mean":  round(snr_mean, 2),  "snr_std":   round(snr_std, 2),
            "snr_min":   round(snr_min, 2),   "snr_max":   round(snr_max, 2),
            "tendance":  get_tendance(slope)
        },
        "prediction": {
            "slope":                round(slope, 4),
            "jours_avant_seuil":    round(jours, 2) if jours is not None else None,
            "snr_threshold":        SNR_THRESHOLD,
            "niveau_risque":        niveau_risque,
            "intervalle_confiance": {
                "lower": round(ci_lower, 2),
                "upper": round(ci_upper, 2)
            }
        },
        "diagnostic": {
            "message":            get_message(niveau_risque, jours, snr_last),
            "risque_identifie":   get_risque_cause(root_cause, niveau_risque),
            "action_recommandee": get_action(niveau_risque, jours)
        }
    }

    results.append(result)

    if niveau_risque == "critique": critique += 1
    elif niveau_risque == "eleve":  eleve    += 1
    elif niveau_risque == "modere": modere   += 1
    elif niveau_risque == "faible": faible   += 1
    else:                           stable   += 1

    jours_str = f"{abs(jours):.2f} mesures" if jours is not None else "N/A"
    print(f"  {device_id:15s} | slope={slope:+.3f} | SNR={snr_last:.2f} dB | "
          f"risque={niveau_risque:8s} | {jours_str}")

# Sauvegarder dans MinIO
print(f"\n💾 Sauvegarde dans MinIO → {OUTPUT_KEY}")
s3.put_object(
    Bucket = BUCKET_RESULTS,
    Key    = OUTPUT_KEY,
    Body   = json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8")
)

print(f"\n{'='*60}")
print(f"✅ SCÉNARIO 3 IGD TERMINÉ")
print(f"{'='*60}")
print(f"  Total        : {len(results)}")
print(f"  🔴 Critique  : {critique}")
print(f"  🟠 Élevé     : {eleve}")
print(f"  🟡 Modéré    : {modere}")
print(f"  🟢 Faible    : {faible}")
print(f"  ✅ Stable    : {stable}")
print(f"  MinIO        : {BUCKET_RESULTS}/{OUTPUT_KEY}")

critiques = [r for r in results if r["prediction"]["niveau_risque"] == "critique"]
if critiques:
    print(f"\n🚨 CRITIQUES ({len(critiques)}) :")
    for r in critiques[:5]:
        print(f"  {r['device_id']:15s} → {r['diagnostic']['message']}")
        print(f"  {'':15s}   {r['diagnostic']['action_recommandee']}")