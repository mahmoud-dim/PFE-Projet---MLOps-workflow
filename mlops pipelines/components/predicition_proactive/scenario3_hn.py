import os
import json
import boto3
import pandas as pd
from io import StringIO

# ============================================================
# scenario3_hn.py
# Scénario 3 : Prédiction Proactive Huawei/Nokia
# Input  : MinIO → datasets/processed/hn_scenario3_processed.csv
# Output : MinIO → results/scenario3/diagnostic_results_scenario3_hn.json
# ============================================================

MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "http://10.98.20.211:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
BUCKET_DATASETS  = "datasets"
BUCKET_RESULTS   = "results"

INPUT_KEY    = "processed/hn_scenario3_processed.csv"
OUTPUT_KEY   = "scenario3/diagnostic_results_scenario3_hn.json"
RX_THRESHOLD = -28.0

s3 = boto3.client(
    "s3",
    endpoint_url          = MINIO_ENDPOINT,
    aws_access_key_id     = MINIO_ACCESS_KEY,
    aws_secret_access_key = MINIO_SECRET_KEY
)

print("="*60)
print("🚀 SCÉNARIO 3 — PRÉDICTION PROACTIVE HUAWEI/NOKIA")
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
    if niveau_risque == "stable":   return "Aucune action requise — RxPower stable ou en amélioration"
    elif niveau_risque == "faible": return "Surveillance périodique recommandée"
    elif niveau_risque == "modere": return "Surveillance proactive — planifier une vérification optique"
    elif niveau_risque == "eleve":  return "Intervention préventive — vérifier connecteurs et fibre"
    elif niveau_risque == "critique": return "Intervention urgente requise — risque de rupture signal GPON"
    return "Surveillance recommandée"


def get_message(niveau_risque: str, jours, rx_last: float) -> str:
    if niveau_risque == "stable":
        return f"RxPower stable à {rx_last:.2f} dBm — aucun dépassement prévu"
    elif jours is not None and not pd.isna(jours):
        if jours <= 0:
            return f"⚠️  Seuil critique déjà dépassé (RxPower={rx_last:.2f} dBm < {RX_THRESHOLD} dBm)"
        else:
            return f"Dépassement du seuil critique prévu dans {abs(jours):.2f} mesures"
    return "Évolution RxPower indéterminée"


def get_risque_cause(root_cause: str, niveau_risque: str) -> str:
    if niveau_risque == "stable":
        return "Aucun risque détecté"
    cause_map = {
        "OPTICAL_AGING":        "Risque élevé de vieillissement matériel — laser GPON dégradé",
        "CRITICAL_FIBER_ISSUE": "Risque critique de rupture ou macro-courbure fibre optique",
        "ENVIRONMENTAL":        "Risque de perturbation environnementale (température, humidité)",
        "POWER_ISSUE":          "Risque d'instabilité électrique — alimentation défaillante",
        "NORMAL":               "Dégradation RxPower sans cause identifiée — surveillance requise"
    }
    for key, val in cause_map.items():
        if key.lower() in str(root_cause).lower():
            return val
    return "Risque de perturbation optique — vérifier connecteurs et câble fibre"


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
print(f"\n🔍 Analyse de {len(df)} équipements...")

results  = []
critique = eleve = modere = faible = stable = 0

for _, row in df.iterrows():
    device_id     = row["device_id"]
    slope         = row["slope"]
    rx_last       = row["rx_last"]
    rx_first      = row["rx_first"]
    rx_mean       = row["rx_mean"]
    rx_std        = row["rx_std"]
    rx_min        = row["rx_min"]
    rx_max        = row["rx_max"]
    ci_lower      = row["ci_lower"]
    ci_upper      = row["ci_upper"]
    niveau_risque = row["niveau_risque"]
    root_cause    = row["root_cause"]
    health_score  = row["health_score"]
    vendor        = row["vendor"]

    jours = row["jours_avant_seuil"]
    if pd.isna(jours):
        jours = None

    rx_evolution = {f"rx_T{i}": round(row[f"rx_T{i}"], 2) for i in range(1, 11)}

    result = {
        "device_id":    device_id,
        "vendor":       vendor,
        "health_score": int(health_score),
        "root_cause":   root_cause,
        "rx_evolution": {
            **rx_evolution,
            "rx_first": round(rx_first, 2), "rx_last":  round(rx_last, 2),
            "rx_mean":  round(rx_mean, 2),  "rx_std":   round(rx_std, 2),
            "rx_min":   round(rx_min, 2),   "rx_max":   round(rx_max, 2),
            "tendance": get_tendance(slope)
        },
        "prediction": {
            "slope":                round(slope, 4),
            "jours_avant_seuil":    round(jours, 2) if jours is not None else None,
            "rx_threshold":         RX_THRESHOLD,
            "niveau_risque":        niveau_risque,
            "intervalle_confiance": {
                "lower": round(ci_lower, 2),
                "upper": round(ci_upper, 2)
            }
        },
        "diagnostic": {
            "message":            get_message(niveau_risque, jours, rx_last),
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
    print(f"  {device_id:20s} | slope={slope:+.3f} | RxPower={rx_last:.2f} dBm | "
          f"risque={niveau_risque:8s} | {jours_str}")

# Sauvegarder dans MinIO
print(f"\n💾 Sauvegarde dans MinIO → {OUTPUT_KEY}")
s3.put_object(
    Bucket = BUCKET_RESULTS,
    Key    = OUTPUT_KEY,
    Body   = json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8")
)

print(f"\n{'='*60}")
print("✅ SCÉNARIO 3 HUAWEI/NOKIA TERMINÉ")
print(f"{'='*60}")
print(f"  Total          : {len(results)}")
print(f"  🔴 Critique    : {critique}")
print(f"  🟠 Élevé       : {eleve}")
print(f"  🟡 Modéré      : {modere}")
print(f"  🟢 Faible      : {faible}")
print(f"  ✅ Stable      : {stable}")
print(f"  MinIO          : {BUCKET_RESULTS}/{OUTPUT_KEY}")

critiques = [r for r in results if r["prediction"]["niveau_risque"] == "critique"]
if critiques:
    print(f"\n🚨 CRITIQUES ({len(critiques)}) :")
    for r in critiques[:5]:
        print(f"  {r['device_id']:20s} → {r['diagnostic']['message']}")
        print(f"  {'':20s}   {r['diagnostic']['action_recommandee']}")