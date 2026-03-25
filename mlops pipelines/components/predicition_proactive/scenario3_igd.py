import os
import json
import pandas as pd
import numpy as np

# ============================================================
# scenario3_igd.py
# Scénario 3 : Prédiction Proactive / Evolution Temporelle IGD
# Méthode    : Time Series + Intervalle de Confiance
# Input      : datasets/processed/igd_scenario3_processed.csv
# Output     : components/scenario3/data/diagnostic_results_scenario3_igd.json
# ============================================================

BASE_DIR    = os.path.dirname(__file__)
INPUT_PATH  = os.path.join(BASE_DIR, "../../datasets/processed/igd_scenario3_processed.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/diagnostic_results_scenario3_igd.json")

# Seuil critique SNR pour DSL
SNR_THRESHOLD = 6.0

print("="*60)
print("🚀 SCÉNARIO 3 — PRÉDICTION PROACTIVE IGD (Time Series)")
print("="*60)

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
print("\n📥 Chargement du dataset preprocessé...")
df = pd.read_csv(INPUT_PATH)
print(f"✅ Dataset chargé : {len(df)} équipements")

# ============================================================
# FONCTIONS DE DIAGNOSTIC
# ============================================================

def get_tendance(slope: float) -> str:
    if slope >= 0.5:
        return "amélioration rapide"
    elif slope >= 0.1:
        return "amélioration légère"
    elif slope >= -0.1:
        return "stable"
    elif slope >= -0.5:
        return "dégradation légère"
    elif slope >= -1.0:
        return "dégradation modérée"
    else:
        return "dégradation rapide"


def get_action(niveau_risque: str, jours: float) -> str:
    if niveau_risque == "stable":
        return "Aucune action requise — SNR stable ou en amélioration"
    elif niveau_risque == "faible":
        return "Surveillance périodique recommandée"
    elif niveau_risque == "modere":
        return "Surveillance proactive — planifier une vérification"
    elif niveau_risque == "eleve":
        return "Intervention préventive recommandée dans les 7 jours"
    elif niveau_risque == "critique":
        return "Intervention urgente requise — seuil critique imminent"
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
        "Line degradation"          : "Risque élevé de dégradation de ligne DSL",
        "High Interference / Noise" : "Risque élevé d'interférences ou bruit électrique",
        "Crosstalk / Saturation"    : "Risque de saturation de zone ou crosstalk",
        "Configuration / Ports overload": "Risque de surcharge configuration ou port DSLAM",
        "NORMAL"                    : "Dégradation SNR sans cause identifiée"
    }

    for key, val in cause_map.items():
        if key.lower() in str(root_cause).lower():
            return val

    return "Risque de perturbation environnementale ou dégradation matérielle"


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
print(f"\n{'='*60}")
print(f"🔍 Analyse de {len(df)} équipements IGD...")
print(f"{'='*60}")

results     = []
critique    = 0
eleve       = 0
modere      = 0
faible      = 0
stable      = 0

snr_cols = [f"snr_T{i}" for i in range(1, 11)]

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

    # SNR evolution list
    snr_evolution = {f"snr_T{i}": round(row[f"snr_T{i}"], 2) for i in range(1, 11)}

    # Construire le résultat
    result = {
        "device_id"   : device_id,
        "health_score": int(health_score),
        "root_cause"  : root_cause,

        "snr_evolution": {
            **snr_evolution,
            "snr_first"  : round(snr_first, 2),
            "snr_last"   : round(snr_last, 2),
            "snr_mean"   : round(snr_mean, 2),
            "snr_std"    : round(snr_std, 2),
            "snr_min"    : round(snr_min, 2),
            "snr_max"    : round(snr_max, 2),
            "tendance"   : get_tendance(slope)
        },

        "prediction": {
            "slope"               : round(slope, 4),
            "jours_avant_seuil"   : round(jours, 2) if jours is not None else None,
            "snr_threshold"       : SNR_THRESHOLD,
            "niveau_risque"       : niveau_risque,
            "intervalle_confiance": {
                "lower": round(ci_lower, 2),
                "upper": round(ci_upper, 2)
            }
        },

        "diagnostic": {
            "message"          : get_message(niveau_risque, jours, snr_last),
            "risque_identifie" : get_risque_cause(root_cause, niveau_risque),
            "action_recommandee": get_action(niveau_risque, jours)
        }
    }

    results.append(result)

    # Compteurs
    if niveau_risque == "critique": critique += 1
    elif niveau_risque == "eleve":  eleve    += 1
    elif niveau_risque == "modere": modere   += 1
    elif niveau_risque == "faible": faible   += 1
    else:                           stable   += 1

    # Affichage
    jours_str = f"{abs(jours):.2f} mesures" if jours is not None else "N/A"
    print(f"  {device_id:15s} | slope={slope:+.3f} | SNR_last={snr_last:.2f} dB | "
          f"risque={niveau_risque:8s} | {jours_str}")

# ============================================================
# SAUVEGARDE
# ============================================================
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# ============================================================
# RÉSUMÉ FINAL
# ============================================================
print(f"\n{'='*60}")
print(f"✅ SCÉNARIO 3 IGD TERMINÉ")
print(f"{'='*60}")
print(f"  Total équipements    : {len(results)}")
print(f"  ─────────────────────────────────")
print(f"  🔴 Critique          : {critique}")
print(f"  🟠 Élevé             : {eleve}")
print(f"  🟡 Modéré            : {modere}")
print(f"  🟢 Faible            : {faible}")
print(f"  ✅ Stable            : {stable}")
print(f"  ─────────────────────────────────")
print(f"  Résultats sauvegardés : {OUTPUT_PATH}")

# Afficher quelques exemples critiques
critiques = [r for r in results if r["prediction"]["niveau_risque"] == "critique"]
if critiques:
    print(f"\n🚨 ÉQUIPEMENTS CRITIQUES ({len(critiques)}) :")
    print(f"{'─'*60}")
    for r in critiques[:5]:
        print(f"  {r['device_id']:15s} → {r['diagnostic']['message']}")
        print(f"  {'':15s}   {r['diagnostic']['action_recommandee']}")