import os
import json
import pandas as pd
import numpy as np

# ============================================================
# scenario3_hn.py
# Scénario 3 : Prédiction Proactive / Evolution Temporelle
# Équipements : Huawei HG8145V5 + Nokia G-2425G-A (GPON)
# Métrique    : rx_power_dbm (seuil critique = -28 dBm)
# Input       : datasets/processed/hn_scenario3_processed.csv
# Output      : components/scenario3/data/diagnostic_results_scenario3_hn.json
# ============================================================

BASE_DIR    = os.path.dirname(__file__)
INPUT_PATH  = os.path.join(BASE_DIR, "../../datasets/processed/hn_scenario3_processed.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/diagnostic_results_scenario3_hn.json")

# Seuil critique RxPower GPON
RX_THRESHOLD = -28.0

print("="*60)
print("🚀 SCÉNARIO 3 — PRÉDICTION PROACTIVE HUAWEI/NOKIA (Time Series)")
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
    """
    Pour RxPower (valeurs négatives) :
    slope négatif = RxPower baisse = dégradation
    slope positif = RxPower monte = amélioration
    """
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


def get_action(niveau_risque: str, jours) -> str:
    if niveau_risque == "stable":
        return "Aucune action requise — RxPower stable ou en amélioration"
    elif niveau_risque == "faible":
        return "Surveillance périodique recommandée"
    elif niveau_risque == "modere":
        return "Surveillance proactive — planifier une vérification optique"
    elif niveau_risque == "eleve":
        return "Intervention préventive recommandée — vérifier connecteurs et fibre"
    elif niveau_risque == "critique":
        return "Intervention urgente requise — risque de rupture signal GPON"
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
        "OPTICAL_AGING"      : "Risque élevé de vieillissement matériel — laser GPON dégradé",
        "CRITICAL_FIBER_ISSUE": "Risque critique de rupture ou macro-courbure fibre optique",
        "ENVIRONMENTAL"      : "Risque de perturbation environnementale (température, humidité)",
        "POWER_ISSUE"        : "Risque d'instabilité électrique — alimentation défaillante",
        "NORMAL"             : "Dégradation RxPower sans cause identifiée — surveillance requise"
    }

    for key, val in cause_map.items():
        if key.lower() in str(root_cause).lower():
            return val

    return "Risque de perturbation optique — vérifier connecteurs et câble fibre"


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
print(f"\n{'='*60}")
print(f"🔍 Analyse de {len(df)} équipements Huawei/Nokia...")
print(f"{'='*60}")

results  = []
critique = 0
eleve    = 0
modere   = 0
faible   = 0
stable   = 0

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

    # RxPower evolution
    rx_evolution = {f"rx_T{i}": round(row[f"rx_T{i}"], 2) for i in range(1, 11)}

    result = {
        "device_id"   : device_id,
        "vendor"      : vendor,
        "health_score": int(health_score),
        "root_cause"  : root_cause,

        "rx_evolution": {
            **rx_evolution,
            "rx_first" : round(rx_first, 2),
            "rx_last"  : round(rx_last, 2),
            "rx_mean"  : round(rx_mean, 2),
            "rx_std"   : round(rx_std, 2),
            "rx_min"   : round(rx_min, 2),
            "rx_max"   : round(rx_max, 2),
            "tendance" : get_tendance(slope)
        },

        "prediction": {
            "slope"               : round(slope, 4),
            "jours_avant_seuil"   : round(jours, 2) if jours is not None else None,
            "rx_threshold"        : RX_THRESHOLD,
            "niveau_risque"       : niveau_risque,
            "intervalle_confiance": {
                "lower": round(ci_lower, 2),
                "upper": round(ci_upper, 2)
            }
        },

        "diagnostic": {
            "message"           : get_message(niveau_risque, jours, rx_last),
            "risque_identifie"  : get_risque_cause(root_cause, niveau_risque),
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
    print(f"  {device_id:20s} | slope={slope:+.3f} | RxPower={rx_last:.2f} dBm | "
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
print(f"✅ SCÉNARIO 3 HUAWEI/NOKIA TERMINÉ")
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
        print(f"  {r['device_id']:20s} → {r['diagnostic']['message']}")
        print(f"  {'':20s}   {r['diagnostic']['action_recommandee']}")