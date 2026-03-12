import os
import json
import pandas as pd
from langchain_ollama import OllamaLLM

# ============================================================
# main_igd.py
# Pipeline RAG complet — Diagnostic Multi-Causal IGD (DSL/ADSL)
# LLM    : Ollama local (llama3.1:8b)
# Input 1: dataset CSV (métriques IGD)
# Input 2: JSON résultats scénario 1 (predicted_class)
# ============================================================

# ----------------------------------------
# Initialisation Ollama LLM
# ----------------------------------------
OLLAMA_MODEL = "llama3.1:8b"

llm = OllamaLLM(
    model=OLLAMA_MODEL,
    base_url="http://localhost:11434"
)

# ----------------------------------------
# Chargement Knowledge Base (9 causes IGD)
# ----------------------------------------
BASE_DIR     = os.path.dirname(__file__)
json_kb_path = os.path.join(BASE_DIR, "data/knowledge_base_igd.json")

with open(json_kb_path, "r", encoding="utf-8") as f:
    causes_data = json.load(f)

print(f"✅ Knowledge Base IGD chargée : {len(causes_data)} causes")
print(f"🤖 Modèle LLM                 : {OLLAMA_MODEL} (local)")


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Charge le dataset CSV IGD.
    Colonnes utilisées :
        device_id, downstream_curr_rate_kbps, downstream_max_rate_kbps,
        snr_margin_down_db, attenuation_down_db, crc_errors_total
    """
    df = pd.read_csv(csv_path)
    df.set_index("device_id", inplace=True)
    print(f"📊 Dataset IGD chargé : {len(df)} équipements au total")
    return df


def load_scenario1_results(json_path: str) -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    filtered = [
        p for p in predictions
        if p["prediction"]["predicted_class"] in ("degraded", "critical")
    ]

    print(f"📋 Résultats scénario 1    : {len(predictions)} équipements au total")
    print(f"⚠️  À diagnostiquer        : {len(filtered)} (degraded + critical)")
    print(f"✅ Normaux ignorés          : {len(predictions) - len(filtered)}")

    return filtered


# ============================================================
# ÉTAPE 1 : Features IGD → texte naturel
# ============================================================
def features_to_query(device_id, predicted_class,
                       downstream_curr, downstream_max,
                       snr_margin, attenuation, crc_errors):
    label_map = {
        "optimal":  "Normal",
        "degraded": "Dégradé",
        "critical": "Critique"
    }
    label = label_map.get(predicted_class, predicted_class)

    query = (
        f"Équipement IGD {device_id} en état {label}. "
        f"Débit descendant actuel={downstream_curr} kbps, "
        f"Débit descendant max={downstream_max} kbps, "
        f"SNR margin descendant={snr_margin} dB, "
        f"Atténuation descendante={attenuation} dB, "
        f"Erreurs CRC total={crc_errors}."
    )
    return query


# ============================================================
# ÉTAPE 2 : Génération diagnostic avec Ollama
# Envoie les 9 causes complètes au LLM → il choisit les 3 meilleures
# ============================================================
def generate_diagnosis(device_id, predicted_class, confidence, query):
    label_map = {
        "optimal":  "Normal",
        "degraded": "Dégradé",
        "critical": "Critique"
    }
    label = label_map.get(predicted_class, predicted_class)

    # ✅ Toutes les 9 causes envoyées directement au LLM
    context = ""
    for i, cause in enumerate(causes_data):
        context += (
            f"Cause {i+1}: {cause['Cause']}\n"
            f"Indicateurs: {', '.join(cause['Indicators'])}\n"
            f"Description: {cause['Description']}\n\n"
        )

    prompt = f"""Tu es un expert en diagnostic réseau télécom spécialisé en DSL/ADSL (équipements IGD).

Équipement analysé : {device_id}
État prédit        : {label} (confiance : {confidence*100:.1f}%)
Mesures IGD        : {query}

Voici les 9 causes possibles de dégradation DSL :
{context}

Analyse précisément les mesures IGD et identifie les 3 causes les plus probables
parmi les 9 causes ci-dessus. Chaque équipement a des métriques différentes,
donc le diagnostic doit être spécifique à ces mesures.

Réponds UNIQUEMENT en JSON valide avec ce format exact :

{{
  "device_id": "{device_id}",
  "predicted_class": "{predicted_class}",
  "health_label": "{label}",
  "diagnostic": {{
    "cause_principale": "nom exact de la cause la plus probable",
    "probabilites": {{
      "cause_1": {{ "nom": "...", "probabilite": "XX%" }},
      "cause_2": {{ "nom": "...", "probabilite": "XX%" }},
      "cause_3": {{ "nom": "...", "probabilite": "XX%" }}
    }},
    "explication": "explication courte basée sur les valeurs mesurées",
    "action_recommandee": "action concrète à effectuer"
  }}
}}

Important :
- Les probabilités des 3 causes doivent totaliser 100%
- Utilise les noms exacts des causes tels qu'ils apparaissent ci-dessus
- Base ton analyse sur les valeurs numériques des métriques IGD
- Réponds UNIQUEMENT avec le JSON, rien d'autre
"""

    print(f"  ⏳ Appel Ollama ({OLLAMA_MODEL})...")
    response = llm.invoke(prompt)
    return response


# ============================================================
# PIPELINE RAG POUR UN SEUL ÉQUIPEMENT IGD
# ============================================================
def rag_diagnosis_single(device_id, predicted_class, confidence, df_metrics):

    if device_id not in df_metrics.index:
        print(f"  ⚠️  {device_id} introuvable dans le dataset CSV — ignoré")
        return None

    row             = df_metrics.loc[device_id]
    downstream_curr = row["downstream_curr_rate_kbps"]
    downstream_max  = row["downstream_max_rate_kbps"]
    snr_margin      = row["snr_margin_down_db"]
    attenuation     = row["attenuation_down_db"]
    crc_errors      = row["crc_errors_total"]

    # Étape 1 : Features → texte
    query = features_to_query(
        device_id, predicted_class,
        downstream_curr, downstream_max,
        snr_margin, attenuation, crc_errors
    )

    # Étape 2 : Ollama analyse les 9 causes et choisit les 3 meilleures
    raw_response = generate_diagnosis(
        device_id, predicted_class, confidence, query
    )

    # Étape 3 : Parser le JSON
    try:
        start  = raw_response.find("{")
        end    = raw_response.rfind("}") + 1
        result = json.loads(raw_response[start:end])

        # Afficher les 3 causes
        probs = result["diagnostic"]["probabilites"]
        print(f"  ✅ Cause principale : {result['diagnostic']['cause_principale']}")
        print(f"     ├─ {probs['cause_1']['nom']} : {probs['cause_1']['probabilite']}")
        print(f"     ├─ {probs['cause_2']['nom']} : {probs['cause_2']['probabilite']}")
        print(f"     └─ {probs['cause_3']['nom']} : {probs['cause_3']['probabilite']}")

        return result

    except json.JSONDecodeError:
        print(f"  ⚠️  JSON non parseable pour {device_id}")
        return {"device_id": device_id, "raw_response": raw_response}


# ============================================================
# PIPELINE COMPLET
# ============================================================
def run_full_pipeline(csv_path: str, scenario1_json_path: str, output_path: str):

    print("\n" + "="*60)
    print("🚀 PIPELINE RAG — DIAGNOSTIC MULTI-CAUSAL IGD (DSL/ADSL)")
    print("="*60)

    df_metrics = load_dataset(csv_path)
    scenario1  = load_scenario1_results(scenario1_json_path)

    print(f"\n{'='*60}")
    print(f"🔍 Lancement du diagnostic sur {len(scenario1)} équipements...")
    print(f"{'='*60}")

    results = []
    success = 0
    errors  = 0

    for i, prediction in enumerate(scenario1):
        device_id       = prediction["device_id"]
        predicted_class = prediction["prediction"]["predicted_class"]
        confidence      = prediction["prediction"]["confidence"]

        print(f"\n[{i+1}/{len(scenario1)}] {device_id} — {predicted_class.upper()}")

        result = rag_diagnosis_single(
            device_id, predicted_class, confidence, df_metrics
        )

        if result:
            result["scenario1_confidence"] = confidence
            result["scenario1_risk_level"] = prediction["health_metrics"]["risk_level"]
            result["scenario1_action"]     = prediction["decision_support"]["recommended_action"]
            results.append(result)
            success += 1
        else:
            errors += 1

    # Sauvegarde
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✅ PIPELINE IGD TERMINÉ")
    print(f"{'='*60}")
    print(f"  Équipements diagnostiqués  : {success}")
    print(f"  Introuvables (erreurs)     : {errors}")
    print(f"  Résultats sauvegardés      : {output_path}")

    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    CSV_PATH       = os.path.join(BASE_DIR, "../../datasets/processed/igd_scenario_1.csv")
    SCENARIO1_PATH = os.path.join(BASE_DIR, "../healthscore_training/evaluate/igd_operational_predictions.json")
    OUTPUT_PATH    = os.path.join(BASE_DIR, "data/diagnostic_results_igd.json")

    results = run_full_pipeline(
        csv_path            = CSV_PATH,
        scenario1_json_path = SCENARIO1_PATH,
        output_path         = OUTPUT_PATH
    )

    # Résumé final
    print(f"\n📊 RÉSUMÉ DES DIAGNOSTICS IGD :")
    print(f"{'─'*60}")
    for r in results:
        if "diagnostic" in r:
            probs = r["diagnostic"]["probabilites"]
            print(
                f"  {r['device_id']:20s} [{r['predicted_class'].upper():8s}] → "
                f"{probs['cause_1']['nom']} ({probs['cause_1']['probabilite']}) | "
                f"{probs['cause_2']['nom']} ({probs['cause_2']['probabilite']}) | "
                f"{probs['cause_3']['nom']} ({probs['cause_3']['probabilite']})"
            )