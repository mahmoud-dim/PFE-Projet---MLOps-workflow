import os
import json
import boto3
import pandas as pd
from io import StringIO
from langchain_ollama import OllamaLLM

# ============================================================
# main_igd.py — RAG Diagnostic IGD
# Input  : MinIO → datasets/processed/IGD_scenario_1.csv
#          MinIO → results/scenario1/IGD_operational_predictions.json
#          MinIO → knowledge-base/knowledge_base_igd.json
# Output : MinIO → results/scenario2/diagnostic_results_igd.json
# ============================================================

MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "http://10.98.20.211:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
OLLAMA_MODEL     = "llama3.1:8b"

BUCKET_DATASETS = "datasets"
BUCKET_KB       = "knowledge-base"
BUCKET_RESULTS  = "results"

CSV_KEY       = "processed/IGD_scenario_1.csv"
SCENARIO1_KEY = "scenario1/IGD_operational_predictions.json"
KB_KEY        = "knowledge_base_igd.json"
OUTPUT_KEY    = "scenario2/diagnostic_results_igd.json"

s3 = boto3.client(
    "s3",
    endpoint_url          = MINIO_ENDPOINT,
    aws_access_key_id     = MINIO_ACCESS_KEY,
    aws_secret_access_key = MINIO_SECRET_KEY
)

llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

print("📂 Chargement Knowledge Base IGD depuis MinIO...")
response    = s3.get_object(Bucket=BUCKET_KB, Key=KB_KEY)
causes_data = json.loads(response["Body"].read().decode("utf-8"))
print(f"✅ Knowledge Base IGD chargée : {len(causes_data)} causes")
print(f"🤖 Modèle LLM                 : {OLLAMA_MODEL} (local)")


def load_dataset():
    print("📥 Loading dataset IGD from MinIO...")
    response = s3.get_object(Bucket=BUCKET_DATASETS, Key=CSV_KEY)
    content  = response["Body"].read().decode("utf-8")
    df       = pd.read_csv(StringIO(content))
    df.set_index("device_id", inplace=True)
    print(f"📊 Dataset IGD chargé : {len(df)} équipements")
    return df


def load_scenario1_results():
    print("📥 Loading scénario 1 results from MinIO...")
    response    = s3.get_object(Bucket=BUCKET_RESULTS, Key=SCENARIO1_KEY)
    predictions = json.loads(response["Body"].read().decode("utf-8"))
    filtered    = [p for p in predictions if p["prediction"]["predicted_class"] in ("degraded", "critical")]
    print(f"📋 Résultats scénario 1    : {len(predictions)} équipements")
    print(f"⚠️  À diagnostiquer        : {len(filtered)} (degraded + critical)")
    print(f"✅ Normaux ignorés          : {len(predictions) - len(filtered)}")
    return filtered


def features_to_query(device_id, predicted_class, downstream_curr,
                       downstream_max, snr_margin, attenuation, crc_errors):
    label_map = {"optimal": "Normal", "degraded": "Dégradé", "critical": "Critique"}
    label = label_map.get(predicted_class, predicted_class)
    return (
        f"Équipement IGD {device_id} en état {label}. "
        f"Débit descendant actuel={downstream_curr} kbps, "
        f"Débit descendant max={downstream_max} kbps, "
        f"SNR margin={snr_margin} dB, Atténuation={attenuation} dB, "
        f"Erreurs CRC={crc_errors}."
    )


def generate_diagnosis(device_id, predicted_class, confidence, query):
    label_map = {"optimal": "Normal", "degraded": "Dégradé", "critical": "Critique"}
    label   = label_map.get(predicted_class, predicted_class)
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

Analyse précisément les mesures IGD et identifie les 3 causes les plus probables.

Réponds UNIQUEMENT en JSON valide :
{{
  "device_id": "{device_id}",
  "predicted_class": "{predicted_class}",
  "health_label": "{label}",
  "diagnostic": {{
    "cause_principale": "nom exact de la cause",
    "probabilites": {{
      "cause_1": {{ "nom": "...", "probabilite": "XX%" }},
      "cause_2": {{ "nom": "...", "probabilite": "XX%" }},
      "cause_3": {{ "nom": "...", "probabilite": "XX%" }}
    }},
    "explication": "explication basée sur les valeurs mesurées",
    "action_recommandee": "action concrète"
  }}
}}
Important : probabilités totalisent 100%, répondre UNIQUEMENT avec le JSON.
"""
    print(f"  ⏳ Appel Ollama ({OLLAMA_MODEL})...")
    return llm.invoke(prompt)


def rag_diagnosis_single(device_id, predicted_class, confidence, df_metrics):
    if device_id not in df_metrics.index:
        print(f"  ⚠️  {device_id} introuvable — ignoré")
        return None

    row   = df_metrics.loc[device_id]
    query = features_to_query(
        device_id, predicted_class,
        row["downstream_curr_rate_kbps"], row["downstream_max_rate_kbps"],
        row["snr_margin_down_db"], row["attenuation_down_db"], row["crc_errors_total"]
    )

    raw_response = generate_diagnosis(device_id, predicted_class, confidence, query)

    try:
        start  = raw_response.find("{")
        end    = raw_response.rfind("}") + 1
        result = json.loads(raw_response[start:end])
        probs  = result["diagnostic"]["probabilites"]
        print(f"  ✅ Cause principale : {result['diagnostic']['cause_principale']}")
        for key, val in probs.items():
            print(f"     ├─ {val.get('nom','?')} : {val.get('probabilite','?')}")
        return result
    except json.JSONDecodeError:
        print(f"  ⚠️  JSON non parseable pour {device_id}")
        return {"device_id": device_id, "raw_response": raw_response}


def run_full_pipeline():
    print("\n" + "="*60)
    print("🚀 PIPELINE RAG — DIAGNOSTIC MULTI-CAUSAL IGD (DSL/ADSL)")
    print("="*60)

    df_metrics = load_dataset()
    scenario1  = load_scenario1_results()

    print(f"\n{'='*60}")
    print(f"🔍 Lancement du diagnostic sur {len(scenario1)} équipements...")
    print(f"{'='*60}")

    results = []
    success = 0
    errors  = 0

    # for i, prediction in enumerate(scenario1):
    for i, prediction in enumerate(scenario1):
        device_id       = prediction["device_id"]
        predicted_class = prediction["prediction"]["predicted_class"]
        confidence      = prediction["prediction"]["confidence"]

        print(f"\n[{i+1}/{len(scenario1)}] {device_id} — {predicted_class.upper()}")

        result = rag_diagnosis_single(device_id, predicted_class, confidence, df_metrics)

        if result:
            result["scenario1_confidence"] = confidence
            result["scenario1_risk_level"] = prediction["health_metrics"]["risk_level"]
            result["scenario1_action"]     = prediction["decision_support"]["recommended_action"]
            results.append(result)
            success += 1
        else:
            errors += 1

    print(f"\n💾 Sauvegarde résultats dans MinIO → {OUTPUT_KEY}")
    s3.put_object(
        Bucket = BUCKET_RESULTS,
        Key    = OUTPUT_KEY,
        Body   = json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8")
    )

    print(f"\n{'='*60}")
    print("✅ PIPELINE IGD TERMINÉ")
    print(f"{'='*60}")
    print(f"  Équipements diagnostiqués : {success}")
    print(f"  Erreurs                   : {errors}")
    print(f"  Résultats MinIO           : {OUTPUT_KEY}")
    return results


if __name__ == "__main__":
    results = run_full_pipeline()

    print("\n📊 RÉSUMÉ DES DIAGNOSTICS IGD :")
    print(f"{'─'*60}")
    for r in results:
        if "diagnostic" in r:
            probs = r["diagnostic"]["probabilites"]
            causes_str = " | ".join(
                f"{v.get('nom','?')} ({v.get('probabilite','?')})"
                for v in probs.values()
            )
            print(f"  {r['device_id']:20s} [{r['predicted_class'].upper():8s}] → {causes_str}")