# import os
# import json
# import pandas as pd
# from langchain_ollama import OllamaLLM

# # ============================================================
# # main_hn.py
# # Pipeline RAG complet — Diagnostic Multi-Causal Huawei/Nokia
# # LLM    : Ollama local (llama3.1:8b)
# # Input 1: dataset CSV (métriques ONT)
# # Input 2: JSON résultats scénario 1 (predicted_class)
# # ============================================================

# # ----------------------------------------
# # Initialisation Ollama LLM
# # ----------------------------------------
# OLLAMA_MODEL = "llama3.1:8b"

# llm = OllamaLLM(
#     model=OLLAMA_MODEL,
#     base_url="http://localhost:11434"
# )

# # ----------------------------------------
# # Chargement Knowledge Base (9 causes)
# # ----------------------------------------
# BASE_DIR     = os.path.dirname(__file__)
# json_kb_path = os.path.join(BASE_DIR, "data/knowledge_base_huawei_nokia.json")

# with open(json_kb_path, "r", encoding="utf-8") as f:
#     causes_data = json.load(f)

# print(f"✅ Knowledge Base chargée : {len(causes_data)} causes")
# print(f"🤖 Modèle LLM             : {OLLAMA_MODEL} (local)")


# # ============================================================
# # CHARGEMENT DES DONNÉES
# # ============================================================

# def load_dataset(csv_path: str) -> pd.DataFrame:
#     df = pd.read_csv(csv_path)
#     df.set_index("device_id", inplace=True)
#     print(f"📊 Dataset chargé : {len(df)} box au total")
#     return df


# def load_scenario1_results(json_path: str) -> list:
#     with open(json_path, "r", encoding="utf-8") as f:
#         predictions = json.load(f)

#     filtered = [
#         p for p in predictions
#         if p["prediction"]["predicted_class"] in ("degraded", "critical")
#     ]

#     print(f"📋 Résultats scénario 1   : {len(predictions)} box au total")
#     print(f"⚠️  Box à diagnostiquer   : {len(filtered)} (degraded + critical)")
#     print(f"✅ Box optimales ignorées  : {len(predictions) - len(filtered)}")

#     return filtered


# # ============================================================
# # ÉTAPE 1 : Features ONT → texte naturel
# # ============================================================
# def features_to_query(device_id, predicted_class, rx_power,
#                        bias_current, temperature, supply_voltage):
#     label_map = {
#         "optimal":  "Optimal",
#         "degraded": "Dégradé",
#         "critical": "Critique"
#     }
#     label = label_map.get(predicted_class, predicted_class)

#     query = (
#         f"Box {device_id} en état {label}. "
#         f"RxPower={rx_power} dBm, "
#         f"BiasCurrent={bias_current} µA, "
#         f"Température={temperature}°C, "
#         f"Supply Voltage={supply_voltage} V."
#     )
#     return query


# # ============================================================
# # ÉTAPE 2 : Génération diagnostic avec Ollama
# # Envoie les 9 causes complètes au LLM → il choisit les 3 meilleures
# # ============================================================
# def generate_diagnosis(device_id, predicted_class, confidence, query):
#     label_map = {
#         "optimal":  "Optimal",
#         "degraded": "Dégradé",
#         "critical": "Critique"
#     }
#     label = label_map.get(predicted_class, predicted_class)

#     # ✅ Toutes les 9 causes envoyées directement au LLM
#     context = ""
#     for i, cause in enumerate(causes_data):
#         context += (
#             f"Cause {i+1}: {cause['Cause']}\n"
#             f"Indicateurs: {', '.join(cause['Indicators'])}\n"
#             f"Description: {cause['Description']}\n\n"
#         )

#     prompt = f"""Tu es un expert en diagnostic réseau télécom spécialisé en GPON (Huawei/Nokia).

# Box analysée    : {device_id}
# État prédit     : {label} (confiance : {confidence*100:.1f}%)
# Mesures ONT     : {query}

# Voici les 9 causes possibles de dégradation GPON :
# {context}

# Analyse précisément les mesures ONT et identifie les 3 causes les plus probables
# parmi les 9 causes ci-dessus. Chaque box a des métriques différentes,
# donc le diagnostic doit être spécifique à ces mesures.

# Réponds UNIQUEMENT en JSON valide avec ce format exact :

# {{
#   "device_id": "{device_id}",
#   "predicted_class": "{predicted_class}",
#   "health_label": "{label}",
#   "diagnostic": {{
#     "cause_principale": "nom exact de la cause la plus probable",
#     "probabilites": {{
#       "cause_1": {{ "nom": "...", "probabilite": "XX%" }},
#       "cause_2": {{ "nom": "...", "probabilite": "XX%" }},
#       "cause_3": {{ "nom": "...", "probabilite": "XX%" }}
#     }},
#     "explication": "explication courte basée sur les valeurs mesurées",
#     "action_recommandee": "action concrète à effectuer"
#   }}
# }}

# Important :
# - Les probabilités des 3 causes doivent totaliser 100%
# - Utilise les noms exacts des causes tels qu'ils apparaissent ci-dessus
# - Base ton analyse sur les valeurs numériques des métriques ONT
# - Réponds UNIQUEMENT avec le JSON, rien d'autre
# """

#     print(f"  ⏳ Appel Ollama ({OLLAMA_MODEL})...")
#     response = llm.invoke(prompt)
#     return response


# # ============================================================
# # PIPELINE RAG POUR UNE SEULE BOX
# # ============================================================
# def rag_diagnosis_single(device_id, predicted_class, confidence, df_metrics):

#     if device_id not in df_metrics.index:
#         print(f"  ⚠️  {device_id} introuvable dans le dataset CSV — ignoré")
#         return None

#     row            = df_metrics.loc[device_id]
#     rx_power       = row["rx_power_dbm"]
#     bias_current   = row["bias_current_ua"]
#     temperature    = row["temperature_c"]
#     supply_voltage = row["supply_voltage_v"]

#     # Étape 1 : Features → texte
#     query = features_to_query(
#         device_id, predicted_class,
#         rx_power, bias_current, temperature, supply_voltage
#     )

#     # Étape 2 : Ollama analyse les 9 causes et choisit les 3 meilleures
#     raw_response = generate_diagnosis(
#         device_id, predicted_class, confidence, query
#     )

#     # Étape 3 : Parser le JSON
#     try:
#         start  = raw_response.find("{")
#         end    = raw_response.rfind("}") + 1
#         result = json.loads(raw_response[start:end])

#         # Afficher les 3 causes
#         probs = result["diagnostic"]["probabilites"]
#         print(f"  ✅ Cause principale : {result['diagnostic']['cause_principale']}")
#         print(f"     ├─ {probs['cause_1']['nom']} : {probs['cause_1']['probabilite']}")
#         print(f"     ├─ {probs['cause_2']['nom']} : {probs['cause_2']['probabilite']}")
#         print(f"     └─ {probs['cause_3']['nom']} : {probs['cause_3']['probabilite']}")

#         return result

#     except json.JSONDecodeError:
#         print(f"  ⚠️  JSON non parseable pour {device_id}")
#         return {"device_id": device_id, "raw_response": raw_response}


# # ============================================================
# # PIPELINE COMPLET
# # ============================================================
# def run_full_pipeline(csv_path: str, scenario1_json_path: str, output_path: str):

#     print("\n" + "="*60)
#     print("🚀 PIPELINE RAG — DIAGNOSTIC MULTI-CAUSAL GPON (Huawei/Nokia)")
#     print("="*60)

#     df_metrics = load_dataset(csv_path)
#     scenario1  = load_scenario1_results(scenario1_json_path)

#     print(f"\n{'='*60}")
#     print(f"🔍 Lancement du diagnostic sur {len(scenario1)} box...")
#     print(f"{'='*60}")

#     results = []
#     success = 0
#     errors  = 0

#     # for i, prediction in enumerate(scenario1):
#     for i, prediction in enumerate(scenario1[:3]):  # Limiter à 3 box pour les tests initiaux
#         device_id       = prediction["device_id"]
#         predicted_class = prediction["prediction"]["predicted_class"]
#         confidence      = prediction["prediction"]["confidence"]

#         print(f"\n[{i+1}/{len(scenario1)}] {device_id} — {predicted_class.upper()}")

#         result = rag_diagnosis_single(
#             device_id, predicted_class, confidence, df_metrics
#         )

#         if result:
#             result["scenario1_confidence"] = confidence
#             result["scenario1_risk_level"] = prediction["health_metrics"]["risk_level"]
#             result["scenario1_action"]     = prediction["decision_support"]["recommended_action"]
#             results.append(result)
#             success += 1
#         else:
#             errors += 1

#     # Sauvegarde
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2, ensure_ascii=False)

#     print(f"\n{'='*60}")
#     print(f"✅ PIPELINE TERMINÉ")
#     print(f"{'='*60}")
#     print(f"  Box diagnostiquées        : {success}")
#     print(f"  Box introuvables (erreurs): {errors}")
#     print(f"  Résultats sauvegardés     : {output_path}")

#     return results


# # ============================================================
# # MAIN
# # ============================================================
# if __name__ == "__main__":

#     CSV_PATH       = os.path.join(BASE_DIR, "../../datasets/processed/huawei_nokia_scenario_1.csv")
#     SCENARIO1_PATH = os.path.join(BASE_DIR, "../healthscore_training/evaluate/huawei_nokia_operational_predictions.json")
#     OUTPUT_PATH    = os.path.join(BASE_DIR, "data/diagnostic_results.json")

#     results = run_full_pipeline(
#         csv_path            = CSV_PATH,
#         scenario1_json_path = SCENARIO1_PATH,
#         output_path         = OUTPUT_PATH
#     )

#     # Résumé final
#     print(f"\n📊 RÉSUMÉ DES DIAGNOSTICS :")
#     print(f"{'─'*60}")
#     for r in results:
#         if "diagnostic" in r:
#             probs = r["diagnostic"]["probabilites"]
#             print(
#                 f"  {r['device_id']:20s} [{r['predicted_class'].upper():8s}] → "
#                 f"{probs['cause_1']['nom']} ({probs['cause_1']['probabilite']}) | "
#                 f"{probs['cause_2']['nom']} ({probs['cause_2']['probabilite']}) | "
#                 f"{probs['cause_3']['nom']} ({probs['cause_3']['probabilite']})"
#             )




import os
import json
import boto3
import pickle
import faiss
import numpy as np
from io import StringIO, BytesIO
import pandas as pd
from langchain_ollama import OllamaLLM

# ============================================================
# main_hn.py — RAG Diagnostic Huawei/Nokia
# Input  : MinIO → datasets/processed/huawei_nokia_scenario_1.csv
#          MinIO → results/scenario1/huawei_nokia_operational_predictions.json
#          MinIO → knowledge-base/knowledge_base_huawei_nokia.json
# Output : MinIO → results/scenario2/diagnostic_results_hn.json
# ============================================================

MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "http://10.98.20.211:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
OLLAMA_MODEL     = "llama3.1:8b"

BUCKET_DATASETS = "datasets"
BUCKET_KB       = "knowledge-base"
BUCKET_MODELS   = "models"
BUCKET_RESULTS  = "results"

CSV_KEY        = "processed/huawei_nokia_scenario_1.csv"
SCENARIO1_KEY  = "scenario1/huawei_nokia_operational_predictions.json"
KB_KEY         = "knowledge_base_huawei_nokia.json"
FAISS_KEY      = "scenario2/faiss_index_huawei_nokia.index"
CHUNKS_KEY     = "scenario2/chunks_huawei_nokia.pkl"
OUTPUT_KEY     = "scenario2/diagnostic_results_hn.json"

s3 = boto3.client(
    "s3",
    endpoint_url          = MINIO_ENDPOINT,
    aws_access_key_id     = MINIO_ACCESS_KEY,
    aws_secret_access_key = MINIO_SECRET_KEY
)

llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

# ----------------------------------------
# Chargement Knowledge Base depuis MinIO
# ----------------------------------------
print("📂 Chargement Knowledge Base depuis MinIO...")
response    = s3.get_object(Bucket=BUCKET_KB, Key=KB_KEY)
causes_data = json.loads(response["Body"].read().decode("utf-8"))
print(f"✅ Knowledge Base chargée : {len(causes_data)} causes")
print(f"🤖 Modèle LLM             : {OLLAMA_MODEL} (local)")


def load_dataset():
    print("📥 Loading dataset from MinIO...")
    response = s3.get_object(Bucket=BUCKET_DATASETS, Key=CSV_KEY)
    content  = response["Body"].read().decode("utf-8")
    df       = pd.read_csv(StringIO(content))
    df.set_index("device_id", inplace=True)
    print(f"📊 Dataset chargé : {len(df)} box au total")
    return df


def load_scenario1_results():
    print("📥 Loading scénario 1 results from MinIO...")
    response    = s3.get_object(Bucket=BUCKET_RESULTS, Key=SCENARIO1_KEY)
    predictions = json.loads(response["Body"].read().decode("utf-8"))

    filtered = [
        p for p in predictions
        if p["prediction"]["predicted_class"] in ("degraded", "critical")
    ]

    print(f"📋 Résultats scénario 1   : {len(predictions)} box au total")
    print(f"⚠️  Box à diagnostiquer   : {len(filtered)} (degraded + critical)")
    print(f"✅ Box optimales ignorées  : {len(predictions) - len(filtered)}")
    return filtered


def features_to_query(device_id, predicted_class, rx_power,
                       bias_current, temperature, supply_voltage):
    label_map = {"optimal": "Optimal", "degraded": "Dégradé", "critical": "Critique"}
    label = label_map.get(predicted_class, predicted_class)
    return (
        f"Box {device_id} en état {label}. "
        f"RxPower={rx_power} dBm, BiasCurrent={bias_current} µA, "
        f"Température={temperature}°C, Supply Voltage={supply_voltage} V."
    )


def generate_diagnosis(device_id, predicted_class, confidence, query):
    label_map = {"optimal": "Optimal", "degraded": "Dégradé", "critical": "Critique"}
    label   = label_map.get(predicted_class, predicted_class)
    context = ""
    for i, cause in enumerate(causes_data):
        context += (
            f"Cause {i+1}: {cause['Cause']}\n"
            f"Indicateurs: {', '.join(cause['Indicators'])}\n"
            f"Description: {cause['Description']}\n\n"
        )

    prompt = f"""Tu es un expert en diagnostic réseau télécom spécialisé en GPON (Huawei/Nokia).

Box analysée    : {device_id}
État prédit     : {label} (confiance : {confidence*100:.1f}%)
Mesures ONT     : {query}

Voici les 9 causes possibles de dégradation GPON :
{context}

Analyse précisément les mesures ONT et identifie les 3 causes les plus probables.

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
        row["rx_power_dbm"], row["bias_current_ua"],
        row["temperature_c"], row["supply_voltage_v"]
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
    print("🚀 PIPELINE RAG — DIAGNOSTIC MULTI-CAUSAL GPON (Huawei/Nokia)")
    print("="*60)

    df_metrics = load_dataset()
    scenario1  = load_scenario1_results()

    print(f"\n{'='*60}")
    print(f"🔍 Lancement du diagnostic sur {len(scenario1)} box...")
    print(f"{'='*60}")

    results = []
    success = 0
    errors  = 0

    # for i, prediction in enumerate(scenario1):
    for i, prediction in enumerate(scenario1[:3]):
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

    # Sauvegarder dans MinIO
    print(f"\n💾 Sauvegarde résultats dans MinIO → {OUTPUT_KEY}")
    s3.put_object(
        Bucket = BUCKET_RESULTS,
        Key    = OUTPUT_KEY,
        Body   = json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8")
    )

    print(f"\n{'='*60}")
    print(f"✅ PIPELINE TERMINÉ")
    print(f"{'='*60}")
    print(f"  Box diagnostiquées : {success}")
    print(f"  Erreurs            : {errors}")
    print(f"  Résultats MinIO    : {OUTPUT_KEY}")
    return results


if __name__ == "__main__":
    results = run_full_pipeline()

    print(f"\n📊 RÉSUMÉ DES DIAGNOSTICS :")
    print(f"{'─'*60}")
    for r in results:
        if "diagnostic" in r:
            probs = r["diagnostic"]["probabilites"]
            causes_str = " | ".join(
                f"{v.get('nom','?')} ({v.get('probabilite','?')})"
                for v in probs.values()
            )
            print(f"  {r['device_id']:20s} [{r['predicted_class'].upper():8s}] → {causes_str}")