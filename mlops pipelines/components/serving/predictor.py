"""
predictor.py
Custom KServe Predictor — MLOps GPON/DSL Diagnostic

Serves the champion model as a REST API returning the FULL operational
diagnostic (predicted_class, health_score, risk_level, recommended_action...),
not just a raw class label.

Implements the KServe V1 protocol:
    GET  /v1/models/{name}            -> model readiness
    POST /v1/models/{name}:predict    -> predictions

Model selection via env:
    MODEL_NAME = "hn"  -> loads champion HN model + HN feature order
    MODEL_NAME = "igd" -> loads champion IGD model + IGD feature order

Run locally (test):
    MODEL_NAME=hn python3 predictor.py
    curl -X POST http://localhost:8080/v1/models/hn:predict \
         -H "Content-Type: application/json" \
         -d '{"instances": [{"temperature_c":45,"rx_power_dbm":-22,"bias_current_ua":10000,"supply_voltage_v":3.3}]}'
"""

import os
import pickle
import logging
import numpy as np
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn


MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "http://10.98.20.211:9000") # NOSONAR
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
BUCKET_MODELS    = "models"

MODEL_NAME = os.getenv("MODEL_NAME", "hn").lower()
PORT       = int(os.getenv("PORT", "8080"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

# champion model keys + the exact feature order the model was trained on
MODEL_CONFIG = {
    "hn": {
        "model_key": "scenario1/champion/huawei_nokia_healthscore.pkl",
        "features":  ["temperature_c", "rx_power_dbm", "bias_current_ua", "supply_voltage_v"],
    },
    "igd": {
        "model_key": "scenario1/champion/IGD_healthscore.pkl",
        "features":  ["downstream_curr_rate_kbps", "downstream_max_rate_kbps",
                       "snr_margin_down_db", "attenuation_down_db", "crc_errors_total"],
    },
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("predictor")

if MODEL_NAME not in MODEL_CONFIG:
    raise ValueError(f"MODEL_NAME must be one of {list(MODEL_CONFIG)}, got '{MODEL_NAME}'")

CFG       = MODEL_CONFIG[MODEL_NAME]
FEATURES  = CFG["features"]
CLASS_MAP = {0: "optimal", 1: "degraded", 2: "critical"}

# ─────────────────────────────────────────────
# Load model from MinIO at startup
# ─────────────────────────────────────────────
s3 = boto3.client(
    "s3",
    endpoint_url          = MINIO_ENDPOINT,
    aws_access_key_id     = MINIO_ACCESS_KEY,
    aws_secret_access_key = MINIO_SECRET_KEY,
)

def load_model():
    log.info(f"Loading champion model '{MODEL_NAME}' from {CFG['model_key']} ...")
    resp = s3.get_object(Bucket=BUCKET_MODELS, Key=CFG["model_key"]) # nosec B301 NOSONAR
    model = pickle.loads(resp["Body"].read())  # nosec B301
    log.info("✅ Model loaded.")
    return model

model = load_model()

# ─────────────────────────────────────────────
# Decision logic (same as evaluate_*.py)
# ─────────────────────────────────────────────
def derive_decision(p_deg, p_crit, risk_score):
    if risk_score < 0.3:
        risk_level = "low"
    elif risk_score < 0.6:
        risk_level = "medium"
    else:
        risk_level = "high"

    if p_crit > 0.7:
        recommended_action = "dispatch_technician"
    elif p_deg > 0.7:
        recommended_action = "monitor"
    else:
        recommended_action = "no_action"

    if risk_score > 0.8:
        priority_level = 1
    elif risk_score > 0.5:
        priority_level = 2
    else:
        priority_level = 3

    return risk_level, recommended_action, priority_level

def build_diagnostic(features_row: Dict[str, Any]) -> Dict[str, Any]:
    # order features exactly as trained
    try:
        x = np.array([[float(features_row[f]) for f in FEATURES]])
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}") from e

    proba = model.predict_proba(x)[0]
    pred  = int(model.predict(x)[0])

    p_opt, p_deg, p_crit = float(proba[0]), float(proba[1]), float(proba[2])
    confidence   = round(float(np.max(proba)), 3)
    health_score = round(1.0*p_opt + 0.5*p_deg + 0.0*p_crit, 3)
    risk_score   = round(1.0 - health_score, 3)
    risk_level, recommended_action, priority_level = derive_decision(p_deg, p_crit, risk_score)

    return {
        "model_version": MODEL_VERSION,
        "prediction": {
            "predicted_class": CLASS_MAP[pred],
            "class_probabilities": {
                "optimal": round(p_opt, 3),
                "degraded": round(p_deg, 3),
                "critical": round(p_crit, 3),
            },
            "confidence": confidence,
        },
        "health_metrics": {
            "health_score": health_score,
            "risk_score": risk_score,
            "risk_level": risk_level,
        },
        "decision_support": {
            "recommended_action": recommended_action,
            "priority_level": priority_level,
            "auto_ticket": bool(p_crit > 0.8 and confidence > 0.8),
        },
    }

# ─────────────────────────────────────────────
# FastAPI app (KServe V1 protocol)
# ─────────────────────────────────────────────
app = FastAPI(title=f"MLOps Predictor — {MODEL_NAME.upper()}")

class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]

@app.get("/v1/models/{name}")
def model_ready(name: str):
    return {"name": name, "ready": True}

@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_NAME, "version": MODEL_VERSION, "features": FEATURES}

@app.post("/v1/models/{name}:predict", responses={400: {"description": "Invalid request"}})
def predict(name: str, req: PredictRequest):
    if not req.instances:
        raise HTTPException(status_code=400, detail="No instances provided")
    predictions = [build_diagnostic(row) for row in req.instances]
    return {"predictions": predictions}

if __name__ == "__main__":
    log.info(f"🚀 Starting predictor for '{MODEL_NAME}' on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)  # nosec B104 NOSONAR