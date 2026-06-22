"""
compare_and_promote.py
Champion / Challenger model promotion — MLOps GPON/DSL Diagnostic

After a retrain, this compares the freshly trained model (the "challenger")
against the current best model (the "champion") stored in MinIO.
The challenger is promoted to champion ONLY if it is better.

Layout in MinIO (bucket: models):
    scenario1/huawei_nokia_healthscore.pkl          <- latest trained (challenger)
    scenario1/huawei_nokia_metrics.json (in results bucket)
    scenario1/champion/huawei_nokia_healthscore.pkl <- current best (champion)
    scenario1/champion/huawei_nokia_metrics.json

Decision rule (per model):
    promote if challenger.accuracy > champion.accuracy
    (tie-break on f1_macro)
First run: if no champion exists yet, the current model becomes champion.

Usage:
    python3 compare_and_promote.py
"""

import os
import json
import logging
import boto3
from botocore.exceptions import ClientError


MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "http://10.98.20.211:9000") # NOSONAR
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
BUCKET_MODELS    = "models"
BUCKET_RESULTS   = "results"

# improvement margin: challenger must beat champion by at least this much
# (0.0 = any improvement promotes; raise to avoid promoting on noise)
MIN_IMPROVEMENT  = float(os.getenv("MIN_IMPROVEMENT", "0.0"))

# models to manage: name -> (model_key, metrics_key)
MODELS = {
    "HN": {
        "model_key":   "scenario1/huawei_nokia_healthscore.pkl",
        "metrics_key": "scenario1/huawei_nokia_metrics.json",
        "champion_model":   "scenario1/champion/huawei_nokia_healthscore.pkl",
        "champion_metrics": "scenario1/champion/huawei_nokia_metrics.json",
    },
    "IGD": {
        "model_key":   "scenario1/IGD_healthscore.pkl",
        "metrics_key": "scenario1/IGD_metrics.json",
        "champion_model":   "scenario1/champion/IGD_healthscore.pkl",
        "champion_metrics": "scenario1/champion/IGD_metrics.json",
    },
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compare-promote")

s3 = boto3.client(
    "s3",
    endpoint_url          = MINIO_ENDPOINT,
    aws_access_key_id     = MINIO_ACCESS_KEY,
    aws_secret_access_key = MINIO_SECRET_KEY,
)




def load_json(bucket, key):
    try:
        resp = s3.get_object(Bucket=bucket, Key=key) # NOSONAR
        return json.loads(resp["Body"].read().decode("utf-8"))
    except ClientError:
        return None
    except Exception as e:
        log.warning(f"Error reading {bucket}/{key}: {e}")
        return None


def object_exists(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key) # NOSONAR
        return True
    except ClientError:
        return False


def copy_object(bucket, src_key, dst_key):
    s3.copy_object(Bucket=bucket, CopySource=f"{bucket}/{src_key}", Key=dst_key) # NOSONAR


def score_of(metrics):
    """Primary score = accuracy, tie-break = f1_macro."""
    t = metrics.get("test", {})
    return t.get("accuracy", 0.0), t.get("f1_macro", 0.0)


def promote(name, cfg, reason):
    """Copy challenger model + metrics into the champion location."""
    # model .pkl lives in BUCKET_MODELS
    copy_object(BUCKET_MODELS, cfg["model_key"], cfg["champion_model"])
    # metrics json lives in BUCKET_RESULTS -> store champion metrics in models bucket
    # we copy the metrics into the models bucket champion path for self-contained record
    metrics = load_json(BUCKET_RESULTS, cfg["metrics_key"])
    s3.put_object( # NOSONAR 
        Bucket=BUCKET_MODELS,
        Key=cfg["champion_metrics"],
        Body=json.dumps(metrics, indent=2).encode("utf-8"),
    )
    log.info(f"🏆 {name}: PROMOTED challenger → champion ({reason})")


# ─────────────────────────────────────────────
# Per-model comparison
# ─────────────────────────────────────────────

def process_model(name, cfg):
    log.info(f"── {name} ──")

    challenger_metrics = load_json(BUCKET_RESULTS, cfg["metrics_key"])
    if not challenger_metrics:
        log.warning(f"{name}: no challenger metrics found — skipping.")
        return

    ch_acc, ch_f1 = score_of(challenger_metrics)
    log.info(f"   Challenger: accuracy={ch_acc:.4f}  f1={ch_f1:.4f}")

    # first run: no champion yet → current becomes champion
    if not object_exists(BUCKET_MODELS, cfg["champion_metrics"]):
        log.info(f"{name}: no champion yet — establishing current model as champion.")
        promote(name, cfg, "initial champion")
        return

    champion_metrics = load_json(BUCKET_MODELS, cfg["champion_metrics"])
    cp_acc, cp_f1 = score_of(champion_metrics)
    log.info(f"   Champion  : accuracy={cp_acc:.4f}  f1={cp_f1:.4f}")

    # decision
    better_acc = ch_acc > (cp_acc + MIN_IMPROVEMENT)
    tie_acc    = abs(ch_acc - cp_acc) <= MIN_IMPROVEMENT
    better_f1  = ch_f1 > cp_f1

    if better_acc:
        promote(name, cfg, f"accuracy {ch_acc:.4f} > {cp_acc:.4f}")
    elif tie_acc and better_f1:
        promote(name, cfg, f"accuracy tie, f1 {ch_f1:.4f} > {cp_f1:.4f}")
    else:
        log.info(f"   ❌ {name}: challenger NOT better — champion kept "
                 f"(challenger acc={ch_acc:.4f} vs champion acc={cp_acc:.4f}).")


def main():
    log.info("="*60)
    log.info("🥊 Champion / Challenger comparison")
    log.info(f"   Min improvement margin: {MIN_IMPROVEMENT}")
    log.info("="*60)

    for name, cfg in MODELS.items():
        try:
            process_model(name, cfg)
        except Exception:
            log.exception(f"{name}: comparison failed")

    log.info("="*60)
    log.info("✅ Comparison complete")
    log.info("="*60)


if __name__ == "__main__":
    main()