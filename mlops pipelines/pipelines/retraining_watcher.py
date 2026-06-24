"""
retraining_watcher.py
Auto-Retraining Watcher — MLOps GPON/DSL Diagnostic

Triggers the Kubeflow pipeline on ANY of three conditions:
  1. New data        — a raw dataset in MinIO has a newer LastModified timestamp
  2. Schedule        — 6 hours elapsed since the last run
  3. Degradation     — Scenario 1 accuracy < 0.80 or degraded_recall < 0.55 (safety net)

A cooldown prevents double-firing.

Usage (on the VM, with port-forward running):
    kubectl port-forward svc/ml-pipeline -n kubeflow 8888:8888 &
    python3 retraining_watcher.py
"""

import os
import json
import time
import logging
import boto3
from kfp import compiler
from kfp.client import Client

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "http://10.98.20.211:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
BUCKET_RESULTS   = "results"
BUCKET_DATASETS  = "datasets"

KFP_HOST         = os.getenv("KFP_HOST", "http://localhost:8888")
PIPELINE_FILE    = os.getenv("PIPELINE_FILE", "mlops_diagnostic_pipeline.yaml")
EXPERIMENT_NAME  = os.getenv("EXPERIMENT_NAME", "auto-retraining")

POLL_INTERVAL    = int(os.getenv("POLL_INTERVAL", "60"))       # seconds between checks
COOLDOWN         = int(os.getenv("COOLDOWN", "1800"))          # 30 min after a trigger
SCHEDULE_EVERY   = int(os.getenv("SCHEDULE_EVERY", "21600"))   # 6 hours, in seconds
ACCURACY_FLOOR   = 0.80

# raw datasets to watch for "new data arrived"
DATASET_KEYS = {
    "HN":  "raw/fibre_combined_realistic.csv",
    "IGD": "raw/igd_realistic.csv",
}

# scenario-1 metrics to watch for degradation (safety net)
METRICS_KEYS = {
    "HN":  "scenario1/huawei_nokia_metrics.json",
    "IGD": "scenario1/IGD_metrics.json",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("retraining-watcher")

s3 = boto3.client(
    "s3",
    endpoint_url          = MINIO_ENDPOINT,
    aws_access_key_id     = MINIO_ACCESS_KEY,
    aws_secret_access_key = MINIO_SECRET_KEY,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_last_modified(bucket, key):
    """Return the LastModified timestamp of a MinIO object, or None."""
    try:
        resp = s3.head_object(Bucket=bucket, Key=key)
        return resp["LastModified"].timestamp()
    except Exception as e:
        log.warning(f"Could not stat {bucket}/{key}: {e}")
        return None


def load_metrics(key):
    try:
        resp = s3.get_object(Bucket=BUCKET_RESULTS, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except Exception as e:
        log.warning(f"Could not load {key}: {e}")
        return None


def check_new_data(seen):
    """Compare current dataset timestamps to last seen. Returns (changed, reasons, updated_seen)."""
    reasons = []
    changed = False
    for name, key in DATASET_KEYS.items():
        ts = get_last_modified(BUCKET_DATASETS, key)
        if ts is None:
            continue
        if name not in seen:
            seen[name] = ts          # first observation, no trigger
        elif ts > seen[name]:
            reasons.append(f"{name}: new data in {key}")
            seen[name] = ts
            changed = True
    return changed, reasons, seen


def check_degradation():
    """Safety-net: returns (needed, reasons) based on scenario-1 metrics."""
    reasons = []
    for name, key in METRICS_KEYS.items():
        m = load_metrics(key)
        if not m:
            continue
        acc = m.get("test", {}).get("accuracy", 1.0)
        trig = m.get("monitoring_kpi", {}).get("trigger_retraining", False)
        if acc < ACCURACY_FLOOR:
            reasons.append(f"{name}: accuracy={acc:.3f} < {ACCURACY_FLOOR}")
        if trig:
            reasons.append(f"{name}: degraded_recall below threshold")
    return (len(reasons) > 0), reasons


def ensure_pipeline_file():
    if os.path.exists(PIPELINE_FILE):
        return
    log.info(f"{PIPELINE_FILE} not found — compiling from pipeline.py ...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("pipeline_mod", "pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    compiler.Compiler().compile(
        pipeline_func=mod.mlops_diagnostic_pipeline,
        package_path=PIPELINE_FILE,
    )
    log.info(f"Compiled → {PIPELINE_FILE}")


def trigger_pipeline(reason):
    ensure_pipeline_file()
    client = Client(host=KFP_HOST)
    try:
        exp = client.get_experiment(experiment_name=EXPERIMENT_NAME)
    except Exception:
        exp = client.create_experiment(name=EXPERIMENT_NAME)
    run_name = f"auto-retrain-{time.strftime('%Y%m%d-%H%M%S')}"
    run = client.run_pipeline(
        experiment_id=exp.experiment_id,
        job_name=run_name,
        pipeline_package_path=PIPELINE_FILE,
    )
    log.info(f"🚀 Pipeline launched: run='{run_name}' id={run.run_id}")
    log.info(f"   Reason: {reason}")
    return run


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def main():
    log.info("="*60)
    log.info("🔁 Auto-Retraining Watcher started")
    log.info(f"   Poll every     : {POLL_INTERVAL}s")
    log.info(f"   Schedule every : {SCHEDULE_EVERY}s ({SCHEDULE_EVERY//3600}h)")
    log.info(f"   Cooldown       : {COOLDOWN}s")
    log.info("   Triggers       : new data | schedule | accuracy < 0.80")
    log.info("="*60)

    seen_timestamps = {}
    last_trigger_time = 0.0
    last_run_time = time.time()   # start the 6h clock now

    # prime dataset timestamps so we don't fire on first loop
    check_new_data(seen_timestamps)

    while True:
        now = time.time()
        in_cooldown = (now - last_trigger_time) < COOLDOWN

        reasons = []

        # 1. new data?
        new_data, data_reasons, seen_timestamps = check_new_data(seen_timestamps)
        if new_data:
            reasons += data_reasons

        # 2. schedule?
        if (now - last_run_time) >= SCHEDULE_EVERY:
            reasons.append(f"scheduled retrain ({SCHEDULE_EVERY//3600}h elapsed)")

        # 3. degradation (safety net)?
        degraded, deg_reasons = check_degradation()
        if degraded:
            reasons += deg_reasons

        if reasons:
            if in_cooldown:
                remaining = int(COOLDOWN - (now - last_trigger_time))
                log.info(f"⏳ Trigger needed but in cooldown ({remaining}s left) — skipping.")
            else:
                full_reason = " ; ".join(reasons)
                log.warning(f"⚠️  RETRAIN TRIGGERED — {full_reason}")
                try:
                    trigger_pipeline(full_reason)
                    last_trigger_time = time.time()
                    last_run_time = time.time()
                except Exception as e:
                    log.error(f"Failed to launch pipeline: {e}")
        else:
            log.info("✅ No trigger (no new data, schedule not due, models healthy).")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()