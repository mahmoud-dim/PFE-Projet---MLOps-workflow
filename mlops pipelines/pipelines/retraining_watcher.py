"""
retraining_watcher.py
Auto-Retraining Watcher — MLOps GPON/DSL Diagnostic

Watches model metrics in MinIO and automatically triggers the Kubeflow
pipeline when a model degrades.

Trigger condition (per model HN / IGD):
    trigger_retraining == true   (degraded_recall < 0.55)
    OR  test accuracy < 0.80

When triggered, launches the compiled Kubeflow pipeline via the KFP v2 client,
then enters a cooldown so it does not relaunch every loop while metrics stay bad.

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

KFP_HOST         = os.getenv("KFP_HOST", "http://localhost:8888")
PIPELINE_FILE    = os.getenv("PIPELINE_FILE", "mlops_diagnostic_pipeline.yaml")
EXPERIMENT_NAME  = os.getenv("EXPERIMENT_NAME", "auto-retraining")

POLL_INTERVAL    = int(os.getenv("POLL_INTERVAL", "60"))      # seconds between checks
COOLDOWN         = int(os.getenv("COOLDOWN", "1800"))         # seconds to wait after a trigger (30 min)

ACCURACY_FLOOR   = 0.80

# metrics files to watch
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

def load_metrics(key):
    """Load a metrics JSON from MinIO. Returns dict or None."""
    try:
        resp = s3.get_object(Bucket=BUCKET_RESULTS, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except Exception as e:
        log.warning(f"Could not load {key}: {e}")
        return None


def check_model(name, metrics):
    """Return (should_retrain: bool, reason: str) for one model."""
    if not metrics:
        return False, "metrics unavailable"

    accuracy = metrics.get("test", {}).get("accuracy", 1.0)
    trigger  = metrics.get("monitoring_kpi", {}).get("trigger_retraining", False)
    recall   = metrics.get("monitoring_kpi", {}).get("degraded_recall", 1.0)

    reasons = []
    if trigger:
        reasons.append(f"degraded_recall={recall:.3f} < 0.55")
    if accuracy < ACCURACY_FLOOR:
        reasons.append(f"accuracy={accuracy:.3f} < {ACCURACY_FLOOR}")

    if reasons:
        return True, f"{name}: " + " | ".join(reasons)
    return False, f"{name}: healthy (acc={accuracy:.3f}, recall={recall:.3f})"


def ensure_pipeline_file():
    """Make sure the compiled pipeline YAML exists; compile it if missing."""
    if os.path.exists(PIPELINE_FILE):
        return
    log.info(f"{PIPELINE_FILE} not found — compiling from pipeline.py ...")
    # import the pipeline function and compile
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
    """Launch the Kubeflow pipeline via KFP v2 client."""
    ensure_pipeline_file()
    client = Client(host=KFP_HOST)

    # ensure experiment exists
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
    log.info(f"   MinIO        : {MINIO_ENDPOINT}")
    log.info(f"   KFP host     : {KFP_HOST}")
    log.info(f"   Poll every   : {POLL_INTERVAL}s")
    log.info(f"   Cooldown     : {COOLDOWN}s after a trigger")
    log.info(f"   Trigger if   : degraded_recall < 0.55  OR  accuracy < {ACCURACY_FLOOR}")
    log.info("="*60)

    last_trigger_time = 0.0

    while True:
        now = time.time()
        in_cooldown = (now - last_trigger_time) < COOLDOWN

        retrain_needed = False
        reasons = []
        for name, key in METRICS_KEYS.items():
            metrics = load_metrics(key)
            should, reason = check_model(name, metrics)
            log.info(f"  {reason}")
            if should:
                retrain_needed = True
                reasons.append(reason)

        if retrain_needed:
            if in_cooldown:
                remaining = int(COOLDOWN - (now - last_trigger_time))
                log.info(f"⏳ Retrain needed but in cooldown ({remaining}s left) — skipping.")
            else:
                full_reason = " ; ".join(reasons)
                log.warning(f"⚠️  RETRAIN TRIGGERED — {full_reason}")
                try:
                    trigger_pipeline(full_reason)
                    last_trigger_time = time.time()
                except Exception as e:
                    log.error(f"Failed to launch pipeline: {e}")
        else:
            log.info("✅ All models healthy — no action.")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()