"""
ml_exporter.py
Prometheus Exporter — MLOps GPON/DSL Diagnostic
Reads results from MinIO and exposes metrics for Prometheus scraping

Metrics exposed:
  - Scenario 1 : Accuracy, F1, distribution optimal/degraded/critical
  - Scenario 2 : Root cause analysis results
  - Scenario 3 : Risk levels, days before failure
  - Gatekeeping : Pass/Fail status

Usage:
  python3 ml_exporter.py
  Access: http://localhost:8000/metrics
"""

import os
import json
import time
import boto3
import logging
from io import BytesIO
from prometheus_client import start_http_server, Gauge, Counter, Info

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "http://10.98.20.211:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
BUCKET_RESULTS   = "results"
EXPORTER_PORT    = int(os.getenv("EXPORTER_PORT", "8000"))
SCRAPE_INTERVAL  = int(os.getenv("SCRAPE_INTERVAL", "30"))  # seconds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# MinIO Client
# ─────────────────────────────────────────────
s3 = boto3.client(
    "s3",
    endpoint_url          = MINIO_ENDPOINT,
    aws_access_key_id     = MINIO_ACCESS_KEY,
    aws_secret_access_key = MINIO_SECRET_KEY
)

# ─────────────────────────────────────────────
# SCENARIO 1 — Classification Metrics
# ─────────────────────────────────────────────

# Accuracy
hn_accuracy  = Gauge("ml_accuracy_hn",  "Random Forest accuracy — Huawei/Nokia GPON")
igd_accuracy = Gauge("ml_accuracy_igd", "Random Forest accuracy — IGD DSL")

# F1-Score
hn_f1  = Gauge("ml_f1_score_hn",  "F1-Score macro — Huawei/Nokia GPON")
igd_f1 = Gauge("ml_f1_score_igd", "F1-Score macro — IGD DSL")

# Overfitting gap
hn_overfit_gap  = Gauge("ml_overfit_gap_hn",  "Train-Test accuracy gap — Huawei/Nokia")
igd_overfit_gap = Gauge("ml_overfit_gap_igd", "Train-Test accuracy gap — IGD")

# Gatekeeping
hn_gatekeeping  = Gauge("ml_gatekeeping_hn",  "Gatekeeping status HN (1=PASSED, 0=FAILED)")
igd_gatekeeping = Gauge("ml_gatekeeping_igd", "Gatekeeping status IGD (1=PASSED, 0=FAILED)")

# Distribution predictions HN
hn_boxes_optimal  = Gauge("ml_boxes_optimal_hn",  "Number of HN boxes predicted as Optimal")
hn_boxes_degraded = Gauge("ml_boxes_degraded_hn", "Number of HN boxes predicted as Degraded")
hn_boxes_critical = Gauge("ml_boxes_critical_hn", "Number of HN boxes predicted as Critical")

# Distribution predictions IGD
igd_boxes_optimal  = Gauge("ml_boxes_optimal_igd",  "Number of IGD boxes predicted as Optimal")
igd_boxes_degraded = Gauge("ml_boxes_degraded_igd", "Number of IGD boxes predicted as Degraded")
igd_boxes_critical = Gauge("ml_boxes_critical_igd", "Number of IGD boxes predicted as Critical")

# Retraining trigger
hn_retraining_trigger  = Gauge("ml_retraining_trigger_hn",  "Retraining needed HN (1=YES, 0=NO)")
igd_retraining_trigger = Gauge("ml_retraining_trigger_igd", "Retraining needed IGD (1=YES, 0=NO)")

# Degraded recall
hn_degraded_recall  = Gauge("ml_degraded_recall_hn",  "Degraded class recall — HN")
igd_degraded_recall = Gauge("ml_degraded_recall_igd", "Degraded class recall — IGD")

# ─────────────────────────────────────────────
# SCENARIO 2 — Root Cause Analysis
# ─────────────────────────────────────────────
hn_diagnosed_total  = Gauge("ml_diagnosed_total_hn",  "Total HN boxes diagnosed by RAG")
igd_diagnosed_total = Gauge("ml_diagnosed_total_igd", "Total IGD boxes diagnosed by RAG")

# Top causes HN (by count)
hn_cause_perturbation    = Gauge("ml_cause_hn_perturbation",    "HN cause: Perturbation Environnementale")
hn_cause_saturation      = Gauge("ml_cause_hn_saturation",      "HN cause: Saturation / Configuration")
hn_cause_attenuation     = Gauge("ml_cause_hn_attenuation",     "HN cause: Atténuation de Ligne")
hn_cause_vieillissement  = Gauge("ml_cause_hn_vieillissement",  "HN cause: Vieillissement Matériel")
hn_cause_macrobend       = Gauge("ml_cause_hn_macrobend",       "HN cause: Macrobend / Connecteur Sale")
hn_cause_rupture         = Gauge("ml_cause_hn_rupture",         "HN cause: Rupture Partielle Fibre")
hn_cause_other           = Gauge("ml_cause_hn_other",           "HN cause: Other causes")

# Top causes IGD (by count)
igd_cause_line_degradation = Gauge("ml_cause_igd_line_degradation", "IGD cause: Line Degradation")
igd_cause_interference     = Gauge("ml_cause_igd_interference",     "IGD cause: High Interference / Noise")
igd_cause_crosstalk        = Gauge("ml_cause_igd_crosstalk",        "IGD cause: Crosstalk / Saturation")
igd_cause_config           = Gauge("ml_cause_igd_config",           "IGD cause: Configuration / Ports Overload")
igd_cause_other            = Gauge("ml_cause_igd_other",            "IGD cause: Other causes")

# ─────────────────────────────────────────────
# SCENARIO 3 — Proactive Prediction
# ─────────────────────────────────────────────

# Risk levels HN
hn_risk_critique = Gauge("ml_risk_critique_hn", "HN boxes with CRITIQUE risk (≤2 days)")
hn_risk_eleve    = Gauge("ml_risk_eleve_hn",    "HN boxes with ÉLEVÉ risk (≤7 days)")
hn_risk_modere   = Gauge("ml_risk_modere_hn",   "HN boxes with MODÉRÉ risk (≤30 days)")
hn_risk_faible   = Gauge("ml_risk_faible_hn",   "HN boxes with FAIBLE risk (>30 days)")
hn_risk_stable   = Gauge("ml_risk_stable_hn",   "HN boxes STABLE (no degradation)")

# Risk levels IGD
igd_risk_critique = Gauge("ml_risk_critique_igd", "IGD boxes with CRITIQUE risk (≤2 days)")
igd_risk_eleve    = Gauge("ml_risk_eleve_igd",    "IGD boxes with ÉLEVÉ risk (≤7 days)")
igd_risk_modere   = Gauge("ml_risk_modere_igd",   "IGD boxes with MODÉRÉ risk (≤30 days)")
igd_risk_faible   = Gauge("ml_risk_faible_igd",   "IGD boxes with FAIBLE risk (>30 days)")
igd_risk_stable   = Gauge("ml_risk_stable_igd",   "IGD boxes STABLE (no degradation)")

# Average slope
hn_avg_slope  = Gauge("ml_avg_slope_hn",  "Average RxPower slope — HN (dBm/measure)")
igd_avg_slope = Gauge("ml_avg_slope_igd", "Average SNR slope — IGD (dB/measure)")

# Urgent devices (≤7 days)
hn_urgent_devices  = Gauge("ml_urgent_devices_hn",  "HN devices needing intervention in ≤7 days")
igd_urgent_devices = Gauge("ml_urgent_devices_igd", "IGD devices needing intervention in ≤7 days")

# Total devices
hn_total_devices  = Gauge("ml_total_devices_hn",  "Total HN devices monitored")
igd_total_devices = Gauge("ml_total_devices_igd", "Total IGD devices monitored")


# ─────────────────────────────────────────────
# FUNCTIONS — Load from MinIO
# ─────────────────────────────────────────────

def load_json_from_minio(bucket, key):
    """Load a JSON file from MinIO and return as dict/list"""
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))
    except Exception as e:
        log.warning(f"Failed to load {bucket}/{key}: {e}")
        return None


def update_scenario1_metrics():
    """Update Scenario 1 metrics from MinIO results"""
    log.info("📊 Updating Scenario 1 metrics...")

    # ── Huawei/Nokia ──
    hn_metrics = load_json_from_minio(BUCKET_RESULTS, "scenario1/huawei_nokia_metrics.json")
    if hn_metrics:
        acc  = hn_metrics["test"]["accuracy"]
        f1   = hn_metrics["test"]["f1_macro"]
        gap  = hn_metrics["overfitting"]["train_test_gap"]
        deg_recall = hn_metrics["monitoring_kpi"]["degraded_recall"]
        trigger    = hn_metrics["monitoring_kpi"]["trigger_retraining"]

        hn_accuracy.set(acc)
        hn_f1.set(f1)
        hn_overfit_gap.set(gap)
        hn_degraded_recall.set(deg_recall)
        hn_gatekeeping.set(1 if acc >= 0.80 and f1 >= 0.75 else 0)
        hn_retraining_trigger.set(1 if trigger else 0)
        log.info(f"  HN → Accuracy={acc:.4f}, F1={f1:.4f}, Gap={gap:.4f}")

    # ── HN Operational Predictions ──
    hn_preds = load_json_from_minio(BUCKET_RESULTS, "scenario1/huawei_nokia_operational_predictions.json")
    if hn_preds:
        optimal  = sum(1 for p in hn_preds if p["prediction"]["predicted_class"] == "optimal")
        degraded = sum(1 for p in hn_preds if p["prediction"]["predicted_class"] == "degraded")
        critical = sum(1 for p in hn_preds if p["prediction"]["predicted_class"] == "critical")
        hn_boxes_optimal.set(optimal)
        hn_boxes_degraded.set(degraded)
        hn_boxes_critical.set(critical)
        log.info(f"  HN boxes → Optimal={optimal}, Degraded={degraded}, Critical={critical}")

    # ── IGD ──
    igd_metrics = load_json_from_minio(BUCKET_RESULTS, "scenario1/IGD_metrics.json")
    if igd_metrics:
        acc  = igd_metrics["test"]["accuracy"]
        f1   = igd_metrics["test"]["f1_macro"]
        gap  = igd_metrics["overfitting"]["train_test_gap"]
        deg_recall = igd_metrics["monitoring_kpi"]["degraded_recall"]
        trigger    = igd_metrics["monitoring_kpi"]["trigger_retraining"]

        igd_accuracy.set(acc)
        igd_f1.set(f1)
        igd_overfit_gap.set(gap)
        igd_degraded_recall.set(deg_recall)
        igd_gatekeeping.set(1 if acc >= 0.80 and f1 >= 0.75 else 0)
        igd_retraining_trigger.set(1 if trigger else 0)
        log.info(f"  IGD → Accuracy={acc:.4f}, F1={f1:.4f}, Gap={gap:.4f}")

    # ── IGD Operational Predictions ──
    igd_preds = load_json_from_minio(BUCKET_RESULTS, "scenario1/IGD_operational_predictions.json")
    if igd_preds:
        optimal  = sum(1 for p in igd_preds if p["prediction"]["predicted_class"] == "optimal")
        degraded = sum(1 for p in igd_preds if p["prediction"]["predicted_class"] == "degraded")
        critical = sum(1 for p in igd_preds if p["prediction"]["predicted_class"] == "critical")
        igd_boxes_optimal.set(optimal)
        igd_boxes_degraded.set(degraded)
        igd_boxes_critical.set(critical)
        log.info(f"  IGD boxes → Optimal={optimal}, Degraded={degraded}, Critical={critical}")


def update_scenario2_metrics():
    """Update Scenario 2 metrics from MinIO results"""
    log.info("🔍 Updating Scenario 2 metrics...")

    # ── HN Root Cause ──
    hn_diag = load_json_from_minio(BUCKET_RESULTS, "scenario2/diagnostic_results_hn.json")
    if hn_diag:
        hn_diagnosed_total.set(len(hn_diag))
        cause_counts = {
            "perturbation": 0, "saturation": 0, "attenuation": 0,
            "vieillissement": 0, "macrobend": 0, "rupture": 0, "other": 0
        }
        for d in hn_diag:
            if "diagnostic" in d:
                cause = d["diagnostic"].get("cause_principale", "").lower()
                if "perturbation" in cause:      cause_counts["perturbation"] += 1
                elif "saturation" in cause:      cause_counts["saturation"] += 1
                elif "atténuation" in cause or "attenuation" in cause: cause_counts["attenuation"] += 1
                elif "vieillissement" in cause:  cause_counts["vieillissement"] += 1
                elif "macrobend" in cause:       cause_counts["macrobend"] += 1
                elif "rupture" in cause:         cause_counts["rupture"] += 1
                else:                            cause_counts["other"] += 1

        hn_cause_perturbation.set(cause_counts["perturbation"])
        hn_cause_saturation.set(cause_counts["saturation"])
        hn_cause_attenuation.set(cause_counts["attenuation"])
        hn_cause_vieillissement.set(cause_counts["vieillissement"])
        hn_cause_macrobend.set(cause_counts["macrobend"])
        hn_cause_rupture.set(cause_counts["rupture"])
        hn_cause_other.set(cause_counts["other"])
        log.info(f"  HN diagnosed: {len(hn_diag)} devices")

    # ── IGD Root Cause ──
    igd_diag = load_json_from_minio(BUCKET_RESULTS, "scenario2/diagnostic_results_igd.json")
    if igd_diag:
        igd_diagnosed_total.set(len(igd_diag))
        cause_counts = {
            "line_degradation": 0, "interference": 0,
            "crosstalk": 0, "config": 0, "other": 0
        }
        for d in igd_diag:
            if "diagnostic" in d:
                cause = d["diagnostic"].get("cause_principale", "").lower()
                if "line degradation" in cause or "dégradation" in cause: cause_counts["line_degradation"] += 1
                elif "interference" in cause or "noise" in cause:          cause_counts["interference"] += 1
                elif "crosstalk" in cause:                                  cause_counts["crosstalk"] += 1
                elif "configuration" in cause or "config" in cause:        cause_counts["config"] += 1
                else:                                                       cause_counts["other"] += 1

        igd_cause_line_degradation.set(cause_counts["line_degradation"])
        igd_cause_interference.set(cause_counts["interference"])
        igd_cause_crosstalk.set(cause_counts["crosstalk"])
        igd_cause_config.set(cause_counts["config"])
        igd_cause_other.set(cause_counts["other"])
        log.info(f"  IGD diagnosed: {len(igd_diag)} devices")


def update_scenario3_metrics():
    """Update Scenario 3 metrics from MinIO results"""
    log.info("⏱️  Updating Scenario 3 metrics...")

    # ── HN Proactive Prediction ──
    hn_s3 = load_json_from_minio(BUCKET_RESULTS, "scenario3/diagnostic_results_scenario3_hn.json")
    if hn_s3:
        counts = {"critique": 0, "eleve": 0, "modere": 0, "faible": 0, "stable": 0}
        slopes = []
        urgent = 0

        for d in hn_s3:
            niveau = d["prediction"]["niveau_risque"]
            counts[niveau] = counts.get(niveau, 0) + 1
            slopes.append(d["prediction"]["slope"])
            jours = d["prediction"]["jours_avant_seuil"]
            if jours is not None and jours <= 7:
                urgent += 1

        hn_risk_critique.set(counts["critique"])
        hn_risk_eleve.set(counts["eleve"])
        hn_risk_modere.set(counts["modere"])
        hn_risk_faible.set(counts["faible"])
        hn_risk_stable.set(counts["stable"])
        hn_avg_slope.set(sum(slopes) / len(slopes) if slopes else 0)
        hn_urgent_devices.set(urgent)
        hn_total_devices.set(len(hn_s3))
        log.info(f"  HN S3 → Critique={counts['critique']}, Élevé={counts['eleve']}, Urgent={urgent}")

    # ── IGD Proactive Prediction ──
    igd_s3 = load_json_from_minio(BUCKET_RESULTS, "scenario3/diagnostic_results_scenario3_igd.json")
    if igd_s3:
        counts = {"critique": 0, "eleve": 0, "modere": 0, "faible": 0, "stable": 0}
        slopes = []
        urgent = 0

        for d in igd_s3:
            niveau = d["prediction"]["niveau_risque"]
            counts[niveau] = counts.get(niveau, 0) + 1
            slopes.append(d["prediction"]["slope"])
            jours = d["prediction"]["jours_avant_seuil"]
            if jours is not None and jours <= 7:
                urgent += 1

        igd_risk_critique.set(counts["critique"])
        igd_risk_eleve.set(counts["eleve"])
        igd_risk_modere.set(counts["modere"])
        igd_risk_faible.set(counts["faible"])
        igd_risk_stable.set(counts["stable"])
        igd_avg_slope.set(sum(slopes) / len(slopes) if slopes else 0)
        igd_urgent_devices.set(urgent)
        igd_total_devices.set(len(igd_s3))
        log.info(f"  IGD S3 → Critique={counts['critique']}, Élevé={counts['eleve']}, Urgent={urgent}")


def update_all_metrics():
    """Update all metrics from MinIO"""
    log.info("="*50)
    log.info("🔄 Refreshing all MLOps metrics from MinIO...")
    try:
        update_scenario1_metrics()
    except Exception as e:
        log.error(f"Scenario 1 update failed: {e}")
    try:
        update_scenario2_metrics()
    except Exception as e:
        log.error(f"Scenario 2 update failed: {e}")
    try:
        update_scenario3_metrics()
    except Exception as e:
        log.error(f"Scenario 3 update failed: {e}")
    log.info("✅ All metrics updated")
    log.info("="*50)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    log.info("🚀 Starting MLOps Prometheus Exporter...")
    log.info(f"   MinIO    : {MINIO_ENDPOINT}")
    log.info(f"   Port     : {EXPORTER_PORT}")
    log.info(f"   Interval : {SCRAPE_INTERVAL}s")

    # Start HTTP server for Prometheus scraping
    start_http_server(EXPORTER_PORT)
    log.info(f"✅ Exporter running on http://localhost:{EXPORTER_PORT}/metrics")

    # Initial load
    update_all_metrics()

    # Continuous update loop
    while True:
        time.sleep(SCRAPE_INTERVAL)
        update_all_metrics()