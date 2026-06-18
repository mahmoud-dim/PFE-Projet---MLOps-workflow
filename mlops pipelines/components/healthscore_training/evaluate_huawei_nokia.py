import numpy as np
import pickle
import json
import os
import boto3
from io import BytesIO
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# evaluate_huawei_nokia.py
# Evaluation + Gatekeeping — Scénario 1
# Input  : MinIO → models/huawei_nokia_*.pkl / *.json
# Output : MinIO → results/scenario1/huawei_nokia_*.json / *.png
# ============================================================

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://10.98.20.211:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
BUCKET_MODELS    = "models"
BUCKET_RESULTS   = "results"

MODEL_KEY      = "scenario1/huawei_nokia_healthscore.pkl"
DATA_SPLIT_KEY = "scenario1/huawei_nokia_data_split.pkl"
TRAIN_INFO_KEY = "scenario1/huawei_nokia_train_info.json"
METRICS_KEY    = "scenario1/huawei_nokia_metrics.json"
REPORT_KEY     = "scenario1/huawei_nokia_classification_report.txt"
OPERATIONAL_KEY= "scenario1/huawei_nokia_operational_predictions.json"
CM_KEY         = "scenario1/huawei_nokia_confusion_matrix.png"

ACCURACY_THRESHOLD = 0.80
F1_THRESHOLD       = 0.75

s3 = boto3.client(
    "s3",
    endpoint_url          = MINIO_ENDPOINT,
    aws_access_key_id     = MINIO_ACCESS_KEY,
    aws_secret_access_key = MINIO_SECRET_KEY
)

# ======================== FONCTIONS MinIO ========================

def load_model_from_minio():
    print(f"📥 Loading model from MinIO... Key: {MODEL_KEY}")
    response = s3.get_object(Bucket=BUCKET_MODELS, Key=MODEL_KEY)
    model    = pickle.loads(response["Body"].read())  # nosec B301
    print("✅ Model loaded successfully")
    return model

def load_data_split_from_minio():
    print(f"📥 Loading data split from MinIO... Key: {DATA_SPLIT_KEY}")
    response   = s3.get_object(Bucket=BUCKET_MODELS, Key=DATA_SPLIT_KEY)
    data_split = pickle.loads(response["Body"].read())  # nosec B301
    X_train, X_test   = data_split['X_train'], data_split['X_test']
    y_train, y_test   = data_split['y_train'], data_split['y_test']
    did_train, did_test = data_split['device_id_train'], data_split['device_id_test']
    meta_train, meta_test = data_split['meta_train'], data_split['meta_test']
    print(f"✅ Data loaded: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, did_train, did_test, meta_train, meta_test

def load_train_info_from_minio():
    print(f"📥 Loading training info from MinIO... Key: {TRAIN_INFO_KEY}")
    response = s3.get_object(Bucket=BUCKET_MODELS, Key=TRAIN_INFO_KEY)
    return json.loads(response["Body"].read().decode("utf-8"))

def save_to_minio(key, body, content_type="application/json"):
    s3.put_object(Bucket=BUCKET_RESULTS, Key=key, Body=body, ContentType=content_type)
    print(f"   ✅ Saved → {key}")

# ======================== FONCTIONS EVALUATION ========================

def evaluate_model(model, X_train, X_test, y_train, y_test):
    print("\n📊 Evaluating model...")
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy  = accuracy_score(y_test, y_test_pred)
    test_f1_macro  = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_f1_w      = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_prec      = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_rec       = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_prec_cls  = precision_score(y_test, y_test_pred, average=None, zero_division=0)
    test_rec_cls   = recall_score(y_test, y_test_pred, average=None, zero_division=0)
    test_f1_cls    = f1_score(y_test, y_test_pred, average=None, zero_division=0)

    overfitting_gap = train_accuracy - test_accuracy
    class_names     = ['Optimal (0)', 'Dégradé (1)', 'Critique (2)']
    cm              = confusion_matrix(y_test, y_test_pred)
    report_str      = classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0)
    report_dict     = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    degraded_recall = report_dict.get('1', {}).get('recall', 0.0)

    print(f"\n🏋️ Train Accuracy : {train_accuracy:.4f}")
    print(f"🎯 Test Accuracy  : {test_accuracy:.4f}")
    print(f"📉 Overfit Gap    : {overfitting_gap:.4f}")
    print(f"\n📋 CLASSIFICATION REPORT:\n{report_str}")
    print(f"🔢 CONFUSION MATRIX:\n{cm}")
    print(f"\n🔔 Dégradé Recall : {degraded_recall:.4f} | Trigger: {'⚠️ YES' if degraded_recall < 0.55 else '✅ NO'}")

    metrics = {
        'train': {'accuracy': float(train_accuracy), 'f1_macro': float(f1_score(y_train, y_train_pred, average='macro', zero_division=0))},
        'test': {
            'accuracy': float(test_accuracy), 'f1_macro': float(test_f1_macro),
            'f1_weighted': float(test_f1_w), 'precision_macro': float(test_prec), 'recall_macro': float(test_rec),
            'per_class': {
                'precision': [float(p) for p in test_prec_cls],
                'recall':    [float(r) for r in test_rec_cls],
                'f1_score':  [float(f) for f in test_f1_cls]
            }
        },
        'overfitting': {
            'train_test_gap': float(overfitting_gap),
            'status': 'good' if overfitting_gap < 0.10 else 'moderate' if overfitting_gap < 0.15 else 'high'
        },
        'confusion_matrix':      cm.tolist(),
        'classification_report': report_dict,
        'monitoring_kpi': {
            'degraded_recall': float(degraded_recall),
            'retraining_threshold': 0.55,
            'trigger_retraining': bool(degraded_recall < 0.55)
        }
    }
    return metrics, cm, report_str

def evaluate_per_vendor(model, X_test, y_test, meta_test):
    print("\n📊 Per-vendor evaluation (Huawei vs Nokia)...")
    y_pred  = model.predict(X_test)
    y_true  = np.array(y_test)
    vendors = np.array(meta_test['vendor'])
    per_vendor = {}
    for v in ['HUAWEI', 'NOKIA']:
        mask = vendors == v
        n = int(mask.sum())
        if n == 0:
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        f1m = f1_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
        per_vendor[v] = {'n_test': n, 'accuracy': float(acc), 'f1_macro': float(f1m)}
        print(f"   {v}: n={n}  acc={acc:.4f}  f1={f1m:.4f}")
    return per_vendor

def check_gatekeeping(metrics):
    print("\n" + "="*75)
    print("🚪 GATEKEEPING - Model Validation")
    print("="*75)
    test_accuracy = metrics['test']['accuracy']
    test_f1       = metrics['test']['f1_macro']
    overfit_status= metrics['overfitting']['status']

    acc_pass     = test_accuracy >= ACCURACY_THRESHOLD
    f1_pass      = test_f1 >= F1_THRESHOLD
    overfit_pass = overfit_status in ['good', 'moderate']

    print(f"  Accuracy  : {test_accuracy:.4f}  {'✅ PASS' if acc_pass else '❌ FAIL'}")
    print(f"  F1-Score  : {test_f1:.4f}  {'✅ PASS' if f1_pass else '❌ FAIL'}")
    print(f"  Overfitting: {overfit_status}  {'✅ PASS' if overfit_pass else '❌ FAIL'}")

    passed = acc_pass and f1_pass and overfit_pass
    if passed:
        print("\n✅ GATEKEEPING PASSED — Ready for deployment")
    else:
        print("\n❌ GATEKEEPING FAILED — Pipeline stopped")
        exit(1)
    return passed

def compute_decision_metrics(model, X, device_ids, metadata):
    print("\n🏭 Generating operational scoring...")
    class_map     = {0: 'optimal', 1: 'degraded', 2: 'critical'}
    probabilities = model.predict_proba(X)
    predictions   = model.predict(X)
    results       = []

    for i, (proba, pred_class) in enumerate(zip(probabilities, predictions)):
        device_id  = str(list(device_ids)[i])
        vendor     = str(metadata.iloc[i]['vendor'])
        city       = str(metadata.iloc[i]['city'])
        p_opt, p_deg, p_crit = float(proba[0]), float(proba[1]), float(proba[2])
        confidence   = round(float(np.max(proba)), 3)
        health_score = round(1.0*p_opt + 0.5*p_deg + 0.0*p_crit, 3)
        risk_score   = round(1.0 - health_score, 3)

        risk_level         = "low" if risk_score < 0.3 else "medium" if risk_score < 0.6 else "high"
        recommended_action = "dispatch_technician" if p_crit > 0.7 else "monitor" if p_deg > 0.7 else "no_action"
        priority_level     = 1 if risk_score > 0.8 else 2 if risk_score > 0.5 else 3

        results.append({
            "device_id": device_id,
            "vendor": vendor,
            "city": city,
            "prediction": {
                "predicted_class": class_map[int(pred_class)],
                "class_probabilities": {"optimal": round(p_opt,3), "degraded": round(p_deg,3), "critical": round(p_crit,3)},
                "confidence": confidence
            },
            "health_metrics": {"health_score": health_score, "risk_score": risk_score, "risk_level": risk_level},
            "decision_support": {"recommended_action": recommended_action, "priority_level": priority_level, "auto_ticket": bool(p_crit > 0.8 and confidence > 0.8)}
        })

    print(f"✅ Operational scoring generated for {len(results)} devices")
    return results

def save_confusion_matrix_to_minio(cm):
    class_names = ['Optimal\n(0)', 'Dégradé\n(1)', 'Critique\n(2)']
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Confusion Matrix - Huawei/Nokia GPON', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0); plt.close()
    save_to_minio(CM_KEY, buffer.getvalue(), "image/png")

# ======================== MAIN ========================

def main():
    print("="*75)
    print("📊 EVALUATION - Scenario 1 : Huawei/Nokia GPON")
    print("="*75)

    model      = load_model_from_minio()
    X_train, X_test, y_train, y_test, did_train, did_test, meta_train, meta_test = load_data_split_from_minio()
    train_info = load_train_info_from_minio()

    metrics, cm, report = evaluate_model(model, X_train, X_test, y_train, y_test)
    metrics['per_vendor'] = evaluate_per_vendor(model, X_test, y_test, meta_test)

    print("\n💾 Saving results to MinIO...")
    save_to_minio(METRICS_KEY,  json.dumps(metrics, indent=2).encode("utf-8"))
    save_to_minio(REPORT_KEY,   report.encode("utf-8"), "text/plain")
    save_confusion_matrix_to_minio(cm)

    passed = check_gatekeeping(metrics)

    operational = compute_decision_metrics(model, X_test, did_test, meta_test)
    print("\n📋 Sample (first 3):")
    for s in operational[:3]:
        print(json.dumps(s, indent=2))

    save_to_minio(OPERATIONAL_KEY, json.dumps(operational, indent=2).encode("utf-8"))

    print("\n" + "="*75)
    print("✅ EVALUATION COMPLETED")
    print("="*75)
    print(f"  Accuracy    : {metrics['test']['accuracy']:.4f}")
    print(f"  F1-Score    : {metrics['test']['f1_macro']:.4f}")
    print(f"  Gatekeeping : {'PASSED ✅' if passed else 'FAILED ❌'}")

    return 0 if passed else 1

if __name__ == "__main__":
    exit(main())