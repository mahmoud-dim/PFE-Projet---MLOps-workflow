

# import pandas as pd
# import numpy as np
# import pickle
# import json
# import os
# from sklearn.metrics import (
#     classification_report, 
#     confusion_matrix, 
#     accuracy_score, 
#     f1_score,
#     precision_score,
#     recall_score
# )
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ======================== CONFIGURATION ========================
# MODEL_PATH = "components/healthscore_training/models/huawei_nokia_healthscore.pkl"
# DATA_SPLIT_PATH = "components/healthscore_training/models/huawei_nokia_data_split.pkl"
# TRAIN_INFO_PATH = "components/healthscore_training/models/huawei_nokia_train_info.json"

# # Outputs
# EVALUATION_DIR = "components/healthscore_training/evaluate"
# METRICS_OUTPUT_PATH = f"{EVALUATION_DIR}/huawei_nokia_metrics.json"
# CONFUSION_MATRIX_PATH = f"{EVALUATION_DIR}/huawei_nokia_confusion_matrix.png"
# CLASSIFICATION_REPORT_PATH = f"{EVALUATION_DIR}/huawei_nokia_classification_report.txt"
# FEATURE_IMPORTANCE_PATH = f"{EVALUATION_DIR}/huawei_nokia_feature_importance.png"

# # Seuils de gatekeeping
# ACCURACY_THRESHOLD = 0.80
# F1_THRESHOLD = 0.75

# # ======================== FONCTIONS ========================

# def load_model(path):
#     """Charge le modèle entraîné"""
#     print(f"📥 Loading model from {path}...")
#     with open(path, 'rb') as f:
#         model = pickle.load(f)
#     print("✅ Model loaded successfully")
#     return model

# def load_data_split(path):
#     """Charge les données train/test"""
#     print(f"📥 Loading data split from {path}...")
#     with open(path, 'rb') as f:
#         data_split = pickle.load(f)
    
#     X_train = data_split['X_train']
#     X_test = data_split['X_test']
#     y_train = data_split['y_train']
#     y_test = data_split['y_test']
    
#     print(f"✅ Data loaded:")
#     print(f"  Train: {X_train.shape[0]} samples")
#     print(f"  Test:  {X_test.shape[0]} samples")
    
#     return X_train, X_test, y_train, y_test

# def load_train_info(path):
#     """Charge les informations d'entraînement"""
#     print(f"📥 Loading training info from {path}...")
#     with open(path, 'r') as f:
#         train_info = json.load(f)
#     print("✅ Training info loaded")
#     return train_info

# def evaluate_model(model, X_train, X_test, y_train, y_test):
#     """Évalue le modèle sur train et test sets"""
#     print("\n📊 Evaluating model...")
    
#     # ========== PRÉDICTIONS ==========
#     print("⏳ Making predictions on train set...")
#     y_train_pred = model.predict(X_train)
    
#     print("⏳ Making predictions on test set...")
#     y_test_pred = model.predict(X_test)
    
#     # ========== MÉTRIQUES TRAIN ==========
#     train_accuracy = accuracy_score(y_train, y_train_pred)
#     train_f1_macro = f1_score(y_train, y_train_pred, average='macro', zero_division=0)
#     train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
#     train_precision_macro = precision_score(y_train, y_train_pred, average='macro', zero_division=0)
#     train_recall_macro = recall_score(y_train, y_train_pred, average='macro', zero_division=0)
    
#     # ========== MÉTRIQUES TEST ==========
#     test_accuracy = accuracy_score(y_test, y_test_pred)
#     test_f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
#     test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
#     test_precision_macro = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
#     test_recall_macro = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
    
#     # ========== MÉTRIQUES PAR CLASSE (TEST) ==========
#     test_precision_per_class = precision_score(y_test, y_test_pred, average=None, zero_division=0)
#     test_recall_per_class = recall_score(y_test, y_test_pred, average=None, zero_division=0)
#     test_f1_per_class = f1_score(y_test, y_test_pred, average=None, zero_division=0)
    
#     # ========== AFFICHAGE ==========
#     print("\n" + "="*75)
#     print("📈 PERFORMANCE METRICS")
#     print("="*75)
    
#     print(f"\n🏋️ TRAINING SET:")
#     print(f"  Accuracy:            {train_accuracy:.4f}")
#     print(f"  Precision (macro):   {train_precision_macro:.4f}")
#     print(f"  Recall (macro):      {train_recall_macro:.4f}")
#     print(f"  F1-Score (macro):    {train_f1_macro:.4f}")
#     print(f"  F1-Score (weighted): {train_f1_weighted:.4f}")
    
#     print(f"\n🎯 TEST SET:")
#     print(f"  Accuracy:            {test_accuracy:.4f}")
#     print(f"  Precision (macro):   {test_precision_macro:.4f}")
#     print(f"  Recall (macro):      {test_recall_macro:.4f}")
#     print(f"  F1-Score (macro):    {test_f1_macro:.4f}")
#     print(f"  F1-Score (weighted): {test_f1_weighted:.4f}")
    
#     # Check overfitting
#     overfitting_gap = train_accuracy - test_accuracy
#     print(f"\n📉 Overfitting Check:")
#     print(f"  Gap (Train - Test):  {overfitting_gap:.4f}")
#     if overfitting_gap > 0.15:
#         print(f"  ⚠️ WARNING: Possible overfitting (gap > 15%)")
#     elif overfitting_gap > 0.10:
#         print(f"  ⚠️ CAUTION: Moderate overfitting (gap > 10%)")
#     else:
#         print(f"  ✅ Good generalization (gap < 10%)")
    
#     # ========== MÉTRIQUES PAR CLASSE ==========
#     print(f"\n📊 PER-CLASS METRICS (Test Set):")
#     class_names = ['Optimal (0)', 'Dégradé (1)', 'Critique (2)']
    
#     for i, name in enumerate(class_names):
#         if i < len(test_precision_per_class):
#             print(f"\n  {name}:")
#             print(f"    Precision: {test_precision_per_class[i]:.4f}")
#             print(f"    Recall:    {test_recall_per_class[i]:.4f}")
#             print(f"    F1-Score:  {test_f1_per_class[i]:.4f}")
    
#     # ========== CLASSIFICATION REPORT ==========
#     print(f"\n📋 CLASSIFICATION REPORT (Test Set):")
#     report_str = classification_report(
#         y_test, y_test_pred, 
#         target_names=class_names,
#         zero_division=0
#     )
#     print(report_str)
    
#     # ========== CONFUSION MATRIX ==========
#     cm = confusion_matrix(y_test, y_test_pred)
#     print(f"\n🔢 CONFUSION MATRIX (Test Set):")
#     print(cm)
    
#     # Confusion matrix avec pourcentages
#     cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
#     print(f"\n🔢 CONFUSION MATRIX (Percentages):")
#     for i, row in enumerate(cm_percent):
#         print(f"  {class_names[i]:12s}: ", end="")
#         for j, val in enumerate(row):
#             print(f"{val:5.1f}%  ", end="")
#         print()
    
#     # ========== CONSTRUIRE DICTIONNAIRE MÉTRIQUES ==========
#     metrics = {
#         'train': {
#             'accuracy': float(train_accuracy),
#             'precision_macro': float(train_precision_macro),
#             'recall_macro': float(train_recall_macro),
#             'f1_macro': float(train_f1_macro),
#             'f1_weighted': float(train_f1_weighted)
#         },
#         'test': {
#             'accuracy': float(test_accuracy),
#             'precision_macro': float(test_precision_macro),
#             'recall_macro': float(test_recall_macro),
#             'f1_macro': float(test_f1_macro),
#             'f1_weighted': float(test_f1_weighted),
#             'per_class': {
#                 'precision': [float(p) for p in test_precision_per_class],
#                 'recall': [float(r) for r in test_recall_per_class],
#                 'f1_score': [float(f) for f in test_f1_per_class]
#             }
#         },
#         'overfitting': {
#             'train_test_gap': float(overfitting_gap),
#             'status': 'good' if overfitting_gap < 0.10 else 'moderate' if overfitting_gap < 0.15 else 'high'
#         },
#         'confusion_matrix': cm.tolist(),
#         'confusion_matrix_percent': cm_percent.tolist(),
#         'classification_report': classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
#     }
    
#     return metrics, cm, report_str

# def plot_confusion_matrix(cm, output_path):
#     """Génère et sauvegarde la matrice de confusion"""
#     print(f"\n📊 Generating confusion matrix plot...")
    
#     class_names = ['Optimal\n(0)', 'Dégradé\n(1)', 'Critique\n(2)']
    
#     fig, ax = plt.subplots(figsize=(10, 8))
    
#     # Heatmap
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=class_names,
#                 yticklabels=class_names,
#                 cbar_kws={'label': 'Count'},
#                 ax=ax)
    
#     ax.set_title('Confusion Matrix - Health Score Classification\nHuawei HG8145V5 + Nokia G-2425G-A (GPON)', 
#                  fontsize=14, fontweight='bold', pad=20)
#     ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
#     ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
#     # Ajouter pourcentages
#     total = cm.sum()
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             percentage = cm[i, j] / total * 100
#             ax.text(j + 0.5, i + 0.75, f'({percentage:.1f}%)', 
#                    ha='center', va='center', fontsize=10, color='gray', style='italic')
    
#     plt.tight_layout()
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"✅ Confusion matrix saved to {output_path}")

# def plot_feature_importance(train_info, output_path):
#     """Génère le graphique de feature importance"""
#     print(f"\n📊 Generating feature importance plot...")
    
#     features = [item['feature'] for item in train_info['feature_importance']]
#     importance = [item['importance'] * 100 for item in train_info['feature_importance']]  # En %
    
#     # Mapper les noms de features pour affichage
#     feature_labels = {
#         'rx_power_dbm': 'RxPower (dBm)',
#         'bias_current_ua': 'Bias Current (uA)',
#         'temperature_c': 'Temperature (°C)',
#         'supply_voltage_v': 'Supply Voltage (V)'
#     }
    
#     labels = [feature_labels.get(f, f) for f in features]
#     colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
#     fig, ax = plt.subplots(figsize=(10, 6))
#     bars = ax.barh(labels, importance, color=colors[:len(labels)])
    
#     ax.set_xlabel('Importance (%)', fontsize=12, fontweight='bold')
#     ax.set_title('Feature Importance - Health Score Classification\nHuawei HG8145V5 + Nokia G-2425G-A (GPON)', 
#                  fontsize=14, fontweight='bold', pad=20)
#     ax.set_xlim(0, max(importance) * 1.15)
#     ax.grid(axis='x', alpha=0.3, linestyle='--')
    
#     # Ajouter valeurs sur les barres
#     for bar, val in zip(bars, importance):
#         ax.text(val + 1, bar.get_y() + bar.get_height()/2, 
#                 f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')
    
#     plt.tight_layout()
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"✅ Feature importance plot saved to {output_path}")

# def save_metrics(metrics, path):
#     """Sauvegarde les métriques en JSON"""
#     print(f"\n💾 Saving metrics to {path}...")
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, 'w') as f:
#         json.dump(metrics, f, indent=2)
#     print("✅ Metrics saved successfully")

# def save_classification_report(report, path):
#     """Sauvegarde le classification report en texte"""
#     print(f"\n💾 Saving classification report to {path}...")
#     os.makedirs(os.path.dirname(path), exist_ok=True)
    
#     with open(path, 'w', encoding='utf-8') as f:
#         f.write("="*75 + "\n")
#         f.write("CLASSIFICATION REPORT - Health Score Classification\n")
#         f.write("Dataset: Huawei HG8145V5 + Nokia G-2425G-A (GPON)\n")
#         f.write("Model: Random Forest Classifier\n")
#         f.write("="*75 + "\n\n")
#         f.write(report)
    
#     print("✅ Classification report saved successfully")

# def check_gatekeeping(metrics, accuracy_threshold=ACCURACY_THRESHOLD, f1_threshold=F1_THRESHOLD):
#     """Vérifie si le modèle passe les seuils de gatekeeping"""
#     print("\n" + "="*75)
#     print("🚪 GATEKEEPING - Model Validation")
#     print("="*75)
    
#     test_accuracy = metrics['test']['accuracy']
#     test_f1 = metrics['test']['f1_macro']
#     overfitting_status = metrics['overfitting']['status']
    
#     print(f"\n📏 Thresholds:")
#     print(f"  Accuracy threshold:  {accuracy_threshold:.2f}")
#     print(f"  F1-Score threshold:  {f1_threshold:.2f}")
#     print(f"  Overfitting status:  Must be 'good' or 'moderate'")
    
#     print(f"\n📊 Model Performance:")
#     acc_pass = test_accuracy >= accuracy_threshold
#     f1_pass = test_f1 >= f1_threshold
#     overfit_pass = overfitting_status in ['good', 'moderate']
    
#     print(f"  Accuracy:  {test_accuracy:.4f}  {'✅ PASS' if acc_pass else '❌ FAIL'}")
#     print(f"  F1-Score:  {test_f1:.4f}  {'✅ PASS' if f1_pass else '❌ FAIL'}")
#     print(f"  Overfitting: {overfitting_status}  {'✅ PASS' if overfit_pass else '❌ FAIL'}")
    
#     passed = acc_pass and f1_pass and overfit_pass
    
#     if passed:
#         print("\n" + "="*75)
#         print("✅ GATEKEEPING PASSED")
#         print("="*75)
#         print("   ✓ Model meets all performance requirements")
#         print("   ✓ Ready for deployment")
#         print("   ✓ Can proceed to KServe serving")
#     else:
#         print("\n" + "="*75)
#         print("❌ GATEKEEPING FAILED")
#         print("="*75)
#         print("   ✗ Model performance below threshold")
#         print("\n   📋 Recommendations:")
#         if not acc_pass:
#             print("     • Accuracy too low - collect more training data")
#         if not f1_pass:
#             print("     • F1-Score too low - check class imbalance (use class_weight)")
#         if not overfit_pass:
#             print("     • High overfitting - increase regularization (min_samples_split)")
#         print("     • Try different hyperparameters (GridSearch)")
#         print("     • Consider ensemble methods (XGBoost, LightGBM)")
    
#     return passed

# # ======================== MAIN ========================

# def main():
#     print("="*75)
#     print("📊 EVALUATION - Scenario 1: Health Score Classification")
#     print("   Model: Random Forest Classifier")
#     print("   Dataset: Huawei HG8145V5 + Nokia G-2425G-A (GPON)")
#     print("="*75)
    
#     # 1. Charger le modèle
#     model = load_model(MODEL_PATH)
    
#     # 2. Charger les données
#     X_train, X_test, y_train, y_test = load_data_split(DATA_SPLIT_PATH)
    
#     # 3. Charger les infos d'entraînement
#     train_info = load_train_info(TRAIN_INFO_PATH)
#     print(f"\n📋 Model Info:")
#     print(f"  Type:              {train_info['model_type']}")
#     print(f"  N Estimators:      {train_info['n_estimators']}")
#     print(f"  Max Depth:         {train_info['max_depth']}")
#     print(f"  Features:          {train_info['n_features']}")
#     print(f"  Training samples:  {train_info['n_samples_train']}")
#     print(f"  Classes:           {train_info['classes']}")
    
#     # 4. Évaluer le modèle
#     metrics, cm, report = evaluate_model(model, X_train, X_test, y_train, y_test)
    
#     # 5. Générer visualisations
#     plot_confusion_matrix(cm, CONFUSION_MATRIX_PATH)
#     plot_feature_importance(train_info, FEATURE_IMPORTANCE_PATH)
    
#     # 6. Sauvegarder les métriques
#     save_metrics(metrics, METRICS_OUTPUT_PATH)
    
#     # 7. Sauvegarder le classification report
#     save_classification_report(report, CLASSIFICATION_REPORT_PATH)
    
#     # 8. Gatekeeping
#     passed = check_gatekeeping(metrics)
    
#     # 9. Résumé final
#     print("\n" + "="*75)
#     print("✅ EVALUATION COMPLETED")
#     print("="*75)
#     print(f"\n📁 Outputs:")
#     print(f"  Metrics (JSON):       {METRICS_OUTPUT_PATH}")
#     print(f"  Confusion Matrix:     {CONFUSION_MATRIX_PATH}")
#     print(f"  Feature Importance:   {FEATURE_IMPORTANCE_PATH}")
#     print(f"  Classification Report: {CLASSIFICATION_REPORT_PATH}")
    
#     print(f"\n🎯 Final Status:")
#     print(f"  Test Accuracy:  {metrics['test']['accuracy']:.4f}")
#     print(f"  Test F1-Score:  {metrics['test']['f1_macro']:.4f}")
#     print(f"  Gatekeeping:    {'PASSED ✅' if passed else 'FAILED ❌'}")
    
#     if passed:
#         print(f"\n💡 Next Steps:")
#         print(f"  1. Review confusion matrix: {CONFUSION_MATRIX_PATH}")
#         print(f"  2. Proceed to IGD training")
#         print(f"  3. Then: Dockerization & Kubeflow pipeline")
#     else:
#         print(f"\n💡 Next Steps:")
#         print(f"  1. Tune hyperparameters in train_huawei_nokia.py")
#         print(f"  2. Re-run training")
#         print(f"  3. Re-evaluate")
    
#     return 0 if passed else 1

# if __name__ == "__main__":
#     exit_code = main()
#     exit(exit_code)




import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ======================== CONFIGURATION ========================
MODEL_PATH      = "components/healthscore_training/models/huawei_nokia_healthscore.pkl"
DATA_SPLIT_PATH = "components/healthscore_training/models/huawei_nokia_data_split.pkl"
TRAIN_INFO_PATH = "components/healthscore_training/models/huawei_nokia_train_info.json"

# Outputs
EVALUATION_DIR             = "components/healthscore_training/evaluate"
METRICS_OUTPUT_PATH        = f"{EVALUATION_DIR}/huawei_nokia_metrics.json"
CONFUSION_MATRIX_PATH      = f"{EVALUATION_DIR}/huawei_nokia_confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = f"{EVALUATION_DIR}/huawei_nokia_classification_report.txt"
FEATURE_IMPORTANCE_PATH    = f"{EVALUATION_DIR}/huawei_nokia_feature_importance.png"
OPERATIONAL_OUTPUT_PATH    = f"{EVALUATION_DIR}/huawei_nokia_operational_predictions.json"  # ← NOUVEAU

# Seuils de gatekeeping
ACCURACY_THRESHOLD = 0.80
F1_THRESHOLD       = 0.75

# ======================== FONCTIONS ========================

def load_model(path):
    print(f"📥 Loading model from {path}...")
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully")
    return model

def load_data_split(path):
    """Charge les données train/test — incluant les device_id"""
    print(f"📥 Loading data split from {path}...")
    with open(path, 'rb') as f:
        data_split = pickle.load(f)

    X_train          = data_split['X_train']
    X_test           = data_split['X_test']
    y_train          = data_split['y_train']
    y_test           = data_split['y_test']
    device_id_train  = data_split['device_id_train']   # ← CHANGEMENT
    device_id_test   = data_split['device_id_test']    # ← CHANGEMENT

    print(f"✅ Data loaded:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test, device_id_train, device_id_test

def load_train_info(path):
    print(f"📥 Loading training info from {path}...")
    with open(path, 'r') as f:
        train_info = json.load(f)
    print("✅ Training info loaded")
    return train_info

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Évalue le modèle sur train et test sets"""
    print("\n📊 Evaluating model...")

    print("⏳ Making predictions on train set...")
    y_train_pred = model.predict(X_train)

    print("⏳ Making predictions on test set...")
    y_test_pred = model.predict(X_test)

    # ========== MÉTRIQUES TRAIN ==========
    train_accuracy        = accuracy_score(y_train, y_train_pred)
    train_f1_macro        = f1_score(y_train, y_train_pred, average='macro', zero_division=0)
    train_f1_weighted     = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
    train_precision_macro = precision_score(y_train, y_train_pred, average='macro', zero_division=0)
    train_recall_macro    = recall_score(y_train, y_train_pred, average='macro', zero_division=0)

    # ========== MÉTRIQUES TEST ==========
    test_accuracy        = accuracy_score(y_test, y_test_pred)
    test_f1_macro        = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_f1_weighted     = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_precision_macro = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_recall_macro    = recall_score(y_test, y_test_pred, average='macro', zero_division=0)

    # ========== MÉTRIQUES PAR CLASSE (TEST) ==========
    test_precision_per_class = precision_score(y_test, y_test_pred, average=None, zero_division=0)
    test_recall_per_class    = recall_score(y_test, y_test_pred, average=None, zero_division=0)
    test_f1_per_class        = f1_score(y_test, y_test_pred, average=None, zero_division=0)

    # ========== AFFICHAGE ==========
    print("\n" + "="*75)
    print("📈 PERFORMANCE METRICS")
    print("="*75)

    print(f"\n🏋️ TRAINING SET:")
    print(f"  Accuracy:            {train_accuracy:.4f}")
    print(f"  Precision (macro):   {train_precision_macro:.4f}")
    print(f"  Recall (macro):      {train_recall_macro:.4f}")
    print(f"  F1-Score (macro):    {train_f1_macro:.4f}")
    print(f"  F1-Score (weighted): {train_f1_weighted:.4f}")

    print(f"\n🎯 TEST SET:")
    print(f"  Accuracy:            {test_accuracy:.4f}")
    print(f"  Precision (macro):   {test_precision_macro:.4f}")
    print(f"  Recall (macro):      {test_recall_macro:.4f}")
    print(f"  F1-Score (macro):    {test_f1_macro:.4f}")
    print(f"  F1-Score (weighted): {test_f1_weighted:.4f}")

    overfitting_gap = train_accuracy - test_accuracy
    print(f"\n📉 Overfitting Check:")
    print(f"  Gap (Train - Test):  {overfitting_gap:.4f}")
    if overfitting_gap > 0.15:
        print(f"  ⚠️ WARNING: Possible overfitting (gap > 15%)")
    elif overfitting_gap > 0.10:
        print(f"  ⚠️ CAUTION: Moderate overfitting (gap > 10%)")
    else:
        print(f"  ✅ Good generalization (gap < 10%)")

    print(f"\n📊 PER-CLASS METRICS (Test Set):")
    class_names = ['Optimal (0)', 'Dégradé (1)', 'Critique (2)']
    for i, name in enumerate(class_names):
        if i < len(test_precision_per_class):
            print(f"\n  {name}:")
            print(f"    Precision: {test_precision_per_class[i]:.4f}")
            print(f"    Recall:    {test_recall_per_class[i]:.4f}")
            print(f"    F1-Score:  {test_f1_per_class[i]:.4f}")

    print(f"\n📋 CLASSIFICATION REPORT (Test Set):")
    report_str = classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0)
    print(report_str)

    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n🔢 CONFUSION MATRIX (Test Set):")
    print(cm)

    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    print(f"\n🔢 CONFUSION MATRIX (Percentages):")
    for i, row in enumerate(cm_percent):
        print(f"  {class_names[i]:12s}: ", end="")
        for j, val in enumerate(row):
            print(f"{val:5.1f}%  ", end="")
        print()

    metrics = {
        'train': {
            'accuracy':        float(train_accuracy),
            'precision_macro': float(train_precision_macro),
            'recall_macro':    float(train_recall_macro),
            'f1_macro':        float(train_f1_macro),
            'f1_weighted':     float(train_f1_weighted)
        },
        'test': {
            'accuracy':        float(test_accuracy),
            'precision_macro': float(test_precision_macro),
            'recall_macro':    float(test_recall_macro),
            'f1_macro':        float(test_f1_macro),
            'f1_weighted':     float(test_f1_weighted),
            'per_class': {
                'precision': [float(p) for p in test_precision_per_class],
                'recall':    [float(r) for r in test_recall_per_class],
                'f1_score':  [float(f) for f in test_f1_per_class]
            }
        },
        'overfitting': {
            'train_test_gap': float(overfitting_gap),
            'status': 'good' if overfitting_gap < 0.10 else 'moderate' if overfitting_gap < 0.15 else 'high'
        },
        'confusion_matrix':          cm.tolist(),
        'confusion_matrix_percent':  cm_percent.tolist(),
        'classification_report':     classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    }

    # ← CHANGEMENT : monitoring_kpi
    report_dict     = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    degraded_recall = report_dict.get('1', {}).get('recall', 0.0)

    metrics['monitoring_kpi'] = {
        'degraded_recall':      float(degraded_recall),
        'retraining_threshold': 0.55,
        'trigger_retraining':   bool(degraded_recall < 0.55)
    }

    print(f"\n🔔 MONITORING KPI:")
    print(f"  Dégradé (1) Recall:   {degraded_recall:.4f}")
    print(f"  Retraining Threshold: 0.55")
    print(f"  Trigger Retraining:   {'⚠️ YES' if degraded_recall < 0.55 else '✅ NO'}")

    return metrics, cm, report_str


def plot_confusion_matrix(cm, output_path):
    print(f"\n📊 Generating confusion matrix plot...")

    class_names = ['Optimal\n(0)', 'Dégradé\n(1)', 'Critique\n(2)']
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax)

    ax.set_title('Confusion Matrix - Health Score Classification\nHuawei HG8145V5 + Nokia G-2425G-A (GPON)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')

    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.75, f'({percentage:.1f}%)',
                   ha='center', va='center', fontsize=10, color='gray', style='italic')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to {output_path}")


def plot_feature_importance(train_info, output_path):
    print(f"\n📊 Generating feature importance plot...")

    features   = [item['feature'] for item in train_info['feature_importance']]
    importance = [item['importance'] * 100 for item in train_info['feature_importance']]

    feature_labels = {
        'rx_power_dbm':     'RxPower (dBm)',
        'bias_current_ua':  'Bias Current (uA)',
        'temperature_c':    'Temperature (°C)',
        'supply_voltage_v': 'Supply Voltage (V)'
    }
    labels = [feature_labels.get(f, f) for f in features]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, importance, color=colors[:len(labels)])

    ax.set_xlabel('Importance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance - Health Score Classification\nHuawei HG8145V5 + Nokia G-2425G-A (GPON)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, max(importance) * 1.15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for bar, val in zip(bars, importance):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Feature importance plot saved to {output_path}")


def save_metrics(metrics, path):
    print(f"\n💾 Saving metrics to {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✅ Metrics saved successfully")


def save_classification_report(report, path):
    print(f"\n💾 Saving classification report to {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("="*75 + "\n")
        f.write("CLASSIFICATION REPORT - Health Score Classification\n")
        f.write("Dataset: Huawei HG8145V5 + Nokia G-2425G-A (GPON)\n")
        f.write("Model: Random Forest Classifier\n")
        f.write("="*75 + "\n\n")
        f.write(report)
    print("✅ Classification report saved successfully")


def check_gatekeeping(metrics, accuracy_threshold=ACCURACY_THRESHOLD, f1_threshold=F1_THRESHOLD):
    print("\n" + "="*75)
    print("🚪 GATEKEEPING - Model Validation")
    print("="*75)

    test_accuracy      = metrics['test']['accuracy']
    test_f1            = metrics['test']['f1_macro']
    overfitting_status = metrics['overfitting']['status']

    acc_pass     = test_accuracy >= accuracy_threshold
    f1_pass      = test_f1 >= f1_threshold
    overfit_pass = overfitting_status in ['good', 'moderate']

    print(f"\n📏 Thresholds:")
    print(f"  Accuracy threshold:  {accuracy_threshold:.2f}")
    print(f"  F1-Score threshold:  {f1_threshold:.2f}")
    print(f"  Overfitting status:  Must be 'good' or 'moderate'")

    print(f"\n📊 Model Performance:")
    print(f"  Accuracy:    {test_accuracy:.4f}  {'✅ PASS' if acc_pass else '❌ FAIL'}")
    print(f"  F1-Score:    {test_f1:.4f}  {'✅ PASS' if f1_pass else '❌ FAIL'}")
    print(f"  Overfitting: {overfitting_status}  {'✅ PASS' if overfit_pass else '❌ FAIL'}")

    passed = acc_pass and f1_pass and overfit_pass

    if passed:
        print("\n" + "="*75)
        print("✅ GATEKEEPING PASSED — Ready for deployment")
        print("="*75)
    else:
        print("\n" + "="*75)
        print("❌ GATEKEEPING FAILED")
        print("="*75)
        if not acc_pass:
            print("  • Accuracy too low → collect more training data")
        if not f1_pass:
            print("  • F1-Score too low → check class imbalance (class_weight)")
        if not overfit_pass:
            print("  • High overfitting → augmenter min_samples_split")

    return passed


# ======================== OPERATIONAL DECISION ENGINE ========================

def compute_decision_metrics(model, X, device_ids):
    """
    ← NOUVEAU : Génère les métriques opérationnelles pour chaque device.
    Utilise les VRAIS device_id extraits du dataset
    (ex: HUAWEI_BOX_042, NOKIA_BOX_015)
    """
    print("\n🏭 Generating operational scoring...")

    class_map       = {0: 'optimal', 1: 'degraded', 2: 'critical'}
    probabilities   = model.predict_proba(X)
    predictions     = model.predict(X)
    device_ids_list = list(device_ids)

    results = []

    for i, (proba, pred_class) in enumerate(zip(probabilities, predictions)):

        device_id  = str(device_ids_list[i])   # ex: "HUAWEI_BOX_042"
        p_optimal  = float(proba[0])
        p_degraded = float(proba[1])
        p_critical = float(proba[2])

        # 1. Confidence
        confidence = round(float(np.max(proba)), 3)

        # 2. Health Score continu (0 → 1)
        health_score = round(
            (1.0 * p_optimal) + (0.5 * p_degraded) + (0.0 * p_critical), 3
        )

        # 3. Risk Score
        risk_score = round(1.0 - health_score, 3)

        # 4. Risk Level
        if risk_score < 0.3:
            risk_level = "low"
        elif risk_score < 0.6:
            risk_level = "medium"
        else:
            risk_level = "high"

        # 5. Recommended Action
        if p_critical > 0.7:
            recommended_action = "dispatch_technician"
        elif p_degraded > 0.7:
            recommended_action = "monitor"
        else:
            recommended_action = "no_action"

        # 6. Priority Level
        if risk_score > 0.8:
            priority_level = 1
        elif risk_score > 0.5:
            priority_level = 2
        else:
            priority_level = 3

        # 7. Auto Ticket
        auto_ticket = bool(p_critical > 0.8 and confidence > 0.8)

        results.append({
            "device_id": device_id,
            "prediction": {
                "predicted_class": class_map[int(pred_class)],
                "class_probabilities": {
                    "optimal":  round(p_optimal,  3),
                    "degraded": round(p_degraded, 3),
                    "critical": round(p_critical, 3)
                },
                "confidence": confidence
            },
            "health_metrics": {
                "health_score": health_score,
                "risk_score":   risk_score,
                "risk_level":   risk_level
            },
            "decision_support": {
                "recommended_action": recommended_action,
                "priority_level":     priority_level,
                "auto_ticket":        auto_ticket
            }
        })

    print(f"✅ Operational scoring generated for {len(results)} devices")
    return results


# ======================== MAIN ========================

def main():
    print("="*75)
    print("📊 EVALUATION - Scenario 1: Health Score Classification")
    print("   Model: Random Forest Classifier")
    print("   Dataset: Huawei HG8145V5 + Nokia G-2425G-A (GPON)")
    print("="*75)

    # 1. Charger le modèle
    model = load_model(MODEL_PATH)

    # 2. Charger les données (avec device_id)
    X_train, X_test, y_train, y_test, device_id_train, device_id_test = load_data_split(DATA_SPLIT_PATH)

    # 3. Charger les infos d'entraînement
    train_info = load_train_info(TRAIN_INFO_PATH)
    print(f"\n📋 Model Info:")
    print(f"  Type:             {train_info['model_type']}")
    print(f"  N Estimators:     {train_info['n_estimators']}")
    print(f"  Max Depth:        {train_info['max_depth']}")
    print(f"  Features:         {train_info['n_features']}")
    print(f"  Training samples: {train_info['n_samples_train']}")
    print(f"  Classes:          {train_info['classes']}")

    # 4. Évaluer le modèle
    metrics, cm, report = evaluate_model(model, X_train, X_test, y_train, y_test)

    # 5. Générer visualisations
    plot_confusion_matrix(cm, CONFUSION_MATRIX_PATH)
    plot_feature_importance(train_info, FEATURE_IMPORTANCE_PATH)

    # 6. Sauvegarder les métriques
    save_metrics(metrics, METRICS_OUTPUT_PATH)

    # 7. Sauvegarder le classification report
    save_classification_report(report, CLASSIFICATION_REPORT_PATH)

    # 8. Gatekeeping
    passed = check_gatekeeping(metrics)

    # 9. ← NOUVEAU : Operational output avec vrais device_id
    print("\n" + "="*75)
    print("🏭 OPERATIONAL PREDICTIONS")
    print("="*75)

    operational_results = compute_decision_metrics(
        model=model,
        X=X_test,
        device_ids=device_id_test   # ← HUAWEI_BOX_042, NOKIA_BOX_015...
    )

    print("\n📋 Sample Predictions (first 3):")
    for sample in operational_results[:3]:
        print(json.dumps(sample, indent=2))

    os.makedirs(os.path.dirname(OPERATIONAL_OUTPUT_PATH), exist_ok=True)
    with open(OPERATIONAL_OUTPUT_PATH, "w") as f:
        json.dump(operational_results, f, indent=2)
    print(f"\n💾 Operational predictions saved to {OPERATIONAL_OUTPUT_PATH}")

    # 10. Résumé final
    print("\n" + "="*75)
    print("✅ EVALUATION COMPLETED")
    print("="*75)
    print(f"\n📁 Outputs:")
    print(f"  Metrics (JSON):          {METRICS_OUTPUT_PATH}")
    print(f"  Confusion Matrix:        {CONFUSION_MATRIX_PATH}")
    print(f"  Feature Importance:      {FEATURE_IMPORTANCE_PATH}")
    print(f"  Classification Report:   {CLASSIFICATION_REPORT_PATH}")
    print(f"  Operational Predictions: {OPERATIONAL_OUTPUT_PATH}")

    print(f"\n🎯 Final Status:")
    print(f"  Test Accuracy:  {metrics['test']['accuracy']:.4f}")
    print(f"  Test F1-Score:  {metrics['test']['f1_macro']:.4f}")
    print(f"  Gatekeeping:    {'PASSED ✅' if passed else 'FAILED ❌'}")

    if passed:
        print(f"\n💡 Next Steps:")
        print(f"  1. Review confusion matrix: {CONFUSION_MATRIX_PATH}")
        print(f"  2. Dockerization & Kubeflow pipeline")
        print(f"  3. KServe serving")
    else:
        print(f"\n💡 Next Steps:")
        print(f"  1. Tune hyperparameters in train_huawei_nokia.py")
        print(f"  2. Re-run training then re-evaluate")

    return 0 if passed else 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)