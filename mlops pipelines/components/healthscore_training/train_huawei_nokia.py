import pandas as pd
import pickle
import json
import os
import boto3
from io import StringIO, BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# train_huawei_nokia.py
# Training — Scénario 1 : Health Score Classification
# Input  : MinIO → datasets/processed/huawei_nokia_scenario_1.csv
# Output : MinIO → models/huawei_nokia_healthscore.pkl
#                  models/huawei_nokia_data_split.pkl
#                  models/huawei_nokia_train_info.json
# ============================================================

# ----------------------------------------
# Configuration MinIO
# ----------------------------------------
# MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "http://minio-service.kubeflow:9000")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://10.98.20.211:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
BUCKET_DATASETS  = "datasets"
BUCKET_MODELS    = "models"

INPUT_KEY      = "processed/huawei_nokia_scenario_1.csv"
# MODEL_KEY      = "huawei_nokia_healthscore.pkl"
# DATA_SPLIT_KEY = "huawei_nokia_data_split.pkl"
# TRAIN_INFO_KEY = "huawei_nokia_train_info.json"
MODEL_KEY      = "scenario1/huawei_nokia_healthscore.pkl"
DATA_SPLIT_KEY = "scenario1/huawei_nokia_data_split.pkl"
TRAIN_INFO_KEY = "scenario1/huawei_nokia_train_info.json"

# Hyperparamètres
RANDOM_STATE      = 42
TEST_SIZE         = 0.2
N_ESTIMATORS      = 100
MAX_DEPTH         = 7
MIN_SAMPLES_SPLIT = 10

# ----------------------------------------
# Client MinIO
# ----------------------------------------
s3 = boto3.client(
    "s3",
    endpoint_url          = MINIO_ENDPOINT,
    aws_access_key_id     = MINIO_ACCESS_KEY,
    aws_secret_access_key = MINIO_SECRET_KEY
)

# ======================== FONCTIONS MinIO ========================

def load_processed_data_from_minio():
    print("📥 Loading processed dataset from MinIO...")
    print(f"   Bucket : {BUCKET_DATASETS}")
    print(f"   Key    : {INPUT_KEY}")

    response = s3.get_object(Bucket=BUCKET_DATASETS, Key=INPUT_KEY)
    content  = response["Body"].read().decode("utf-8")
    df       = pd.read_csv(StringIO(content))

    print(f"   Dataset shape: {df.shape}")
    print(f"   Features: {list(df.columns)}")
    return df


def save_model_to_minio(model):
    print("\n💾 Saving model to MinIO...")
    print(f"   Bucket : {BUCKET_MODELS}")
    print(f"   Key    : {MODEL_KEY}")

    buffer = BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)

    s3.put_object(Bucket=BUCKET_MODELS, Key=MODEL_KEY, Body=buffer.getvalue())
    print("   ✅ Model saved successfully")


# def save_data_split_to_minio(X_train, X_test, y_train, y_test, device_id_train, device_id_test):
def save_data_split_to_minio(X_train, X_test, y_train, y_test,
                             device_id_train, device_id_test,
                             meta_train, meta_test):
    print("\n💾 Saving data split to MinIO...")
    print(f"   Bucket : {BUCKET_MODELS}")
    print(f"   Key    : {DATA_SPLIT_KEY}")

    data_split = {
        'X_train':         X_train,
        'X_test':          X_test,
        'y_train':         y_train,
        'y_test':          y_test,
        'device_id_train': device_id_train,
        'device_id_test':  device_id_test,
        'meta_train':      meta_train,
        'meta_test':       meta_test
    }

    buffer = BytesIO()
    pickle.dump(data_split, buffer)
    buffer.seek(0)

    s3.put_object(Bucket=BUCKET_MODELS, Key=DATA_SPLIT_KEY, Body=buffer.getvalue())
    print("   ✅ Data split saved successfully")


def save_train_info_to_minio(model, X_train):
    print("\n💾 Saving training info to MinIO...")
    print(f"   Bucket : {BUCKET_MODELS}")
    print(f"   Key    : {TRAIN_INFO_KEY}")

    feature_importance = []
    for feature, importance in zip(X_train.columns, model.feature_importances_):
        feature_importance.append({
            'feature':    feature,
            'importance': float(importance)
        })
    feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)

    train_info = {
        'model_type':         'RandomForestClassifier',
        'n_estimators':       N_ESTIMATORS,
        'max_depth':          MAX_DEPTH,
        'min_samples_split':  MIN_SAMPLES_SPLIT,
        'class_weight':       'balanced',
        'random_state':       RANDOM_STATE,
        'n_features':         X_train.shape[1],
        'n_samples_train':    X_train.shape[0],
        'feature_names':      list(X_train.columns),
        'feature_importance': feature_importance,
        'classes':            model.classes_.tolist()
    }

    s3.put_object(
        Bucket = BUCKET_MODELS,
        Key    = TRAIN_INFO_KEY,
        Body   = json.dumps(train_info, indent=2).encode("utf-8")
    )
    print("   ✅ Training info saved successfully")

    print("\n🔍 FEATURE IMPORTANCE:")
    for item in feature_importance:
        print(f"  {item['feature']:20s}: {item['importance']:.4f}")


# ======================== FONCTIONS TRAINING ========================

# def split_features_target(df):
#     print("\n🎯 Splitting features and target...")

#     device_ids = df['device_id'].copy()
#     X = df.drop(['health_score', 'device_id'], axis=1)
#     y = df['health_score']

#     print(f"   Features shape: {X.shape}")
#     print(f"   Target shape: {y.shape}")
#     print("\n   Class distribution:")
#     print(y.value_counts().sort_index())

#     return X, y, device_ids

def split_features_target(df):
    print("\n🎯 Splitting features and target...")

    device_ids = df['device_id'].copy()
    metadata = df[['vendor', 'city']].copy()   # kept aside, NOT features
    X = df.drop(['health_score', 'device_id', 'vendor', 'city'], axis=1)
    y = df['health_score']

    print(f"   Features shape: {X.shape}")
    print(f"   Feature columns: {list(X.columns)}")
    print(f"   Target shape: {y.shape}")
    print("\n   Class distribution:")
    print(y.value_counts().sort_index())

    return X, y, device_ids, metadata


# def split_train_test(X, y, device_ids):
#     print(f"\n✂️ Splitting data (test_size={TEST_SIZE})...")

#     X_train, X_test, y_train, y_test, device_id_train, device_id_test = train_test_split(
#         X, y, device_ids,
#         test_size    = TEST_SIZE,
#         random_state = RANDOM_STATE,
#         stratify     = y
#     )

#     print(f"   Training set: {X_train.shape[0]} samples")
#     print(f"   Test set: {X_test.shape[0]} samples")
#     print("\n   Train distribution:")
#     print(y_train.value_counts().sort_index())
#     print("\n   Test distribution:")
#     print(y_test.value_counts().sort_index())

#     return X_train, X_test, y_train, y_test, device_id_train, device_id_test
def split_train_test(X, y, device_ids, metadata):
    print(f"\n✂️ Splitting data (test_size={TEST_SIZE})...")

    (X_train, X_test, y_train, y_test,
     device_id_train, device_id_test,
     meta_train, meta_test) = train_test_split(
        X, y, device_ids, metadata,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y
    )

    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    print("\n   Train distribution:")
    print(y_train.value_counts().sort_index())
    print("\n   Test distribution:")
    print(y_test.value_counts().sort_index())

    return (X_train, X_test, y_train, y_test,
            device_id_train, device_id_test, meta_train, meta_test)


def train_random_forest(X_train, y_train):
    print("\n🌲 Training Random Forest Classifier...")
    print("   Hyperparameters:")
    print(f"     - n_estimators:      {N_ESTIMATORS}")
    print(f"     - max_depth:         {MAX_DEPTH}")
    print(f"     - min_samples_split: {MIN_SAMPLES_SPLIT}")
    print("     - class_weight:      balanced")
    print(f"     - random_state:      {RANDOM_STATE}")

    model = RandomForestClassifier(
        n_estimators      = N_ESTIMATORS,
        max_depth         = MAX_DEPTH,
        min_samples_split = MIN_SAMPLES_SPLIT,
        random_state      = RANDOM_STATE,
        class_weight      = 'balanced',
        n_jobs            = -1,
        verbose           = 1
    )

    print("\n⏳ Training in progress...")
    model.fit(X_train, y_train)
    print("✅ Training completed")

    train_acc = model.score(X_train, y_train)
    print(f"   Training accuracy: {train_acc:.4f}")

    return model


# ======================== MAIN ========================

def main():
    print("="*60)
    print("🚀 TRAINING - Scenario 1: Health Score Classification")
    print("   Model: Random Forest Classifier")
    print("   Dataset: Huawei + Nokia (GPON)")
    print("="*60)

    df = load_processed_data_from_minio()
    # X, y, device_ids = split_features_target(df)
    # X_train, X_test, y_train, y_test, device_id_train, device_id_test = split_train_test(X, y, device_ids)
    X, y, device_ids, metadata = split_features_target(df)
    (X_train, X_test, y_train, y_test,
     device_id_train, device_id_test,
     meta_train, meta_test) = split_train_test(X, y, device_ids, metadata)
    model = train_random_forest(X_train, y_train)
    save_model_to_minio(model)
    # save_data_split_to_minio(X_train, X_test, y_train, y_test, device_id_train, device_id_test)
    save_data_split_to_minio(X_train, X_test, y_train, y_test,
                             device_id_train, device_id_test,
                             meta_train, meta_test)
    save_train_info_to_minio(model, X_train)

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED")
    print("="*60)
    print(f"\n  Outputs in MinIO bucket '{BUCKET_MODELS}':")
    print(f"    Model      : {MODEL_KEY}")
    print(f"    Data split : {DATA_SPLIT_KEY}")
    print(f"    Info       : {TRAIN_INFO_KEY}")
    print("\n💡 Next step: Run evaluate_huawei_nokia.py")


if __name__ == "__main__":
    main()