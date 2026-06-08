import pandas as pd
import os
import boto3
from io import StringIO

# ============================================================
# preprocess_igd.py
# Preprocessing IGD — Scénario 1
# Input  : MinIO → datasets/raw/igd_realistic.csv
# Output : MinIO → datasets/processed/IGD_scenario_1.csv
# ============================================================

# ----------------------------------------
# Configuration MinIO
# ----------------------------------------
# MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "http://minio-service.kubeflow:9000")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://10.98.20.211:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
BUCKET_DATASETS  = "datasets"

INPUT_KEY  = "raw/igd_realistic.csv"
OUTPUT_KEY = "processed/IGD_scenario_1.csv"

# ----------------------------------------
# Client MinIO (compatible S3)
# ----------------------------------------
s3 = boto3.client(
    "s3",
    endpoint_url          = MINIO_ENDPOINT,
    aws_access_key_id     = MINIO_ACCESS_KEY,
    aws_secret_access_key = MINIO_SECRET_KEY
)

# ----------------- Fonctions MinIO -----------------

def load_data_from_minio():
    print("📥 Loading raw dataset from MinIO...")
    print(f"   Bucket : {BUCKET_DATASETS}")
    print(f"   Key    : {INPUT_KEY}")

    response = s3.get_object(Bucket=BUCKET_DATASETS, Key=INPUT_KEY)
    content  = response["Body"].read().decode("utf-8")
    df       = pd.read_csv(StringIO(content))

    print(f"   Dataset shape: {df.shape}")
    return df


def save_data_to_minio(df):
    print("\n💾 Saving processed dataset to MinIO...")
    print(f"   Bucket : {BUCKET_DATASETS}")
    print(f"   Key    : {OUTPUT_KEY}")

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    s3.put_object(
        Bucket = BUCKET_DATASETS,
        Key    = OUTPUT_KEY,
        Body   = csv_buffer.getvalue().encode("utf-8")
    )

    print("   ✅ Saved successfully")
    print(f"   Final shape: {df.shape}")


# ----------------- Fonctions Preprocessing -----------------

def clean_data(df):
    print("🧹 Cleaning data...")
    numeric_cols = [
        "downstream_curr_rate_kbps",
        "downstream_max_rate_kbps",
        "snr_margin_down_db",
        "attenuation_down_db",
        "crc_errors_total"
    ]

    df["device_id"] = df["device_id"].astype(str)

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)

    print(f"   After cleaning: {df.shape}")
    return df


def keep_most_recent(df):
    print("\n🕐 Keeping most recent record per device_id...")

    before = df.shape[0]

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", ascending=True)
    df = df.drop_duplicates(subset="device_id", keep="last")

    after = df.shape[0]

    print(f"   Before : {before} rows")
    print(f"   After  : {after} rows  (supprimé {before - after} duplications)")
    print(f"   Unique devices : {after}")

    return df


def select_features(df):
    print("\n🎯 Selecting features for Scenario 1...")
    features = [
        "device_id",
        "downstream_curr_rate_kbps",
        "downstream_max_rate_kbps",
        "snr_margin_down_db",
        "attenuation_down_db",
        "crc_errors_total"
    ]
    target = "health_score"
    df_s1 = df[features + [target]]

    print(f"   Features : {features}")
    print(f"   Target   : {target}")
    print("\n   Class distribution:")
    print(df_s1[target].value_counts().sort_index())
    return df_s1


# ----------------- Main -----------------
if __name__ == "__main__":

    print("="*60)
    print("🚀 PREPROCESSING — IGD Scénario 1")
    print("="*60)

    df    = load_data_from_minio()
    df    = clean_data(df)
    df    = keep_most_recent(df)
    df_s1 = select_features(df)
    save_data_to_minio(df_s1)

    print("\n✅ Preprocessing Scenario 1 (IGD) completed successfully.")