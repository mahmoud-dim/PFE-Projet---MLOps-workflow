# import pandas as pd
# import os

# # Chemins fixes
# INPUT_PATH = "datasets/raw/fibre_combined_realistic.csv"
# OUTPUT_PATH = "datasets/processed/huawei_nokia_scenario_1.csv"

# # ----------------- Fonctions -----------------

# def load_data(path):
#     print("📥 Loading raw dataset...")
#     df = pd.read_csv(path)
#     print(f"Dataset shape: {df.shape}")
#     return df

# def clean_data(df):
#     print("🧹 Cleaning data...")
#     numeric_cols = ["temperature_c", "rx_power_dbm", "bias_current_ua", "supply_voltage_v"]

#     # Conversion en numérique
#     for col in numeric_cols:
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     # Remplacer NaN par la moyenne
#     for col in numeric_cols:
#         if df[col].isnull().sum() > 0:
#             df[col].fillna(df[col].mean(), inplace=True)

#     # Supprimer duplications
#     df = df.drop_duplicates()


#     print(f"After cleaning: {df.shape}")
#     return df

# def select_features(df):
#     print("🎯 Selecting features for Scenario 1...")
#     features = ["temperature_c", "rx_power_dbm", "bias_current_ua", "supply_voltage_v"]
#     target = "health_score"
#     df_s1 = df[features + [target]]
#     print("Class distribution:")
#     print(df_s1[target].value_counts())
#     return df_s1

# def save_data(df, path):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     df.to_csv(path, index=False)
#     print(f"💾 Saved processed dataset to {path}")

# # ----------------- Main -----------------
# df = load_data(INPUT_PATH)
# df = clean_data(df)
# df_s1 = select_features(df)
# save_data(df_s1, OUTPUT_PATH)

# print("✅ Preprocessing Scenario 1 completed successfully.")




import pandas as pd
import os

# Chemins fixes
INPUT_PATH  = "datasets/raw/fibre_combined_realistic.csv"
OUTPUT_PATH = "datasets/processed/huawei_nokia_scenario_1.csv"

# ----------------- Fonctions -----------------

def load_data(path):
    print("📥 Loading raw dataset...")
    df = pd.read_csv(path)
    print(f"   Dataset shape: {df.shape}")

    # Répartition par fabricant
    huawei_count = df[df["device_id"].str.startswith("HUAWEI")].shape[0]
    nokia_count  = df[df["device_id"].str.startswith("NOKIA")].shape[0]
    print(f"   Huawei rows : {huawei_count}")
    print(f"   Nokia rows  : {nokia_count}")
    return df

def clean_data(df):
    print("\n🧹 Cleaning data...")

    numeric_cols = ["temperature_c", "rx_power_dbm", "bias_current_ua", "supply_voltage_v"]

    # device_id en string (ex: HUAWEI_BOX_012, NOKIA_BOX_005)
    df["device_id"] = df["device_id"].astype(str)

    # Conversion en numérique
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remplacer NaN par la moyenne
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)

    print(f"   After cleaning: {df.shape}")
    return df

def keep_most_recent(df):
    """
    Pour chaque device_id, garde uniquement la ligne
    avec le timestamp le plus récent (état actuel de la box).
    """
    print("\n🕐 Keeping most recent record per device_id...")

    before = df.shape[0]

    # Convertir timestamp en datetime pour un tri correct
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Trier par timestamp croissant puis garder la dernière ligne par device
    df = df.sort_values("timestamp", ascending=True)
    df = df.drop_duplicates(subset="device_id", keep="last")

    after = df.shape[0]

    print(f"   Before : {before} rows")
    print(f"   After  : {after} rows  (supprimé {before - after} duplications)")
    print(f"   Unique devices : {after}")

    # Répartition par fabricant après déduplication
    huawei_count = df[df["device_id"].str.startswith("HUAWEI")].shape[0]
    nokia_count  = df[df["device_id"].str.startswith("NOKIA")].shape[0]
    print(f"   Huawei devices : {huawei_count}")
    print(f"   Nokia devices  : {nokia_count}")

    return df

def select_features(df):
    print("\n🎯 Selecting features for Scenario 1...")

    # device_id inclus (string, non numérique — ex: HUAWEI_BOX_012)
    features = [
        "device_id",
        "temperature_c",
        "rx_power_dbm",
        "bias_current_ua",
        "supply_voltage_v"
    ]
    target = "health_score"

    df_s1 = df[features + [target]]

    print(f"   Features : {features}")
    print(f"   Target   : {target}")
    print(f"\n   Class distribution:")
    print(df_s1[target].value_counts().sort_index())
    print(f"\n   Sample device_ids:")
    print(df_s1["device_id"].head(5).tolist())

    return df_s1

def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n💾 Saved processed dataset to {path}")
    print(f"   Final shape: {df.shape}")

# ----------------- Main -----------------
df    = load_data(INPUT_PATH)
df    = clean_data(df)
df    = keep_most_recent(df)   # ← garde le plus récent par device_id
df_s1 = select_features(df)
save_data(df_s1, OUTPUT_PATH)

print("\n✅ Preprocessing Scenario 1 (Huawei + Nokia) completed successfully.")