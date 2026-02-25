# import pandas as pd
# import os


# # chemin input and output 
# INPUT_PATH = "datasets/raw/igd_realistic.csv"
# OUTPUT_PATH = "datasets/processed/IGD_scenario_1.csv"

# #Foncitons

# def load_data(path):
#     print("📥 Loading raw dataset...")
#     df = pd.read_csv(path)
#     print(f"Dataset shape: {df.shape}")
#     return df

# def clean_data(df):
#     print("🧹 Cleaning data...")
#     numeric_cols = ["downstream_curr_rate_kbps", "downstream_max_rate_kbps", "snr_margin_down_db", "attenuation_down_db","crc_errors_total"]
#     df["device_id"] = df["device_id"].astype(str)

#     #En numérique 
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
#     features = ["device_id","downstream_curr_rate_kbps", "downstream_max_rate_kbps", "snr_margin_down_db", "attenuation_down_db","crc_errors_total"]
#     target = "health_score"
#     df_si1 = df[features + [target]]
#     print("Class distribution:")
#     print(df_si1[target].value_counts())
#     return df_si1

# def save_data(df, path):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     df.to_csv(path, index=False)
#     print(f"💾 Saved processed dataset to {path}")

# # ----------------- Main -----------------
# df = load_data(INPUT_PATH)
# df = clean_data(df)
# df_si1 = select_features(df)
# save_data(df_si1, OUTPUT_PATH)

# print("✅ Preprocessing Scenario 1 completed successfully.")




import pandas as pd
import os


# chemin input and output
INPUT_PATH = "datasets/raw/igd_realistic.csv"
OUTPUT_PATH = "datasets/processed/IGD_scenario_1.csv"

# Fonctions

def load_data(path):
    print("📥 Loading raw dataset...")
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    return df

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

    # En numérique
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remplacer NaN par la moyenne
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)

    print(f"After cleaning: {df.shape}")
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

    print(f"  Before : {before} rows")
    print(f"  After  : {after} rows  (supprimé {before - after} duplications)")
    print(f"  Unique devices : {after}")

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

    print("Class distribution:")
    print(df_s1[target].value_counts().sort_index())
    return df_s1

def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n💾 Saved processed dataset to {path}")
    print(f"   Final shape: {df.shape}")

# ----------------- Main -----------------
df = load_data(INPUT_PATH)
df = clean_data(df)
df = keep_most_recent(df)   # ← garde le plus récent par device_id
df_s1 = select_features(df)
save_data(df_s1, OUTPUT_PATH)

print("\n✅ Preprocessing Scenario 1 completed successfully.")