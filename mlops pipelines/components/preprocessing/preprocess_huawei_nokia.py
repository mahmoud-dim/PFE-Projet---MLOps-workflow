import pandas as pd
import os

# Chemins fixes
INPUT_PATH = "datasets/raw/fibre_combined_realistic.csv"
OUTPUT_PATH = "datasets/processed/huawei_nokia_scenario_1.csv"

# ----------------- Fonctions -----------------

def load_data(path):
    print("📥 Loading raw dataset...")
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    return df

def clean_data(df):
    print("🧹 Cleaning data...")
    numeric_cols = ["temperature_c", "rx_power_dbm", "bias_current_ua", "supply_voltage_v"]

    # Conversion en numérique
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remplacer NaN par la moyenne
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)

    # Supprimer duplications
    df = df.drop_duplicates()


    print(f"After cleaning: {df.shape}")
    return df

def select_features(df):
    print("🎯 Selecting features for Scenario 1...")
    features = ["temperature_c", "rx_power_dbm", "bias_current_ua", "supply_voltage_v"]
    target = "health_score"
    df_s1 = df[features + [target]]
    print("Class distribution:")
    print(df_s1[target].value_counts())
    return df_s1

def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"💾 Saved processed dataset to {path}")

# ----------------- Main -----------------
df = load_data(INPUT_PATH)
df = clean_data(df)
df_s1 = select_features(df)
save_data(df_s1, OUTPUT_PATH)

print("✅ Preprocessing Scenario 1 completed successfully.")





# import pandas as pd




# def load_data(input_path:str):
#     print("📥 Loading raw dataset...")
#     df = pd.read_csv(input_path)
#     print(f"Dataset shape: {df.shape}")
#     return df
# df= load_data("datasets/raw/fibre_combined_timeseries.csv")
