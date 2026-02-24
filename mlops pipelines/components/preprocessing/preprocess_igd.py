import pandas as pd
import os


# chemin input and output 
INPUT_PATH = "datasets/raw/igd_realistic.csv"
OUTPUT_PATH = "datasets/processed/IGD_scenario_1.csv"

#Foncitons

def load_data(path):
    print("📥 Loading raw dataset...")
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    return df

def clean_data(df):
    print("🧹 Cleaning data...")
    numeric_cols = ["downstream_curr_rate_kbps", "downstream_max_rate_kbps", "snr_margin_down_db", "attenuation_down_db","crc_errors_total"]

    #En numérique 
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
    features = ["downstream_curr_rate_kbps", "downstream_max_rate_kbps", "snr_margin_down_db", "attenuation_down_db","crc_errors_total"]
    target = "health_score"
    df_si1 = df[features + [target]]
    print("Class distribution:")
    print(df_si1[target].value_counts())
    return df_si1

def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"💾 Saved processed dataset to {path}")

# ----------------- Main -----------------
df = load_data(INPUT_PATH)
df = clean_data(df)
df_si1 = select_features(df)
save_data(df_si1, OUTPUT_PATH)

print("✅ Preprocessing Scenario 1 completed successfully.")