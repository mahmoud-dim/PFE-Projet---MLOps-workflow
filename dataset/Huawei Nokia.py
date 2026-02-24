import pandas as pd

# Charger les datasets
df_huawei = pd.read_csv("datasets/huawei_realistic.csv")
df_nokia  = pd.read_csv("datasets/nokia_realistic.csv")

# =========================
# AJOUT DE LA COLONNE VENDOR
# =========================
df_huawei["vendor"] = "HUAWEI"
df_nokia["vendor"]  = "NOKIA"

print("Huawei columns:", df_huawei.columns)
print("Nokia columns:", df_nokia.columns)

# Vérifier que les colonnes (hors vendor) sont identiques
common_columns_match = all(
    df_huawei.drop(columns=["vendor"]).columns ==
    df_nokia.drop(columns=["vendor"]).columns
)
print("Columns match (hors vendor):", common_columns_match)

# =========================
# FUSION DES DATASETS
# =========================
df_combined = pd.concat([df_huawei, df_nokia], ignore_index=True)

print("Nombre de lignes Huawei:", len(df_huawei))
print("Nombre de lignes Nokia:", len(df_nokia))
print("Nombre de lignes combinées:", len(df_combined))

print(df_combined.head())

# =========================
# SAUVEGARDE
# =========================
df_combined.to_csv("datasets/fibre_combined_realistic.csv", index=False)

print("✅ Datasets Huawei et Nokia fusionnés avec succès avec colonne vendor !")
