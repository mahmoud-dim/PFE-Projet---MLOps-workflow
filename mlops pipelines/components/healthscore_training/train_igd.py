import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from imblearn.over_sampling import SMOTE

INPUT_PATH = "datasets/processed/IGD_scenario_1.csv"
MODEL_OUTPUT_PATH = "components/healthscore_training/models/IGD_healthscore.pkl"
DATA_SPLIT_OUTPUT_PATH = "components/healthscore_training/models/IGD_data_split.pkl"
TRAIN_INFO_PATH = "components/healthscore_training/models/IGD_train_info.json"

# Hyperparamètres
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 6
MIN_SAMPLES_SPLIT = 15

# ======================== FONCTIONS ========================

def load_processed_data(path):
    """Charge le dataset preprocessed"""
    print("📥 Loading processed dataset...")
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    return df

def split_features_target(df):
    """Sépare features et target, et garde device_id séparé"""
    print("\n🎯 Splitting features and target...")

    device_ids = df['device_id'].copy()  # conserver pour référence
    X = df.drop(['health_score', 'device_id'], axis=1)  # device_id exclu du modèle
    y = df['health_score']

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nClass distribution:")
    print(y.value_counts().sort_index())

    return X, y, device_ids

def split_train_test(X, y, device_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split dataset en train/test avec stratification et conservation des device_id"""
    print(f"\n✂️ Splitting data (test_size={test_size})...")

    X_train, X_test, y_train, y_test, device_id_train, device_id_test = train_test_split(
        X, y, device_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Afficher distribution par classe
    print("\nTrain distribution:")
    print(y_train.value_counts().sort_index())
    print("\nTest distribution:")
    print(y_test.value_counts().sort_index())

    return X_train, X_test, y_train, y_test, device_id_train, device_id_test

def train_random_forest(X_train, y_train):
    """Entraîne le modèle Random Forest"""
    print("\n🌲 Training Random Forest Classifier...")
    print(f"Hyperparameters:")
    print(f"  - n_estimators: {N_ESTIMATORS}")
    print(f"  - max_depth: {MAX_DEPTH}")
    print(f"  - min_samples_split: {MIN_SAMPLES_SPLIT}")
    print(f"  - random_state: {RANDOM_STATE}")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1,
        verbose=1
    )

    print("\n⏳ Training in progress...")
    model.fit(X_train, y_train)
    print("✅ Training completed")

    train_acc = model.score(X_train, y_train)
    print(f"Training accuracy: {train_acc:.4f}")

    return model

def save_model(model, path):
    """Sauvegarde le modèle entraîné"""
    print(f"\n💾 Saving model to {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print("✅ Model saved successfully")

def save_data_split(X_train, X_test, y_train, y_test, device_id_train, device_id_test, path):
    """Sauvegarde les données de train/test pour l'évaluation"""
    print(f"\n💾 Saving data split to {path}...")

    data_split = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'device_id_train': device_id_train,
        'device_id_test': device_id_test
    }

    with open(path, 'wb') as f:
        pickle.dump(data_split, f)
    print("✅ Data split saved successfully")

def save_train_info(model, X_train, path):
    """Sauvegarde les informations d'entraînement"""
    print(f"\n💾 Saving training info to {path}...")

    feature_importance = []
    for feature, importance in zip(X_train.columns, model.feature_importances_):
        feature_importance.append({
            'feature': feature,
            'importance': float(importance)
        })
    feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)

    train_info = {
        'model_type': 'RandomForestClassifier',
        'n_estimators': N_ESTIMATORS,
        'max_depth': MAX_DEPTH,
        'min_samples_split': MIN_SAMPLES_SPLIT,
        'random_state': RANDOM_STATE,
        'n_features': X_train.shape[1],
        'n_samples_train': X_train.shape[0],
        'feature_names': list(X_train.columns),
        'feature_importance': feature_importance,
        'classes': model.classes_.tolist()
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(train_info, f, indent=2)
    print("✅ Training info saved successfully")

    print("\n🔍 FEATURE IMPORTANCE:")
    for item in feature_importance:
        print(f"  {item['feature']:20s}: {item['importance']:.4f}")

# ======================== MAIN ========================

def main():
    print("="*60)
    print("🚀 TRAINING - Scenario 1: Health Score Classification")
    print("   Model: Random Forest Classifier")
    print("   Dataset: IGD ")
    print("="*60)

    # 1. Charger les données
    df = load_processed_data(INPUT_PATH)

    # 2. Séparer features, target et device_id
    X, y, device_ids = split_features_target(df)

    # 3. Split train/test
    X_train, X_test, y_train, y_test, device_id_train, device_id_test = split_train_test(X, y, device_ids)

    # 4. Entraîner le modèle
    model = train_random_forest(X_train, y_train)

    # 5. Sauvegarder le modèle
    save_model(model, MODEL_OUTPUT_PATH)

    # 6. Sauvegarder les données de split (pour évaluation)
    save_data_split(X_train, X_test, y_train, y_test, device_id_train, device_id_test, DATA_SPLIT_OUTPUT_PATH)

    # 7. Sauvegarder les infos d'entraînement
    save_train_info(model, X_train, TRAIN_INFO_PATH)

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  Model:      {MODEL_OUTPUT_PATH}")
    print(f"  Data split: {DATA_SPLIT_OUTPUT_PATH}")
    print(f"  Info:       {TRAIN_INFO_PATH}")
    print("\n💡 Next step: Run evaluate_igd.py")

if __name__ == "__main__":
    main()
