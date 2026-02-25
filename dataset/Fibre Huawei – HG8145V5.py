"""
Générateur de dataset réaliste pour Huawei HG8145V5 (GPON)
300 devices × 10 samples = 3000 lignes
Basé sur normes physiques réelles

Usage:
  - Scénario 1 : preprocess garde le snapshot le plus récent → 300 lignes
  - Scénario 3 : preprocess garde toutes les lignes → 3000 lignes (time series)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# ======================== CONFIGURATION ========================
OUTPUT_PATH = "dataset/datasets/huawei_realistic.csv"
N_DEVICES = 300                 # ← 300 devices uniques
N_SAMPLES_PER_DEVICE = 10       # ← 10 mesures par device (toutes les 6h)
MEASUREMENT_INTERVAL_HOURS = 6  # → couvre 2.5 jours par device

# Normes GPON Huawei
NORMS = {
    'rx_power_lower':       -28,    # dBm
    'rx_power_upper':        -8,    # dBm
    'rx_power_optimal_min': -20,
    'rx_power_optimal_max': -12,
    'bias_current_normal':  10000,  # uA
    'bias_current_max':     15000,  # uA
    'temp_normal':             50,  # °C
    'temp_max':                70,  # °C
    'voltage_nominal':        3.3,  # V
}

# ======================== FONCTIONS ========================

def generate_device_timeseries(device_id, n_samples):
    """
    Génère série temporelle pour un device Huawei GPON.
    RxPower = DRIVER PRINCIPAL de la santé
    BiasCurrent = indicateur santé laser
    Temperature = secondaire (environnement + stress laser)
    """

    # Profil du device (fixe pour toute la série)
    profile = random.choices(
        ['stable_optimal', 'fiber_degradation', 'critical_failure', 'laser_aging'],
        weights=[0.50, 0.25, 0.10, 0.15]
    )[0]

    # Point de départ : il y a 2.5 jours
    start_time = datetime.now() - timedelta(days=2, hours=12)
    samples = []

    for i in range(n_samples):
        timestamp = start_time + timedelta(hours=MEASUREMENT_INTERVAL_HOURS * i)
        hour = timestamp.hour

        # ========== PROFIL 1: STABLE OPTIMAL (50%) ==========
        if profile == 'stable_optimal':

            # RxPower excellent (signal fort)
            rx_power = np.random.uniform(-20, -12)

            # BiasCurrent normal (laser en bonne santé)
            bias_current = np.random.uniform(8000, 11000)

            # Température suit cycle jour/nuit
            if 10 <= hour <= 18:
                temperature = np.random.uniform(48, 58)
            else:
                temperature = np.random.uniform(38, 48)

            supply_voltage = np.random.uniform(3.25, 3.35)
            tx_power       = np.random.uniform(2, 4)

            health_score = 0
            root_cause   = "NORMAL"

        # ========== PROFIL 2: DÉGRADATION FIBRE (25%) ==========
        elif profile == 'fiber_degradation':

            # Progression temporelle : signal se dégrade sur les 10 mesures
            progress = i / (n_samples - 1)  # 0.0 → 1.0

            # RxPower part de -18 → -27 dBm
            rx_power = -18 - (progress * 9)

            # Laser force pour compenser → Bias augmente
            bias_current = 9000 + (progress * 4500)

            # Laser qui force → Température augmente
            temperature = 48 + (progress * 18)
            if 10 <= hour <= 18:
                temperature += np.random.uniform(5, 10)

            supply_voltage = np.random.uniform(3.2, 3.35)
            tx_power       = np.random.uniform(2.5, 5)

            # Health basé sur RxPower (seuils réels GPON)
            if rx_power > -24:
                health_score = 0
                root_cause   = "NORMAL"
            elif rx_power > -26:
                health_score = 1
                root_cause   = "OPTICAL_AGING"
            else:
                health_score = 1
                root_cause   = "OPTICAL_AGING"

        # ========== PROFIL 3: PANNE CRITIQUE (10%) ==========
        elif profile == 'critical_failure':

            # Signal très faible (fibre cassée / connecteur sale)
            rx_power = np.random.uniform(-32, -28)

            # Laser au maximum (essaie de compenser)
            bias_current = np.random.uniform(14500, 18000)

            # Laser surchargé → Température très élevée
            temperature = np.random.uniform(68, 78)

            supply_voltage = np.random.uniform(3.05, 3.25)
            tx_power       = np.random.uniform(4, 6.5)

            health_score = 2
            root_cause   = "CRITICAL_FIBER_ISSUE"

        # ========== PROFIL 4: VIEILLISSEMENT LASER (15%) ==========
        else:  # laser_aging

            # Signal acceptable mais pas optimal
            rx_power = np.random.uniform(-24, -18)

            # BiasCurrent élevé (laser vieux force)
            bias_current = np.random.uniform(12500, 14500)

            # Température légèrement élevée
            temperature = np.random.uniform(55, 68)
            if 10 <= hour <= 18:
                temperature += np.random.uniform(3, 8)

            supply_voltage = np.random.uniform(3.15, 3.30)
            tx_power       = np.random.uniform(3, 5)

            if bias_current > 13500:
                health_score = 1
                root_cause   = "OPTICAL_AGING"
            else:
                health_score = 0
                root_cause   = "NORMAL"

        # Bruit réaliste (mesures physiques jamais parfaites)
        rx_power       += np.random.normal(0, 0.5)
        bias_current   += np.random.normal(0, 150)
        temperature    += np.random.normal(0, 2)
        supply_voltage += np.random.normal(0, 0.03)
        tx_power       += np.random.normal(0, 0.3)

        samples.append({
            'device_id':        device_id,
            'product_class':    'HG8145V5',
            'timestamp':        timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'rx_power_dbm':     round(rx_power, 2),
            'tx_power_dbm':     round(tx_power, 2),
            'bias_current_ua':  round(bias_current, 2),
            'supply_voltage_v': round(supply_voltage, 2),
            'temperature_c':    round(temperature, 2),
            'rx_power_lower':   NORMS['rx_power_lower'],
            'rx_power_upper':   NORMS['rx_power_upper'],
            'health_score':     health_score,
            'root_cause':       root_cause
        })

    return samples


# ======================== MAIN ========================

def main():
    print("="*70)
    print("🚀 HUAWEI HG8145V5 DATASET GENERATOR")
    print(f"   {N_DEVICES} devices × {N_SAMPLES_PER_DEVICE} samples = "
          f"{N_DEVICES * N_SAMPLES_PER_DEVICE} lignes total")
    print("   Realistic GPON Fiber Optics Behavior")
    print("="*70)

    all_samples = []

    print(f"\n📊 Generating data for {N_DEVICES} devices...")
    print(f"   {N_SAMPLES_PER_DEVICE} measurements per device "
          f"(every {MEASUREMENT_INTERVAL_HOURS}h → covers 2.5 days)")

    for i in range(N_DEVICES):
        device_id = f"HUAWEI_BOX_{i+1:03d}"
        samples   = generate_device_timeseries(device_id, N_SAMPLES_PER_DEVICE)
        all_samples.extend(samples)

        if (i + 1) % 50 == 0:
            print(f"   ✅ Generated {i+1}/{N_DEVICES} devices...")

    # Créer DataFrame
    df = pd.DataFrame(all_samples)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Statistiques
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total rows    : {len(df)}")
    print(f"   Unique devices: {df['device_id'].nunique()}")
    print(f"   Rows/device   : {N_SAMPLES_PER_DEVICE}")
    print(f"   Date range    : {df['timestamp'].min()} → {df['timestamp'].max()}")

    print(f"\n🏥 Health Score Distribution:")
    health_dist = df['health_score'].value_counts().sort_index()
    for health, count in health_dist.items():
        label = "Optimal" if health == 0 else "Dégradé" if health == 1 else "Critique"
        pct   = count / len(df) * 100
        print(f"   {health} ({label:8s}): {count:4d} samples ({pct:.1f}%)")

    print(f"\n🔍 Root Cause Distribution:")
    for cause, count in df['root_cause'].value_counts().items():
        pct = count / len(df) * 100
        print(f"   {cause:30s}: {count:4d} ({pct:.1f}%)")

    print(f"\n📊 Signal Quality Statistics:")
    print(f"   RxPower    → mean: {df['rx_power_dbm'].mean():.2f} dBm"
          f"  | min: {df['rx_power_dbm'].min():.2f}"
          f"  | max: {df['rx_power_dbm'].max():.2f}")
    print(f"   BiasCurr   → mean: {df['bias_current_ua'].mean():.0f} uA"
          f"  | min: {df['bias_current_ua'].min():.0f}"
          f"  | max: {df['bias_current_ua'].max():.0f}")
    print(f"   Temp       → mean: {df['temperature_c'].mean():.1f} °C"
          f"  | min: {df['temperature_c'].min():.1f}"
          f"  | max: {df['temperature_c'].max():.1f}")

    # Sauvegarder
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n💾 Dataset saved to: {OUTPUT_PATH}")
    print("\n" + "="*70)
    print("✅ GENERATION COMPLETED")
    print("="*70)
    print(f"\n🎯 Key Points:")
    print(f"   ✅ {N_DEVICES * N_SAMPLES_PER_DEVICE} lignes total")
    print(f"   ✅ {N_DEVICES} devices uniques (HUAWEI_BOX_001 → HUAWEI_BOX_300)")
    print(f"   ✅ RxPower = driver principal (40-50% importance expected)")
    print(f"   ✅ BiasCurrent = indicateur santé laser (35-45% expected)")
    print(f"   ✅ Temperature = secondaire (10-15% expected)")
    print(f"\n💡 Next steps:")
    print(f"   Scénario 1 → preprocess_huawei_nokia_s1.py  (garde le plus récent → 600 lignes)")
    print(f"   Scénario 3 → preprocess_huawei_nokia_s3.py  (garde tout → 6000 lignes)")

if __name__ == "__main__":
    main()