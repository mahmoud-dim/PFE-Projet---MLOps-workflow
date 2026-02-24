"""
Générateur de dataset réaliste pour Nokia G-2425G-A / G-1425G-B (GPON)
Minimum 300 lignes - Basé sur normes physiques réelles
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# ======================== CONFIGURATION ========================
OUTPUT_PATH = "datasets/nokia_realistic.csv"
N_DEVICES = 30
N_SAMPLES_PER_DEVICE = 10  # 30 devices × 10 samples = 300 lignes minimum
MEASUREMENT_INTERVAL_HOURS = 6

# Normes GPON Nokia (1490nm RX, 1310nm TX)
NORMS = {
    'rx_power_lower': -28,    # dBm à 1490nm
    'rx_power_upper': -8,     # dBm
    'rx_power_optimal_min': -22,
    'rx_power_optimal_max': -10,
    'bias_current_normal': 12000,  # uA
    'bias_current_max': 15000,     # uA
    'temp_normal': 50,        # °C
    'temp_max': 70,           # °C
    'voltage_nominal': 3.3,   # mV (affiché en V dans dataset)
}

# ======================== FONCTIONS ========================

def generate_device_timeseries(device_id, product_class, n_samples):
    """
    Génère série temporelle pour un device Nokia
    RxPower = DRIVER PRINCIPAL de la santé
    """
    
    # Choisir profil du device
    profile = random.choices(
        ['stable_optimal', 'macrobend', 'connector_dirty', 'severe_loss', 'intermittent'],
        weights=[0.45, 0.25, 0.15, 0.08, 0.07]
    )[0]
    
    start_time = datetime.now() - timedelta(days=2, hours=12)
    samples = []
    
    for i in range(n_samples):
        timestamp = start_time + timedelta(hours=MEASUREMENT_INTERVAL_HOURS * i)
        hour = timestamp.hour
        
        # ========== PROFIL 1: STABLE OPTIMAL (45%) ==========
        if profile == 'stable_optimal':
            # RxPower excellent (1490nm)
            rx_power = np.random.uniform(-21, -10)
            
            # BiasCurrent normal
            bias_current = np.random.uniform(10000, 13000)
            
            # Température environnementale
            if 10 <= hour <= 18:
                temperature = np.random.uniform(45, 60)
            else:
                temperature = np.random.uniform(35, 50)
            
            supply_voltage = np.random.uniform(3.20, 3.40)
            tx_power = np.random.uniform(1.5, 3.5)
            
            health_score = 0
            root_cause = "NORMAL"
        
        # ========== PROFIL 2: MACROBEND (25%) ==========
        elif profile == 'macrobend':
            # RxPower dégradé progressivement (fibre courbée)
            progress = i / (n_samples - 1)
            rx_power = -19 - (progress * 8)  # -19 → -27 dBm
            
            # Laser compense
            bias_current = 11000 + (progress * 3500)
            
            # Température augmente avec effort laser
            temperature = 48 + (progress * 15)
            if 10 <= hour <= 18:
                temperature += 5
            
            supply_voltage = np.random.uniform(3.15, 3.35)
            tx_power = np.random.uniform(2, 4.5)
            
            # Health basé sur RxPower
            if rx_power > -23:
                health_score = 0
                root_cause = "NORMAL"
            elif rx_power > -26:
                health_score = 1
                root_cause = "MACROBEND_OR_CONNECTOR"
            else:
                health_score = 1
                root_cause = "MACROBEND_OR_CONNECTOR"
        
        # ========== PROFIL 3: CONNECTEUR SALE (15%) ==========
        elif profile == 'connector_dirty':
            # Signal moyen-faible (pertes au connecteur)
            rx_power = np.random.uniform(-26, -22)
            
            # Bias légèrement élevé
            bias_current = np.random.uniform(12500, 14500)
            
            # Température normale à légèrement élevée
            temperature = np.random.uniform(50, 65)
            if 10 <= hour <= 18:
                temperature += np.random.uniform(3, 8)
            
            supply_voltage = np.random.uniform(3.10, 3.30)
            tx_power = np.random.uniform(2.5, 4)
            
            health_score = 1
            root_cause = "MACROBEND_OR_CONNECTOR"
        
        # ========== PROFIL 4: PERTE SÉVÈRE (8%) ==========
        elif profile == 'severe_loss':
            # Signal très faible
            rx_power = np.random.uniform(-30, -27)
            
            # Laser au maximum
            bias_current = np.random.uniform(14000, 19500)
            
            # Température élevée
            temperature = np.random.uniform(65, 80)
            
            supply_voltage = np.random.uniform(3.05, 3.25)
            tx_power = np.random.uniform(3.5, 5.5)
            
            health_score = 2
            root_cause = "SEVERE_SIGNAL_LOSS"
        
        # ========== PROFIL 5: INTERMITTENT (7%) ==========
        else:  # intermittent
            # Alterne bon/mauvais
            if i % 3 == 0:  # 1/3 mauvais
                rx_power = np.random.uniform(-27, -24)
                bias_current = np.random.uniform(13000, 15500)
                temperature = np.random.uniform(58, 70)
                health_score = 1
                root_cause = "MACROBEND_OR_CONNECTOR"
            else:  # 2/3 bon
                rx_power = np.random.uniform(-21, -13)
                bias_current = np.random.uniform(10000, 12500)
                temperature = np.random.uniform(40, 55)
                health_score = 0
                root_cause = "NORMAL"
            
            supply_voltage = np.random.uniform(3.15, 3.35)
            tx_power = np.random.uniform(2, 4)
        
        # Bruit réaliste
        rx_power += np.random.normal(0, 0.6)
        bias_current += np.random.normal(0, 200)
        temperature += np.random.normal(0, 2.5)
        supply_voltage += np.random.normal(0, 0.04)
        tx_power += np.random.normal(0, 0.35)
        
        samples.append({
            'device_id': device_id,
            'product_class': product_class,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'rx_power_dbm': round(rx_power, 2),
            'tx_power_dbm': round(tx_power, 2),
            'bias_current_ua': round(bias_current, 2),
            'supply_voltage_v': round(supply_voltage, 2),
            'temperature_c': round(temperature, 2),
            'rx_power_lower': NORMS['rx_power_lower'],
            'rx_power_upper': NORMS['rx_power_upper'],
            'health_score': health_score,
            'root_cause': root_cause
        })
    
    return samples

# ======================== MAIN ========================

def main():
    print("="*70)
    print("🚀 NOKIA G-2425G-A / G-1425G-B DATASET GENERATOR")
    print("   Realistic GPON Fiber Optics Behavior")
    print("   (1490nm RX / 1310nm TX)")
    print("="*70)
    
    all_samples = []
    
    print(f"\n📊 Generating data for {N_DEVICES} devices...")
    print(f"   {N_SAMPLES_PER_DEVICE} measurements per device (every {MEASUREMENT_INTERVAL_HOURS}h)")
    
    for i in range(N_DEVICES):
        device_id = f"NOKIA_BOX_{i+1:03d}"
        # Alterner entre les 2 modèles
        product_class = "G-2425G-A" if i % 2 == 0 else "G-1425G-B"
        samples = generate_device_timeseries(device_id, product_class, N_SAMPLES_PER_DEVICE)
        all_samples.extend(samples)
        
        if (i+1) % 10 == 0:
            print(f"   ✅ Generated {i+1}/{N_DEVICES} devices...")
    
    # Créer DataFrame
    df = pd.DataFrame(all_samples)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Statistiques
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total samples: {len(df)}")
    print(f"   Total devices: {df['device_id'].nunique()}")
    print(f"   Product classes:")
    for pc, count in df['product_class'].value_counts().items():
        print(f"     - {pc}: {count} samples")
    print(f"   Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    
    print(f"\n🏥 Health Score Distribution:")
    health_dist = df['health_score'].value_counts().sort_index()
    for health, count in health_dist.items():
        label = "Optimal" if health == 0 else "Dégradé" if health == 1 else "Critique"
        percentage = (count / len(df)) * 100
        print(f"   {health} ({label:8s}): {count:3d} samples ({percentage:5.1f}%)")
    
    print(f"\n🔍 Root Cause Distribution:")
    cause_dist = df['root_cause'].value_counts()
    for cause, count in cause_dist.items():
        percentage = (count / len(df)) * 100
        print(f"   {cause:30s}: {count:3d} ({percentage:5.1f}%)")
    
    # Sauvegarder
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n💾 Dataset saved to: {OUTPUT_PATH}")
    
    print("\n" + "="*70)
    print("✅ GENERATION COMPLETED")
    print("="*70)
    print("\n🎯 Key Features:")
    print("   ✅ RxPower@1490nm = PRIMARY driver")
    print("   ✅ TxPower@1310nm wavelength specific")
    print("   ✅ Realistic fiber issues (macrobend, connectors)")
    print("   ✅ Intermittent problems included")
    
    print(f"\n💡 Next step:")
    print(f"   python components/preprocessing/preprocess_huawei_nokia.py")

if __name__ == "__main__":
    main()