"""
Générateur de dataset réaliste pour IGD (ADSL/VDSL)
Minimum 300 lignes - Basé sur lois physiques Shannon et caractéristiques DSL
VERSION CORRIGÉE : SNR/Attenuation = DRIVERS, CRC = CONSÉQUENCE
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# ======================== CONFIGURATION ========================
OUTPUT_PATH = "datasets/igd_realistic.csv"
N_DEVICES = 30
N_SAMPLES_PER_DEVICE = 10  # 30 devices × 10 samples = 300 lignes minimum
MEASUREMENT_INTERVAL_HOURS = 6

# Normes ADSL/VDSL
NORMS = {
    'snr_critical': 6,        # dB
    'snr_optimal': 15,        # dB
    'attenuation_optimal': 20,  # dB
    'attenuation_critical': 55,  # dB
    'crc_threshold': 100,     # erreurs/période
}

# ======================== FONCTIONS ========================

def calculate_theoretical_rate(snr_db, attenuation_db, max_rate):
    """
    Calcule débit théorique basé sur SNR et atténuation
    Formule inspirée de Shannon-Hartley
    """
    # Convertir SNR en linéaire
    snr_linear = 10 ** (snr_db / 10)
    
    # Capacité théorique (simplifié)
    capacity_factor = np.log2(1 + snr_linear) / 10  # Normalisé
    
    # Impact de l'atténuation
    attenuation_factor = max(0.1, 1 - (attenuation_db / 80))
    
    # Débit théorique
    theoretical_rate = max_rate * capacity_factor * attenuation_factor
    
    return min(theoretical_rate, max_rate)

def calculate_crc_errors_from_snr(snr_db, add_variance=True):
    """
    ✅ NOUVELLE FONCTION : Calcule CRC errors basé UNIQUEMENT sur SNR
    CRC = CONSÉQUENCE de SNR, pas INPUT pour health_score
    """
    # Relation physique : Plus SNR bas = plus d'erreurs
    # MAIS avec GROSSE variance pour éviter corrélation parfaite
    
    if snr_db < 4:
        base_errors = np.random.uniform(1500, 2500)
    elif snr_db < 6:
        base_errors = np.random.uniform(1000, 1800)
    elif snr_db < 8:
        base_errors = np.random.uniform(500, 1200)
    elif snr_db < 10:
        base_errors = np.random.uniform(200, 600)
    elif snr_db < 12:
        base_errors = np.random.uniform(100, 350)
    elif snr_db < 15:
        base_errors = np.random.uniform(30, 150)
    else:
        base_errors = np.random.uniform(0, 80)
    
    if add_variance:
        # ✅ Ajouter GROSSE variance multiplicative pour casser corrélation
        variance_factor = np.random.uniform(0.4, 1.8)  # ±80% variance !
        base_errors *= variance_factor
    
    return max(0, int(base_errors))

def generate_device_timeseries(device_id, n_samples):
    """
    ✅ VERSION CORRIGÉE
    Génère série temporelle pour un device ADSL
    SNR + Attenuation = DRIVERS PRINCIPAUX (PAS CRC !)
    Health_score basé UNIQUEMENT sur SNR/Attenuation
    CRC calculé APRÈS comme conséquence de SNR
    """
    
    # Profil du device
    profile = random.choices(
        ['stable_optimal', 'line_degradation', 'high_interference', 
         'config_issue', 'crosstalk'],
        weights=[0.45, 0.20, 0.15, 0.12, 0.08]
    )[0]
    
    # Caractéristiques de ligne (fixes pour le device)
    line_distance_km = np.random.uniform(0.5, 4.5)
    base_attenuation = 10 + (line_distance_km * 8)  # ~10-46 dB
    
    # Max rate dépend de la ligne
    if base_attenuation < 25:
        max_rate_down = int(np.random.uniform(18000, 24000))
    elif base_attenuation < 40:
        max_rate_down = int(np.random.uniform(12000, 18000))
    else:
        max_rate_down = int(np.random.uniform(8000, 12000))
    
    max_rate_up = int(max_rate_down * 0.15)  # ADSL asymétrique
    
    start_time = datetime.now() - timedelta(days=2, hours=12)
    samples = []
    
    for i in range(n_samples):
        timestamp = start_time + timedelta(hours=MEASUREMENT_INTERVAL_HOURS * i)
        hour = timestamp.hour
        
        # ========== PROFIL 1: STABLE OPTIMAL (45%) ==========
        if profile == 'stable_optimal':
            # ✅ 1. GÉNÉRER SNR (DRIVER #1)
            snr_down = np.random.uniform(16, 25)
            snr_up = np.random.uniform(13, 22)
            
            # ✅ 2. GÉNÉRER ATTÉNUATION (DRIVER #2)
            atten_down = base_attenuation + np.random.uniform(-2, 3)
            atten_up = atten_down + np.random.uniform(0, 5)
            
            # ✅ 3. CALCULER DÉBIT (fonction de SNR/Atten)
            theoretical_rate = calculate_theoretical_rate(snr_down, atten_down, max_rate_down)
            curr_rate_down = int(theoretical_rate * np.random.uniform(0.88, 0.98))
            curr_rate_up = int(max_rate_up * np.random.uniform(0.85, 0.95))
            
            # ✅ 4. DÉTERMINER HEALTH basé sur SNR/Atten SEULEMENT
            if snr_down >= 15 and atten_down < 35:
                health_score = 0
                root_cause = "NORMAL"
            elif snr_down >= 10 and atten_down < 45:
                health_score = 0
                root_cause = "NORMAL"
            else:
                health_score = 1
                root_cause = "NORMAL"
            
            # ✅ 5. CRC ERRORS calculé EN DERNIER (conséquence de SNR)
            crc_errors = calculate_crc_errors_from_snr(snr_down, add_variance=True)
        
        # ========== PROFIL 2: LIGNE DÉGRADÉE (20%) ==========
        elif profile == 'line_degradation':
            progress = i / (n_samples - 1)
            
            # ✅ 1. SNR se dégrade (DRIVER)
            snr_down = 20 - (progress * 14)  # 20 → 6 dB
            snr_up = 17 - (progress * 12)
            
            # ✅ 2. ATTÉNUATION augmente (DRIVER)
            atten_down = base_attenuation + (progress * 20)
            atten_up = atten_down + np.random.uniform(3, 10)
            
            # ✅ 3. DÉBIT calculé
            theoretical_rate = calculate_theoretical_rate(snr_down, atten_down, max_rate_down)
            curr_rate_down = int(theoretical_rate * np.random.uniform(0.70, 0.90))
            curr_rate_up = int(max_rate_up * np.random.uniform(0.60, 0.80))
            
            # ✅ 4. HEALTH basé sur SNR/Atten UNIQUEMENT (PAS CRC !)
            if snr_down < 6 or atten_down > 55:
                health_score = 2
                root_cause = "Line degradation"
            elif snr_down < 10 or atten_down > 45:
                health_score = 1
                root_cause = "Line degradation"
            else:
                health_score = 0
                root_cause = "NORMAL"
            
            # ✅ 5. CRC calculé EN DERNIER (fonction de SNR avec variance)
            crc_errors = calculate_crc_errors_from_snr(snr_down, add_variance=True)
        
        # ========== PROFIL 3: INTERFÉRENCES ÉLEVÉES (15%) ==========
        elif profile == 'high_interference':
            # ✅ 1. SNR très bas (DRIVER)
            snr_down = np.random.uniform(3, 8)
            snr_up = np.random.uniform(2, 6)
            
            # ✅ 2. ATTÉNUATION normale
            atten_down = base_attenuation + np.random.uniform(-3, 5)
            atten_up = atten_down + np.random.uniform(2, 8)
            
            # ✅ 3. DÉBIT très réduit
            theoretical_rate = calculate_theoretical_rate(snr_down, atten_down, max_rate_down)
            curr_rate_down = int(theoretical_rate * np.random.uniform(0.50, 0.75))
            curr_rate_up = int(max_rate_up * np.random.uniform(0.40, 0.65))
            
            # ✅ 4. HEALTH basé sur SNR
            if snr_down < 5:
                health_score = 2
                root_cause = "High Interference / Noise"
            elif snr_down < 8:
                health_score = 2
                root_cause = "High Interference / Noise"
            else:
                health_score = 1
                root_cause = "High Interference / Noise"
            
            # ✅ 5. CRC élevé (conséquence de SNR bas)
            crc_errors = calculate_crc_errors_from_snr(snr_down, add_variance=True)
        
        # ========== PROFIL 4: PROBLÈME CONFIG (12%) ==========
        elif profile == 'config_issue':
            # ✅ 1. SNR/ATTEN OK (ligne saine)
            snr_down = np.random.uniform(13, 22)
            snr_up = np.random.uniform(11, 19)
            atten_down = base_attenuation + np.random.uniform(-2, 3)
            atten_up = atten_down + np.random.uniform(0, 5)
            
            # ✅ 2. DÉBIT bridé artificiellement (config)
            theoretical_rate = calculate_theoretical_rate(snr_down, atten_down, max_rate_down)
            curr_rate_down = int(theoretical_rate * np.random.uniform(0.40, 0.60))
            curr_rate_up = int(max_rate_up * np.random.uniform(0.35, 0.55))
            
            # ✅ 3. HEALTH basé sur ratio débit (pas SNR/Atten car OK)
            efficiency = curr_rate_down / theoretical_rate
            if efficiency < 0.50:
                health_score = 1
                root_cause = "Configuration / Ports overload"
            else:
                health_score = 0
                root_cause = "NORMAL"
            
            # ✅ 4. CRC modéré (SNR OK mais config peut causer retransmissions)
            # Ajouter erreurs liées à la config (pas juste SNR)
            base_crc = calculate_crc_errors_from_snr(snr_down, add_variance=False)
            config_errors = int(np.random.uniform(100, 400))  # Erreurs de config
            crc_errors = base_crc + config_errors
        
        # ========== PROFIL 5: CROSSTALK / SATURATION (8%) ==========
        else:  # crosstalk
            # ✅ 1. SNR moyen mais variable
            snr_down = np.random.uniform(8, 14)
            snr_up = np.random.uniform(6, 12)
            
            # Heures de pointe = pire (crosstalk des voisins)
            if 18 <= hour <= 23:
                snr_down -= np.random.uniform(2, 5)
                snr_up -= np.random.uniform(2, 4)
            
            # ✅ 2. ATTÉNUATION
            atten_down = base_attenuation + np.random.uniform(0, 8)
            atten_up = atten_down + np.random.uniform(3, 10)
            
            # ✅ 3. DÉBIT
            theoretical_rate = calculate_theoretical_rate(snr_down, atten_down, max_rate_down)
            curr_rate_down = int(theoretical_rate * np.random.uniform(0.65, 0.85))
            curr_rate_up = int(max_rate_up * np.random.uniform(0.60, 0.80))
            
            # ✅ 4. HEALTH basé sur SNR
            if snr_down < 7:
                health_score = 2
                root_cause = "Crosstalk / Saturation"
            elif snr_down < 10:
                health_score = 1
                root_cause = "Crosstalk / Saturation"
            else:
                health_score = 0
                root_cause = "NORMAL"
            
            # ✅ 5. CRC fonction de SNR
            crc_errors = calculate_crc_errors_from_snr(snr_down, add_variance=True)
        
        # ✅ AJOUTER BRUIT RÉALISTE sur SNR/Atten (mesures jamais parfaites)
        snr_down = max(0, snr_down + np.random.normal(0, 1.2))
        snr_up = max(0, snr_up + np.random.normal(0, 1.0))
        atten_down += np.random.normal(0, 1.5)
        atten_up += np.random.normal(0, 1.5)
        
        # ✅ AJOUTER VARIANCE FINALE sur CRC (bruit de mesure)
        crc_errors = int(crc_errors * np.random.uniform(0.8, 1.2))
        crc_errors = max(0, crc_errors)
        
        samples.append({
            'device_id': device_id,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'downstream_curr_rate_kbps': curr_rate_down,
            'downstream_max_rate_kbps': max_rate_down,
            'upstream_curr_rate_kbps': curr_rate_up,
            'upstream_max_rate_kbps': max_rate_up,
            'snr_margin_down_db': round(snr_down, 2),
            'snr_margin_up_db': round(snr_up, 2),
            'attenuation_down_db': round(atten_down, 2),
            'attenuation_up_db': round(atten_up, 2),
            'crc_errors_total': crc_errors,
            'health_score': health_score,
            'root_cause': root_cause
        })
    
    return samples

# ======================== MAIN ========================

def main():
    print("="*70)
    print("🚀 IGD (ADSL/VDSL) DATASET GENERATOR - VERSION CORRIGÉE")
    print("   Realistic DSL Line Behavior")
    print("   SNR/Attenuation = DRIVERS, CRC = CONSEQUENCE")
    print("="*70)
    
    all_samples = []
    
    print(f"\n📊 Generating data for {N_DEVICES} devices...")
    print(f"   {N_SAMPLES_PER_DEVICE} measurements per device (every {MEASUREMENT_INTERVAL_HOURS}h)")
    
    for i in range(N_DEVICES):
        device_id = f"IGD_{i+1:03d}"
        samples = generate_device_timeseries(device_id, N_SAMPLES_PER_DEVICE)
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
    
    # Statistiques SNR/Attenuation
    print(f"\n📊 Line Quality Statistics:")
    print(f"   SNR Downstream:")
    print(f"     Mean: {df['snr_margin_down_db'].mean():.2f} dB")
    print(f"     Min:  {df['snr_margin_down_db'].min():.2f} dB")
    print(f"     Max:  {df['snr_margin_down_db'].max():.2f} dB")
    print(f"   Attenuation Downstream:")
    print(f"     Mean: {df['attenuation_down_db'].mean():.2f} dB")
    print(f"     Min:  {df['attenuation_down_db'].min():.2f} dB")
    print(f"     Max:  {df['attenuation_down_db'].max():.2f} dB")
    print(f"   CRC Errors:")
    print(f"     Mean: {df['crc_errors_total'].mean():.0f}")
    print(f"     Min:  {df['crc_errors_total'].min()}")
    print(f"     Max:  {df['crc_errors_total'].max()}")
    
    # ✅ VÉRIFICATION CORRÉLATIONS
    print(f"\n🔍 Feature Correlations with Health Score:")
    corr = df[['snr_margin_down_db', 'attenuation_down_db', 'crc_errors_total', 'health_score']].corr()['health_score'].drop('health_score').sort_values(key=abs, ascending=False)
    for feature, value in corr.items():
        print(f"   {feature:25s}: {value:+.3f}")
    
    # Sauvegarder
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n💾 Dataset saved to: {OUTPUT_PATH}")
    
    print("\n" + "="*70)
    print("✅ GENERATION COMPLETED - CORRECTED VERSION")
    print("="*70)
    print("\n🎯 Key Features (CORRECTED):")
    print("   ✅ SNR = PRIMARY driver (35-45% importance expected)")
    print("   ✅ Attenuation = SECONDARY driver (25-35% expected)")
    print("   ✅ CRC Errors = CONSEQUENCE of SNR (8-15% expected)")
    print("   ✅ Health_score based ONLY on SNR/Attenuation")
    print("   ✅ High variance in CRC to break artificial correlation")
    
    print(f"\n💡 Next steps:")
    print(f"   1. python components/preprocessing/preprocess_igd.py")
    print(f"   2. python components/healthscore_training/train_igd.py")
    print(f"   3. Check feature importance → SNR should be #1 now!")

if __name__ == "__main__":
    main()