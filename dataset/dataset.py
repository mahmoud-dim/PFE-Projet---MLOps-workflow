import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# -----------------------------
# CONFIG
# -----------------------------
N_DEVICES_PER_CLASS = 30
N_TIMESTEPS = 5

product_classes = ["HG8145V5", "IGD", "G-2425G-A"]

start_time = datetime.now()

data = []

# -----------------------------
# HEALTH SCORE LOGIC
# -----------------------------
def compute_health_score(row, product_class):
    score = 0

    if product_class in ["HG8145V5", "G-2425G-A"]:
        if row["Temperature"] > 65 or row["BiasCurrent"] > 15000:
            score = 2
        elif row["Temperature"] > 55 or row["BiasCurrent"] > 13000:
            score = 1

    if product_class == "IGD":
        ratio = row["DownstreamCurrRate"] / row["DownstreamMaxRate"]
        if ratio < 0.3 or row["CRCErrors"] > 3000:
            score = 2
        elif ratio < 0.6 or row["CRCErrors"] > 1000:
            score = 1

    return score

# -----------------------------
# DATA GENERATION
# -----------------------------
for product_class in product_classes:
    for device_id in range(N_DEVICES_PER_CLASS):
        device_uid = f"{product_class}_BOX_{device_id+1}"

        for t in range(N_TIMESTEPS):
            timestamp = start_time + timedelta(minutes=30*t)

            if product_class == "HG8145V5":
                row = {
                    "DeviceID": device_uid,
                    "ProductClass": product_class,
                    "Timestamp": timestamp,
                    "RXPower": np.random.uniform(-28, -18),
                    "TXPower": np.random.uniform(1.5, 3.5),
                    "BiasCurrent": np.random.uniform(7000, 16000),
                    "SupplyVoltage": np.random.uniform(3100, 3400),
                    "Temperature": np.random.uniform(40, 75),
                }

            elif product_class == "G-2425G-A":
                row = {
                    "DeviceID": device_uid,
                    "ProductClass": product_class,
                    "Timestamp": timestamp,
                    "RXPower": np.random.uniform(-29, -19),
                    "TXPower": np.random.uniform(1.5, 3.5),
                    "BiasCurrent": np.random.uniform(7500, 16000),
                    "SupplyVoltage": np.random.uniform(3100, 3400),
                    "Temperature": np.random.uniform(40, 75),
                }

            elif product_class == "IGD":
                max_rate = np.random.uniform(15000, 25000)
                curr_rate = max_rate * np.random.uniform(0.2, 1.0)

                row = {
                    "DeviceID": device_uid,
                    "ProductClass": product_class,
                    "Timestamp": timestamp,
                    "DownstreamCurrRate": curr_rate,
                    "DownstreamMaxRate": max_rate,
                    "SNRMargin": np.random.uniform(3, 12),
                    "Attenuation": np.random.uniform(20, 55),
                    "CRCErrors": np.random.randint(0, 5000),
                }

            # Compute Health Score
            row["Health_Score"] = compute_health_score(row, product_class)

            data.append(row)

# -----------------------------
# CREATE DATAFRAME
# -----------------------------
df = pd.DataFrame(data)

# Save dataset
df.to_csv("fake_genieacs_dataset.csv", index=False)

print("✅ Dataset generated successfully!")
print(df.head())
print(f"\nTotal rows: {len(df)}")
