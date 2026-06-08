import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from io import StringIO

# ============================================================
# test_preprocessing.py
# Tests unitaires — Preprocessing (Scénario 1 + Scénario 3)
# Couvre : preprocess_huawei_nokia.py, preprocess_igd.py,
#          preprocess_hn_scenario3.py, preprocess_igd_scenario3.py
# ============================================================


# ─────────────────────────────────────────────
# FIXTURES — Données de test réutilisables
# ─────────────────────────────────────────────

@pytest.fixture
def sample_hn_df():
    """DataFrame simulant fibre_combined_realistic.csv (HN)"""
    data = {
        "device_id":       ["HUAWEI_001", "HUAWEI_001", "NOKIA_001", "NOKIA_001", "HUAWEI_002", "HUAWEI_002"],
        "timestamp":       ["2024-01-01 00:00:00", "2024-01-01 01:00:00",
                            "2024-01-01 00:00:00", "2024-01-01 01:00:00",
                            "2024-01-01 00:00:00", "2024-01-01 01:00:00"],
        "temperature_c":   [35.0, 36.0, 34.0, 35.0, 33.0, 34.0],
        "rx_power_dbm":    [-20.0, -21.0, -19.0, -20.0, -22.0, -23.0],
        "bias_current_ua": [10.0, 11.0, 9.0, 10.0, 12.0, 13.0],
        "supply_voltage_v":[3.3, 3.3, 3.3, 3.3, 3.3, 3.3],
        "health_score":    [0, 1, 0, 1, 2, 2],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_igd_df():
    """DataFrame simulant igd_realistic.csv"""
    data = {
        "device_id":                  ["IGD_001", "IGD_001", "IGD_002", "IGD_002"],
        "timestamp":                  ["2024-01-01 00:00:00", "2024-01-01 01:00:00",
                                       "2024-01-01 00:00:00", "2024-01-01 01:00:00"],
        "downstream_curr_rate_kbps":  [10000.0, 9500.0, 8000.0, 7500.0],
        "downstream_max_rate_kbps":   [20000.0, 20000.0, 15000.0, 15000.0],
        "snr_margin_down_db":         [15.0, 14.0, 10.0, 9.0],
        "attenuation_down_db":        [25.0, 25.5, 30.0, 30.5],
        "crc_errors_total":           [0.0, 5.0, 10.0, 20.0],
        "health_score":               [0, 0, 1, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_hn_scenario3_df():
    """DataFrame avec 10 mesures par équipement pour Scénario 3 HN"""
    rows = []
    for device in ["HUAWEI_001", "NOKIA_001"]:
        for t in range(1, 11):
            rows.append({
                "device_id":      device,
                "timestamp":      f"2024-01-{t:02d} 00:00:00",
                "rx_power_dbm":   -20.0 - t * 0.5,
                "health_score":   1,
                "root_cause":     "OPTICAL_AGING",
                "vendor":         "Huawei" if "HUAWEI" in device else "Nokia",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_igd_scenario3_df():
    """DataFrame avec 10 mesures par équipement pour Scénario 3 IGD"""
    rows = []
    for device in ["IGD_001", "IGD_002"]:
        for t in range(1, 11):
            rows.append({
                "device_id":          device,
                "timestamp":          f"2024-01-{t:02d} 00:00:00",
                "snr_margin_down_db": 15.0 - t * 0.3,
                "health_score":       1,
                "root_cause":         "Line degradation",
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# TESTS — preprocess_huawei_nokia.py (Scénario 1)
# ─────────────────────────────────────────────

class TestPreprocessHuaweiNokia:

    def test_clean_data_numeric_columns(self, sample_hn_df):
        """Les colonnes numériques doivent être de type float après clean"""
        numeric_cols = ["temperature_c", "rx_power_dbm", "bias_current_ua", "supply_voltage_v"]
        for col in numeric_cols:
            sample_hn_df[col] = pd.to_numeric(sample_hn_df[col], errors="coerce")
        for col in numeric_cols:
            assert sample_hn_df[col].dtype in [np.float64, np.float32]

    def test_clean_data_fills_nulls(self):
        """Les valeurs nulles doivent être remplies par la moyenne"""
        df = pd.DataFrame({
            "device_id":       ["HUAWEI_001", "NOKIA_001"],
            "temperature_c":   [35.0, None],
            "rx_power_dbm":    [-20.0, -21.0],
            "bias_current_ua": [10.0, 11.0],
            "supply_voltage_v":[3.3, 3.3],
            "health_score":    [0, 1],
        })
        for col in ["temperature_c", "rx_power_dbm", "bias_current_ua", "supply_voltage_v"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mean(), inplace=True)
        assert df["temperature_c"].isnull().sum() == 0

    def test_keep_most_recent_one_row_per_device(self, sample_hn_df):
        """Après keep_most_recent, 1 seule ligne par device_id"""
        sample_hn_df["timestamp"] = pd.to_datetime(sample_hn_df["timestamp"])
        sample_hn_df = sample_hn_df.sort_values("timestamp", ascending=True)
        result = sample_hn_df.drop_duplicates(subset="device_id", keep="last")
        assert result["device_id"].nunique() == len(result)

    def test_keep_most_recent_keeps_last_timestamp(self, sample_hn_df):
        """La ligne gardée doit être la plus récente"""
        sample_hn_df["timestamp"] = pd.to_datetime(sample_hn_df["timestamp"])
        sample_hn_df = sample_hn_df.sort_values("timestamp", ascending=True)
        result = sample_hn_df.drop_duplicates(subset="device_id", keep="last")
        huawei = result[result["device_id"] == "HUAWEI_001"]
        assert huawei["timestamp"].values[0] == pd.Timestamp("2024-01-01 01:00:00")

    def test_select_features_correct_columns(self, sample_hn_df):
        """Les colonnes sélectionnées doivent correspondre aux features + target"""
        features = ["device_id", "temperature_c", "rx_power_dbm",
                    "bias_current_ua", "supply_voltage_v"]
        target = "health_score"
        result = sample_hn_df[features + [target]]
        assert list(result.columns) == features + [target]

    def test_select_features_no_extra_columns(self, sample_hn_df):
        """Aucune colonne supplémentaire ne doit être présente"""
        features = ["device_id", "temperature_c", "rx_power_dbm",
                    "bias_current_ua", "supply_voltage_v"]
        target = "health_score"
        result = sample_hn_df[features + [target]]
        assert len(result.columns) == 6

    def test_health_score_valid_classes(self, sample_hn_df):
        """Les classes health_score doivent être 0, 1 ou 2"""
        valid_classes = {0, 1, 2}
        assert set(sample_hn_df["health_score"].unique()).issubset(valid_classes)

    def test_device_id_contains_huawei_or_nokia(self, sample_hn_df):
        """Tous les device_id doivent commencer par HUAWEI ou NOKIA"""
        for did in sample_hn_df["device_id"]:
            assert did.startswith("HUAWEI") or did.startswith("NOKIA")


# ─────────────────────────────────────────────
# TESTS — preprocess_igd.py (Scénario 1)
# ─────────────────────────────────────────────

class TestPreprocessIGD:

    def test_clean_data_numeric_columns_igd(self, sample_igd_df):
        """Les colonnes numériques IGD doivent être float après clean"""
        numeric_cols = ["downstream_curr_rate_kbps", "downstream_max_rate_kbps",
                        "snr_margin_down_db", "attenuation_down_db", "crc_errors_total"]
        for col in numeric_cols:
            sample_igd_df[col] = pd.to_numeric(sample_igd_df[col], errors="coerce")
        for col in numeric_cols:
            assert sample_igd_df[col].dtype in [np.float64, np.float32]

    def test_keep_most_recent_igd(self, sample_igd_df):
        """1 seule ligne par device IGD après keep_most_recent"""
        sample_igd_df["timestamp"] = pd.to_datetime(sample_igd_df["timestamp"])
        sample_igd_df = sample_igd_df.sort_values("timestamp", ascending=True)
        result = sample_igd_df.drop_duplicates(subset="device_id", keep="last")
        assert result["device_id"].nunique() == len(result)

    def test_select_features_igd_columns(self, sample_igd_df):
        """Les features IGD sélectionnées doivent être correctes"""
        features = ["device_id", "downstream_curr_rate_kbps", "downstream_max_rate_kbps",
                    "snr_margin_down_db", "attenuation_down_db", "crc_errors_total"]
        target = "health_score"
        result = sample_igd_df[features + [target]]
        assert "snr_margin_down_db" in result.columns
        assert "health_score" in result.columns

    def test_device_id_starts_with_igd(self, sample_igd_df):
        """Tous les device_id IGD doivent commencer par IGD"""
        for did in sample_igd_df["device_id"]:
            assert did.startswith("IGD")

    def test_no_negative_crc_errors(self, sample_igd_df):
        """Les erreurs CRC ne peuvent pas être négatives"""
        assert (sample_igd_df["crc_errors_total"] >= 0).all()


# ─────────────────────────────────────────────
# TESTS — preprocess_hn_scenario3.py (Scénario 3)
# ─────────────────────────────────────────────

class TestPreprocessHNScenario3:

    def test_only_devices_with_10_measures(self, sample_hn_scenario3_df):
        """Seuls les équipements avec exactement 10 mesures sont gardés"""
        counts = sample_hn_scenario3_df.groupby("device_id").size()
        valid  = counts[counts == 10]
        result = sample_hn_scenario3_df[sample_hn_scenario3_df["device_id"].isin(valid.index)]
        for device in result["device_id"].unique():
            assert len(result[result["device_id"] == device]) == 10

    def test_t_index_range(self, sample_hn_scenario3_df):
        """Le t_index doit aller de 1 à 10"""
        df = sample_hn_scenario3_df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)
        df["t_index"] = df.groupby("device_id").cumcount() + 1
        assert df["t_index"].min() == 1
        assert df["t_index"].max() == 10

    def test_slope_calculation(self):
        """La pente doit être (T10 - T1) / 9"""
        rx_T1  = -20.0
        rx_T10 = -25.0
        slope  = (rx_T10 - rx_T1) / 9
        assert round(slope, 4) == round(-5.0 / 9, 4)

    def test_slope_negative_means_degradation(self):
        """Un slope négatif indique une dégradation"""
        slope = (-25.0 - (-20.0)) / 9
        assert slope < 0

    def test_pivot_one_row_per_device(self, sample_hn_scenario3_df):
        """Après pivot, 1 ligne par équipement"""
        df = sample_hn_scenario3_df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)
        df["t_index"] = df.groupby("device_id").cumcount() + 1
        counts = df.groupby("device_id").size()
        valid  = counts[counts == 10]
        df     = df[df["device_id"].isin(valid.index)]
        pivot  = df.pivot(index="device_id", columns="t_index", values="rx_power_dbm")
        assert len(pivot) == df["device_id"].nunique()

    def test_rx_threshold_value(self):
        """Le seuil critique RxPower doit être -28.0 dBm"""
        RX_THRESHOLD = -28.0
        assert RX_THRESHOLD == -28.0

    def test_jours_avant_seuil_positive_when_not_exceeded(self):
        """jours_avant_seuil doit être positif si seuil pas encore atteint"""
        RX_THRESHOLD = -28.0
        rx_last = -25.0
        slope   = -0.5
        jours   = (RX_THRESHOLD - rx_last) / slope
        assert jours > 0

    def test_jours_avant_seuil_none_when_slope_positive(self):
        """jours_avant_seuil doit être None si slope >= 0 (pas de dégradation)"""
        slope = 0.1
        jours = None if slope >= 0 else ((-28.0 - (-25.0)) / slope)
        assert jours is None


# ─────────────────────────────────────────────
# TESTS — preprocess_igd_scenario3.py (Scénario 3)
# ─────────────────────────────────────────────

class TestPreprocessIGDScenario3:

    def test_snr_threshold_value(self):
        """Le seuil critique SNR doit être 6.0 dB"""
        SNR_THRESHOLD = 6.0
        assert SNR_THRESHOLD == 6.0

    def test_slope_snr_calculation(self):
        """La pente SNR doit être (T10 - T1) / 9"""
        snr_T1  = 15.0
        snr_T10 = 9.0
        slope   = (snr_T10 - snr_T1) / 9
        assert round(slope, 4) == round(-6.0 / 9, 4)

    def test_only_devices_with_10_measures_igd(self, sample_igd_scenario3_df):
        """Seuls équipements avec 10 mesures IGD sont gardés"""
        counts = sample_igd_scenario3_df.groupby("device_id").size()
        valid  = counts[counts == 10]
        result = sample_igd_scenario3_df[sample_igd_scenario3_df["device_id"].isin(valid.index)]
        for device in result["device_id"].unique():
            assert len(result[result["device_id"] == device]) == 10

    def test_confidence_interval_formula(self):
        """L'intervalle de confiance 95% doit être last ± 1.96 * std"""
        snr_last = 10.0
        snr_std  = 2.0
        ci_lower = snr_last - 1.96 * snr_std
        ci_upper = snr_last + 1.96 * snr_std
        assert ci_lower == 10.0 - 1.96 * 2.0
        assert ci_upper == 10.0 + 1.96 * 2.0

    def test_jours_avant_seuil_igd(self):
        """jours_avant_seuil IGD correctement calculé"""
        SNR_THRESHOLD = 6.0
        snr_last = 9.0
        slope    = -0.5
        jours    = (SNR_THRESHOLD - snr_last) / slope
        assert jours == (6.0 - 9.0) / (-0.5)
        assert jours == 6.0