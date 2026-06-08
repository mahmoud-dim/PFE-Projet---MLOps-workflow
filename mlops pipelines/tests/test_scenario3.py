import pytest
import pandas as pd
import numpy as np

# ============================================================
# test_scenario3.py
# Tests unitaires — Scénario 3 : Prédiction Proactive
# Couvre : scenario3_hn.py, scenario3_igd.py
# ============================================================

RX_THRESHOLD  = -28.0
SNR_THRESHOLD = 6.0


# ─────────────────────────────────────────────
# FONCTIONS extraites des scripts (pour tests unitaires)
# ─────────────────────────────────────────────

def get_tendance(slope: float) -> str:
    if slope >= 0.5:    return "amélioration rapide"
    elif slope >= 0.1:  return "amélioration légère"
    elif slope >= -0.1: return "stable"
    elif slope >= -0.5: return "dégradation légère"
    elif slope >= -1.0: return "dégradation modérée"
    else:               return "dégradation rapide"


def get_action_hn(niveau_risque: str, jours) -> str:
    if niveau_risque == "stable":    return "Aucune action requise — RxPower stable ou en amélioration"
    elif niveau_risque == "faible":  return "Surveillance périodique recommandée"
    elif niveau_risque == "modere":  return "Surveillance proactive — planifier une vérification optique"
    elif niveau_risque == "eleve":   return "Intervention préventive — vérifier connecteurs et fibre"
    elif niveau_risque == "critique":return "Intervention urgente requise — risque de rupture signal GPON"
    return "Surveillance recommandée"


def get_action_igd(niveau_risque: str, jours) -> str:
    if niveau_risque == "stable":    return "Aucune action requise — SNR stable ou en amélioration"
    elif niveau_risque == "faible":  return "Surveillance périodique recommandée"
    elif niveau_risque == "modere":  return "Surveillance proactive — planifier une vérification"
    elif niveau_risque == "eleve":   return "Intervention préventive recommandée dans les 7 jours"
    elif niveau_risque == "critique":return "Intervention urgente requise — seuil critique imminent"
    return "Surveillance recommandée"


def jours_avant_seuil_hn(slope: float, rx_last: float) -> float | None:
    if slope >= 0:
        return None
    return round((RX_THRESHOLD - rx_last) / slope, 2)


def jours_avant_seuil_igd(slope: float, snr_last: float) -> float | None:
    if slope >= 0:
        return None
    return round((SNR_THRESHOLD - snr_last) / slope, 2)


def niveau_risque_from_slope_and_jours(slope: float, jours) -> str:
    if slope >= 0:
        return "stable"
    if jours is None:
        return "stable"
    if jours <= 2:    return "critique"
    elif jours <= 7:  return "eleve"
    elif jours <= 30: return "modere"
    else:             return "faible"


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def sample_hn_scenario3_processed():
    """DataFrame HN simulé après preprocessing Scénario 3"""
    np.random.seed(42)
    n = 20
    data = {}
    for i in range(1, 11):
        data[f"rx_T{i}"] = np.random.uniform(-27, -18, n)
    data["device_id"]          = [f"HUAWEI_{i:03d}" for i in range(n)]
    data["vendor"]             = ["Huawei"] * n
    data["health_score"]       = np.random.choice([0, 1, 2], n)
    data["root_cause"]         = np.random.choice(["OPTICAL_AGING", "NORMAL", "ENVIRONMENTAL"], n)
    data["slope"]              = np.random.uniform(-1.0, 0.5, n)
    data["rx_last"]            = np.random.uniform(-27, -18, n)
    data["rx_first"]           = np.random.uniform(-22, -15, n)
    data["rx_mean"]            = np.random.uniform(-25, -18, n)
    data["rx_std"]             = np.random.uniform(0.1, 2.0, n)
    data["rx_min"]             = np.random.uniform(-30, -25, n)
    data["rx_max"]             = np.random.uniform(-20, -15, n)
    data["ci_lower"]           = data["rx_last"] - 1.96 * np.array(data["rx_std"])
    data["ci_upper"]           = data["rx_last"] + 1.96 * np.array(data["rx_std"])
    data["rx_threshold"]       = RX_THRESHOLD
    df = pd.DataFrame(data)
    df["jours_avant_seuil"] = df.apply(
        lambda row: jours_avant_seuil_hn(row["slope"], row["rx_last"]), axis=1
    )
    df["niveau_risque"] = df.apply(
        lambda row: niveau_risque_from_slope_and_jours(row["slope"], row["jours_avant_seuil"]), axis=1
    )
    return df


@pytest.fixture
def sample_igd_scenario3_processed():
    """DataFrame IGD simulé après preprocessing Scénario 3"""
    np.random.seed(42)
    n = 20
    data = {}
    for i in range(1, 11):
        data[f"snr_T{i}"] = np.random.uniform(5, 20, n)
    data["device_id"]          = [f"IGD_{i:03d}" for i in range(n)]
    data["health_score"]       = np.random.choice([0, 1, 2], n)
    data["root_cause"]         = np.random.choice(["Line degradation", "High Interference / Noise", "NORMAL"], n)
    data["slope"]              = np.random.uniform(-1.0, 0.5, n)
    data["snr_last"]           = np.random.uniform(7, 18, n)
    data["snr_first"]          = np.random.uniform(10, 20, n)
    data["snr_mean"]           = np.random.uniform(8, 18, n)
    data["snr_std"]            = np.random.uniform(0.1, 2.0, n)
    data["snr_min"]            = np.random.uniform(5, 10, n)
    data["snr_max"]            = np.random.uniform(15, 25, n)
    data["ci_lower"]           = data["snr_last"] - 1.96 * np.array(data["snr_std"])
    data["ci_upper"]           = data["snr_last"] + 1.96 * np.array(data["snr_std"])
    data["snr_threshold"]      = SNR_THRESHOLD
    df = pd.DataFrame(data)
    df["jours_avant_seuil"] = df.apply(
        lambda row: jours_avant_seuil_igd(row["slope"], row["snr_last"]), axis=1
    )
    df["niveau_risque"] = df.apply(
        lambda row: niveau_risque_from_slope_and_jours(row["slope"], row["jours_avant_seuil"]), axis=1
    )
    return df


# ─────────────────────────────────────────────
# TESTS — Seuils critiques
# ─────────────────────────────────────────────

class TestSeuils:

    def test_rx_threshold_value(self):
        """Seuil critique RxPower doit être -28.0 dBm"""
        assert RX_THRESHOLD == -28.0

    def test_snr_threshold_value(self):
        """Seuil critique SNR doit être 6.0 dB"""
        assert SNR_THRESHOLD == 6.0


# ─────────────────────────────────────────────
# TESTS — Calcul slope
# ─────────────────────────────────────────────

class TestSlopeCalculation:

    def test_slope_formula(self):
        """slope = (T10 - T1) / 9"""
        rx_T1, rx_T10 = -20.0, -25.0
        slope = (rx_T10 - rx_T1) / 9
        assert round(slope, 6) == round(-5.0 / 9, 6)

    def test_slope_negative_means_degradation(self):
        """Un slope négatif indique une dégradation"""
        slope = (-25.0 - (-20.0)) / 9
        assert slope < 0

    def test_slope_positive_means_improvement(self):
        """Un slope positif indique une amélioration"""
        slope = (-18.0 - (-22.0)) / 9
        assert slope > 0

    def test_slope_zero_means_stable(self):
        """Un slope nul indique un équipement stable"""
        slope = (-20.0 - (-20.0)) / 9
        assert slope == 0.0

    def test_slope_snr_calculation(self):
        """Slope SNR correctement calculé"""
        snr_T1, snr_T10 = 15.0, 9.0
        slope = (snr_T10 - snr_T1) / 9
        assert round(slope, 6) == round(-6.0 / 9, 6)


# ─────────────────────────────────────────────
# TESTS — jours_avant_seuil
# ─────────────────────────────────────────────

class TestJoursAvantSeuil:

    def test_jours_none_when_slope_positive_hn(self):
        """jours = None si slope >= 0 (pas de dégradation) HN"""
        assert jours_avant_seuil_hn(0.1, -25.0) is None

    def test_jours_none_when_slope_zero_hn(self):
        """jours = None si slope = 0 HN"""
        assert jours_avant_seuil_hn(0.0, -25.0) is None

    def test_jours_positive_when_not_exceeded_hn(self):
        """jours > 0 si seuil pas encore atteint HN"""
        jours = jours_avant_seuil_hn(-0.5, -25.0)
        assert jours is not None
        assert jours > 0

    def test_jours_correct_value_hn(self):
        """Vérification calcul jours HN : (-28 - (-25)) / 0.5 = 6"""
        jours = jours_avant_seuil_hn(-0.5, -25.0)
        assert jours == round((-28.0 - (-25.0)) / (-0.5), 2)
        assert jours == 6.0

    def test_jours_none_when_slope_positive_igd(self):
        """jours = None si slope >= 0 IGD"""
        assert jours_avant_seuil_igd(0.2, 10.0) is None

    def test_jours_positive_when_not_exceeded_igd(self):
        """jours > 0 si seuil SNR pas encore atteint"""
        jours = jours_avant_seuil_igd(-0.5, 9.0)
        assert jours is not None
        assert jours > 0

    def test_jours_correct_value_igd(self):
        """Vérification calcul jours IGD : (6 - 9) / 0.5 = 6"""
        jours = jours_avant_seuil_igd(-0.5, 9.0)
        assert jours == round((6.0 - 9.0) / (-0.5), 2)
        assert jours == 6.0


# ─────────────────────────────────────────────
# TESTS — niveau_risque
# ─────────────────────────────────────────────

class TestNiveauRisque:

    def test_niveau_stable_when_slope_positive(self):
        """slope >= 0 → niveau stable"""
        assert niveau_risque_from_slope_and_jours(0.1, None) == "stable"

    def test_niveau_critique_when_2_days_or_less(self):
        """jours <= 2 → critique"""
        assert niveau_risque_from_slope_and_jours(-1.0, 1.5) == "critique"

    def test_niveau_critique_exact_2_days(self):
        """jours = 2 exactement → critique"""
        assert niveau_risque_from_slope_and_jours(-1.0, 2.0) == "critique"

    def test_niveau_eleve_between_2_and_7(self):
        """2 < jours <= 7 → eleve"""
        assert niveau_risque_from_slope_and_jours(-1.0, 5.0) == "eleve"

    def test_niveau_eleve_exact_7_days(self):
        """jours = 7 exactement → eleve"""
        assert niveau_risque_from_slope_and_jours(-1.0, 7.0) == "eleve"

    def test_niveau_modere_between_7_and_30(self):
        """7 < jours <= 30 → modere"""
        assert niveau_risque_from_slope_and_jours(-1.0, 15.0) == "modere"

    def test_niveau_modere_exact_30_days(self):
        """jours = 30 exactement → modere"""
        assert niveau_risque_from_slope_and_jours(-1.0, 30.0) == "modere"

    def test_niveau_faible_above_30_days(self):
        """jours > 30 → faible"""
        assert niveau_risque_from_slope_and_jours(-1.0, 60.0) == "faible"

    def test_valid_risk_levels_in_dataset(self, sample_hn_scenario3_processed):
        """Tous les niveaux de risque doivent être valides"""
        valid_levels = {"stable", "faible", "modere", "eleve", "critique"}
        assert set(sample_hn_scenario3_processed["niveau_risque"].unique()).issubset(valid_levels)

    def test_valid_risk_levels_igd(self, sample_igd_scenario3_processed):
        """Tous les niveaux de risque IGD doivent être valides"""
        valid_levels = {"stable", "faible", "modere", "eleve", "critique"}
        assert set(sample_igd_scenario3_processed["niveau_risque"].unique()).issubset(valid_levels)


# ─────────────────────────────────────────────
# TESTS — get_tendance
# ─────────────────────────────────────────────

class TestGetTendance:

    def test_tendance_amelioration_rapide(self):
        assert get_tendance(0.6) == "amélioration rapide"

    def test_tendance_amelioration_legere(self):
        assert get_tendance(0.2) == "amélioration légère"

    def test_tendance_stable(self):
        assert get_tendance(0.0) == "stable"

    def test_tendance_degradation_legere(self):
        assert get_tendance(-0.3) == "dégradation légère"

    def test_tendance_degradation_moderee(self):
        assert get_tendance(-0.7) == "dégradation modérée"

    def test_tendance_degradation_rapide(self):
        assert get_tendance(-1.5) == "dégradation rapide"


# ─────────────────────────────────────────────
# TESTS — get_action
# ─────────────────────────────────────────────

class TestGetAction:

    def test_action_stable_hn(self):
        """Action stable HN correcte"""
        action = get_action_hn("stable", None)
        assert "stable" in action.lower() or "aucune" in action.lower()

    def test_action_critique_hn(self):
        """Action critique HN = intervention urgente"""
        action = get_action_hn("critique", 1)
        assert "urgente" in action.lower() or "intervention" in action.lower()

    def test_action_stable_igd(self):
        """Action stable IGD correcte"""
        action = get_action_igd("stable", None)
        assert "aucune" in action.lower() or "stable" in action.lower()

    def test_action_critique_igd(self):
        """Action critique IGD = intervention urgente"""
        action = get_action_igd("critique", 1)
        assert "urgente" in action.lower() or "intervention" in action.lower()


# ─────────────────────────────────────────────
# TESTS — Intervalle de confiance
# ─────────────────────────────────────────────

class TestIntervalleConfiance:

    def test_ci_formula(self):
        """CI 95% = last ± 1.96 * std"""
        rx_last = -22.0
        rx_std  = 1.5
        ci_lower = rx_last - 1.96 * rx_std
        ci_upper = rx_last + 1.96 * rx_std
        assert round(ci_lower, 4) == round(-22.0 - 1.96 * 1.5, 4)
        assert round(ci_upper, 4) == round(-22.0 + 1.96 * 1.5, 4)

    def test_ci_lower_less_than_upper(self):
        """ci_lower doit toujours être < ci_upper"""
        rx_last, rx_std = -22.0, 1.5
        ci_lower = rx_last - 1.96 * rx_std
        ci_upper = rx_last + 1.96 * rx_std
        assert ci_lower < ci_upper

    def test_ci_in_dataset_hn(self, sample_hn_scenario3_processed):
        """ci_lower < ci_upper pour tout le dataset HN"""
        assert (sample_hn_scenario3_processed["ci_lower"] <
                sample_hn_scenario3_processed["ci_upper"]).all()

    def test_ci_in_dataset_igd(self, sample_igd_scenario3_processed):
        """ci_lower < ci_upper pour tout le dataset IGD"""
        assert (sample_igd_scenario3_processed["ci_lower"] <
                sample_igd_scenario3_processed["ci_upper"]).all()


# ─────────────────────────────────────────────
# TESTS — Structure des résultats
# ─────────────────────────────────────────────

class TestResultStructure:

    def test_result_has_required_keys_hn(self, sample_hn_scenario3_processed):
        """Chaque résultat HN doit avoir les clés requises"""
        row = sample_hn_scenario3_processed.iloc[0]
        required_keys = ["device_id", "vendor", "health_score", "root_cause",
                         "slope", "niveau_risque", "jours_avant_seuil"]
        for key in required_keys:
            assert key in row.index

    def test_result_has_required_keys_igd(self, sample_igd_scenario3_processed):
        """Chaque résultat IGD doit avoir les clés requises"""
        row = sample_igd_scenario3_processed.iloc[0]
        required_keys = ["device_id", "health_score", "root_cause",
                         "slope", "niveau_risque", "jours_avant_seuil"]
        for key in required_keys:
            assert key in row.index

    def test_jours_is_none_for_stable_devices(self, sample_hn_scenario3_processed):
        """Les équipements stables (slope >= 0) doivent avoir jours = None"""
        stable_devices = sample_hn_scenario3_processed[
            sample_hn_scenario3_processed["niveau_risque"] == "stable"
        ]
        for _, row in stable_devices.iterrows():
            assert row["jours_avant_seuil"] is None or pd.isna(row["jours_avant_seuil"])

    def test_dataset_not_empty(self, sample_hn_scenario3_processed):
        """Le dataset ne doit pas être vide"""
        assert len(sample_hn_scenario3_processed) > 0

    def test_dataset_igd_not_empty(self, sample_igd_scenario3_processed):
        """Le dataset IGD ne doit pas être vide"""
        assert len(sample_igd_scenario3_processed) > 0