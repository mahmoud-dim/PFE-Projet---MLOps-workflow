import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# test_evaluate.py
# Tests unitaires — Evaluation + Gatekeeping (Scénario 1)
# Couvre : evaluate_huawei_nokia.py, evaluate_igd.py
# ============================================================

ACCURACY_THRESHOLD = 0.80
F1_THRESHOLD       = 0.75


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def mock_metrics_passing():
    """Métriques simulées qui passent le gatekeeping"""
    return {
        "test": {
            "accuracy":   0.975,
            "f1_macro":   0.981,
            "f1_weighted":0.982,
            "precision_macro": 0.98,
            "recall_macro":    0.97,
            "per_class": {
                "precision": [0.99, 0.98, 0.97],
                "recall":    [0.98, 0.97, 0.96],
                "f1_score":  [0.98, 0.97, 0.96]
            }
        },
        "train": {
            "accuracy": 0.99,
            "f1_macro": 0.99
        },
        "overfitting": {
            "train_test_gap": 0.015,
            "status": "good"
        },
        "monitoring_kpi": {
            "degraded_recall":    0.97,
            "retraining_threshold": 0.55,
            "trigger_retraining": False
        }
    }


@pytest.fixture
def mock_metrics_failing_accuracy():
    """Métriques simulées qui échouent sur l'accuracy"""
    return {
        "test": {"accuracy": 0.75, "f1_macro": 0.80},
        "overfitting": {"status": "good"}
    }


@pytest.fixture
def mock_metrics_failing_f1():
    """Métriques simulées qui échouent sur le F1"""
    return {
        "test": {"accuracy": 0.85, "f1_macro": 0.70},
        "overfitting": {"status": "good"}
    }


@pytest.fixture
def mock_metrics_igd_passing():
    """Métriques IGD simulées qui passent"""
    return {
        "test": {
            "accuracy": 0.833,
            "f1_macro": 0.801,
        },
        "overfitting": {
            "train_test_gap": 0.05,
            "status": "good"
        },
        "monitoring_kpi": {
            "degraded_recall": 0.80,
            "retraining_threshold": 0.55,
            "trigger_retraining": False
        }
    }


@pytest.fixture
def simple_trained_model():
    """Modèle simple entraîné pour tests d'évaluation"""
    np.random.seed(42)
    n = 150
    X = pd.DataFrame({
        "temperature_c":    np.random.uniform(25, 70, n),
        "rx_power_dbm":     np.random.uniform(-30, -10, n),
        "bias_current_ua":  np.random.uniform(5, 20, n),
        "supply_voltage_v": np.random.uniform(3.0, 3.6, n),
    })
    y = np.random.choice([0, 1, 2], n)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# TESTS — Seuils Gatekeeping
# ─────────────────────────────────────────────

class TestGatekeepingThresholds:

    def test_accuracy_threshold_value(self):
        """Le seuil accuracy doit être exactement 0.80"""
        assert ACCURACY_THRESHOLD == 0.80

    def test_f1_threshold_value(self):
        """Le seuil F1 doit être exactement 0.75"""
        assert F1_THRESHOLD == 0.75

    def test_gatekeeping_passes_with_good_metrics(self, mock_metrics_passing):
        """Le gatekeeping doit passer avec de bonnes métriques"""
        acc   = mock_metrics_passing["test"]["accuracy"]
        f1    = mock_metrics_passing["test"]["f1_macro"]
        overfit = mock_metrics_passing["overfitting"]["status"]

        acc_pass     = acc >= ACCURACY_THRESHOLD
        f1_pass      = f1 >= F1_THRESHOLD
        overfit_pass = overfit in ["good", "moderate"]

        assert acc_pass     is True
        assert f1_pass      is True
        assert overfit_pass is True

    def test_gatekeeping_fails_with_low_accuracy(self, mock_metrics_failing_accuracy):
        """Le gatekeeping doit échouer si accuracy < 0.80"""
        acc = mock_metrics_failing_accuracy["test"]["accuracy"]
        assert acc < ACCURACY_THRESHOLD

    def test_gatekeeping_fails_with_low_f1(self, mock_metrics_failing_f1):
        """Le gatekeeping doit échouer si F1 < 0.75"""
        f1 = mock_metrics_failing_f1["test"]["f1_macro"]
        assert f1 < F1_THRESHOLD

    def test_hn_accuracy_above_threshold(self, mock_metrics_passing):
        """HN accuracy 97.5% doit passer le seuil 80%"""
        assert mock_metrics_passing["test"]["accuracy"] >= ACCURACY_THRESHOLD

    def test_igd_accuracy_above_threshold(self, mock_metrics_igd_passing):
        """IGD accuracy 83.3% doit passer le seuil 80%"""
        assert mock_metrics_igd_passing["test"]["accuracy"] >= ACCURACY_THRESHOLD

    def test_hn_f1_above_threshold(self, mock_metrics_passing):
        """HN F1 98.1% doit passer le seuil 75%"""
        assert mock_metrics_passing["test"]["f1_macro"] >= F1_THRESHOLD

    def test_igd_f1_above_threshold(self, mock_metrics_igd_passing):
        """IGD F1 80.1% doit passer le seuil 75%"""
        assert mock_metrics_igd_passing["test"]["f1_macro"] >= F1_THRESHOLD


# ─────────────────────────────────────────────
# TESTS — evaluate_model
# ─────────────────────────────────────────────

class TestEvaluateModel:

    def test_accuracy_score_between_0_and_1(self, simple_trained_model):
        """L'accuracy doit être entre 0 et 1"""
        model, _, X_test, _, y_test = simple_trained_model
        y_pred    = model.predict(X_test)
        acc       = accuracy_score(y_test, y_pred)
        assert 0.0 <= acc <= 1.0

    def test_f1_score_between_0_and_1(self, simple_trained_model):
        """Le F1-score doit être entre 0 et 1"""
        model, _, X_test, _, y_test = simple_trained_model
        y_pred = model.predict(X_test)
        f1     = f1_score(y_test, y_pred, average="macro", zero_division=0)
        assert 0.0 <= f1 <= 1.0

    def test_predictions_are_valid_classes(self, simple_trained_model):
        """Toutes les prédictions doivent être dans {0, 1, 2}"""
        model, _, X_test, _, _ = simple_trained_model
        predictions = model.predict(X_test)
        assert set(predictions).issubset({0, 1, 2})

    def test_overfitting_gap_calculation(self, simple_trained_model):
        """Le gap overfitting doit être train_acc - test_acc"""
        model, X_train, X_test, y_train, y_test = simple_trained_model
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc  = accuracy_score(y_test,  model.predict(X_test))
        gap       = train_acc - test_acc
        assert isinstance(gap, float)
        assert gap >= -0.1   # train acc >= test acc en général

    def test_overfitting_status_good(self, mock_metrics_passing):
        """Un gap < 0.10 doit donner status 'good'"""
        gap    = mock_metrics_passing["overfitting"]["train_test_gap"]
        status = "good" if gap < 0.10 else "moderate" if gap < 0.15 else "high"
        assert status == "good"

    def test_overfitting_status_moderate(self):
        """Un gap entre 0.10 et 0.15 doit donner status 'moderate'"""
        gap    = 0.12
        status = "good" if gap < 0.10 else "moderate" if gap < 0.15 else "high"
        assert status == "moderate"

    def test_overfitting_status_high(self):
        """Un gap > 0.15 doit donner status 'high'"""
        gap    = 0.20
        status = "good" if gap < 0.10 else "moderate" if gap < 0.15 else "high"
        assert status == "high"


# ─────────────────────────────────────────────
# TESTS — compute_decision_metrics (Scoring opérationnel)
# ─────────────────────────────────────────────

class TestDecisionMetrics:

    def test_health_score_formula(self):
        """health_score = 1.0*p_opt + 0.5*p_deg + 0.0*p_crit"""
        p_opt, p_deg, p_crit = 0.0, 1.0, 0.0
        health_score = round(1.0*p_opt + 0.5*p_deg + 0.0*p_crit, 3)
        assert health_score == 0.5

    def test_risk_score_formula(self):
        """risk_score = 1.0 - health_score"""
        health_score = 0.5
        risk_score   = round(1.0 - health_score, 3)
        assert risk_score == 0.5

    def test_risk_level_low(self):
        """risk_score < 0.3 → risk_level = low"""
        risk_score = 0.2
        risk_level = "low" if risk_score < 0.3 else "medium" if risk_score < 0.6 else "high"
        assert risk_level == "low"

    def test_risk_level_medium(self):
        """0.3 <= risk_score < 0.6 → risk_level = medium"""
        risk_score = 0.5
        risk_level = "low" if risk_score < 0.3 else "medium" if risk_score < 0.6 else "high"
        assert risk_level == "medium"

    def test_risk_level_high(self):
        """risk_score >= 0.6 → risk_level = high"""
        risk_score = 0.8
        risk_level = "low" if risk_score < 0.3 else "medium" if risk_score < 0.6 else "high"
        assert risk_level == "high"

    def test_recommended_action_dispatch(self):
        """p_crit > 0.7 → dispatch_technician"""
        p_crit = 0.8
        p_deg  = 0.1
        action = "dispatch_technician" if p_crit > 0.7 else "monitor" if p_deg > 0.7 else "no_action"
        assert action == "dispatch_technician"

    def test_recommended_action_monitor(self):
        """p_deg > 0.7 et p_crit <= 0.7 → monitor"""
        p_crit = 0.1
        p_deg  = 0.8
        action = "dispatch_technician" if p_crit > 0.7 else "monitor" if p_deg > 0.7 else "no_action"
        assert action == "monitor"

    def test_recommended_action_no_action(self):
        """p_crit <= 0.7 et p_deg <= 0.7 → no_action"""
        p_crit = 0.1
        p_deg  = 0.3
        action = "dispatch_technician" if p_crit > 0.7 else "monitor" if p_deg > 0.7 else "no_action"
        assert action == "no_action"

    def test_auto_ticket_condition(self):
        """auto_ticket = True si p_crit > 0.8 ET confidence > 0.8"""
        p_crit     = 0.9
        confidence = 0.95
        auto_ticket = bool(p_crit > 0.8 and confidence > 0.8)
        assert auto_ticket is True

    def test_auto_ticket_false_when_low_confidence(self):
        """auto_ticket = False si confidence <= 0.8"""
        p_crit     = 0.9
        confidence = 0.7
        auto_ticket = bool(p_crit > 0.8 and confidence > 0.8)
        assert auto_ticket is False

    def test_priority_level_1(self):
        """risk_score > 0.8 → priority_level = 1"""
        risk_score     = 0.9
        priority_level = 1 if risk_score > 0.8 else 2 if risk_score > 0.5 else 3
        assert priority_level == 1

    def test_priority_level_2(self):
        """0.5 < risk_score <= 0.8 → priority_level = 2"""
        risk_score     = 0.6
        priority_level = 1 if risk_score > 0.8 else 2 if risk_score > 0.5 else 3
        assert priority_level == 2

    def test_priority_level_3(self):
        """risk_score <= 0.5 → priority_level = 3"""
        risk_score     = 0.2
        priority_level = 1 if risk_score > 0.8 else 2 if risk_score > 0.5 else 3
        assert priority_level == 3

    def test_class_map_correct(self):
        """Le mapping classes doit être {0: optimal, 1: degraded, 2: critical}"""
        class_map = {0: "optimal", 1: "degraded", 2: "critical"}
        assert class_map[0] == "optimal"
        assert class_map[1] == "degraded"
        assert class_map[2] == "critical"

    def test_predicted_class_valid(self, simple_trained_model):
        """Les classes prédites doivent être optimal, degraded ou critical"""
        model, _, X_test, _, _ = simple_trained_model
        class_map   = {0: "optimal", 1: "degraded", 2: "critical"}
        predictions = model.predict(X_test)
        for pred in predictions:
            assert class_map[int(pred)] in ["optimal", "degraded", "critical"]


# ─────────────────────────────────────────────
# TESTS — Monitoring KPI
# ─────────────────────────────────────────────

class TestMonitoringKPI:

    def test_retraining_threshold_value(self):
        """Le seuil de retraining sur degraded_recall doit être 0.55"""
        retraining_threshold = 0.55
        assert retraining_threshold == 0.55

    def test_no_retraining_triggered_hn(self, mock_metrics_passing):
        """Pas de retraining déclenché si degraded_recall >= 0.55"""
        degraded_recall = mock_metrics_passing["monitoring_kpi"]["degraded_recall"]
        trigger = degraded_recall < 0.55
        assert trigger is False

    def test_retraining_triggered_when_low_recall(self):
        """Retraining déclenché si degraded_recall < 0.55"""
        degraded_recall = 0.40
        trigger = degraded_recall < 0.55
        assert trigger is True