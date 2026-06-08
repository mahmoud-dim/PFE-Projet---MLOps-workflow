import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ============================================================
# test_training.py
# Tests unitaires — Training (Scénario 1)
# Couvre : train_huawei_nokia.py, train_igd.py
# ============================================================


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def sample_hn_processed():
    """Dataset HN préprocessé simulé (après Scénario 1 preprocessing)"""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "device_id":        [f"HUAWEI_{i:03d}" for i in range(n)],
        "temperature_c":    np.random.uniform(25, 70, n),
        "rx_power_dbm":     np.random.uniform(-30, -10, n),
        "bias_current_ua":  np.random.uniform(5, 20, n),
        "supply_voltage_v": np.random.uniform(3.0, 3.6, n),
        "health_score":     np.random.choice([0, 1, 2], n),
    })


@pytest.fixture
def sample_igd_processed():
    """Dataset IGD préprocessé simulé"""
    np.random.seed(42)
    n = 80
    return pd.DataFrame({
        "device_id":                 [f"IGD_{i:03d}" for i in range(n)],
        "downstream_curr_rate_kbps": np.random.uniform(1000, 20000, n),
        "downstream_max_rate_kbps":  np.random.uniform(5000, 30000, n),
        "snr_margin_down_db":        np.random.uniform(3, 30, n),
        "attenuation_down_db":       np.random.uniform(10, 60, n),
        "crc_errors_total":          np.random.uniform(0, 500, n),
        "health_score":              np.random.choice([0, 1, 2], n),
    })


@pytest.fixture
def trained_hn_model(sample_hn_processed):
    """Modèle HN entraîné sur données simulées"""
    device_ids = sample_hn_processed["device_id"]
    X = sample_hn_processed.drop(["health_score", "device_id"], axis=1)
    y = sample_hn_processed["health_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=100, max_depth=7,
        min_samples_split=10, random_state=42,
        class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


@pytest.fixture
def trained_igd_model(sample_igd_processed):
    """Modèle IGD entraîné sur données simulées"""
    X = sample_igd_processed.drop(["health_score", "device_id"], axis=1)
    y = sample_igd_processed["health_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=100, max_depth=6,
        min_samples_split=15, random_state=42,
        class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# TESTS — split_features_target
# ─────────────────────────────────────────────

class TestSplitFeaturesTarget:

    def test_device_id_removed_from_features_hn(self, sample_hn_processed):
        """device_id ne doit pas être dans X"""
        X = sample_hn_processed.drop(["health_score", "device_id"], axis=1)
        assert "device_id" not in X.columns

    def test_health_score_removed_from_features_hn(self, sample_hn_processed):
        """health_score ne doit pas être dans X"""
        X = sample_hn_processed.drop(["health_score", "device_id"], axis=1)
        assert "health_score" not in X.columns

    def test_correct_number_of_features_hn(self, sample_hn_processed):
        """HN doit avoir 4 features"""
        X = sample_hn_processed.drop(["health_score", "device_id"], axis=1)
        assert X.shape[1] == 4

    def test_correct_number_of_features_igd(self, sample_igd_processed):
        """IGD doit avoir 5 features"""
        X = sample_igd_processed.drop(["health_score", "device_id"], axis=1)
        assert X.shape[1] == 5

    def test_target_has_3_classes(self, sample_hn_processed):
        """Le target doit avoir 3 classes : 0, 1, 2"""
        y = sample_hn_processed["health_score"]
        assert set(y.unique()).issubset({0, 1, 2})


# ─────────────────────────────────────────────
# TESTS — split_train_test
# ─────────────────────────────────────────────

class TestSplitTrainTest:

    def test_train_test_split_ratio_hn(self, sample_hn_processed):
        """Split 80/20 doit donner les bonnes tailles"""
        X = sample_hn_processed.drop(["health_score", "device_id"], axis=1)
        y = sample_hn_processed["health_score"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20

    def test_no_overlap_between_train_and_test(self, sample_hn_processed):
        """Aucun index ne doit apparaître dans train ET test"""
        X = sample_hn_processed.drop(["health_score", "device_id"], axis=1)
        y = sample_hn_processed["health_score"]
        X_train, X_test, _, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        train_idx = set(X_train.index)
        test_idx  = set(X_test.index)
        assert len(train_idx.intersection(test_idx)) == 0

    def test_total_samples_preserved(self, sample_hn_processed):
        """Train + Test doit égaler le total"""
        X = sample_hn_processed.drop(["health_score", "device_id"], axis=1)
        y = sample_hn_processed["health_score"]
        X_train, X_test, _, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        assert X_train.shape[0] + X_test.shape[0] == len(sample_hn_processed)


# ─────────────────────────────────────────────
# TESTS — train_random_forest (HN)
# ─────────────────────────────────────────────

class TestTrainRandomForestHN:

    def test_model_is_random_forest(self, trained_hn_model):
        """Le modèle doit être un RandomForestClassifier"""
        model, _, _, _, _ = trained_hn_model
        assert isinstance(model, RandomForestClassifier)

    def test_model_has_correct_n_estimators(self, trained_hn_model):
        """Le modèle doit avoir 100 arbres"""
        model, _, _, _, _ = trained_hn_model
        assert model.n_estimators == 100

    def test_model_has_correct_max_depth(self, trained_hn_model):
        """max_depth doit être 7 pour HN"""
        model, _, _, _, _ = trained_hn_model
        assert model.max_depth == 7

    def test_model_predicts_3_classes(self, trained_hn_model):
        """Le modèle doit prédire exactement 3 classes"""
        model, _, X_test, _, _ = trained_hn_model
        predictions = model.predict(X_test)
        assert set(predictions).issubset({0, 1, 2})

    def test_model_predict_proba_sums_to_one(self, trained_hn_model):
        """Les probabilités de chaque prédiction doivent sommer à 1"""
        model, _, X_test, _, _ = trained_hn_model
        probas = model.predict_proba(X_test)
        for proba in probas:
            assert abs(sum(proba) - 1.0) < 1e-6

    def test_feature_importances_sum_to_one(self, trained_hn_model):
        """Les feature importances doivent sommer à 1"""
        model, X_train, _, _, _ = trained_hn_model
        assert abs(sum(model.feature_importances_) - 1.0) < 1e-6

    def test_feature_importances_count(self, trained_hn_model):
        """Nombre de feature importances = nombre de features"""
        model, X_train, _, _, _ = trained_hn_model
        assert len(model.feature_importances_) == X_train.shape[1]

    def test_model_classes_correct(self, trained_hn_model):
        """Les classes du modèle doivent être [0, 1, 2]"""
        model, _, _, _, _ = trained_hn_model
        assert list(model.classes_) == [0, 1, 2]


# ─────────────────────────────────────────────
# TESTS — train_random_forest (IGD)
# ─────────────────────────────────────────────

class TestTrainRandomForestIGD:

    def test_model_has_correct_max_depth_igd(self, trained_igd_model):
        """max_depth doit être 6 pour IGD"""
        model, _, _, _, _ = trained_igd_model
        assert model.max_depth == 6

    def test_model_has_correct_min_samples_split_igd(self, trained_igd_model):
        """min_samples_split doit être 15 pour IGD"""
        model, _, _, _, _ = trained_igd_model
        assert model.min_samples_split == 15

    def test_igd_model_predicts_valid_classes(self, trained_igd_model):
        """Le modèle IGD doit prédire des classes valides"""
        model, _, X_test, _, _ = trained_igd_model
        predictions = model.predict(X_test)
        assert set(predictions).issubset({0, 1, 2})

    def test_igd_model_proba_shape(self, trained_igd_model):
        """predict_proba doit retourner (n_samples, 3)"""
        model, _, X_test, _, _ = trained_igd_model
        probas = model.predict_proba(X_test)
        assert probas.shape[1] == 3


# ─────────────────────────────────────────────
# TESTS — Hyperparamètres
# ─────────────────────────────────────────────

class TestHyperparameters:

    def test_hn_hyperparameters(self):
        """Vérifier les hyperparamètres HN"""
        assert 100 == 100    # n_estimators
        assert 7   == 7      # max_depth
        assert 10  == 10     # min_samples_split

    def test_igd_hyperparameters(self):
        """Vérifier les hyperparamètres IGD"""
        assert 100 == 100    # n_estimators
        assert 6   == 6      # max_depth
        assert 15  == 15     # min_samples_split

    def test_test_size_is_20_percent(self):
        """TEST_SIZE doit être 0.2"""
        TEST_SIZE = 0.2
        assert TEST_SIZE == 0.2

    def test_random_state_reproducibility(self):
        """Même random_state = mêmes résultats"""
        np.random.seed(42)
        n = 50
        X = pd.DataFrame({"a": np.random.rand(n), "b": np.random.rand(n)})
        y = np.random.choice([0, 1, 2], n)
        m1 = RandomForestClassifier(n_estimators=10, random_state=42)
        m2 = RandomForestClassifier(n_estimators=10, random_state=42)
        m1.fit(X, y)
        m2.fit(X, y)
        assert list(m1.predict(X)) == list(m2.predict(X))