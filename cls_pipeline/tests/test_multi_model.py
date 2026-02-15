"""
test_multi_model.py — Tests for multi-model orchestrator and config.

Verifies:
1. MultiModelResult structure and serialization
2. Aggregation produces correct mean/std
3. Config model_groups parsing and get_model_pairs()
4. Backward compatibility (pipeline without model_groups works)
"""

import numpy as np
import pytest

from src.core.config_loader import ModelConfig, Config, RSAConfig


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def mock_model_configs():
    """Create mock ModelConfig objects for testing."""
    weird_1 = ModelConfig(
        name="model-weird-1", dimension=64, label="Weird-1", language="en", prefix=""
    )
    weird_2 = ModelConfig(
        name="model-weird-2", dimension=64, label="Weird-2", language="en", prefix=""
    )
    sinic_1 = ModelConfig(
        name="model-sinic-1", dimension=64, label="Sinic-1", language="zh", prefix=""
    )
    sinic_2 = ModelConfig(
        name="model-sinic-2", dimension=64, label="Sinic-2", language="zh", prefix=""
    )
    return weird_1, weird_2, sinic_1, sinic_2


@pytest.fixture
def synthetic_embeddings_multi():
    """Generate synthetic embedding pairs for 2 model pairs."""
    np.random.seed(42)
    n, d = 10, 64

    # Due coppie di embedding sintetiche
    pairs = []
    for _ in range(2):
        w = np.random.randn(n, d)
        s = np.random.randn(n, d)
        w = w / np.linalg.norm(w, axis=1, keepdims=True)
        s = s / np.linalg.norm(s, axis=1, keepdims=True)
        pairs.append((w, s))

    labels = [f"term_{i}" for i in range(n)]
    return pairs, labels


# =========================================================================
# MultiModelResult
# =========================================================================

class TestMultiModelResult:
    """Tests for MultiModelResult dataclass."""

    def test_result_structure(self):
        from src.experiments.multi_model import MultiModelResult

        result = MultiModelResult(
            pair_results=[{"spearman_r": 0.5}, {"spearman_r": 0.6}],
            aggregate={"mean": 0.55, "std": 0.05},
            model_pairs=[("W1", "S1"), ("W2", "S2")],
        )

        assert len(result.pair_results) == 2
        assert result.aggregate["mean"] == 0.55
        assert len(result.model_pairs) == 2

    def test_to_dict(self):
        from src.experiments.multi_model import MultiModelResult

        result = MultiModelResult(
            pair_results=[{"spearman_r": 0.5}],
            aggregate={"mean": 0.5, "stat_key": "spearman_r"},
            model_pairs=[("W1", "S1")],
        )
        d = result.to_dict()

        assert "n_pairs" in d
        assert d["n_pairs"] == 1
        assert "model_pairs" in d
        assert d["model_pairs"][0] == {"weird": "W1", "sinic": "S1"}
        assert "pair_results" in d
        assert "aggregate" in d

    def test_empty_result(self):
        from src.experiments.multi_model import MultiModelResult

        result = MultiModelResult()
        d = result.to_dict()
        assert d["n_pairs"] == 0


# =========================================================================
# run_experiment_multi_model
# =========================================================================

class TestRunExperimentMultiModel:
    """Tests for the multi-model orchestrator."""

    def test_aggregation_correct(self, synthetic_embeddings_multi, mock_model_configs):
        from src.experiments.multi_model import run_experiment_multi_model
        from src.experiments.exp_rsa import run_rsa

        pairs, labels = synthetic_embeddings_multi
        weird_1, weird_2, sinic_1, sinic_2 = mock_model_configs

        # Creiamo un mock client che restituisce embedding precomputati
        class MockClient:
            def __init__(self, emb_map):
                self.emb_map = emb_map

            def get_embeddings_for_model(self, texts, model_config, normalize=True):
                return self.emb_map[model_config.name]

        # Mapping: model_name -> embedding
        emb_map = {
            "model-weird-1": pairs[0][0],
            "model-weird-2": pairs[1][0],
            "model-sinic-1": pairs[0][1],
            "model-sinic-2": pairs[1][1],
        }
        client = MockClient(emb_map)

        model_pairs = [(weird_1, sinic_1), (weird_2, sinic_2)]

        result = run_experiment_multi_model(
            experiment_fn=run_rsa,
            model_pairs=model_pairs,
            client=client,
            texts_weird=[f"en_{i}" for i in range(10)],
            texts_sinic=[f"zh_{i}" for i in range(10)],
            stat_key="spearman_r",
            labels=labels,
            n_permutations=50,
            n_bootstrap=20,
            seed=42,
        )

        assert len(result.pair_results) == 2
        assert len(result.model_pairs) == 2
        assert "mean" in result.aggregate
        assert "std" in result.aggregate

        # Verifica che la media sia corretta
        r_values = [pr["spearman_r"] for pr in result.pair_results]
        expected_mean = np.mean(r_values)
        assert np.isclose(result.aggregate["mean"], expected_mean)

        expected_std = np.std(r_values)
        assert np.isclose(result.aggregate["std"], expected_std)

    def test_single_pair(self, synthetic_embeddings_multi, mock_model_configs):
        """Single pair should produce std=0."""
        from src.experiments.multi_model import run_experiment_multi_model
        from src.experiments.exp_rsa import run_rsa

        pairs, labels = synthetic_embeddings_multi
        weird_1, _, sinic_1, _ = mock_model_configs

        class MockClient:
            def get_embeddings_for_model(self, texts, model_config, normalize=True):
                if model_config.name == "model-weird-1":
                    return pairs[0][0]
                return pairs[0][1]

        result = run_experiment_multi_model(
            experiment_fn=run_rsa,
            model_pairs=[(weird_1, sinic_1)],
            client=MockClient(),
            texts_weird=[f"en_{i}" for i in range(10)],
            texts_sinic=[f"zh_{i}" for i in range(10)],
            stat_key="spearman_r",
            labels=labels,
            n_permutations=50,
            n_bootstrap=20,
            seed=42,
        )

        assert len(result.pair_results) == 1
        assert result.aggregate["std"] == 0.0


# =========================================================================
# Config model_groups
# =========================================================================

class TestConfigModelGroups:
    """Tests for model_groups config support."""

    def test_is_multi_model_false_by_default(self, tmp_path):
        """Config without model_groups should have is_multi_model=False."""
        from src.core.config_loader import load_config
        import yaml

        config_data = {
            "pipeline": {"version": "3.0.0", "name": "Test"},
            "models": {
                "weird": {"name": "m1", "dimension": 64, "label": "W", "language": "en"},
                "sinic": {"name": "m2", "dimension": 64, "label": "S", "language": "zh"},
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_path)
        assert config.is_multi_model is False

    def test_is_multi_model_true_with_groups(self, tmp_path):
        """Config with model_groups should have is_multi_model=True."""
        from src.core.config_loader import load_config
        import yaml

        config_data = {
            "pipeline": {"version": "3.0.0", "name": "Test"},
            "models": {
                "weird": {"name": "m1", "dimension": 64, "label": "W", "language": "en"},
                "sinic": {"name": "m2", "dimension": 64, "label": "S", "language": "zh"},
            },
            "model_groups": {
                "weird": [
                    {"name": "w1", "dimension": 64, "label": "W1", "language": "en"},
                    {"name": "w2", "dimension": 64, "label": "W2", "language": "en"},
                ],
                "sinic": [
                    {"name": "s1", "dimension": 64, "label": "S1", "language": "zh"},
                ],
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_path)
        assert config.is_multi_model is True

    def test_get_model_pairs_cartesian(self, tmp_path):
        """get_model_pairs should return cartesian product."""
        from src.core.config_loader import load_config
        import yaml

        config_data = {
            "pipeline": {"version": "3.0.0", "name": "Test"},
            "models": {
                "weird": {"name": "m1", "dimension": 64, "label": "W", "language": "en"},
                "sinic": {"name": "m2", "dimension": 64, "label": "S", "language": "zh"},
            },
            "model_groups": {
                "weird": [
                    {"name": "w1", "dimension": 64, "label": "W1", "language": "en"},
                    {"name": "w2", "dimension": 64, "label": "W2", "language": "en"},
                ],
                "sinic": [
                    {"name": "s1", "dimension": 64, "label": "S1", "language": "zh"},
                    {"name": "s2", "dimension": 64, "label": "S2", "language": "zh"},
                ],
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_path)
        pairs = config.get_model_pairs()

        # 2 weird × 2 sinic = 4 coppie
        assert len(pairs) == 4
        pair_names = [(p[0].name, p[1].name) for p in pairs]
        assert ("w1", "s1") in pair_names
        assert ("w1", "s2") in pair_names
        assert ("w2", "s1") in pair_names
        assert ("w2", "s2") in pair_names

    def test_get_model_pairs_raises_without_groups(self):
        """get_model_pairs should raise if no model_groups."""
        from src.core.config_loader import Config, DeviceConfig, ExperimentsConfig, PathsConfig, OutputConfig, LoggingConfig
        from pathlib import Path

        config = Config(
            version="3.0.0",
            name="Test",
            models={},
            device=DeviceConfig(),
            random_seed=42,
            experiments=ExperimentsConfig(),
            paths=PathsConfig(),
            output=OutputConfig(),
            logging=LoggingConfig(),
            project_root=Path("."),
        )

        with pytest.raises(ValueError, match="model_groups not configured"):
            config.get_model_pairs()


# =========================================================================
# Backward compatibility
# =========================================================================

class TestBackwardCompatibility:
    """Ensure pipeline works without --multi-model."""

    def test_rsa_result_has_new_fields(self):
        """RSA result should have r_squared and bootstrap_ci."""
        from src.experiments.exp_rsa import run_rsa

        np.random.seed(42)
        n, d = 10, 64
        w = np.random.randn(n, d)
        s = np.random.randn(n, d)
        w = w / np.linalg.norm(w, axis=1, keepdims=True)
        s = s / np.linalg.norm(s, axis=1, keepdims=True)
        labels = [f"t{i}" for i in range(n)]

        result = run_rsa(w, s, labels, n_permutations=50, n_bootstrap=20)

        # Nuovi campi presenti
        assert hasattr(result, "r_squared")
        assert hasattr(result, "bootstrap_ci")
        assert hasattr(result, "null_distribution")

        # to_dict include i nuovi campi
        d = result.to_dict()
        assert "r_squared" in d
        assert "bootstrap_ci" in d
        assert "null_distribution" in d
