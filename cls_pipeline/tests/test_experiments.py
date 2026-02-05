"""
test_experiments.py — Unit tests for all experiment modules.

Uses synthetic data to verify output shapes, mathematical correctness,
and statistical test validity.
"""

import numpy as np
import pytest


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def synthetic_embeddings():
    """Generate synthetic normalized embeddings (N=15, D=64)."""
    np.random.seed(42)
    n, d = 15, 64

    weird = np.random.randn(n, d)
    sinic = np.random.randn(n, d)

    weird = weird / np.linalg.norm(weird, axis=1, keepdims=True)
    sinic = sinic / np.linalg.norm(sinic, axis=1, keepdims=True)

    labels = [f"term_{i}" for i in range(n)]
    return weird, sinic, labels


@pytest.fixture
def large_corpus():
    """Generate larger corpus for NDA tests (50 core + 100 background)."""
    np.random.seed(42)
    d = 64
    n_core, n_bg = 10, 40

    core_w = np.random.randn(n_core, d)
    core_s = np.random.randn(n_core, d)
    bg_w = np.random.randn(n_bg, d)
    bg_s = np.random.randn(n_bg, d)

    all_w = np.vstack([core_w, bg_w])
    all_s = np.vstack([core_s, bg_s])

    # Normalize
    core_w = core_w / np.linalg.norm(core_w, axis=1, keepdims=True)
    core_s = core_s / np.linalg.norm(core_s, axis=1, keepdims=True)
    all_w = all_w / np.linalg.norm(all_w, axis=1, keepdims=True)
    all_s = all_s / np.linalg.norm(all_s, axis=1, keepdims=True)

    core_labels = [f"core_{i}" for i in range(n_core)]
    all_labels = core_labels + [f"bg_{i}" for i in range(n_bg)]

    return core_w, core_s, core_labels, all_w, all_s, all_labels


# =========================================================================
# Statistical utilities
# =========================================================================

class TestStatistical:
    """Tests for statistical utility functions."""

    def test_permutation_test_basic(self):
        from src.experiments.statistical import permutation_test

        np.random.seed(42)
        data = np.random.randn(20)
        observed = np.mean(data)

        result = permutation_test(
            observed, data,
            stat_fn=lambda x: np.mean(x),
            n_permutations=500,
        )

        assert 0 <= result.p_value <= 1
        assert result.n_permutations == 500
        assert len(result.null_distribution) == 500

    def test_bootstrap_ci_basic(self):
        from src.experiments.statistical import bootstrap_ci

        np.random.seed(42)
        data = np.random.randn(50)

        result = bootstrap_ci(
            data,
            stat_fn=np.mean,
            n_bootstrap=500,
        )

        assert result.ci_lower < result.estimate < result.ci_upper
        assert result.n_bootstrap == 500
        assert result.alpha == 0.05

    def test_mantel_test_identical_matrices(self):
        from src.experiments.statistical import mantel_test

        np.random.seed(42)
        n = 10
        rdm = np.random.rand(n, n)
        rdm = (rdm + rdm.T) / 2
        np.fill_diagonal(rdm, 0)

        result = mantel_test(rdm, rdm, n_permutations=500)

        # Identical matrices should have r = 1.0
        assert np.isclose(result.observed, 1.0)
        assert result.p_value < 0.05

    def test_mantel_test_random_matrices(self):
        from src.experiments.statistical import mantel_test

        np.random.seed(42)
        n = 10
        rdm_a = np.random.rand(n, n)
        rdm_a = (rdm_a + rdm_a.T) / 2
        np.fill_diagonal(rdm_a, 0)

        rdm_b = np.random.rand(n, n)
        rdm_b = (rdm_b + rdm_b.T) / 2
        np.fill_diagonal(rdm_b, 0)

        result = mantel_test(rdm_a, rdm_b, n_permutations=500)

        assert -1 <= result.observed <= 1
        assert 0 <= result.p_value <= 1


# =========================================================================
# Experiment 1: RSA
# =========================================================================

class TestExpRSA:
    """Tests for RSA + Mantel test experiment."""

    def test_compute_rdm_shape(self, synthetic_embeddings):
        from src.experiments.exp_rsa import compute_rdm

        weird, _, _ = synthetic_embeddings
        rdm = compute_rdm(weird)

        assert rdm.shape == (15, 15)
        assert np.allclose(np.diag(rdm), 0)  # Diagonal is zero
        assert np.allclose(rdm, rdm.T)  # Symmetric

    def test_compute_rdm_values(self, synthetic_embeddings):
        from src.experiments.exp_rsa import compute_rdm

        weird, _, _ = synthetic_embeddings
        rdm = compute_rdm(weird)

        # Cosine distances should be in [0, 2]
        assert np.all(rdm >= -1e-10)
        assert np.all(rdm <= 2 + 1e-10)

    def test_run_rsa_output(self, synthetic_embeddings):
        from src.experiments.exp_rsa import run_rsa

        weird, sinic, labels = synthetic_embeddings
        result = run_rsa(weird, sinic, labels, n_permutations=100)

        assert -1 <= result.spearman_r <= 1
        assert 0 <= result.p_value <= 1
        assert result.n_pairs == 15 * 14 // 2  # = 105
        assert result.rdm_weird.shape == (15, 15)
        assert result.rdm_sinic.shape == (15, 15)
        assert len(result.labels) == 15

    def test_rsa_to_dict(self, synthetic_embeddings):
        from src.experiments.exp_rsa import run_rsa

        weird, sinic, labels = synthetic_embeddings
        result = run_rsa(weird, sinic, labels, n_permutations=100)
        d = result.to_dict()

        assert "spearman_r" in d
        assert "p_value" in d
        assert "n_pairs" in d
        assert "significant" in d
        assert "labels" in d
        assert "rdm_weird" in d
        assert "rdm_sinic" in d
        assert len(d["rdm_weird"]) == 15
        assert len(d["rdm_sinic"]) == 15

    def test_rsa_identical_spaces(self):
        """Identical spaces should have r close to 1."""
        from src.experiments.exp_rsa import run_rsa

        np.random.seed(42)
        vectors = np.random.randn(10, 32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        labels = [f"t{i}" for i in range(10)]

        result = run_rsa(vectors, vectors, labels, n_permutations=100)
        assert result.spearman_r > 0.99


# =========================================================================
# Experiment 2: Gromov-Wasserstein
# =========================================================================

class TestExpGW:
    """Tests for Gromov-Wasserstein experiment."""

    def test_gw_distance_output(self, synthetic_embeddings):
        from src.experiments.exp_gw import gromov_wasserstein_distance

        weird, sinic, _ = synthetic_embeddings
        result = gromov_wasserstein_distance(
            weird, sinic, n_permutations=50,
        )

        assert isinstance(result.distance, float)
        assert result.distance >= 0
        assert result.transport_plan.shape == (15, 15)
        assert 0 <= result.p_value <= 1

    def test_gw_transport_plan_valid(self, synthetic_embeddings):
        from src.experiments.exp_gw import gromov_wasserstein_distance

        weird, sinic, _ = synthetic_embeddings
        result = gromov_wasserstein_distance(
            weird, sinic, n_permutations=10,
        )

        # All entries non-negative
        assert np.all(result.transport_plan >= -1e-10)

        # Marginals sum to uniform
        n = 15
        row_sums = result.transport_plan.sum(axis=1)
        col_sums = result.transport_plan.sum(axis=0)
        np.testing.assert_array_almost_equal(row_sums, np.ones(n) / n, decimal=4)
        np.testing.assert_array_almost_equal(col_sums, np.ones(n) / n, decimal=4)

    def test_gw_to_dict(self, synthetic_embeddings):
        from src.experiments.exp_gw import gromov_wasserstein_distance

        weird, sinic, _ = synthetic_embeddings
        result = gromov_wasserstein_distance(
            weird, sinic, n_permutations=10,
        )
        d = result.to_dict()

        assert "distance" in d
        assert "p_value" in d
        assert "significant" in d
        assert "transport_plan" in d
        assert isinstance(d["significant"], bool)


# =========================================================================
# Experiment 3: Axes
# =========================================================================

class TestExpAxes:
    """Tests for axiological axis projection experiment."""

    def test_build_kozlowski_axis(self):
        from src.experiments.exp_axes import build_kozlowski_axis

        np.random.seed(42)
        pairs = [
            (np.random.randn(64), np.random.randn(64))
            for _ in range(5)
        ]

        axis = build_kozlowski_axis(pairs)
        assert axis.shape == (64,)
        assert np.isclose(np.linalg.norm(axis), 1.0)

    def test_project_on_axis(self):
        from src.experiments.exp_axes import build_kozlowski_axis, project_on_axis

        np.random.seed(42)
        d = 64
        pairs = [(np.random.randn(d), np.random.randn(d)) for _ in range(5)]
        axis = build_kozlowski_axis(pairs)

        vectors = np.random.randn(10, d)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        labels = [f"t{i}" for i in range(10)]

        scores = project_on_axis(vectors, labels, axis)

        assert len(scores) == 10
        for score in scores.values():
            assert -1.0 <= score <= 1.0

    def test_run_axes_experiment(self, synthetic_embeddings):
        from src.experiments.exp_axes import run_axes_experiment

        weird, sinic, labels = synthetic_embeddings

        # Mock embedding functions
        np.random.seed(42)
        def mock_embed_weird(texts):
            v = np.random.randn(len(texts), 64)
            return v / np.linalg.norm(v, axis=1, keepdims=True)

        def mock_embed_sinic(texts):
            v = np.random.randn(len(texts), 64)
            return v / np.linalg.norm(v, axis=1, keepdims=True)

        value_axes = {
            "test_axis": {
                "en_pairs": [["a", "b"], ["c", "d"], ["e", "f"]],
                "zh_pairs": [["x", "y"], ["w", "z"], ["u", "v"]],
            }
        }

        result = run_axes_experiment(
            weird, sinic, labels, value_axes,
            mock_embed_weird, mock_embed_sinic,
            n_bootstrap=50,
        )

        assert len(result.axes) == 1
        ax = result.axes[0]
        assert ax.axis_name == "test_axis"
        assert len(ax.weird_scores) == 15
        assert len(ax.sinic_scores) == 15
        assert -1 <= ax.spearman_r <= 1
        assert ax.bootstrap_ci is not None
        assert ax.bootstrap_ci.ci_lower <= ax.bootstrap_ci.ci_upper

    def test_axes_to_dict(self, synthetic_embeddings):
        from src.experiments.exp_axes import run_axes_experiment

        weird, sinic, labels = synthetic_embeddings
        np.random.seed(42)

        def mock_embed(texts):
            v = np.random.randn(len(texts), 64)
            return v / np.linalg.norm(v, axis=1, keepdims=True)

        result = run_axes_experiment(
            weird, sinic, labels,
            {"ax1": {"en_pairs": [["a", "b"]], "zh_pairs": [["c", "d"]]}},
            mock_embed, mock_embed, n_bootstrap=20,
        )

        d = result.to_dict()
        assert "n_terms" in d
        assert "n_axes" in d
        assert "axes" in d


# =========================================================================
# Experiment 4: Clustering
# =========================================================================

class TestExpClustering:
    """Tests for hierarchical clustering experiment."""

    def test_hierarchical_clustering(self, synthetic_embeddings):
        from src.experiments.exp_clustering import hierarchical_clustering

        weird, _, labels = synthetic_embeddings
        result = hierarchical_clustering(weird, labels)

        assert result.linkage_matrix.shape == (14, 4)  # n-1 x 4
        assert result.labels == labels

    def test_run_clustering_experiment(self, synthetic_embeddings):
        from src.experiments.exp_clustering import run_clustering_experiment

        weird, sinic, labels = synthetic_embeddings
        result = run_clustering_experiment(
            weird, sinic, labels,
            k_values=[3, 5],
            n_permutations=50,
        )

        assert len(result.fm_results) == 2
        for fm in result.fm_results:
            assert 0 <= fm.fm_index <= 1
            assert 0 <= fm.p_value <= 1
            assert fm.k in (3, 5)

    def test_clustering_identical(self):
        """Same embeddings should give FM = 1."""
        from src.experiments.exp_clustering import run_clustering_experiment

        np.random.seed(42)
        vectors = np.random.randn(15, 64)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        labels = [f"t{i}" for i in range(15)]

        result = run_clustering_experiment(
            vectors, vectors, labels,
            k_values=[3], n_permutations=50,
        )

        assert np.isclose(result.fm_results[0].fm_index, 1.0)

    def test_clustering_to_dict(self, synthetic_embeddings):
        from src.experiments.exp_clustering import run_clustering_experiment

        weird, sinic, labels = synthetic_embeddings
        result = run_clustering_experiment(
            weird, sinic, labels,
            k_values=[3], n_permutations=20,
        )
        d = result.to_dict()

        assert "n_terms" in d
        assert "fm_results" in d
        assert "labels" in d
        assert "linkage_weird" in d
        assert "linkage_sinic" in d
        assert len(d["fm_results"]) == 1

    def test_clustering_k_filter(self, synthetic_embeddings):
        """k values >= n should be filtered out."""
        from src.experiments.exp_clustering import run_clustering_experiment

        weird, sinic, labels = synthetic_embeddings
        result = run_clustering_experiment(
            weird, sinic, labels,
            k_values=[3, 5, 100],  # 100 > n=15
            n_permutations=20,
        )

        # k=100 should be filtered
        k_tested = [fm.k for fm in result.fm_results]
        assert 100 not in k_tested
        assert 3 in k_tested
        assert 5 in k_tested


# =========================================================================
# Experiment 5: NDA
# =========================================================================

class TestExpNDA:
    """Tests for Neighborhood Divergence Analysis."""

    def test_nda_part_a(self, large_corpus):
        from src.experiments.exp_nda import run_nda_part_a

        core_w, core_s, core_labels, all_w, all_s, all_labels = large_corpus

        result = run_nda_part_a(
            core_w, core_s, core_labels,
            all_w, all_s, all_labels,
            k=5, n_permutations=50,
        )

        assert len(result.term_results) == len(core_labels)
        assert 0 <= result.mean_jaccard <= 1
        assert 0 <= result.p_value <= 1
        assert result.k == 5

        for tr in result.term_results:
            assert 0 <= tr.jaccard <= 1
            assert len(tr.weird_neighbors) <= 5
            assert len(tr.sinic_neighbors) <= 5

    def test_nda_part_a_identical_spaces(self, large_corpus):
        """Identical spaces should have high Jaccard."""
        from src.experiments.exp_nda import run_nda_part_a

        core_w, _, core_labels, all_w, _, all_labels = large_corpus

        result = run_nda_part_a(
            core_w, core_w, core_labels,
            all_w, all_w, all_labels,
            k=5, n_permutations=20,
        )

        # Identical spaces should have Jaccard = 1
        assert result.mean_jaccard > 0.99

    def test_nda_part_b(self, large_corpus):
        from src.experiments.exp_nda import run_nda_part_b

        _, _, _, all_w, all_s, all_labels = large_corpus

        np.random.seed(42)
        def mock_embed_w(texts):
            v = np.random.randn(len(texts), 64)
            return v / np.linalg.norm(v, axis=1, keepdims=True)

        def mock_embed_s(texts):
            v = np.random.randn(len(texts), 64)
            return v / np.linalg.norm(v, axis=1, keepdims=True)

        decomps = [
            {
                "id": "test_decomp",
                "operation": "subtract",
                "en_a": "Law", "en_b": "State",
                "zh_a": "法律", "zh_b": "国家",
                "hypothesis_weird": "Justice",
                "hypothesis_sinic": "Void",
                "jurisprudential_question": "Test question?",
            }
        ]

        result = run_nda_part_b(
            decomps, mock_embed_w, mock_embed_s,
            all_w, all_s, all_labels, k=5,
        )

        assert len(result.decompositions) == 1
        d = result.decompositions[0]
        assert d.decomposition_id == "test_decomp"
        assert 0 <= d.jaccard <= 1
        assert len(d.weird_neighbors) == 5
        assert len(d.sinic_neighbors) == 5

    def test_nda_part_a_to_dict(self, large_corpus):
        from src.experiments.exp_nda import run_nda_part_a

        core_w, core_s, core_labels, all_w, all_s, all_labels = large_corpus
        result = run_nda_part_a(
            core_w, core_s, core_labels,
            all_w, all_s, all_labels,
            k=5, n_permutations=20,
        )
        d = result.to_dict()

        assert "mean_jaccard" in d
        assert "p_value" in d
        assert "false_friends" in d
        assert "per_term" in d
        # Verify per_term includes full neighbor lists
        for item in d["per_term"]:
            assert "weird_neighbors" in item
            assert "sinic_neighbors" in item
            assert "shared_neighbors" in item

    def test_nda_part_b_to_dict(self, large_corpus):
        from src.experiments.exp_nda import run_nda_part_b

        _, _, _, all_w, all_s, all_labels = large_corpus
        np.random.seed(42)
        def mock_embed(texts):
            v = np.random.randn(len(texts), 64)
            return v / np.linalg.norm(v, axis=1, keepdims=True)

        decomps = [{
            "id": "test", "operation": "subtract",
            "en_a": "A", "en_b": "B", "zh_a": "C", "zh_b": "D",
            "hypothesis_weird": "", "hypothesis_sinic": "",
            "jurisprudential_question": "?",
        }]

        result = run_nda_part_b(decomps, mock_embed, mock_embed, all_w, all_s, all_labels, k=3)
        d = result.to_dict()

        assert "n_decompositions" in d
        assert "decompositions" in d
        assert "mean_jaccard" in d


# =========================================================================
# Dataset builder
# =========================================================================

class TestBuildDataset:
    """Tests for dataset construction."""

    def test_build_dataset_structure(self):
        from src.data.build_dataset import build_dataset

        dataset = build_dataset()

        assert "core_terms" in dataset
        assert "background_terms" in dataset
        assert "control_terms" in dataset
        assert "value_axes" in dataset
        assert "normative_decompositions" in dataset
        assert "metadata" in dataset

    def test_core_terms_count(self):
        from src.data.build_dataset import build_core_terms

        terms = build_core_terms()
        assert len(terms) >= 380

    def test_background_terms_count(self):
        from src.data.build_dataset import build_background_terms

        terms = build_background_terms()
        assert len(terms) >= 300

    def test_control_terms_count(self):
        from src.data.build_dataset import build_control_terms

        terms = build_control_terms()
        assert len(terms) >= 45

    def test_value_axes_structure(self):
        from src.data.build_dataset import build_value_axes

        axes = build_value_axes()
        assert len(axes) == 3
        assert "individual_collective" in axes
        assert "rights_duties" in axes
        assert "public_private" in axes

        for axis_name, axis_def in axes.items():
            assert "en_pairs" in axis_def
            assert "zh_pairs" in axis_def
            assert len(axis_def["en_pairs"]) == len(axis_def["zh_pairs"])
            assert len(axis_def["en_pairs"]) >= 5

    def test_normative_decompositions(self):
        from src.data.build_dataset import build_normative_decompositions

        decomps = build_normative_decompositions()
        assert len(decomps) == 5

        for d in decomps:
            assert "id" in d
            assert "en_a" in d
            assert "en_b" in d
            assert "zh_a" in d
            assert "zh_b" in d
            assert "jurisprudential_question" in d

    def test_no_duplicate_ids(self):
        from src.data.build_dataset import build_core_terms, build_background_terms, build_control_terms

        core = build_core_terms()
        bg = build_background_terms()
        ctrl = build_control_terms()

        core_ids = [t["id"] for t in core]
        bg_ids = [t["id"] for t in bg]
        ctrl_ids = [t["id"] for t in ctrl]

        assert len(core_ids) == len(set(core_ids)), "Duplicate core IDs"
        assert len(bg_ids) == len(set(bg_ids)), "Duplicate background IDs"
        assert len(ctrl_ids) == len(set(ctrl_ids)), "Duplicate control IDs"

        overlap_cb = set(core_ids) & set(bg_ids)
        assert len(overlap_cb) == 0, f"Core/Background overlap: {overlap_cb}"
        overlap_cc = set(core_ids) & set(ctrl_ids)
        assert len(overlap_cc) == 0, f"Core/Control overlap: {overlap_cc}"
        overlap_bc = set(bg_ids) & set(ctrl_ids)
        assert len(overlap_bc) == 0, f"Background/Control overlap: {overlap_bc}"

    def test_total_terms(self):
        from src.data.build_dataset import build_dataset

        dataset = build_dataset()
        n = dataset["metadata"]["n_total_terms"]
        assert n >= 700

    def test_no_expected_divergence(self):
        from src.data.build_dataset import build_core_terms

        terms = build_core_terms()
        for t in terms:
            assert "expected_divergence" not in t, f"Term {t['id']} has expected_divergence"

    def test_all_terms_have_source(self):
        from src.data.build_dataset import build_core_terms, build_background_terms, build_control_terms

        for t in build_core_terms() + build_background_terms() + build_control_terms():
            assert "source" in t, f"Term {t['id']} missing source field"
            assert t["source"] in ("HK DOJ", "CC-CEDICT"), f"Unknown source: {t['source']}"
