"""Construction invariants of the three matched arms."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from gatek.metrics import euclidean_distances, relative_contrast
from gatek.synth import ARMS, build_arm, label_letters, make_latent

LATENT = make_latent(
    seed=0, n_demo=60, n_query=24, cluster_offset=2.0, cluster_sigma=0.45, margin=0.75
)


def arms_at(dim: int) -> dict:
    return {
        a: build_arm(a, dim, LATENT, nuisance_scale=1.0, dense_column_scale="unit_std")
        for a in ARMS
    }


def test_label_rule_is_xor_on_the_latent_coordinates() -> None:
    expected = (LATENT.demo_z[:, 0] * LATENT.demo_z[:, 1] > 0).astype(int)
    assert np.array_equal(LATENT.demo_y, expected)


def test_class_balance_is_exact() -> None:
    assert (LATENT.demo_y == 1).sum() == (LATENT.demo_y == 0).sum()
    assert (LATENT.query_y == 1).sum() == (LATENT.query_y == 0).sum()


def test_arms_share_latent_rows_labels_names_and_shapes() -> None:
    arms = arms_at(32)
    ref = arms["sparse_zero_pad"]
    for arm in arms.values():
        assert np.array_equal(arm.latent.demo_z, ref.latent.demo_z)
        assert np.array_equal(arm.demo_y, ref.demo_y)
        assert arm.feature_names == ref.feature_names
        assert arm.demo_x.shape == ref.demo_x.shape
        assert arm.query_x.shape == ref.query_x.shape


def test_sparse_arm_pads_with_zeros_and_keeps_the_latent_coordinates() -> None:
    arm = arms_at(16)["sparse_zero_pad"]
    assert np.array_equal(arm.demo_x[:, :2], LATENT.demo_z)
    assert np.all(arm.demo_x[:, 2:] == 0.0)


def test_dense_projection_adds_no_independent_information() -> None:
    """A flexible classical learner stays strong on the dense arm."""
    arm = arms_at(64)["dense_redundant_projection"]
    model = RandomForestClassifier(n_estimators=200, random_state=0).fit(arm.demo_x, arm.demo_y)
    assert (model.predict(arm.query_x) == arm.query_y).mean() >= 0.85


def test_independent_nuisance_creates_a_raw_distance_contrast() -> None:
    arms = arms_at(64)
    sparse, nuisance = arms["sparse_zero_pad"], arms["independent_nuisance"]
    rc_sparse = np.mean(
        [relative_contrast(euclidean_distances(q, sparse.demo_x)) for q in sparse.query_x]
    )
    rc_nuisance = np.mean(
        [relative_contrast(euclidean_distances(q, nuisance.demo_x)) for q in nuisance.query_x]
    )
    assert rc_sparse / rc_nuisance >= 1.5


def test_latent_space_concentration_is_identical_across_arms() -> None:
    arms = arms_at(64)
    values = {
        name: float(
            np.mean(
                [
                    relative_contrast(euclidean_distances(q, a.latent.demo_z))
                    for q in a.latent.query_z
                ]
            )
        )
        for name, a in arms.items()
    }
    assert max(values.values()) - min(values.values()) < 1e-12


def test_generation_is_deterministic() -> None:
    again = make_latent(
        seed=0, n_demo=60, n_query=24, cluster_offset=2.0, cluster_sigma=0.45, margin=0.75
    )
    assert np.array_equal(again.demo_z, LATENT.demo_z)
    a = build_arm("independent_nuisance", 32, LATENT, 1.0, "unit_std")
    b = build_arm("independent_nuisance", 32, again, 1.0, "unit_std")
    assert np.array_equal(a.demo_x, b.demo_x)


def test_label_letters_map_zero_to_a() -> None:
    assert label_letters(np.array([0, 1, 1])) == ["A", "B", "B"]


def test_dim_below_two_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_arm("sparse_zero_pad", 1, LATENT, 1.0, "unit_std")
