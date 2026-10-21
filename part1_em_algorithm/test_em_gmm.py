"""Tests for em_gmm. Run from the part1_em_algorithm folder: python -m pytest -q"""

from pathlib import Path

import numpy as np
import pytest

from em_gmm import (
    VARIANCE_FLOOR,
    GMMParams,
    decision_boundary,
    e_step,
    fit_em,
    load_heights,
    mean_split_estimate,
    posterior,
)

DATA = Path(__file__).parent / "data" / "GaltonFamilies.csv"


@pytest.fixture
def well_separated():
    rng = np.random.default_rng(42)
    x = np.concatenate([rng.normal(60, 2, 3000), rng.normal(75, 3, 1000)])
    return x


def test_recovers_known_parameters_on_synthetic_data(well_separated):
    final = fit_em(well_separated, mean_split_estimate(well_separated))[-1].params
    np.testing.assert_allclose(final.mu, [60, 75], atol=0.2)
    np.testing.assert_allclose(np.sqrt(final.var), [2, 3], atol=0.2)
    np.testing.assert_allclose(final.pi, [0.75, 0.25], atol=0.02)


def test_log_likelihood_never_decreases(well_separated):
    lls = [s.log_likelihood for s in fit_em(well_separated, mean_split_estimate(well_separated))]
    assert all(b >= a - 1e-9 for a, b in zip(lls, lls[1:]))


def test_responsibilities_are_valid_probabilities(well_separated):
    r, _ = e_step(well_separated, mean_split_estimate(well_separated))
    assert np.all((r >= 0) & (r <= 1))
    np.testing.assert_allclose(r.sum(axis=1), 1.0)


def test_posterior_sums_to_one_and_cm_matches_inches():
    params = GMMParams(mu=np.array([64.0, 69.5]), var=np.array([5.3, 6.0]), pi=np.array([0.43, 0.57]))
    p_in = posterior(70, params, unit="in")
    np.testing.assert_allclose(p_in.sum(), 1.0)
    np.testing.assert_allclose(posterior(70 * 2.54, params, unit="cm"), p_in)


@pytest.mark.parametrize("height,unit", [(float("nan"), "in"), (-5, "in"), (0, "in"), (70, "feet")])
def test_posterior_rejects_bad_input(height, unit):
    params = GMMParams(mu=np.array([64.0, 69.5]), var=np.array([5.3, 6.0]), pi=np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        posterior(height, params, unit=unit)


def test_mean_split_rejects_constant_data():
    with pytest.raises(ValueError):
        mean_split_estimate(np.full(10, 65.0))


def test_variance_floor_prevents_collapse():
    # 50 identical heights plus a spread cluster: the tight component tends to var -> 0.
    x = np.concatenate([np.full(50, 60.0), np.linspace(70, 80, 50)])
    final = fit_em(x, mean_split_estimate(x))[-1].params
    assert np.all(final.var >= VARIANCE_FLOOR)
    assert np.all(np.isfinite(final.mu))


def test_decision_boundary_is_where_posteriors_are_equal():
    params = GMMParams(mu=np.array([64.0, 69.5]), var=np.array([5.3, 6.0]), pi=np.array([0.43, 0.57]))
    (boundary,) = decision_boundary(params, low=40, high=100)
    np.testing.assert_allclose(posterior(boundary, params), [0.5, 0.5], atol=1e-6)


@pytest.mark.skipif(not DATA.exists(), reason="Galton CSV not present")
def test_parents_are_deduplicated_by_family():
    heights, labels = load_heights(DATA, parent="father")
    assert (labels == 0).sum() == 934  # one row per child
    assert (labels == 1).sum() == 205  # one father per family, not one per child
    assert heights.size == 1139


def test_load_heights_rejects_unknown_parent():
    with pytest.raises(ValueError):
        load_heights(DATA, parent="uncle")
