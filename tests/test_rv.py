import numpy as np
import pytest

from bobspec.io import Spec
from bobspec.rv import cross_correlate, monte_carlo


@pytest.mark.parametrize("rv", [-450.0, -120.0, 0.0, 150.0, 400.0])
def test_recovers_velocity_without_relativistic_bias(line_model, template, obs_wave, rv) -> None:
    result = cross_correlate(Spec(obs_wave, line_model(obs_wave, rv)), template)
    assert result.fit_ok
    assert result.rv == pytest.approx(rv, abs=0.01)


def test_nan_gap_and_tellurics_do_not_bias(line_model, template, obs_wave) -> None:
    flux = line_model(obs_wave, 150.0)
    flux[3000:3400] = np.nan
    result = cross_correlate(Spec(obs_wave, flux), template, exclude_regions=[(5200.0, 5220.0)])
    assert result.rv == pytest.approx(150.0, abs=0.02)


def test_wave_range_restricts_correlation(line_model, template, obs_wave) -> None:
    result = cross_correlate(Spec(obs_wave, line_model(obs_wave, 50.0)), template, wave_range=(5100, 5150))
    assert result.wave_range == pytest.approx((5100.0, 5150.0))
    assert result.rv == pytest.approx(50.0, abs=0.02)


def test_td_error_grows_with_noise(line_model, template, obs_wave) -> None:
    rng = np.random.default_rng(0)
    errors = []
    for noise in (0.005, 0.05):
        flux = line_model(obs_wave, 30.0) + rng.normal(0, noise, obs_wave.size)
        errors.append(cross_correlate(Spec(obs_wave, flux), template).rv_err_td)
    assert 0 < errors[0] < errors[1]


def test_no_overlap_raises(template) -> None:
    far = Spec(np.arange(6000.0, 6100.0, 0.03), np.ones(3334))
    with pytest.raises(ValueError, match="overlap"):
        cross_correlate(far, template)


def test_narrow_velocity_window_raises(line_model, template, obs_wave) -> None:
    with pytest.raises(ValueError, match="narrower"):
        cross_correlate(Spec(obs_wave, line_model(obs_wave)), template, v_range=(0.0, 1.0))


def test_featureless_spectrum_raises(template, obs_wave) -> None:
    with pytest.raises(ValueError, match="no features"):
        cross_correlate(Spec(obs_wave, np.ones(obs_wave.size)), template)


def test_monte_carlo_reproducible_and_scales_with_snr(template, obs_wave) -> None:
    kwargs = {"v_range": (-200.0, 200.0)}
    a = monte_carlo(template, obs_wave, 80.0, 20.0, 30, np.random.default_rng(5), **kwargs)
    b = monte_carlo(template, obs_wave, 80.0, 20.0, 30, np.random.default_rng(5), **kwargs)
    c = monte_carlo(template, obs_wave, 80.0, 80.0, 30, np.random.default_rng(5), **kwargs)
    np.testing.assert_array_equal(a.rvs, b.rvs)
    assert a.n_failed == 0
    assert np.mean(a.rvs) == pytest.approx(80.0, abs=0.05)
    assert a.std / c.std == pytest.approx(4.0, rel=0.35)


def test_monte_carlo_needs_two_realisations(template, obs_wave) -> None:
    with pytest.raises(ValueError):
        monte_carlo(template, obs_wave, 0.0, 30.0, 1, np.random.default_rng())


def test_same_coverage_as_template_is_unbiased(template) -> None:
    # Observed spectrum covering exactly the template range: the template edges
    # (one-sided high-pass filter) must not leak into the correlation.
    shifted = template.with_wave(template.wave * (1 + 141.0 / 299792.458))
    result = cross_correlate(shifted, template)
    assert result.rv == pytest.approx(141.0, abs=0.01)


def test_monte_carlo_matches_empirical_scatter(line_model, template, obs_wave) -> None:
    clean = Spec(obs_wave, line_model(obs_wave, 60.0))
    rng = np.random.default_rng(11)
    empirical = [
        cross_correlate(clean.with_flux(clean.flux + rng.normal(0, 1 / 15, clean.flux.size)), template).rv
        for _ in range(40)
    ]
    mc = monte_carlo(template, obs_wave, 60.0, 15.0, 40, np.random.default_rng(12))
    assert np.mean(empirical) == pytest.approx(60.0, abs=3 * np.std(empirical) / np.sqrt(40))
    assert mc.std == pytest.approx(np.std(empirical, ddof=1), rel=0.35)
