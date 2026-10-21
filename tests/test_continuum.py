import numpy as np
import pytest

from bobspec import continuum as cont
from bobspec.io import Spec


@pytest.fixture
def tilted(line_model):
    """Spectrum = slowly varying continuum x absorption lines + noise."""
    wave = np.arange(5010.0, 5290.0, 0.03)
    true_cont = 1000.0 * (1.0 + 0.3 * (wave - 5150.0) / 140.0 - 0.2 * ((wave - 5150.0) / 140.0) ** 2)
    rng = np.random.default_rng(2)
    flux = true_cont * line_model(wave) * (1.0 + rng.normal(0, 0.005, wave.size))
    return wave, flux, true_cont


@pytest.mark.parametrize("model", [cont.chebyshev_model(3), cont.spline_model(3, 8)])
def test_iterative_fit_recovers_continuum(tilted, model) -> None:
    wave, flux, true_cont = tilted
    fit = cont.iterative_fit(wave, flux, model, sigma_low=2.0, sigma_up=3.0)
    assert fit.converged
    assert np.median(np.abs(fit.continuum / true_cont - 1.0)) < 0.01


def test_mask_excludes_pixels(tilted) -> None:
    wave, flux, _ = tilted
    mask = (wave < 5100) | (wave > 5120)
    fit = cont.iterative_fit(wave, flux, cont.chebyshev_model(3), sigma_low=2, sigma_up=3, mask=mask)
    assert not fit.used[~mask].any()


def test_nan_pixels_are_ignored(tilted) -> None:
    wave, flux, _ = tilted
    flux = flux.copy()
    flux[::50] = np.nan
    fit = cont.iterative_fit(wave, flux, cont.chebyshev_model(3), sigma_low=1.5, sigma_up=3)
    assert np.all(np.isfinite(fit.continuum))


@pytest.mark.parametrize(("low", "up", "it"), [(0, 3, 5), (2, -1, 5), (2, 3, 0)])
def test_iterative_fit_invalid_parameters(tilted, low, up, it) -> None:
    wave, flux, _ = tilted
    with pytest.raises(ValueError):
        cont.iterative_fit(wave, flux, cont.chebyshev_model(2), sigma_low=low, sigma_up=up, max_iter=it)


def test_too_few_points_raises() -> None:
    with pytest.raises(ValueError, match="too few"):
        cont.iterative_fit(np.arange(3.0), np.ones(3), cont.chebyshev_model(5), sigma_low=2, sigma_up=2)


@pytest.mark.parametrize("bad", [0, 6])
def test_spline_degree_validated(bad: int) -> None:
    with pytest.raises(ValueError):
        cont.spline_model(bad)


def test_anchor_curve_respects_degree_and_clamps() -> None:
    curve = cont.anchor_curve([1.0, 2.0, 3.0], [1.0, 3.0, 1.0], k=1)
    assert curve(np.array([1.5]))[0] == pytest.approx(2.0)  # linear, not cubic
    assert curve(np.array([10.0]))[0] == pytest.approx(1.0)  # held constant outside


def test_anchor_curve_duplicates_are_averaged() -> None:
    curve = cont.anchor_curve([1.0, 1.0, 2.0], [0.0, 2.0, 1.0], k=1)
    assert curve(np.array([1.0]))[0] == pytest.approx(1.0)


def test_anchor_curve_needs_k_plus_one_points() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        cont.anchor_curve([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], k=3)


def test_smooth_continuum_upper_envelope(tilted) -> None:
    wave, flux, true_cont = tilted
    spec = Spec(wave, flux)
    estimate = cont.smooth_continuum(spec, 5.0, percentile=95)
    assert np.median(np.abs(estimate / true_cont - 1.0)) < 0.02


@pytest.mark.parametrize(("window", "iters"), [(0.0, 2), (5.0, 0)])
def test_smooth_continuum_invalid(tilted, window, iters) -> None:
    wave, flux, _ = tilted
    with pytest.raises(ValueError):
        cont.smooth_continuum(Spec(wave, flux), window, max_iter=iters)


def test_normalize_handles_nonpositive_continuum() -> None:
    spec = Spec([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
    out = cont.normalize(spec, np.array([1.0, 0.0, -1.0]), "test")
    assert out.flux[0] == 2.0 and np.isnan(out.flux[1:]).all()
