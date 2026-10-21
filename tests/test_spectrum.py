import numpy as np
import pytest
from conftest import gaussian_line
from scipy.integrate import trapezoid

from bobspec import spectrum as sp
from bobspec.constants import C_KMS
from bobspec.io import Spec


def _moments(spec: Spec, centre: float, half: float) -> tuple[float, float, float]:
    """Equivalent width, centroid and velocity variance of an absorption line."""
    win = np.abs(spec.wave - centre) < half
    w, depth = spec.wave[win], 1.0 - spec.flux[win]
    ew = trapezoid(depth, w)
    mean = trapezoid(depth * w, w) / ew
    var = trapezoid(depth * (w - mean) ** 2, w) / ew
    return ew, mean, var * (C_KMS / mean) ** 2


# ----------------------------------------------------------------------------- grids
def test_uniform_grid_includes_stop() -> None:
    np.testing.assert_allclose(sp.uniform_grid(0.0, 1.0, 0.1), np.linspace(0, 1, 11))


@pytest.mark.parametrize(("start", "stop", "step"), [(0, 1, 0), (0, 1, -1), (1, 0, 0.1)])
def test_uniform_grid_invalid(start: float, stop: float, step: float) -> None:
    with pytest.raises(ValueError):
        sp.uniform_grid(start, stop, step)


def test_log_grid_constant_velocity_step() -> None:
    grid = sp.log_grid(5000.0, 6000.0, 2.0)
    np.testing.assert_allclose(np.diff(np.log(grid)) * C_KMS, 2.0)


def test_interpolate_outside_is_nan() -> None:
    spec = Spec([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    out = sp.interpolate(spec, np.array([0.5, 1.5, 3.5]))
    assert np.isnan(out[0]) and np.isnan(out[2]) and out[1] == pytest.approx(1.5)


# ----------------------------------------------------------------------------- rebinning
def test_rebin_conserves_integrated_flux() -> None:
    rng = np.random.default_rng(0)
    wave = np.arange(5000.0, 5100.0, 0.01)
    spec = Spec(wave, 1.0 + rng.normal(0, 0.1, wave.size))
    coarse = sp.rebin(spec, np.arange(5001.0, 5099.0, 0.25))
    inside = (wave >= 5001.0 - 0.125) & (wave < 5099.0 - 0.25 + 0.125)
    assert np.nansum(coarse.flux) * 0.25 == pytest.approx(spec.flux[inside].sum() * 0.01, rel=1e-3)


def test_rebin_reduces_noise_like_binning() -> None:
    rng = np.random.default_rng(1)
    wave = np.arange(5000.0, 5100.0, 0.01)
    spec = Spec(wave, 1.0 + rng.normal(0, 0.1, wave.size))
    coarse = sp.rebin(spec, np.arange(5001.0, 5099.0, 0.16))  # 16 pixels per bin
    assert np.nanstd(coarse.flux) == pytest.approx(0.1 / 4, rel=0.15)


def test_rebin_ignores_nan_and_marks_uncovered() -> None:
    flux = np.ones(100)
    flux[40:60] = np.nan
    spec = Spec(np.arange(100.0), flux)
    out = sp.rebin(spec, np.arange(5.0, 95.0, 10.0))
    assert np.isnan(out.flux[4])  # bin 45 fully inside the gap
    np.testing.assert_allclose(out.flux[np.isfinite(out.flux)], 1.0)


def test_resample_picks_rebin_for_coarser_grid() -> None:
    spec = Spec(np.arange(0.0, 10.0, 0.01), np.tile([0.0, 2.0], 500))
    assert np.allclose(sp.resample(spec, np.arange(1.0, 9.0, 0.5)).flux, 1.0)


# ----------------------------------------------------------------------------- Doppler
def test_rest_frame_inverts_shift() -> None:
    spec = Spec([5000.0, 5001.0], [1.0, 1.0])
    shifted = spec.with_wave(spec.wave * sp.doppler_factor(250.0))
    np.testing.assert_allclose(sp.to_rest_frame(shifted, 250.0).wave, spec.wave)


def test_doppler_rejects_superluminal() -> None:
    with pytest.raises(ValueError):
        sp.doppler_factor(4e5)


def test_exclusion_mask() -> None:
    wave = np.arange(10.0)
    np.testing.assert_array_equal(sp.exclusion_mask(wave, [(2, 3), (8, 7)]), [1, 1, 0, 0, 1, 1, 1, 0, 0, 1])


# ----------------------------------------------------------------------------- broadening
@pytest.mark.parametrize("centre", [4000.0, 8000.0])
def test_gaussian_broaden_gives_constant_resolution(centre: float) -> None:
    wave = np.arange(centre - 20, centre + 20, 0.005)
    spec = Spec(wave, gaussian_line(wave, centre, 0.02, 0.5))
    out = sp.gaussian_broaden(spec, 20_000)
    ew0, _, var0 = _moments(spec, centre, 5)
    ew1, mean1, var1 = _moments(out, centre, 5)
    sigma_r = C_KMS / 20_000 / 2.354820045
    assert ew1 == pytest.approx(ew0, rel=1e-3)
    assert mean1 == pytest.approx(centre, abs=1e-4)
    assert np.sqrt(var1 - var0) == pytest.approx(sigma_r, rel=0.01)


def test_gaussian_broaden_invalid_resolution() -> None:
    with pytest.raises(ValueError):
        sp.gaussian_broaden(Spec([1.0, 2.0], [1.0, 1.0]), 0)


def test_rotation_zero_vsini_is_identity() -> None:
    spec = Spec(np.arange(10.0), np.arange(10.0))
    assert sp.rotational_broaden(spec, 0.0, 0.6) is spec


@pytest.mark.parametrize(("vsini", "eps"), [(-1.0, 0.5), (10.0, 1.5), (10.0, -0.1)])
def test_rotation_invalid_parameters(vsini: float, eps: float) -> None:
    with pytest.raises(ValueError):
        sp.rotational_broaden(Spec([1.0, 2.0], [1.0, 1.0]), vsini, eps)


def test_rotation_kernel_normalised_and_symmetric() -> None:
    kernel = sp.rotation_kernel(30.0, 0.6, 0.5)
    assert kernel.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(kernel, kernel[::-1])


@pytest.mark.parametrize("centre", [4500.0, 9000.0])
def test_rotation_width_is_exact_at_any_wavelength(centre: float) -> None:
    # epsilon = 0: the kernel is a half-ellipse with velocity variance vsini^2 / 4.
    wave = np.arange(centre - 30, centre + 30, 0.005)
    spec = Spec(wave, gaussian_line(wave, centre, 0.02, 0.6))
    out = sp.rotational_broaden(spec, 40.0, 0.0)
    ew0, _, var0 = _moments(spec, centre, 10)
    ew1, _, var1 = _moments(out, centre, 10)
    assert ew1 == pytest.approx(ew0, rel=1e-3)
    assert np.sqrt(var1 - var0) == pytest.approx(20.0, rel=0.01)


# ----------------------------------------------------------------------------- noise
def test_add_noise_reproducible_and_scaled() -> None:
    spec = Spec(np.arange(20_000.0), np.full(20_000, 1000.0))
    a = sp.add_noise(spec, 50.0, np.random.default_rng(7))
    b = sp.add_noise(spec, 50.0, np.random.default_rng(7))
    np.testing.assert_array_equal(a.flux, b.flux)
    assert np.std(a.flux) == pytest.approx(1000.0 / 50.0, rel=0.03)


def test_photon_noise_smaller_in_line_core() -> None:
    flux = np.where(np.arange(40_000) < 20_000, 1.0, 0.25)
    spec = Spec(np.arange(40_000.0), flux)
    noisy = sp.add_noise(spec, 100.0, np.random.default_rng(3), model="photon", reference=1.0)
    assert np.std(noisy.flux[20_000:] - 0.25) == pytest.approx(0.5 / 100.0, rel=0.05)


@pytest.mark.parametrize("snr", [0.0, -5.0])
def test_add_noise_rejects_bad_snr(snr: float) -> None:
    with pytest.raises(ValueError):
        sp.add_noise(Spec([1.0, 2.0], [1.0, 1.0]), snr, np.random.default_rng())
