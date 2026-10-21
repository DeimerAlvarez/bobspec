import numpy as np
import pytest
from conftest import gaussian_line

from bobspec.io import Spec
from bobspec.lines import measure_line
from bobspec.snr import der_snr, region_snr, snr_profile


def test_der_snr_white_noise() -> None:
    flux = 1.0 + np.random.default_rng(0).normal(0, 0.01, 20_000)
    assert der_snr(flux) == pytest.approx(100.0, rel=0.03)


def test_der_snr_insensitive_to_well_sampled_lines() -> None:
    wave = np.arange(5000.0, 5100.0, 0.03)
    flux = np.ones(wave.size)
    for centre in np.linspace(5005, 5095, 20):
        flux *= gaussian_line(wave, centre, 0.15, 0.5)
    flux += np.random.default_rng(1).normal(0, 0.01, wave.size)
    assert der_snr(flux) == pytest.approx(100.0, rel=0.15)
    assert region_snr(flux) < 20  # mean/std is ruined by the lines


def test_der_snr_needs_points() -> None:
    with pytest.raises(ValueError):
        der_snr(np.array([1.0, np.nan, 2.0]))


def test_region_snr() -> None:
    flux = 10.0 + np.random.default_rng(2).normal(0, 0.5, 5000)
    assert region_snr(flux) == pytest.approx(20.0, rel=0.05)


def test_snr_profile_skips_masked_gaps() -> None:
    flux = 1.0 + np.random.default_rng(3).normal(0, 0.02, 1000)
    mask = np.ones(1000, bool)
    mask[150:160] = False
    centres, values = snr_profile(np.arange(1000.0), flux, mask, chunk=100)
    assert centres.size == 9  # 150 good pixels -> 1 chunk (50 left over), 840 -> 8 chunks
    assert np.all((centres < 150) | (centres > 160))
    assert np.median(values) == pytest.approx(50.0, rel=0.15)


def test_measure_line_matches_analytic_ew() -> None:
    wave = np.arange(6000.0, 6004.0, 0.01)
    sigma, depth = 0.08, 0.6
    spec = Spec(wave, 500.0 * gaussian_line(wave, 6002.0, sigma, depth))
    m = measure_line(spec, 6000.5, 6003.5, resolution=30_000)
    expected = depth * sigma * np.sqrt(2 * np.pi) * 1000
    assert m.fit_ok
    assert m.center == pytest.approx(6002.0, abs=1e-4)
    assert m.fwhm == pytest.approx(2.3548 * sigma, rel=1e-3)
    assert m.ew_gauss == pytest.approx(expected, rel=1e-3)
    assert m.ew_int == pytest.approx(expected, rel=1e-2)


def test_measure_line_too_narrow_window() -> None:
    spec = Spec(np.arange(6000.0, 6004.0, 0.01), np.ones(400))
    with pytest.raises(ValueError):
        measure_line(spec, 6001.0, 6001.03)
