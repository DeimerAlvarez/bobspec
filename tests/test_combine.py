import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import bobspec.cli  # noqa: F401  (registers the commands)
from bobspec.combine import combine_spectra, grow_mask, local_robust_sigma, running_nanmedian
from bobspec.io import Spec, read_spectrum, write_spectrum
from bobspec.registry import execute, resolve

sys.path.insert(0, str(Path(__file__).parent / "reference"))
import combine_original

WAVE = np.arange(6000.0, 6050.0, 0.025)
PARANAL = {"HIERARCH ESO TEL GEOLON": -70.4045, "HIERARCH ESO TEL GEOLAT": -24.6268, "HIERARCH ESO TEL GEOELEV": 2648.0}


def truth() -> np.ndarray:
    flux = 1000.0 * np.ones(WAVE.size)
    for centre in np.linspace(6003, 6047, 12):
        flux *= 1 - 0.6 * np.exp(-0.5 * ((WAVE - centre) / 0.08) ** 2)
    return flux


def exposures(n: int = 5, seed: int = 0, cosmics: int = 4, noise: float = 10.0):
    """n exposures with different flux levels, Gaussian noise and injected cosmic rays."""
    rng = np.random.default_rng(seed)
    clean = truth()
    spectra, hits = [], []
    for i in range(n):
        level = 0.8 + 0.1 * i
        flux = level * clean + rng.normal(0, noise, WAVE.size)
        pix = rng.choice(np.arange(60, WAVE.size - 60), cosmics, replace=False)
        flux[pix] += rng.uniform(300, 3000, cosmics)
        hits.append(pix)
        header = fits.Header({"MJD-OBS": 60000.0 + 0.03 * i, "EXPTIME": 2400.0})
        spectra.append(Spec(WAVE, flux, header, name=f"exp{i + 1}"))
    return spectra, hits


# ----------------------------------------------------------------------------- building blocks
def test_running_nanmedian_matches_loop() -> None:
    values = np.random.default_rng(1).normal(size=300)
    values[[5, 6, 150]] = np.nan
    expected = combine_original.running_median(values, 11)
    np.testing.assert_allclose(running_nanmedian(values, 11), expected)


@pytest.mark.parametrize("window", [0, 4, -3])
def test_running_nanmedian_invalid_window(window: int) -> None:
    with pytest.raises(ValueError):
        running_nanmedian(np.ones(10), window)


def test_local_sigma_estimates_noise() -> None:
    residual = np.random.default_rng(2).normal(0, 3.0, 5000)
    assert np.median(local_robust_sigma(residual, 51)) == pytest.approx(3.0, rel=0.05)


def test_local_sigma_constant_input_raises() -> None:
    with pytest.raises(ValueError):
        local_robust_sigma(np.zeros(100), 11)


def test_grow_mask() -> None:
    mask = np.zeros(9, bool)
    mask[4] = True
    np.testing.assert_array_equal(grow_mask(mask, 2), [0, 0, 1, 1, 1, 1, 1, 0, 0])


# ----------------------------------------------------------------------------- combination
def test_same_result_as_original_code(tmp_path: Path) -> None:
    spectra, _ = exposures()
    files = []
    for spec in spectra:  # original reads FITS images with a linear WCS
        path = tmp_path / f"{spec.name}.fits"
        fits.PrimaryHDU(spec.flux, fits.Header({"CRVAL1": WAVE[0], "CDELT1": 0.025, "CRPIX1": 1.0})).writeto(path)
        files.append(str(path))
    _, expected, expected_err, expected_n, expected_masks = combine_original.combine_spectra(
        files, str(tmp_path / "orig.fits"), verbose=False
    )
    result = combine_spectra([read_spectrum(f) for f in files], reference="median")
    np.testing.assert_allclose(result.spectrum.flux, expected, rtol=1e-12)
    np.testing.assert_allclose(result.error, expected_err, rtol=1e-12)
    np.testing.assert_array_equal(result.n_used, expected_n)
    np.testing.assert_array_equal(result.rejected, expected_masks)


def test_cosmics_rejected_lines_kept() -> None:
    spectra, hits = exposures()
    result = combine_spectra(spectra)
    for i, pix in enumerate(hits):
        assert result.rejected[i, pix].all()
    line_core = np.abs(WAVE - 6003.0) < 0.05
    assert not result.rejected[:, line_core].any()  # absorption lines are never "cosmics"
    np.testing.assert_allclose(result.scale / result.scale[0], [1.0, 1.125, 1.25, 1.375, 1.5], rtol=0.01)
    residual = result.spectrum.flux - result.scale[2] ** 0 * np.median(result.scale) * truth()
    assert np.std(residual) < 10.0  # noise ~10/sqrt(5) after scaling, no cosmic spikes


def test_reference_spectrum_with_zero_residuals() -> None:
    # Without rescaling, with an odd number of inputs at increasing flux levels the
    # middle one is the median everywhere (zero residuals): this used to crash.
    spectra, _ = exposures(n=5, cosmics=0)
    result = combine_spectra(spectra, rescale=False)
    assert np.isfinite(result.noise).all()


def test_snr_gain_and_error() -> None:
    rng = np.random.default_rng(8)
    clean = truth()
    spectra = [Spec(WAVE, clean + rng.normal(0, 20.0, WAVE.size)) for _ in range(9)]
    result = combine_spectra(spectra)
    single = np.std(spectra[0].flux - clean)
    combined = np.std(result.spectrum.flux - clean)
    assert single / combined == pytest.approx(3.0, rel=0.15)  # sqrt(9)
    assert np.nanmedian(result.error) == pytest.approx(20.0 / 3.0, rel=0.15)


def test_methods() -> None:
    spectra, _ = exposures()
    mean = combine_spectra(spectra).spectrum.flux
    total = combine_spectra(spectra, method="sum").spectrum.flux
    median = combine_spectra(spectra, method="median").spectrum.flux
    np.testing.assert_allclose(total, 5 * mean)
    assert np.nanmedian(np.abs(median / mean - 1)) < 0.01


def test_weighted_mean_helps_with_unequal_noise() -> None:
    rng = np.random.default_rng(5)
    clean = truth()
    spectra = [Spec(WAVE, clean + rng.normal(0, s, WAVE.size)) for s in (5.0, 5.0, 5.0, 60.0)]
    plain = combine_spectra(spectra, sigma=None, rescale=False).spectrum.flux
    weighted = combine_spectra(spectra, sigma=None, rescale=False, weighted=True).spectrum.flux
    assert np.std(weighted - clean) < 0.7 * np.std(plain - clean)


def test_input_validation() -> None:
    spectra, _ = exposures(n=2)
    with pytest.raises(ValueError, match="at least 3"):
        combine_spectra(spectra)
    assert combine_spectra(spectra, sigma=None).spectrum.flux.size == WAVE.size
    with pytest.raises(ValueError, match="at least 2"):
        combine_spectra(spectra[:1], sigma=None)
    with pytest.raises(ValueError, match="odd"):
        combine_spectra(exposures()[0], window=50)


def test_misaligned_grid_is_interpolated() -> None:
    spectra, _ = exposures(n=3)
    shifted = spectra[2].with_wave(spectra[2].wave + 0.3 * 0.025)
    result = combine_spectra([*spectra[:2], shifted])
    assert result.resampled == (False, False, True)


def test_header_keywords() -> None:
    spectra, _ = exposures()
    header = combine_spectra(spectra).spectrum.header
    assert header["NCOMBINE"] == 5 and header["EXPTOT"] == 12_000.0
    scale = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    mids = 60000.0 + 0.03 * np.arange(5) + 1200.0 / 86400.0
    assert header["MJD-MID"] == pytest.approx(np.average(mids, weights=scale), abs=1e-4)


# ----------------------------------------------------------------------------- command
def _write_star(tmp_path: Path, star: str, seed: int, eso: bool = False) -> None:
    spectra, _ = exposures(seed=seed)
    for i, spec in enumerate(spectra, start=1):
        header = spec.header.copy()
        if eso:
            header.update({"RA": 283.76, "DEC": -30.48, **PARANAL})
        write_spectrum(tmp_path / f"M54_h15_{i}_{star}_SH.fits", Spec(spec.wave, spec.flux, header))


def test_combine_command_with_ids_and_wildcards(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for seed, star in enumerate(("001", "002")):
        _write_star(tmp_path, star, seed)
    Path("lista.id").write_text("# ids\n001\n002\n999\n")
    rows = execute(resolve("combine"), ["m54_hr15_{id}.fits", "M54_h15_*_{id}_SH.fits", "--ids", "lista.id"])
    assert [r["n_input"] for r in rows] == [5, 5]  # star 999 has no files: reported, skipped
    assert Path("m54_hr15_001.fits").exists() and Path("m54_hr15_001_cosmics.txt").exists()
    assert Path("lista_combine.txt").read_text().count("\n") == 3
    assert rows[0]["der_snr_combined"] > 1.8 * rows[0]["der_snr_input"]


def test_combine_barycentric_and_helio(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_star(tmp_path, "007", 3, eso=True)
    execute(resolve("combine"), ["out.fits", "M54_h15_*_007_SH.fits", "--barycentric", "--save-error"])
    combined = read_spectrum("out.fits")
    assert combined.header["BARYFRM"] is True
    assert Path("out_error.fits").exists()
    row = execute(resolve("helio_eso"), ["out.fits"])
    assert row["mjd_mid"] == pytest.approx(combined.header["MJD-MID"])


def test_combine_input_list_and_batch_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_star(tmp_path, "001", 0)
    _write_star(tmp_path, "002", 1)
    Path("exp001.txt").write_text("# exposures\n" + "\n".join(f"M54_h15_{i}_001_SH.fits" for i in range(1, 4)) + "\n")
    row = execute(resolve("combine"), ["a.fits", "@exp001.txt"])  # @ in the inputs: list of exposures
    assert row["n_input"] == 3
    Path("stars.txt").write_text('s1.fits "M54_h15_*_001_SH.fits"\ns2.fits "M54_h15_*_002_SH.fits" --method median\n')
    rows = execute(resolve("combine"), ["@stars.txt"])  # @ first: one combination per line
    assert [r["output"] for r in rows] == ["s1.fits", "s2.fits"]
    assert read_spectrum("s2.fits").header["COMBMETH"] == "median"


def test_combine_requires_odd_window_and_id_placeholder(tmp_path: Path) -> None:
    from bobspec.registry import UsageError

    with pytest.raises(UsageError):
        execute(resolve("combine"), ["o.fits", "a.fits", "b.fits", "--window", "50"])
    with pytest.raises(UsageError):
        execute(resolve("combine"), ["o.fits", "a_{id}.fits", "--ids", str(tmp_path / "x")])
