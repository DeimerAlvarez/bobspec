"""ew: DAOSPEC/4DAO-like equivalent widths, tested on spectra with known lines."""

from pathlib import Path

import numpy as np
import pytest

import bobspec.cli  # noqa: F401  (registers the commands)
from bobspec import ew
from bobspec.commands import nrv_name
from bobspec.constants import C_KMS
from bobspec.io import Spec, read_spectrum, write_spectrum
from bobspec.registry import UsageError, execute, resolve

R, RHO, RV = 20000.0, 0.10, 120.0
WAVE = np.arange(5600.0, 5800.0, 0.05)
RNG = np.random.default_rng(7)
CENTRES = np.sort(RNG.uniform(5606, 5794, 40))
EWS = RNG.uniform(15, 150, 40)  # mA
CODES = np.where(np.arange(40) % 3 == 0, 26.01, 26.00)


def amplitudes(sigma: np.ndarray) -> np.ndarray:
    grid = np.logspace(-3, 2, 3000)
    return np.interp(EWS / 1000 / sigma, ew.ew_factor(grid, RHO), grid)


def make_spectrum(noise: float = 0.01, seed: int = 0, continuum: bool = True) -> Spec:
    sigma = CENTRES / R / 2.3548
    flux = np.ones_like(WAVE)
    for c, s, a in zip(CENTRES * (1 + RV / C_KMS), sigma, amplitudes(sigma), strict=True):
        flux -= ew.saturated(a * np.exp(-0.5 * ((WAVE - c) / s) ** 2), RHO)
    flux *= 1 + np.random.default_rng(seed).normal(0, noise, WAVE.size)
    if continuum:
        flux *= 900.0 * (1 + 0.15 * (WAVE - 5700) / 100 - 0.05 * ((WAVE - 5700) / 100) ** 2)
    return Spec(WAVE, flux, name="synthetic")


# ----------------------------------------------------------------------------- building blocks
def test_saturated_profile_limits() -> None:
    assert ew.saturated(np.array([1e-6]), 0.1)[0] == pytest.approx(1e-6, rel=1e-5)
    assert ew.saturated(np.array([1e9]), 0.1)[0] == pytest.approx(0.9, rel=1e-6)
    small = ew.ew_factor(np.array([1e-4]), 0.1)[0]
    assert small == pytest.approx(1e-4 * np.sqrt(2 * np.pi), rel=1e-4)
    grid = np.array([0.5, 1.0, 2.0])
    numeric = (ew.ew_factor(grid * 1.0001, 0.1) - ew.ew_factor(grid * 0.9999, 0.1)) / (2e-4 * grid)
    np.testing.assert_allclose(ew.ew_factor_derivative(grid, 0.1), numeric, rtol=1e-4)


def test_masks_follow_the_rv() -> None:
    wave = np.arange(6500.0, 6620.0, 0.1)
    halpha = [m for m in ew.DEFAULT_MASKS if m.label == "H alpha"]
    shifted = ew.mask_pixels(wave, halpha, 2000.0)
    assert wave[shifted].min() == pytest.approx(6552.8 * (1 + 2000 / C_KMS), abs=0.1)
    obs = [ew.MaskRegion(6550, 6560)]
    assert np.array_equal(ew.mask_pixels(wave, obs, 0.0), ew.mask_pixels(wave, obs, 300.0))


def test_read_masks(tmp_path: Path) -> None:
    path = tmp_path / "mask.txt"
    path.write_text("# start end\n6000 6010\n6100 6105\n")
    masks = ew.read_masks(str(path))
    assert [(m.start, m.end, m.frame) for m in masks] == [(6000, 6010, "obs"), (6100, 6105, "obs")]


def test_continuum_bases() -> None:
    wave = np.linspace(5000, 5100, 500)
    leg = ew.continuum_basis(wave, "legendre", 4, 0)
    spl = ew.continuum_basis(wave, "spline", 0, 6)
    assert leg.shape == (500, 5)
    np.testing.assert_allclose(spl.sum(axis=1), 1.0)  # B-splines: partition of unity


def test_initial_continuum_ignores_absorption() -> None:
    spec = make_spectrum(noise=0.003)
    basis = ew.continuum_basis(spec.wave, "legendre", 3, 0)
    cont = ew.initial_continuum(spec.flux, basis, np.ones(spec.wave.size, bool))
    true = 900.0 * (1 + 0.15 * (WAVE - 5700) / 100 - 0.05 * ((WAVE - 5700) / 100) ** 2)
    assert np.median(cont / true) == pytest.approx(1.0, abs=0.01)


def test_correlation_factor() -> None:
    rng = np.random.default_rng(3)
    white = rng.normal(size=20000)
    smooth = np.convolve(rng.normal(size=20004), [0.2, 0.2, 0.2, 0.2, 0.2], mode="valid")
    assert ew.correlation_factor(white, 5.0) == pytest.approx(1.0, abs=0.05)
    assert ew.correlation_factor(smooth, 5.0) > 1.8
    assert ew.correlation_factor(white[:30], 5.0) == 1.0  # too short: no correction


def test_find_lines_and_rv() -> None:
    spec = make_spectrum(noise=0.005, continuum=False)
    valid = np.ones(spec.wave.size, bool)
    noise = ew.pixel_noise(spec.flux, valid)
    mu, depth = ew.find_lines(spec.wave, 1 - spec.flux, valid, 5700 / R / 2.3548 / 0.05, noise)
    assert mu.size >= 35 and np.all(depth > 0)
    pre = ew.preliminary_rv(mu, CENTRES, -500, 500, 3.0)
    assert pre is not None and pre[0] == pytest.approx(RV, abs=1.0) and pre[1] >= 30
    lab_idx, fit_idx, rv, sigma = ew.identify(mu, CENTRES, pre[0], 4.0, 3.0)
    assert lab_idx.size >= 30 and rv == pytest.approx(RV, abs=0.5) and sigma < 2.0
    assert len(set(fit_idx.tolist())) == fit_idx.size  # one fitted line per laboratory line


def test_preliminary_rv_without_matches() -> None:
    assert ew.preliminary_rv(np.array([5000.0]), np.array([6000.0]), -500, 500, 3.0) is None
    assert ew.preliminary_rv(np.empty(0), np.array([6000.0]), -500, 500, 3.0) is None


def test_settings_validation() -> None:
    for bad in (
        {"resolution": 0},
        {"resolution": 1e4, "core": 100},
        {"resolution": 1e4, "order": -2},
        {"resolution": 1e4, "vmin": 10, "vmax": 0},
    ):
        with pytest.raises(ValueError):
            ew.EWSettings(**bad)


# ----------------------------------------------------------------------------- full measurement
@pytest.fixture(scope="module")
def result() -> ew.EWResult:
    return ew.measure_ew(make_spectrum(), CENTRES, CODES, ew.EWSettings(masks=()))


def test_measure_recovers_fwhm_rv_and_ews(result: ew.EWResult) -> None:
    run = result.run
    assert result.convergence == 1
    assert result.resolution == pytest.approx(R, rel=0.02)
    assert run.rv == pytest.approx(RV, abs=0.3)
    assert len(run.lines) >= 36
    by_wave = {round(line.wave, 3): line for line in run.lines}
    found = [i for i, c in enumerate(CENTRES) if round(c, 3) in by_wave]
    measured = np.array([by_wave[round(CENTRES[i], 3)].ew for i in found])
    errors = np.array([by_wave[round(CENTRES[i], 3)].ew_err for i in found])
    pull = (measured - EWS[found]) / errors
    assert abs(np.median(measured - EWS[found])) < 1.0  # no bias (mA)
    assert 0.5 < np.std(pull) < 2.0  # errors of the right size
    assert all(by_wave[round(CENTRES[i], 3)].code == CODES[i] for i in found)


def test_normalized_rest_frame_spectrum(result: ew.EWResult) -> None:
    norm = result.normalized_rest()
    clean = make_spectrum(noise=0.0, continuum=False)
    expected = np.interp(norm.wave, clean.wave / (1 + RV / C_KMS), clean.flux)
    assert np.median(np.abs(norm.flux - expected)) < 0.01


def test_fixed_fwhm_and_normalised_input() -> None:
    spec = make_spectrum(continuum=False)
    res = ew.measure_ew(spec, CENTRES, CODES, ew.EWSettings(resolution=R, order=-1, fix_fwhm=True, masks=()))
    assert res.convergence == 1 and len(res.history) == 1
    assert res.resolution == pytest.approx(R, rel=1e-6)
    np.testing.assert_allclose(res.run.continuum, 1.0)


def test_spline_continuum_and_known_rv() -> None:
    settings = ew.EWSettings(resolution=R, continuum="spline", knots=4, rv=RV, masks=())
    res = ew.measure_ew(make_spectrum(seed=1), CENTRES, CODES, settings)
    assert res.run.rv == pytest.approx(RV, abs=0.5)
    assert res.history[0][2] == pytest.approx(RV - 20) and res.history[0][4] == pytest.approx(RV + 20)


def test_no_identified_line_raises() -> None:
    with pytest.raises(ew.EWError, match="no line of the list"):
        ew.measure_ew(make_spectrum(), CENTRES + 900.0, CODES, ew.EWSettings(resolution=R, masks=()))
    with pytest.raises(ew.EWError, match="no laboratory line"):
        ew.measure_ew(
            make_spectrum(), CENTRES + 50.0, CODES, ew.EWSettings(resolution=R, masks=(), vmin=-100, vmax=100)
        )


def test_prepare_trims_edges_and_resamples() -> None:
    flux = np.ones(200)
    flux[:5] = 0.0
    flux[-7:] = -1.0
    wave, kept = ew.prepare(Spec(np.arange(200.0), flux), None, None)
    assert wave[0] == 5.0 and wave[-1] == 192.0 and np.all(kept > 0)
    uneven = np.cumsum(np.linspace(1.0, 1.1, 200))
    wave, _ = ew.prepare(Spec(uneven, np.ones(200)), None, None)
    assert np.allclose(np.diff(wave), np.diff(wave)[0])
    with pytest.raises(ew.EWError):
        ew.prepare(Spec(np.arange(100.0), np.zeros(100)), None, None)


def test_format_table_like_4dao_plus_width() -> None:
    line = ew.MeasuredLine(
        wave=5600.123,
        code=26.01,
        ew=45.678,
        ew_err=1.234,
        fwhm=0.2801,
        resolution=19993.2,
        q=0.9,
        wave_fit=5600.130,
        rv=120.4,
    )
    text = ew.format_table([line])
    header, row = text.splitlines()
    assert header.split() == ["#", "wave", "EW", "error_Ew", "elem", "FWHM", "Resol"]
    assert row.split() == ["5600.123", "45.678", "1.234", "26.01", "0.2801", "19993"]


# ----------------------------------------------------------------------------- command
def test_ew_command_and_batch_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_spectrum("star1.fits", make_spectrum(seed=2))
    write_spectrum("star2.txt", make_spectrum(seed=3))
    np.savetxt("lines.txt", np.column_stack([CENTRES, CODES]), fmt=("%.3f", "%.2f"))
    row = execute(resolve("ew"), ["star1.fits", "lines.txt", "--no-mask"])  # only spectrum and line list
    assert row["CONV"] == 1 and row["Nlin"] >= 32  # a few list lines are closer than FWHM/2
    assert row["R_lines"] == pytest.approx(R, rel=0.05)
    table = np.loadtxt("star1_ew.txt")
    assert table.shape[1] == 6 and np.all(table[:, 1] > 0) and np.all(table[:, 4] > 0)
    np.testing.assert_allclose(table[:, 5], table[:, 0] / table[:, 4], rtol=1e-3)
    assert read_spectrum("star1_nrv.fits").wave[0] < 5600.0  # shifted to rest frame
    Path("stars.txt").write_text("star1.fits lines.txt\nstar2.txt lines.txt\nmissing.fits lines.txt\n")
    rows = execute(resolve("ew"), ["@stars.txt", "--no-mask"])
    assert Path("star2_nrv.txt").exists()
    assert len(rows) == 2
    log_lines = Path("bob_ew.log").read_text().splitlines()
    assert log_lines[0].split()[:4] == ["#", "line", "star", "R_lines"] and len(log_lines) == 3
    with pytest.raises(UsageError):
        execute(resolve("ew"), ["star1.fits", "lines.txt", "--fix-fwhm"])


# ----------------------------------------------------------------------------- widths
def test_profile_fwhm() -> None:
    sigma = 0.1
    assert ew.profile_fwhm(1e-6, sigma, 0.1) == pytest.approx(2.3548 * sigma, rel=1e-4)
    assert ew.profile_fwhm(0.0, sigma, 0.1) == pytest.approx(2.3548 * sigma, rel=1e-4)
    strong = ew.profile_fwhm(5.0, sigma, 0.1)
    x = np.linspace(-1, 1, 200001)
    depth = ew.saturated(5.0 * np.exp(-0.5 * (x / sigma) ** 2), 0.1)
    above = x[depth >= depth.max() / 2]
    assert strong == pytest.approx(above[-1] - above[0], rel=1e-4)  # half-depth width of the profile
    assert strong > 2.3548 * sigma


def test_estimate_fwhm_without_resolution() -> None:
    spec = make_spectrum(noise=0.005)
    fwhm, n = ew.estimate_fwhm(spec.wave, spec.flux, np.ones(spec.wave.size, bool))
    assert n >= 10 and fwhm == pytest.approx(5700 / R, rel=0.15)
    with pytest.raises(ew.EWError, match="clean lines"):
        ew.estimate_fwhm(WAVE, np.ones(WAVE.size), np.ones(WAVE.size, bool))


def test_line_fwhm_matches_true_profiles(result: ew.EWResult) -> None:
    sigma = CENTRES / R / 2.3548
    amp = amplitudes(sigma)
    core = {round(c, 3): 2.3548 * s for c, s in zip(CENTRES, sigma, strict=True)}
    half = {round(c, 3): ew.profile_fwhm(a, s, RHO) for c, s, a in zip(CENTRES, sigma, amp, strict=True)}
    rel_core = [line.fwhm / core[round(line.wave, 3)] - 1 for line in result.run.lines]
    rel_half = [line.fwhm_half_depth / half[round(line.wave, 3)] - 1 for line in result.run.lines]
    assert abs(np.median(rel_core)) < 0.02 and np.std(rel_core) < 0.08
    assert abs(np.median(rel_half)) < 0.02 and np.std(rel_half) < 0.08
    resolutions = np.array([line.resolution for line in result.run.lines])
    assert np.median(resolutions) == pytest.approx(R, rel=0.02)  # same R for weak and strong lines
    strong = np.array([line.ew > 100 for line in result.run.lines])
    assert np.median(resolutions[strong]) == pytest.approx(np.median(resolutions[~strong]), rel=0.04)


def test_half_depth_table_option() -> None:
    line = ew.MeasuredLine(
        wave=5600.0,
        code=26.0,
        ew=150.0,
        ew_err=2.0,
        fwhm=0.28,
        resolution=20000.0,
        q=1.0,
        wave_fit=5600.0,
        rv=0.0,
        fwhm_half_depth=0.31,
    )
    assert ew.format_table([line]).splitlines()[1].split()[4] == "0.2800"
    assert ew.format_table([line], half_depth=True).splitlines()[1].split()[4:] == ["0.3100", "18065"]


def test_noiseless_spectrum_detects_only_real_lines() -> None:
    res = ew.measure_ew(make_spectrum(noise=0.0), CENTRES, CODES, ew.EWSettings(masks=()))
    assert res.run.n_detected <= 45  # 40 lines, a few blends split
    assert res.run.rv == pytest.approx(RV, abs=0.05)


def test_identify_with_one_fitted_line() -> None:
    lab_idx, fit_idx, rv, _ = ew.identify(
        np.array([5000.0 * (1 + 10 / C_KMS)]), np.array([5000.0, 5100.0]), 10.0, 5.0, 3.0
    )
    assert lab_idx.tolist() == [0] and fit_idx.tolist() == [0] and rv == pytest.approx(10.0, abs=1e-6)
    assert ew.identify(np.empty(0), np.array([5000.0]), 0.0, 5.0, 3.0)[0].size == 0


@pytest.mark.parametrize(
    ("name", "expected"),
    [("a.fits", "a_nrv.fits"), ("dir/b.txt", "dir/b_nrv.txt"), ("c.dat", "c_nrv.dat"), ("d.fits.gz", "d_nrv.fits")],
)
def test_nrv_name(name: str, expected: str) -> None:
    assert nrv_name(name) == Path(expected)


# ----------------------------------------------------------------------------- noisy spectra and spikes
def add_spikes(spec: Spec, seed: int = 0, n: int = 25) -> tuple[Spec, np.ndarray]:
    """Cosmic rays (1-3 pixels) and sky-line residuals (one FWHM wide) up to 20x the continuum."""
    rng = np.random.default_rng(seed)
    flux = spec.flux.copy()
    centres = rng.integers(20, flux.size - 20, n)
    level = np.median(flux)
    for i, c in enumerate(centres):
        if i % 2:
            flux[c : c + rng.integers(1, 4)] += level * rng.uniform(2, 20)
        else:
            x = np.arange(flux.size) - c
            flux += level * rng.uniform(2, 20) * np.exp(-0.5 * (x / 2.4) ** 2)
    return Spec(spec.wave, flux, name="spiky"), centres


def test_find_spikes_flags_only_narrow_or_huge_excesses() -> None:
    base = np.full(1000, 100.0)
    flux = base + np.random.default_rng(0).normal(0, 1, 1000)
    flux[100:102] += 50  # cosmic ray
    flux[300:340] += 10  # broad, moderate excess: continuum structure, kept
    flux[600:640] += 300  # broad but huge (sky doublet): spike
    flux[800:806] -= 60  # absorption line: never a spike
    spikes = ew.find_spikes(flux, np.ones(1000, bool), base, max_width=8, grow=2)
    assert spikes[100:102].all() and spikes[98] and spikes[103]  # grown by 2 pixels
    assert not spikes[300:340].any()
    assert spikes[600:640].all()
    assert not spikes[800:806].any()


def test_noisy_spiky_spectrum_converges_without_resolution() -> None:
    spec, _ = add_spikes(make_spectrum(noise=0.05, seed=3))
    res = ew.measure_ew(spec, CENTRES, CODES, ew.EWSettings(masks=()))
    assert res.convergence == 1
    assert res.run.rv == pytest.approx(RV, abs=1.5)
    assert len(res.run.lines) >= 20
    assert res.run.residual_pct < 8.0  # spikes excluded (noise is 5%)
    assert res.resolution == pytest.approx(R, rel=0.1)


def test_estimate_fwhm_on_noisy_spectrum() -> None:
    spec, _ = add_spikes(make_spectrum(noise=0.05, seed=5))
    fwhm, n = ew.estimate_fwhm(spec.wave, spec.flux, np.ones(spec.wave.size, bool))
    assert n >= 3 and fwhm == pytest.approx(5700 / R, rel=0.35)  # a first guess: the iterations refine it


def test_first_guess_falls_back_to_default(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def fail(*_args: object, **_kwargs: object) -> tuple[float, int]:
        raise ew.EWError("only 0 clean lines to estimate the FWHM")

    monkeypatch.setattr(ew, "estimate_fwhm", fail)
    with caplog.at_level("WARNING"):
        res = ew.measure_ew(make_spectrum(), CENTRES, CODES, ew.EWSettings(masks=()))
    assert "starting from R = 20000" in caplog.text
    assert res.convergence == 1 and res.resolution == pytest.approx(R, rel=0.03)
