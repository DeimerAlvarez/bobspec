"""ew 1.2: fixed FWHM, FWHM relation from the list lines, protection against split lines."""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from test_ew import CENTRES, CODES, RHO, RV, R, make_spectrum

import bobspec.cli  # noqa: F401  (registers the commands)
from bobspec import ew
from bobspec.constants import C_KMS
from bobspec.io import Spec, write_spectrum
from bobspec.registry import UsageError, execute, resolve


def line(wave: float, fwhm: float, *, blended: bool = False, fitted: bool = True) -> ew.MeasuredLine:
    return ew.MeasuredLine(
        wave=wave,
        code=26.0,
        ew=50.0,
        ew_err=1.0,
        fwhm=fwhm,
        resolution=wave / fwhm,
        q=1.0,
        wave_fit=wave,
        rv=0.0,
        width_fitted=fitted,
        blended=blended,
    )


# ----------------------------------------------------------------------------- the FWHM relation
def test_fwhm_relation_linear_constant_and_clipping() -> None:
    waves = np.linspace(5600, 5800, 20)
    truth = 0.27 + 0.06 * (waves - 5600) / 200
    lines = [line(w, f) for w, f in zip(waves, truth, strict=True)]
    lines += [line(5650.0, 0.45), line(5750.0, 0.15)]  # an unflagged blend and a narrow feature
    lines += [line(5700.0, 0.9, blended=True), line(5710.0, 0.9, fitted=False)]  # never candidates
    model = ew.fit_fwhm_relation(lines, rv=0.0, lam_c=5700.0)
    assert model.source == "list" and model.n_candidates == 22 and len(model.used) == 20
    np.testing.assert_allclose(model.at(np.array([5600.0, 5700.0, 5800.0])), [0.27, 0.30, 0.33], rtol=1e-6)
    assert model.slope == pytest.approx(0.06 / 200 * 5700 / 0.30)
    flat = ew.fit_fwhm_relation(lines, rv=0.0, lam_c=5700.0, constant=True)
    assert flat.slope == 0.0 and flat.fwhm == pytest.approx(np.median(truth), rel=0.02)


def test_fwhm_relation_observed_frame_and_too_few_lines() -> None:
    lines = [line(w, 0.3) for w in (5600.0, 5650.0, 5700.0, 5750.0, 5800.0)]
    model = ew.fit_fwhm_relation(lines, rv=300.0, lam_c=5700.0)
    assert model.fwhm == pytest.approx(0.3 * (1 + 300 / C_KMS))  # rest-frame widths, observed-frame model
    with pytest.raises(ew.EWError, match="unblended"):
        ew.fit_fwhm_relation(lines[:2], rv=0.0, lam_c=5700.0)


def test_fwhm_model_shapes() -> None:
    lam = np.array([5600.0, 5700.0, 5800.0])
    assert ew.FwhmModel(0.3, 1.0, 5700.0, "all").at(lam) == pytest.approx(0.3 * lam / 5700.0)
    assert ew.FwhmModel(0.3, 0.0, 5700.0, "all").at(lam) == pytest.approx([0.3, 0.3, 0.3])


def test_settings_for_fixed_and_list_fwhm() -> None:
    with pytest.raises(ValueError, match="positive"):
        ew.EWSettings(fwhm=0.0)
    with pytest.raises(ValueError, match="exclude"):
        ew.EWSettings(fwhm=0.3, fwhm_from="list")
    with pytest.raises(ValueError, match="fwhm_from"):
        ew.EWSettings(fwhm_from="some")  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- measurements
def test_fixed_fwhm_is_used_as_given() -> None:
    res = ew.measure_ew(make_spectrum(), CENTRES, CODES, ew.EWSettings(masks=(), fwhm=0.29))
    assert res.fwhm_model.source == "fixed" and res.run.fwhm == pytest.approx(0.29)
    assert len(res.history) == 1


def test_list_mode_recovers_constant_r() -> None:
    res = ew.measure_ew(make_spectrum(), CENTRES, CODES, ew.EWSettings(masks=(), fwhm_from="list"))
    model = res.fwhm_model
    assert model.source == "list" and len(model.used) >= 15 and res.convergence == 1
    lam = np.array([5610.0, 5790.0]) * (1 + RV / C_KMS)
    np.testing.assert_allclose(model.at(lam), lam / R, rtol=0.03)


def narrow_contaminated_spectrum() -> tuple[Spec, np.ndarray]:
    """List lines with a linear FWHM, plus narrow lines (60% of the width) that are not in the list."""
    rng = np.random.default_rng(4)
    wave = np.arange(5600.0, 5800.0, 0.05)
    width = lambda lam: 0.27 + 0.06 * (lam - 5600) / 200  # noqa: E731
    grid = np.logspace(-3, 2, 3000)
    table = ew.ew_factor(grid, RHO)
    lab = np.sort(rng.uniform(5606, 5794, 40))
    flux = np.ones_like(wave)
    narrow = rng.uniform(5606, 5794, 25)
    for centres, ews, factor in ((lab, rng.uniform(20, 140, 40), 1.0), (narrow, rng.uniform(30, 90, 25), 0.6)):
        for c, e in zip(centres, ews, strict=True):
            s = factor * width(c) / 2.3548
            a = np.interp(e / 1000 / s, table, grid)
            flux -= ew.saturated(a * np.exp(-0.5 * ((wave - c) / s) ** 2), RHO)
    flux *= 1 + rng.normal(0, 0.005, wave.size)
    return Spec(wave, 1000 * flux), lab


def test_common_fwhm_does_not_collapse_onto_narrow_lines() -> None:
    spec, lab = narrow_contaminated_spectrum()
    res = ew.measure_ew(spec, lab, np.full(lab.size, 26.0), ew.EWSettings(masks=()))
    assert res.run.fwhm == pytest.approx(0.30, rel=0.12)  # before 1.2 it collapsed to 0.18 A
    assert res.run.n_detected < 80  # 65 lines: no splitting into narrow components


def test_list_mode_recovers_a_linear_fwhm_despite_narrow_lines() -> None:
    spec, lab = narrow_contaminated_spectrum()
    res = ew.measure_ew(spec, lab, np.full(lab.size, 26.0), ew.EWSettings(masks=(), fwhm_from="list"))
    np.testing.assert_allclose(res.fwhm_model.at(np.array([5610.0, 5700.0, 5790.0])), [0.273, 0.300, 0.327], rtol=0.03)


def test_curvature_peak_marks_shoulders_not_wings() -> None:
    wave = np.arange(5000.0, 5010.0, 0.05)
    sigma = 0.3 / 2.3548
    single = 1 - 0.5 * np.exp(-0.5 * ((wave - 5005) / sigma) ** 2)
    pair = single - 0.3 * np.exp(-0.5 * ((wave - 5005.3) / sigma) ** 2)  # shoulder one FWHM away
    valid = np.ones(wave.size, bool)
    wing = np.array([5005.0 + 0.74 * 0.3])
    assert not ew.has_curvature_peak(wave, single, valid, wing, np.array([0.075]), sigma / 0.05)[0]
    assert ew.has_curvature_peak(wave, pair, valid, np.array([5005.3]), np.array([0.075]), sigma / 0.05)[0]


def test_blended_flag_for_a_close_pair() -> None:
    res = ew.measure_ew(make_spectrum(), CENTRES, CODES, ew.EWSettings(masks=()))
    lines = sorted(res.run.lines, key=lambda ln: ln.wave)
    for a, b in pairwise(lines):
        if b.wave - a.wave < 1.0 * a.fwhm:
            assert a.blended and b.blended
    isolated = [ln for ln in lines if all(abs(ln.wave - o.wave) > 4 * ln.fwhm for o in lines if o is not ln)]
    assert isolated and not any(ln.blended for ln in isolated)


# ----------------------------------------------------------------------------- command
def test_command_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    write_spectrum("star.fits", make_spectrum(seed=8))
    np.savetxt("lines.txt", np.column_stack([CENTRES, CODES]), fmt=("%.3f", "%.2f"))
    row = execute(resolve("ew"), ["star.fits", "lines.txt", "--no-mask", "--no-pdf", "--fwhm-from", "list"])
    assert row["FWHM_from"] == "list" and row["N_FWHM"] >= 15
    assert "FWHM relation from" in capsys.readouterr().out
    row = execute(resolve("ew"), ["star.fits", "lines.txt", "--no-mask", "--no-pdf", "--fwhm", "0.3"])
    assert row["FWHM_from"] == "fixed" and "Fixed FWHM" in capsys.readouterr().out
    for bad in (["--fwhm", "0.3", "--fwhm-from", "list"], ["--fwhm", "0.3", "--resol", "20000"]):
        with pytest.raises(UsageError):
            execute(resolve("ew"), ["star.fits", "lines.txt", *bad])
