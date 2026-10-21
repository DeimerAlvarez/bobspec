from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from bobspec import commands
from bobspec.io import Spec, read_spectrum, write_spectrum
from bobspec.registry import SESSION, UsageError, execute, resolve


def run(name: str, *args: object):
    return execute(resolve(name), [str(a) for a in args])


@pytest.fixture
def spec_file(tmp_path: Path, line_model) -> Path:
    wave = np.arange(5010.0, 5290.0, 0.03)
    header = fits.Header(
        {
            "OBJECT": "star",
            "RA": 283.763875,
            "DEC": -30.479861,
            "MJD-OBS": 60000.0,
            "EXPTIME": 1200.0,
            "HIERARCH ESO TEL GEOLON": -70.4045,
            "HIERARCH ESO TEL GEOLAT": -24.6268,
            "HIERARCH ESO TEL GEOELEV": 2648.0,
        }
    )
    flux = 500.0 * line_model(wave, 60.0) * (1 + np.random.default_rng(9).normal(0, 0.01, wave.size))
    return write_spectrum(tmp_path / "star.fits", Spec(wave, flux, header))


def test_companion_paths() -> None:
    assert commands.companion("../out/star.fits", ".snr") == Path("../out/star.snr")
    assert commands.companion("a.b.fits.gz", "_ccf.png") == Path("a.b_ccf.png")
    assert commands.companion("noext", ".snr") == Path("noext.snr")


def test_format_conversions(tmp_path: Path, spec_file: Path) -> None:
    run("totxt", spec_file, tmp_path / "s")  # no suffix -> kept as given (ASCII)
    run("tofits", tmp_path / "s", tmp_path / "back")
    back = read_spectrum(tmp_path / "back.fits")
    assert len(back) == len(read_spectrum(spec_file))


def test_cut_and_shift(tmp_path: Path, spec_file: Path) -> None:
    run("cut", spec_file, 5100, 5200, tmp_path / "cut.txt")
    part = read_spectrum(tmp_path / "cut.txt")
    assert part.wave[0] >= 5100 and part.wave[-1] <= 5200
    with pytest.raises(ValueError):
        run("cut", spec_file, 6000, 6100, tmp_path / "x.txt")
    run("shift_rv", tmp_path / "cut.txt", 100, tmp_path / "rest.txt")
    assert read_spectrum(tmp_path / "rest.txt").wave[0] == pytest.approx(part.wave[0] / (1 + 100 / 299792.458))


def test_rotate_and_smooth_and_ispline(tmp_path: Path, spec_file: Path) -> None:
    run("rotate", spec_file, 0.6, 20, tmp_path / "rot.fits")
    assert np.nanstd(read_spectrum(tmp_path / "rot.fits").flux) < np.nanstd(read_spectrum(spec_file).flux)
    run("smooth", spec_file, 5, tmp_path / "sm.fits", "--auto")
    run("ispline", spec_file, 3, 2, 3, tmp_path / "isp.fits", "--auto", "--knots", 5)
    for name in ("sm", "isp"):
        assert np.nanpercentile(read_spectrum(tmp_path / f"{name}.fits").flux, 90) == pytest.approx(1.0, abs=0.03)
        assert (tmp_path / f"{name}.snr").exists()


def test_header_and_key_list(tmp_path: Path, spec_file: Path, capsys) -> None:
    run("header", spec_file)
    assert "OBJECT" in capsys.readouterr().out
    assert run("header", spec_file, "object") == {"file": str(spec_file), "object": "star"}
    lst = tmp_path / "keys.txt"
    lst.write_text(f"{spec_file} OBJECT\n{spec_file} 'ESO TEL GEOLAT'\n{spec_file} NOPE\n")
    rows = run("header", f"@{lst}")
    assert [r.get("OBJECT", r.get("ESO TEL GEOLAT", r.get("NOPE"))) for r in rows] == ["star", -24.6268, ""]


def test_helio_eso_and_manual_agree(spec_file: Path) -> None:
    eso = run("helio_eso", spec_file)
    manual = run("helio_manual", "18:55:03.33", "-30:28:47.5", 60000.0, -70.4045, -24.6268, 2648, "--exptime", 1200)
    assert eso["v_bary_corr"] == pytest.approx(manual["v_bary_corr"], abs=1e-3)


def test_snr_range(spec_file: Path) -> None:
    row = run("snr", spec_file, "--range", 5100, 5110)
    assert row["der_snr"] > 0


def test_fitline_requires_wavelengths(spec_file: Path) -> None:
    with pytest.raises(UsageError):
        run("fitline", spec_file)


def test_manual_spline_cancel_saves_nothing(tmp_path: Path, spec_file: Path, monkeypatch) -> None:
    monkeypatch.setattr("bobspec.interactive.anchor_editor", lambda spec, k: None)
    run("manual_spline", spec_file, 3, tmp_path / "out.fits")
    assert not (tmp_path / "out.fits").exists()


def test_rv_interactive_quit_saves_nothing(tmp_path: Path, spec_file: Path, template, monkeypatch) -> None:
    tpl = write_spectrum(tmp_path / "tpl.fits", template)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)
    answers = iter(["n", "-300", "300", "", "", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert run("rv", spec_file, tpl, tmp_path / "rest.fits") is None
    assert not (tmp_path / "rest.fits").exists()


def test_tellurics_setting_is_used(tmp_path: Path, spec_file: Path, template, monkeypatch) -> None:
    tpl = write_spectrum(tmp_path / "tpl.fits", template)
    monkeypatch.setattr(SESSION, "telluric_regions", ((5000.0, 5300.0),))
    with pytest.raises(ValueError, match="usable pixels"):
        run("rv", spec_file, tpl, tmp_path / "r.fits", "--auto", "--nsim", 0)
