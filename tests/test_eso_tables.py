"""read_eso_fits: column extraction from FITS binary tables (ESO Phase-3 layout and others)."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import bobspec.cli  # noqa: F401  (registers the commands)
from bobspec.io import SpectrumReadError, list_table_columns, read_spectrum, read_table_columns
from bobspec.registry import UsageError, execute, resolve

NPIX = 500
WAVE_NM = np.linspace(450.0, 460.0, NPIX)
FLUX = 1.0 + 0.1 * np.sin(np.arange(NPIX) / 10)
ERR = np.full(NPIX, 0.02)
PRIMARY = {
    "RA": 283.763875,
    "DEC": -30.479861,
    "MJD-OBS": 57253.1,
    "EXPTIME": 1300.0,
    "PRODCATG": "SCIENCE.SPECTRUM",
    "HIERARCH ESO TEL GEOLON": -70.4045,
    "HIERARCH ESO TEL GEOLAT": -24.6268,
    "HIERARCH ESO TEL GEOELEV": 2648.0,
}


def run(*args: object):
    return execute(resolve("read_eso_fits"), [str(a) for a in args])


@pytest.fixture
def phase3(tmp_path: Path) -> Path:
    """ESO Phase-3 layout: one row, each cell an array; wavelength in nm."""
    cols = [
        fits.Column(name="WAVE", format=f"{NPIX}D", unit="nm", array=[WAVE_NM]),
        fits.Column(name="FLUX_REDUCED", format=f"{NPIX}E", unit="adu", array=[1000 * FLUX]),
        fits.Column(name="ERR_REDUCED", format=f"{NPIX}E", unit="adu", array=[1000 * ERR]),
        fits.Column(name="FLUX", format=f"{NPIX}E", unit="erg cm**-2 s**-1 angstrom**-1", array=[FLUX]),
        fits.Column(name="ERR", format=f"{NPIX}E", array=[ERR]),
        fits.Column(name="QUAL", format=f"{NPIX}J", array=[np.zeros(NPIX, int)]),
    ]
    path = tmp_path / "ADP.2017-01-01T00:00:00.000.fits"
    fits.HDUList([fits.PrimaryHDU(header=fits.Header(PRIMARY)), fits.BinTableHDU.from_columns(cols)]).writeto(path)
    return path


def test_same_arrays_as_eso_recipe(phase3: Path) -> None:
    # ESO archive recipe: scidata = hdulist[1].data; wave = scidata[0][0]; arr1 = scidata[0][1]
    with fits.open(phase3) as hdul:
        scidata = hdul[1].data
        wave, arr1 = scidata[0][0], scidata[0][1]
    spec, cx, cy = read_table_columns(phase3, 0, 1, convert_wave=False)
    np.testing.assert_allclose(spec.wave, wave)
    np.testing.assert_allclose(spec.flux, arr1)
    assert (cx.name, cy.name) == ("WAVE", "FLUX_REDUCED")


def test_list_columns(phase3: Path, capsys) -> None:
    ext, nrows, columns = list_table_columns(phase3)
    assert (ext, nrows) == (1, 1)
    assert [c.name for c in columns] == ["WAVE", "FLUX_REDUCED", "ERR_REDUCED", "FLUX", "ERR", "QUAL"]
    assert columns[0].unit == "nm" and columns[0].length == NPIX
    assert run(phase3) is None
    out = capsys.readouterr().out
    assert "FLUX_REDUCED" in out and "extension 1" in out


def test_extract_by_index_to_txt_converts_nm(phase3: Path, tmp_path: Path) -> None:
    out = tmp_path / "star.txt"
    row = run(phase3, 0, 1, out)
    data = np.loadtxt(out)
    np.testing.assert_allclose(data[:, 0], WAVE_NM * 10.0, rtol=1e-9)
    np.testing.assert_allclose(data[:, 1], 1000 * FLUX, rtol=1e-6)
    assert out.read_text().startswith("# WAVE[Angstrom] FLUX_REDUCED[adu]")
    assert row["n_points"] == NPIX


def test_extract_by_name_to_dat_and_keep_units(phase3: Path, tmp_path: Path) -> None:
    out = tmp_path / "star_err.dat"
    run(phase3, "wave", "ERR", out, "--keep-units")
    data = np.loadtxt(out)
    np.testing.assert_allclose(data[:, 0], WAVE_NM, rtol=0, atol=1e-6)  # ASCII keeps 6 decimals
    np.testing.assert_allclose(data[:, 1], ERR, rtol=1e-6)


def test_extract_to_fits_keeps_primary_header(phase3: Path, tmp_path: Path) -> None:
    out = tmp_path / "star.fits"
    run(phase3, 0, 3, out)
    spec = read_spectrum(out)
    assert spec.wave[0] == pytest.approx(4500.0) and spec.wave[-1] == pytest.approx(4600.0, abs=1e-6)
    assert spec.header["PRODCATG"] == "SCIENCE.SPECTRUM"
    assert any("FLUX (3)" in str(h) for h in spec.header["HISTORY"])
    row = execute(resolve("helio_eso"), [str(out)])  # ESO keywords survived
    assert abs(row["v_bary_corr"]) < 32


def test_row_per_pixel_table(tmp_path: Path) -> None:
    cols = [
        fits.Column(name="LAMBDA", format="D", unit="Angstrom", array=WAVE_NM * 10),
        fits.Column(name="F", format="D", array=FLUX),
    ]
    path = tmp_path / "rows.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(path)
    spec, *_ = read_table_columns(path, "LAMBDA", "F")
    assert len(spec) == NPIX


def test_several_spectra_choose_row(tmp_path: Path) -> None:
    cols = [
        fits.Column(name="WAVE", format=f"{NPIX}D", array=[WAVE_NM, WAVE_NM]),
        fits.Column(name="FLUX", format=f"{NPIX}D", array=[FLUX, 2 * FLUX]),
    ]
    path = tmp_path / "two.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(path)
    spec, *_ = read_table_columns(path, 0, 1, row=1)
    np.testing.assert_allclose(spec.flux, 2 * FLUX)
    with pytest.raises(SpectrumReadError, match="row 5"):
        read_table_columns(path, 0, 1, row=5)


def test_non_finite_wavelengths_dropped(tmp_path: Path) -> None:
    wave = WAVE_NM.copy()
    wave[:3] = np.nan
    cols = [
        fits.Column(name="WAVE", format=f"{NPIX}D", array=[wave]),
        fits.Column(name="FLUX", format=f"{NPIX}D", array=[FLUX]),
    ]
    path = tmp_path / "nan.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(path)
    assert len(read_table_columns(path, 0, 1)[0]) == NPIX - 3


@pytest.mark.parametrize(
    ("xcol", "ycol", "message"), [(0, 9, "column 9 does not exist"), ("WAVE", "NOPE", "no column named")]
)
def test_bad_columns(phase3: Path, xcol: object, ycol: object, message: str) -> None:
    with pytest.raises(SpectrumReadError, match=message):
        read_table_columns(phase3, xcol, ycol)


def test_bad_extension_and_image_file(phase3: Path, tmp_path: Path) -> None:
    with pytest.raises(SpectrumReadError, match="not a binary table"):
        list_table_columns(phase3, ext=0)
    image = tmp_path / "image.fits"
    fits.PrimaryHDU(np.ones(10)).writeto(image)
    with pytest.raises(SpectrumReadError, match="no binary table"):
        list_table_columns(image)


def test_command_argument_checks(phase3: Path, tmp_path: Path) -> None:
    with pytest.raises(UsageError, match="alone"):
        run(phase3, 0, 1)
    with pytest.raises(UsageError, match=r"\.txt, \.dat or \.fits"):
        run(phase3, 0, 1, tmp_path / "out.csv")


def test_batch_list(phase3: Path, tmp_path: Path) -> None:
    lst = tmp_path / "eso.txt"
    lines = [f"{phase3} 0 1 {tmp_path / 'a.txt'}", f"{phase3} WAVE ERR {tmp_path / 'b.fits'}", f"{phase3} 0 99 x.txt"]
    lst.write_text("\n".join(lines) + "\n")
    rows = execute(resolve("read_eso_fits"), [f"@{lst}"])
    assert [r["y"] for r in rows] == ["FLUX_REDUCED", "ERR"]
