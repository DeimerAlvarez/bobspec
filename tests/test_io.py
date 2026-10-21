from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from bobspec.io import Spec, SpectrumReadError, is_fits, read_header, read_spectrum, wavelength_from_wcs, write_spectrum


# ----------------------------------------------------------------------------- Spec
def test_spec_sorts_descending_input() -> None:
    spec = Spec([3.0, 2.0, 1.0], [30.0, 20.0, 10.0])
    np.testing.assert_array_equal(spec.wave, [1, 2, 3])
    np.testing.assert_array_equal(spec.flux, [10, 20, 30])


@pytest.mark.parametrize(
    ("wave", "flux", "message"),
    [
        ([1.0], [1.0], "at least two"),
        ([1.0, 2.0], [1.0], "differ"),
        ([1.0, 1.0], [1.0, 2.0], "duplicated"),
        ([1.0, np.nan], [1.0, 2.0], "non-finite"),
        (np.ones((2, 2)), np.ones((2, 2)), "1-D"),
    ],
)
def test_spec_rejects_invalid_arrays(wave: object, flux: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Spec(wave, flux)


def test_spec_is_immutable() -> None:
    spec = Spec([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        spec.flux[0] = 5.0


def test_select_empty_range_raises() -> None:
    spec = Spec(np.arange(10.0), np.ones(10))
    with pytest.raises(ValueError, match="fewer than 2 pixels"):
        spec.select(100, 200)


def test_select_accepts_reversed_limits() -> None:
    assert len(Spec(np.arange(10.0), np.ones(10)).select(6, 2)) == 5


# ----------------------------------------------------------------------------- reading
@pytest.mark.parametrize("name", ["a.fits", "a.FIT", "a.fts", "a.fits.gz"])
def test_is_fits_suffixes(name: str) -> None:
    assert is_fits(name)


def test_is_fits_rejects_ascii() -> None:
    assert not is_fits("a.txt")


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(SpectrumReadError, match="not found"):
        read_spectrum(tmp_path / "nope.fits")


def test_garbage_ascii_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("hello world\n")
    with pytest.raises(SpectrumReadError):
        read_spectrum(path)


def test_binary_file_with_ascii_suffix_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_bytes(bytes(range(256)))
    with pytest.raises(SpectrumReadError):
        read_spectrum(path)


def test_ascii_round_trip(tmp_path: Path) -> None:
    spec = Spec(np.linspace(5000, 5010, 11), np.arange(11.0))
    out = read_spectrum(write_spectrum(tmp_path / "s.txt", spec))
    np.testing.assert_allclose(out.wave, spec.wave)
    np.testing.assert_allclose(out.flux, spec.flux)


def test_fits_round_trip_keeps_header(tmp_path: Path) -> None:
    header = fits.Header({"OBJECT": "M54-1", "MJD-OBS": 60000.5})
    spec = Spec(5000 + 0.05 * np.arange(100), np.ones(100), header).with_history("test step")
    out = read_spectrum(write_spectrum(tmp_path / "s.fits", spec))
    np.testing.assert_allclose(out.wave, spec.wave, rtol=0, atol=1e-9)
    assert out.header["OBJECT"] == "M54-1"
    assert any("test step" in str(card) for card in out.header["HISTORY"])


def test_fits_output_resamples_nonuniform_grid(tmp_path: Path) -> None:
    wave = np.exp(np.linspace(np.log(5000), np.log(5100), 500))  # log-uniform
    spec = Spec(wave, wave / 5000.0)
    out = read_spectrum(write_spectrum(tmp_path / "s.fits", spec))
    assert out.is_uniform()
    assert out.wave[0] == pytest.approx(wave[0])
    assert out.wave[-1] == pytest.approx(wave[-1], abs=np.diff(wave).min())
    np.testing.assert_allclose(out.flux, out.wave / 5000.0, rtol=1e-6)


def test_fits_output_tiny_step_does_not_crash(tmp_path: Path) -> None:
    spec = Spec(5000 + 4e-5 * np.arange(50), np.ones(50))
    out = read_spectrum(write_spectrum(tmp_path / "s.fits", spec))
    assert len(out) == 50


def _image(tmp_path: Path, header: dict[str, object], npix: int = 10) -> Path:
    path = tmp_path / "img.fits"
    fits.PrimaryHDU(np.arange(npix, dtype=float), header=fits.Header(header)).writeto(path)
    return path


def test_linear_wcs_with_crpix(tmp_path: Path) -> None:
    spec = read_spectrum(_image(tmp_path, {"CRVAL1": 5000.0, "CDELT1": 0.5, "CRPIX1": 3.0}))
    assert spec.wave[2] == pytest.approx(5000.0)
    assert spec.wave[0] == pytest.approx(4999.0)


def test_cd1_1_and_nm_units(tmp_path: Path) -> None:
    spec = read_spectrum(_image(tmp_path, {"CRVAL1": 500.0, "CD1_1": 0.01, "CUNIT1": "nm"}))
    assert spec.wave[0] == pytest.approx(5000.0)
    assert spec.wave[1] - spec.wave[0] == pytest.approx(0.1)


def test_missing_dispersion_raises(tmp_path: Path) -> None:
    with pytest.raises(SpectrumReadError, match="CDELT1"):
        read_spectrum(_image(tmp_path, {"CRVAL1": 5000.0}))


def test_iraf_log10_axis() -> None:
    header = fits.Header({"CRVAL1": np.log10(5000.0), "CDELT1": 1e-5, "DC-FLAG": 1})
    wave = wavelength_from_wcs(header, 3)
    np.testing.assert_allclose(wave, 5000.0 * 10 ** (1e-5 * np.arange(3)))


def test_fits_standard_log_axis() -> None:
    header = fits.Header({"CRVAL1": 5000.0, "CDELT1": 0.1, "CTYPE1": "WAVE-LOG"})
    wave = wavelength_from_wcs(header, 3)
    np.testing.assert_allclose(wave, 5000.0 * np.exp(0.1 * np.arange(3) / 5000.0))


def test_multidimensional_image_rejected(tmp_path: Path) -> None:
    path = tmp_path / "echelle.fits"
    fits.PrimaryHDU(np.ones((3, 10)), header=fits.Header({"CRVAL1": 1.0, "CDELT1": 1.0})).writeto(path)
    with pytest.raises(SpectrumReadError, match="multi-dimensional"):
        read_spectrum(path)


def test_eso_phase3_binary_table(tmp_path: Path) -> None:
    wave_nm = np.linspace(500.0, 510.0, 50)
    cols = [
        fits.Column(name="WAVE", format="50D", unit="nm", array=[wave_nm]),
        fits.Column(name="FLUX", format="50D", array=[np.arange(50.0)]),
    ]
    primary = fits.PrimaryHDU(header=fits.Header({"RA": 283.76, "OBJECT": "star"}))
    path = tmp_path / "eso.fits"
    fits.HDUList([primary, fits.BinTableHDU.from_columns(cols)]).writeto(path)
    spec = read_spectrum(path)
    np.testing.assert_allclose(spec.wave, wave_nm * 10.0)
    assert spec.header["OBJECT"] == "star"


def test_read_header_rejects_ascii(tmp_path: Path) -> None:
    path = tmp_path / "s.txt"
    path.write_text("1 2\n3 4\n")
    with pytest.raises(SpectrumReadError, match="not a FITS"):
        read_header(path)
