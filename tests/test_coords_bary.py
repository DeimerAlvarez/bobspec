import pytest
from astropy.io import fits

from bobspec.barycentric import (
    MissingKeywordError,
    Observation,
    apply_correction,
    barycentric_correction,
    mid_exposure_mjd,
    observation_from_eso_header,
)
from bobspec.coords import parse_coordinates


@pytest.mark.parametrize(
    ("ra", "dec"),
    [("187.5", "-12.3"), ("187.5", "+12.3"), ("1.875e2", "-12.3"), ("12:30:00", "-12:18:00"), ("12h30m00s", "+12d18m")],
)
def test_parse_coordinates_formats(ra: str, dec: str) -> None:
    c = parse_coordinates(ra, dec)
    assert c.ra_deg == pytest.approx(187.5)
    assert abs(c.dec_deg) == pytest.approx(12.3)


def test_parse_coordinates_round_trip() -> None:
    c = parse_coordinates("283.763875", "-30.479861")
    back = parse_coordinates(c.ra_hms, c.dec_dms)
    assert back.ra_deg == pytest.approx(c.ra_deg, abs=1e-6)
    assert back.dec_deg == pytest.approx(c.dec_deg, abs=1e-6)


@pytest.mark.parametrize(
    ("ra", "dec"), [("187.5", "-12:18"), ("400", "0"), ("10", "95"), ("25:00:00", "10:00:00"), ("abc", "def")]
)
def test_parse_coordinates_rejects(ra: str, dec: str) -> None:
    with pytest.raises(ValueError):
        parse_coordinates(ra, dec)


def test_mid_exposure() -> None:
    assert mid_exposure_mjd(60000.0, 3600.0) == pytest.approx(60000.0 + 1800 / 86400)
    with pytest.raises(ValueError):
        mid_exposure_mjd(60000.0, -1.0)


def test_observation_validation() -> None:
    with pytest.raises(ValueError):
        Observation(10.0, 100.0, 60000.0, 0.0, 0.0, 0.0)


PARANAL = {"HIERARCH ESO TEL GEOLON": -70.4045, "HIERARCH ESO TEL GEOLAT": -24.6268, "HIERARCH ESO TEL GEOELEV": 2648.0}


def test_eso_header_uses_mid_exposure() -> None:
    header = fits.Header({"RA": 283.763875, "DEC": -30.479861, "MJD-OBS": 60000.0, "EXPTIME": 1800.0, **PARANAL})
    obs = observation_from_eso_header(header)
    assert obs.mjd_mid == pytest.approx(60000.0 + 900 / 86400)
    v = barycentric_correction(obs)
    assert abs(v) < 32.0  # Earth orbital (~30) + rotation (<0.5) km/s


def test_eso_header_missing_keywords_listed() -> None:
    with pytest.raises(MissingKeywordError, match="MJD-OBS"):
        observation_from_eso_header(fits.Header({"RA": 1.0, "DEC": 1.0}))


def test_annual_modulation_sign() -> None:
    # Target near the ecliptic: corrections half a year apart have opposite signs.
    obs = [Observation(90.0, 23.4, mjd, 0.0, 0.0, 0.0) for mjd in (60000.0, 60182.6)]
    v1, v2 = (barycentric_correction(o) for o in obs)
    assert v1 * v2 < 0 and abs(v1) > 20 and abs(v2) > 20


def test_apply_correction_first_order() -> None:
    assert apply_correction(100.0, 10.0) == pytest.approx(110.0 + 1000.0 / 299792.458)
