"""Barycentric velocity corrections (astropy implementation).

The correction ``v_corr`` is the projection of the observatory velocity onto
the line of sight, so that ``v_bary = v_topo + v_corr + v_topo * v_corr / c``
(see :func:`apply_correction`). It must be evaluated at the *middle* of the
exposure: using the start time (``MJD-OBS``) biases it by up to a few tens of
m/s for hour-long exposures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time

from bobspec.constants import C_KMS

log = logging.getLogger(__name__)

_ESO_KEYS = {
    "ra": "RA",
    "dec": "DEC",
    "mjd": "MJD-OBS",
    "lon": "HIERARCH ESO TEL GEOLON",
    "lat": "HIERARCH ESO TEL GEOLAT",
    "alt": "HIERARCH ESO TEL GEOELEV",
}


class MissingKeywordError(KeyError):
    """Raised when required header keywords are absent."""


@dataclass(frozen=True, slots=True)
class Observation:
    """Target, time and site of an observation.

    Attributes:
        ra_deg: ICRS right ascension (deg).
        dec_deg: ICRS declination (deg).
        mjd_mid: Mid-exposure Modified Julian Date (UTC).
        lon_deg: Geodetic longitude of the site (deg, east positive).
        lat_deg: Geodetic latitude of the site (deg, north positive).
        alt_m: Altitude of the site (m).
    """

    ra_deg: float
    dec_deg: float
    mjd_mid: float
    lon_deg: float
    lat_deg: float
    alt_m: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.ra_deg < 360.0:
            raise ValueError(f"RA must be in [0, 360) deg, got {self.ra_deg}")
        if not -90.0 <= self.dec_deg <= 90.0:
            raise ValueError(f"Dec must be in [-90, 90] deg, got {self.dec_deg}")
        if not -90.0 <= self.lat_deg <= 90.0:
            raise ValueError(f"latitude must be in [-90, 90] deg, got {self.lat_deg}")
        if not -180.0 <= self.lon_deg <= 360.0:
            raise ValueError(f"longitude out of range: {self.lon_deg}")


def mid_exposure_mjd(mjd_start: float, exptime_s: float) -> float:
    """Return the mid-exposure MJD given the start MJD and exposure time (s)."""
    if exptime_s < 0:
        raise ValueError(f"exposure time must be >= 0, got {exptime_s}")
    return mjd_start + 0.5 * exptime_s / 86_400.0


def barycentric_correction(obs: Observation) -> float:
    """Barycentric radial-velocity correction in km/s (astropy/ERFA)."""
    location = EarthLocation.from_geodetic(lon=obs.lon_deg * u.deg, lat=obs.lat_deg * u.deg, height=obs.alt_m * u.m)
    target = SkyCoord(ra=obs.ra_deg * u.deg, dec=obs.dec_deg * u.deg, frame="icrs")
    time = Time(obs.mjd_mid, format="mjd", scale="utc")
    corr = target.radial_velocity_correction(kind="barycentric", obstime=time, location=location)
    return float(corr.to_value(u.km / u.s))


def apply_correction(rv_topo: float, v_corr: float) -> float:
    """Combine a measured velocity with a barycentric correction (to first order in v/c)."""
    return rv_topo + v_corr + rv_topo * v_corr / C_KMS


def observation_from_eso_header(header: fits.Header) -> Observation:
    """Build an :class:`Observation` from an ESO primary header.

    Uses ``RA``, ``DEC`` (deg), ``MJD-OBS`` + ``EXPTIME/2`` (or ``MJD-MID``,
    written by ``combine``) and the ``ESO TEL GEO*`` site keywords.

    Raises:
        MissingKeywordError: Listing every missing keyword.
    """
    missing = [key for key in _ESO_KEYS.values() if key not in header]
    if missing:
        raise MissingKeywordError(f"missing keyword(s): {', '.join(missing)}")
    if header.get("BARYFRM", False):
        log.warning("This spectrum was already shifted to the barycentric frame (BARYFRM = T).")
    values = {name: float(header[key]) for name, key in _ESO_KEYS.items()}
    if "MJD-MID" in header:  # combined spectra: flux-weighted mid-exposure time
        mjd_mid = float(header["MJD-MID"])
    else:
        exptime = header.get("EXPTIME")
        if exptime is None:
            log.warning("EXPTIME missing: using MJD-OBS (exposure start) as observation time.")
            exptime = 0.0
        mjd_mid = mid_exposure_mjd(values["mjd"], float(exptime))
    return Observation(
        ra_deg=values["ra"],
        dec_deg=values["dec"],
        mjd_mid=mjd_mid,
        lon_deg=values["lon"],
        lat_deg=values["lat"],
        alt_m=values["alt"],
    )
