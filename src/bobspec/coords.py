"""Conversion between decimal-degree and sexagesimal equatorial coordinates."""

from __future__ import annotations

from dataclasses import dataclass

from astropy import units as u
from astropy.coordinates import Angle, SkyCoord


@dataclass(frozen=True, slots=True)
class Coordinates:
    """Equatorial coordinates in both representations."""

    ra_deg: float
    dec_deg: float
    ra_hms: str
    dec_dms: str

    def as_row(self) -> dict[str, float | str]:
        """Return a flat mapping for result tables."""
        return {"ra_deg": self.ra_deg, "dec_deg": self.dec_deg, "ra_hms": self.ra_hms, "dec_dms": self.dec_dms}


def _as_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def parse_coordinates(ra: str, dec: str, *, precision: int = 3) -> Coordinates:
    """Parse RA/Dec given either both in decimal degrees or both sexagesimal.

    Decimal values accept any float syntax (``+12.3``, ``1.875e2``).
    Sexagesimal RA is in hours (``12:30:00`` or ``12h30m00s``) and Dec in
    degrees (``-12:18:00`` or ``-12d18m00s``).

    Raises:
        ValueError: For mixed formats, unparsable strings or out-of-range values.
    """
    ra_f, dec_f = _as_float(ra), _as_float(dec)
    if (ra_f is None) != (dec_f is None):
        raise ValueError("give RA and Dec both in decimal degrees or both in sexagesimal format")
    if ra_f is not None and dec_f is not None:
        if not 0.0 <= ra_f < 360.0:
            raise ValueError(f"RA must be in [0, 360) deg, got {ra_f}")
        if not -90.0 <= dec_f <= 90.0:
            raise ValueError(f"Dec must be in [-90, 90] deg, got {dec_f}")
        coord = SkyCoord(ra=ra_f * u.deg, dec=dec_f * u.deg)
    else:
        ra_angle = Angle(ra, unit=u.hourangle)  # Angle does not wrap, so it can be checked
        if not 0.0 <= ra_angle.hour < 24.0:
            raise ValueError(f"RA must be in [0, 24) h, got {ra}")
        coord = SkyCoord(ra=ra_angle, dec=Angle(dec, unit=u.deg))
    return Coordinates(
        ra_deg=float(coord.ra.deg),
        dec_deg=float(coord.dec.deg),
        ra_hms=str(coord.ra.to_string(unit=u.hour, sep=":", precision=precision, pad=True)),
        dec_dms=str(coord.dec.to_string(unit=u.deg, sep=":", precision=precision, pad=True, alwayssign=True)),
    )
