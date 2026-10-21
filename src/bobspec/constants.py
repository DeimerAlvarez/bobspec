"""Physical constants and package-wide defaults."""

from typing import Final

C_KMS: Final[float] = 299_792.458
"""Speed of light in km/s."""

FWHM_TO_SIGMA: Final[float] = 1.0 / 2.354_820_045_030_949
"""Conversion factor FWHM -> Gaussian sigma (1 / (2 sqrt(2 ln 2)))."""

MAD_TO_SIGMA: Final[float] = 1.482_602_218_505_602
"""Scale factor turning a median absolute deviation into a Gaussian sigma."""

Region = tuple[float, float]
"""A closed wavelength interval ``(start, end)`` in Angstrom."""

DEFAULT_TELLURIC_REGIONS: Final[tuple[Region, ...]] = (
    (6860.0, 6950.0),  # O2 Fraunhofer B band
    (7590.0, 7700.0),  # O2 Fraunhofer A band
)
"""Observed-frame regions excluded from continuum and CCF fits by default."""
