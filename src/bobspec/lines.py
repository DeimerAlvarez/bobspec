"""Single-line measurements: centre, FWHM and equivalent width."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit

from bobspec.constants import FWHM_TO_SIGMA
from bobspec.io import Spec

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LineMeasurement:
    """Measurement of one line between ``w1`` and ``w2``.

    Equivalent widths are in milli-Angstrom and positive for absorption.
    ``ew_int`` integrates ``1 - F / F_c`` with a linear local continuum through
    the two window edges; ``ew_gauss`` is the analytic area of the fitted
    Gaussian, ``depth * sigma * sqrt(2 pi)``. A large difference between them
    signals blends or a poor continuum.
    """

    w1: float
    w2: float
    center: float
    fwhm: float
    depth: float
    ew_int: float
    ew_gauss: float
    fit_ok: bool
    continuum: tuple[float, float] = (np.nan, np.nan)
    """Local continuum at ``w1`` and ``w2`` (flux units of the input)."""

    def profile(self, wave: np.ndarray) -> np.ndarray:
        """Fitted profile in the flux units of the input spectrum."""
        cont = np.interp(wave, [self.w1, self.w2], self.continuum)
        return cont * _absorption(wave, self.depth, self.center, self.fwhm * FWHM_TO_SIGMA)

    def as_row(self) -> dict[str, float | bool]:
        """Return the measurement as a flat mapping (for result tables)."""
        return {
            "w1": self.w1,
            "w2": self.w2,
            "center": self.center,
            "fwhm": self.fwhm,
            "depth": self.depth,
            "ew_int_mA": self.ew_int,
            "ew_gauss_mA": self.ew_gauss,
            "fit_ok": self.fit_ok,
        }


def _absorption(x: np.ndarray, depth: float, mu: float, sigma: float) -> np.ndarray:
    return 1.0 - depth * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def measure_line(
    spec: Spec, w1: float, w2: float, *, resolution: float | None = None, edge_pixels: int = 3
) -> LineMeasurement:
    """Measure the line contained in ``[w1, w2]``.

    The local continuum is the straight line through the medians of the
    ``edge_pixels`` pixels closest to each window edge (more robust than a
    single pixel).

    Args:
        spec: Spectrum (normalised or not).
        w1: One window edge (Angstrom).
        w2: The other window edge (Angstrom).
        resolution: Resolving power used for the FWHM first guess.
        edge_pixels: Pixels averaged at each edge for the continuum.

    Raises:
        ValueError: If the window holds fewer than ``2 * edge_pixels + 3`` finite
            pixels or the continuum is not positive.
    """
    seg = spec.select(w1, w2)
    good = seg.finite
    x, y = seg.wave[good], seg.flux[good]
    if x.size < 2 * edge_pixels + 3:
        raise ValueError(f"window {w1:.3f}-{w2:.3f} A contains only {x.size} finite pixels")
    c1, c2 = np.median(y[:edge_pixels]), np.median(y[-edge_pixels:])
    if not (c1 > 0 and c2 > 0):
        raise ValueError("local continuum is not positive")
    rel = y / np.interp(x, [x[0], x[-1]], [c1, c2])
    ew_int = float(trapezoid(1.0 - rel, x) * 1000.0)

    imin = int(np.argmin(rel))
    width = (x[imin] / resolution) * FWHM_TO_SIGMA if resolution else (x[-1] - x[0]) / 6.0
    step = float(np.median(np.diff(x)))
    lower, upper = [-10.0, x[0], step / 4.0], [1.0, x[-1], x[-1] - x[0]]
    p0 = np.clip([1.0 - rel[imin], x[imin], width], lower, upper)
    try:
        (depth, mu, sigma), _ = curve_fit(_absorption, x, rel, p0=p0, bounds=(lower, upper), maxfev=10_000)
        fit_ok = True
    except (RuntimeError, ValueError) as exc:
        log.warning("Gaussian fit failed in %.3f-%.3f A: %s", w1, w2, exc)
        depth = mu = sigma = np.nan
        fit_ok = False
    return LineMeasurement(
        w1=float(x[0]),
        w2=float(x[-1]),
        center=float(mu),
        fwhm=float(sigma / FWHM_TO_SIGMA),
        depth=float(depth),
        ew_int=ew_int,
        ew_gauss=float(depth * sigma * np.sqrt(2 * np.pi) * 1000.0),
        fit_ok=fit_ok,
        continuum=(float(c1), float(c2)),
    )
