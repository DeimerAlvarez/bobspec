"""Continuum estimation and normalisation.

All fitters are pure functions returning the continuum on the full input
grid; plotting and interaction live in :mod:`bobspec.interactive`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import Chebyshev
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline, make_interp_spline
from scipy.ndimage import gaussian_filter1d, percentile_filter

from bobspec.constants import MAD_TO_SIGMA
from bobspec.io import Spec

Curve = Callable[[np.ndarray], np.ndarray]
"""A fitted continuum: maps wavelengths to continuum flux."""

ContinuumModel = Callable[[np.ndarray, np.ndarray], Curve]
"""Fits ``(wave, flux)`` samples and returns the fitted :data:`Curve`."""


@dataclass(frozen=True, slots=True)
class ContinuumFit:
    """Result of :func:`iterative_fit`.

    Attributes:
        continuum: Continuum evaluated on the full input grid.
        used: Mask of the pixels used in the final fit.
        n_iter: Number of fit/clip iterations performed.
        converged: True if the clipping mask stopped changing.
    """

    continuum: np.ndarray = field(repr=False)
    used: np.ndarray = field(repr=False)
    n_iter: int
    converged: bool


def chebyshev_model(deg: int) -> ContinuumModel:
    """Return a least-squares Chebyshev fitter of degree ``deg`` (>= 1)."""
    if deg < 1:
        raise ValueError(f"degree must be >= 1, got {deg}")

    def fit(x: np.ndarray, y: np.ndarray) -> Curve:
        if x.size < deg + 2:
            raise ValueError(f"{x.size} points are too few for a degree-{deg} polynomial")
        return Chebyshev.fit(x, y, deg)

    return fit


def spline_model(k: int, n_knots: int = 20) -> ContinuumModel:
    """Return a least-squares spline fitter of degree ``k`` with quantile-placed knots.

    Interior knots sit at quantiles of the *retained* wavelengths, so they
    never fall inside masked gaps (Schoenberg-Whitney conditions). If the LSQ
    spline still fails, a smoothing spline with a MAD-based smoothing factor
    is used.
    """
    if not 1 <= k <= 5:
        raise ValueError(f"spline degree must be in 1..5, got {k}")
    if n_knots < 1:
        raise ValueError(f"n_knots must be >= 1, got {n_knots}")

    def fit(x: np.ndarray, y: np.ndarray) -> Curve:
        if x.size < k + n_knots + 2:
            raise ValueError(f"{x.size} points are too few for {n_knots} knots and k={k}")
        knots = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_knots + 2)[1:-1]))
        knots = knots[(knots > x[0]) & (knots < x[-1])]
        try:
            return LSQUnivariateSpline(x, y, knots, k=k)
        except ValueError:
            sigma = MAD_TO_SIGMA * np.median(np.abs(y - np.median(y)))
            return UnivariateSpline(x, y, k=k, s=x.size * sigma**2)

    return fit


def iterative_fit(
    wave: np.ndarray,
    flux: np.ndarray,
    model: ContinuumModel,
    *,
    sigma_low: float,
    sigma_up: float,
    mask: np.ndarray | None = None,
    max_iter: int = 20,
    rtol: float = 1e-4,
) -> ContinuumFit:
    """Fit the continuum with asymmetric sigma clipping of relative residuals.

    At each iteration the model is fitted to the retained pixels, residuals
    ``flux / continuum - 1`` are computed, and pixels outside
    ``[median - sigma_low * s, median + sigma_up * s]`` (``s`` = MAD sigma)
    are rejected. Iteration stops when the mask no longer changes or the
    continuum changes by less than ``rtol`` (masks can cycle by a few pixels).

    Args:
        wave: Wavelengths (increasing).
        flux: Fluxes; non-finite pixels are always ignored.
        model: Fitter, e.g. :func:`chebyshev_model` or :func:`spline_model`.
        sigma_low: Rejection threshold below the continuum (> 0).
        sigma_up: Rejection threshold above the continuum (> 0).
        mask: Pixels allowed in the fit (False = excluded, e.g. tellurics).
        max_iter: Maximum number of iterations (>= 1).
        rtol: Convergence threshold on the relative continuum change.

    Raises:
        ValueError: For invalid thresholds or when too few pixels remain.
    """
    if not (sigma_low > 0 and sigma_up > 0):
        raise ValueError(f"sigma thresholds must be positive, got {sigma_low}, {sigma_up}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    allowed = np.isfinite(flux) if mask is None else (np.isfinite(flux) & mask)
    used = allowed.copy()
    previous = np.full(wave.shape, np.nan)
    for iteration in range(1, max_iter + 1):
        continuum = np.asarray(model(wave[used], flux[used])(wave), dtype=float)
        if np.nanmax(np.abs(continuum / previous - 1.0), initial=np.inf) < rtol:
            return ContinuumFit(continuum, used, iteration, True)
        previous = continuum
        with np.errstate(divide="ignore", invalid="ignore"):
            resid = np.where(continuum > 0, flux / continuum - 1.0, np.nan)
        r = resid[used]
        med = np.nanmedian(r)
        scale = max(MAD_TO_SIGMA * np.nanmedian(np.abs(r - med)), 1e-12)
        new = allowed & (resid > med - sigma_low * scale) & (resid < med + sigma_up * scale)
        if np.array_equal(new, used):
            return ContinuumFit(continuum, used, iteration, True)
        used = new
    return ContinuumFit(continuum, used, max_iter, False)


def anchor_curve(x: Sequence[float], y: Sequence[float], k: int = 3) -> Curve:
    """Interpolating spline of degree ``k`` through user-selected anchor points.

    Duplicate abscissae are averaged. Outside the anchors the curve is held
    constant at the end values instead of being extrapolated.

    Raises:
        ValueError: If ``k`` is outside 1..5 or there are fewer than ``k + 1``
            distinct anchors.
    """
    if not 1 <= k <= 5:
        raise ValueError(f"spline degree must be in 1..5, got {k}")
    xs, inverse = np.unique(np.asarray(x, float), return_inverse=True)
    ys = np.bincount(inverse, weights=np.asarray(y, float)) / np.bincount(inverse)
    if xs.size < k + 1:
        raise ValueError(f"need at least {k + 1} distinct anchor points for k={k}, got {xs.size}")
    spline = make_interp_spline(xs, ys, k=k)
    return lambda w: spline(np.clip(w, xs[0], xs[-1]))


def smooth_continuum(
    spec: Spec,
    window_aa: float,
    *,
    percentile: float = 90.0,
    sigma_pix: float = 2.0,
    clip_low: float = 3.0,
    clip_high: float = 5.0,
    max_iter: int = 2,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Upper-envelope continuum: running percentile followed by a Gaussian filter.

    Clipped pixels are replaced by the current continuum before the next
    iteration. Pixels outside ``mask`` (or non-finite) are skipped and filled
    by linear interpolation of the final continuum.

    Args:
        spec: Input spectrum.
        window_aa: Width of the running-percentile window (Angstrom, > 0).
        percentile: Percentile of the running window (upper envelope).
        sigma_pix: Gaussian smoothing applied to the envelope (pixels).
        clip_low: Clipping threshold below the continuum (MAD sigmas).
        clip_high: Clipping threshold above the continuum (MAD sigmas).
        max_iter: Number of envelope/clip iterations (>= 1).
        mask: Pixels allowed in the estimate.

    Returns:
        The continuum on ``spec.wave``.
    """
    if not window_aa > 0:
        raise ValueError(f"window must be positive, got {window_aa}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    good = spec.finite if mask is None else (spec.finite & mask)
    x, y = spec.wave[good], spec.flux[good]
    if x.size < 10:
        raise ValueError("fewer than 10 usable pixels")
    size = max(7, round(window_aa / np.median(np.diff(x)))) | 1
    work = y.copy()
    for _ in range(max_iter):
        envelope = percentile_filter(work, percentile=percentile, size=size, mode="nearest")
        cont = gaussian_filter1d(envelope, sigma_pix, mode="nearest")
        cont = np.where(cont > 0, cont, np.median(y))
        rel = y / cont - 1.0
        med = np.median(rel)
        scale = max(MAD_TO_SIGMA * np.median(np.abs(rel - med)), 1e-12)
        keep = (rel > med - clip_low * scale) & (rel < med + clip_high * scale)
        work = np.where(keep, y, cont)
    return np.interp(spec.wave, x, cont)


def normalize(spec: Spec, continuum: np.ndarray, label: str) -> Spec:
    """Divide by ``continuum`` (non-positive continuum -> NaN) and record history."""
    with np.errstate(divide="ignore", invalid="ignore"):
        flux = np.where(continuum > 0, spec.flux / continuum, np.nan)
    return spec.with_flux(flux).with_history(f"continuum normalised ({label})")
