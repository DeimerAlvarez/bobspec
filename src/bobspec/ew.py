"""Automatic equivalent widths following DAOSPEC and 4DAO.

The algorithm follows the published description of DAOSPEC (Stetson & Pancino
2008, PASP 120, 1332) and of the 4DAO wrapper (Mucciarelli 2013, 4DAO
Cookbook). It is an independent re-implementation, not a translation of the
Fortran code, so results agree within the uncertainties but are not identical.

One *run* (DAOSPEC):

1. a first continuum is fitted (Legendre polynomial or cubic spline) with
   asymmetric clipping;
2. absorption lines are found with a matched filter tuned to the FWHM
   (a Gaussian minus its mean, as in DAOPHOT/FIND);
3. a preliminary radial velocity is the shift that matches the largest number
   of detected lines with the laboratory list;
4. five times: all lines are fitted *simultaneously* (saturated Gaussians with
   a common FWHM, constant or proportional to wavelength), then the
   *effective continuum* is refitted to the spectrum divided by the line model
   with a symmetric robust fit, which statistically absorbs the unresolved
   weak lines; new lines are searched in the residuals;
5. lines are identified with the laboratory list and the RV is computed with
   iterative sigma clipping; EW, its formal error and the quality parameter
   Q (local / global residual rms) are computed for every identified line.

The *4DAO loop* repeats the run, feeding back the output FWHM and narrowing
the RV window to RV +- 5 sigma, until the FWHM converges; a final run keeps
the FWHM fixed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import Literal

import numpy as np
from numpy.polynomial import legendre
from scipy import sparse
from scipy.integrate import trapezoid
from scipy.interpolate import BSpline
from scipy.ndimage import binary_dilation, gaussian_filter1d, median_filter
from scipy.optimize import least_squares

from bobspec.constants import C_KMS, FWHM_TO_SIGMA, MAD_TO_SIGMA
from bobspec.io import Spec
from bobspec.spectrum import resample, to_rest_frame, uniform_grid

log = logging.getLogger(__name__)

MIN_MATCHES = 3
"""Laboratory lines that must match detected lines to accept the preliminary RV."""

BLEND_FRACTION = 0.10
"""A line is blended if its neighbours absorb more than this fraction of its depth within +-1 FWHM ..."""

BLEND_FLOOR = 0.005
"""... and more than this fraction of the continuum."""

RESIDUAL_MIN_SEPARATION = 0.75
"""Lines found in the residuals are added only this many FWHM away from existing lines ..."""

CURVATURE_SMOOTHING = 0.5
"""Sigma of the smoothing used for the curvature test, in units of the line sigma."""

RESIDUAL_ISOLATION = 1.5
"""... and, closer than this many FWHM, only if the curvature of the spectrum peaks there
(:func:`has_curvature_peak`): on the wing of a line wider than the model the residuals show
a dip that is not a line, and adding it would split the line and let the common FWHM
collapse."""

COLLAPSE_RATIO = 0.7
"""Warn when the converged common FWHM is below this fraction of the first guess."""

LIST_MAX_ITER = 10
"""Maximum number of runs when the FWHM relation is derived from the list lines."""

MIN_RV_HALF_WINDOW = 10.0
"""Smallest half width (km/s) of the RV window used after the first iteration."""

ContinuumKind = Literal["legendre", "spline"]
Frame = Literal["obs", "rest"]


# ----------------------------------------------------------------------------- masks
@dataclass(frozen=True, slots=True)
class MaskRegion:
    """Wavelength interval excluded from the analysis.

    ``frame = "obs"`` for features fixed in the observed frame (telluric bands,
    interstellar lines); ``"rest"`` for stellar features, shifted with the RV.
    """

    start: float
    end: float
    frame: Frame = "obs"
    label: str = ""

    def observed(self, rv: float) -> tuple[float, float]:
        """Interval in the observed frame for radial velocity ``rv``."""
        factor = 1.0 + rv / C_KMS if self.frame == "rest" else 1.0
        return self.start * factor, self.end * factor


DEFAULT_MASKS: tuple[MaskRegion, ...] = (
    MaskRegion(5889.0, 5890.5, "obs", "Na D1 interstellar"),
    MaskRegion(5895.0, 5896.5, "obs", "Na D2 interstellar"),
    MaskRegion(5887.0, 5899.0, "rest", "Na D photospheric"),
    MaskRegion(6552.8, 6572.8, "rest", "H alpha"),
    MaskRegion(6860.0, 6930.0, "obs", "O2 B band"),
    MaskRegion(7590.0, 7700.0, "obs", "O2 A band"),
)
"""Default masks: Na D (interstellar and photospheric), H alpha, Fraunhofer B and A bands."""


def read_masks(path: str) -> tuple[MaskRegion, ...]:
    """Read a two-column mask file (observed-frame start and end in Angstrom)."""
    data = np.loadtxt(path, comments="#", ndmin=2, usecols=(0, 1))
    return tuple(MaskRegion(float(a), float(b), "obs", "user") for a, b in data)


def mask_pixels(wave: np.ndarray, masks: Sequence[MaskRegion], rv: float) -> np.ndarray:
    """Boolean array, True where a pixel falls inside any mask."""
    masked = np.zeros(wave.shape, dtype=bool)
    for region in masks:
        lo, hi = sorted(region.observed(rv))
        masked |= (wave >= lo) & (wave <= hi)
    return masked


# ----------------------------------------------------------------------------- line profile
def saturated(g: np.ndarray, rho: float) -> np.ndarray:
    """Saturated Gaussian depth ``g / (1 + g / (1 - rho))`` (DAOSPEC eq. 9)."""
    return g / (1.0 + g / (1.0 - rho))


_X = np.linspace(-12.0, 12.0, 4801)
_E = np.exp(-0.5 * _X**2)


def ew_factor(amp: np.ndarray, rho: float) -> np.ndarray:
    """``EW / sigma`` of a saturated Gaussian of amplitude ``amp`` (vectorised)."""
    amp = np.atleast_1d(np.asarray(amp, float))
    return trapezoid(saturated(amp[:, None] * _E[None, :], rho), _X, axis=1)


def ew_factor_derivative(amp: np.ndarray, rho: float) -> np.ndarray:
    """``d(EW / sigma) / d amp``."""
    amp = np.atleast_1d(np.asarray(amp, float))
    g = amp[:, None] * _E[None, :]
    return trapezoid(_E[None, :] / (1.0 + g / (1.0 - rho)) ** 2, _X, axis=1)


# ----------------------------------------------------------------------------- continuum
def continuum_basis(wave: np.ndarray, kind: ContinuumKind, order: int, knots: int) -> np.ndarray:
    """Design matrix of the continuum: Legendre polynomials or cubic B-splines.

    Legendre and Chebyshev polynomials of the same order span the same functions,
    so a least-squares fit gives the same continuum with either basis.
    """
    x = 2.0 * (wave - wave[0]) / (wave[-1] - wave[0]) - 1.0
    if kind == "legendre":
        return legendre.legvander(x, order)
    inner = np.linspace(-1.0, 1.0, knots + 2)
    t = np.concatenate([[-1.0] * 3, inner, [1.0] * 3])
    return BSpline.design_matrix(np.clip(x, -1.0, 1.0), t, 3).toarray()


def _robust_scale(values: np.ndarray) -> float:
    med = np.median(values)
    return float(max(MAD_TO_SIGMA * np.median(np.abs(values - med)), 1e-12))


def initial_continuum(
    flux: np.ndarray, basis: np.ndarray, valid: np.ndarray, low: float = 1.5, high: float = 3.0, iterations: int = 15
) -> np.ndarray:
    """First continuum: least squares with asymmetric clipping (absorption lines rejected)."""
    use = valid.copy()
    coef = np.linalg.lstsq(basis[use], flux[use], rcond=None)[0]
    for _ in range(iterations):
        model = basis @ coef
        resid = flux - model
        scale = _robust_scale(resid[use])
        new = valid & (resid > -low * scale) & (resid < high * scale)
        if np.array_equal(new, use) or np.count_nonzero(new) <= basis.shape[1]:
            break
        use = new
        coef = np.linalg.lstsq(basis[use], flux[use], rcond=None)[0]
    return basis @ coef


def effective_continuum(
    flux: np.ndarray, line_model: np.ndarray, basis: np.ndarray, valid: np.ndarray, iterations: int = 8
) -> np.ndarray:
    """Effective continuum: robust (Tukey biweight) fit of ``flux = C * line_model``.

    The residuals are balanced around zero, so the flux removed by weak
    unrecognised lines is statistically included in the continuum
    (Stetson & Pancino 2008, Sect. 3.2).
    """
    design = basis * line_model[:, None]
    weights = valid.astype(float)
    coef = np.zeros(basis.shape[1])
    for _ in range(iterations):
        sw = np.sqrt(weights)
        coef = np.linalg.lstsq(design * sw[:, None], flux * sw, rcond=None)[0]
        resid = flux - design @ coef
        scale = _robust_scale(resid[valid])
        u = resid / (4.685 * scale)
        weights = np.where(valid & (np.abs(u) < 1.0), (1.0 - u**2) ** 2, 0.0)
    return basis @ coef


# ----------------------------------------------------------------------------- spikes
SPIKE_SIGMA = 5.0
"""Positive outliers above this many sigma are treated as spikes (cosmic rays, sky residuals)."""

MIN_LINE_FWHM_PIX = 2.5
"""Features narrower than this (pixels) are not considered real lines by the first-guess estimator."""

DEFAULT_RESOLUTION = 20000.0
"""First-guess resolving power when it cannot be estimated from the spectrum."""


def _fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace invalid pixels by linear interpolation of the valid ones."""
    if valid.all() or not valid.any():
        return values.copy()
    idx = np.arange(values.size)
    return np.interp(idx, idx[valid], values[valid])


def find_spikes(values: np.ndarray, valid: np.ndarray, baseline: np.ndarray, max_width: float, grow: int) -> np.ndarray:
    """Positive outliers: cosmic rays and residuals of sky emission lines.

    Pixels more than ``SPIKE_SIGMA`` robust sigmas above ``baseline`` (the
    continuum, or the line model) are flagged; only *narrow* excesses, at most
    ``max_width`` pixels wide at half their height, count as spikes (a sky line
    is about one FWHM wide however bright it is), since wider ones are
    structure of the continuum; excesses more than 50% above the baseline
    (e.g. sky doublets) are spikes whatever their width. The mask is then grown by ``grow`` pixels on each side.
    Absorption lines lie below the baseline and are never flagged.
    """
    filled = _fill(values, valid)
    resid = filled - baseline
    sigma = pixel_noise(resid, valid)
    if not np.isfinite(sigma) or sigma <= 0:
        return np.zeros(values.shape, dtype=bool)
    high = valid & (resid > SPIKE_SIGMA * sigma)
    if not high.any():
        return high
    edges = np.flatnonzero(np.diff(np.concatenate([[0], high.astype(np.int8), [0]])))
    spikes = np.zeros(values.shape, dtype=bool)
    for start, stop in zip(edges[0::2], edges[1::2], strict=True):
        excess = resid[start:stop]
        narrow = np.count_nonzero(excess > 0.5 * excess.max()) <= max_width  # width at half height
        huge = excess.max() > 0.5 * float(np.max(np.abs(baseline[start:stop])))  # > 50% above the baseline
        if narrow or huge:
            spikes[start:stop] = True
    if grow > 0 and spikes.any():
        spikes = binary_dilation(spikes, iterations=grow)
    return spikes & valid


# ----------------------------------------------------------------------------- noise and finder
def correlation_factor(resid: np.ndarray, fwhm_pix: float) -> float:
    """Underestimate of least-squares line errors caused by correlated residuals.

    Rebinning during data reduction correlates neighbouring pixels, and
    unresolved weak lines add structure on the scale of a line. The factor
    compares the scatter of the residuals *weighted by a Gaussian line
    profile* (the weights of the amplitude fit), measured at every position,
    with the value expected for independent pixels; it is clipped at 1.
    """
    r = resid[np.isfinite(resid)]
    sigma_pix = fwhm_pix * FWHM_TO_SIGMA
    half = max(1, int(np.ceil(3.0 * sigma_pix)))
    profile = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_pix) ** 2)
    if r.size < 20 * profile.size:
        return 1.0
    weighted = np.convolve(r - np.mean(r), profile, mode="valid")
    expected = np.std(r) * np.sqrt(np.sum(profile**2))
    return float(max(1.0, np.std(weighted) / expected))


NOISE_FLOOR = 1e-4
"""Smallest noise used for line detection (noiseless synthetic spectra)."""


def pixel_noise(values: np.ndarray, valid: np.ndarray) -> float:
    """Per-pixel noise from second differences (the DER_SNR noise estimator), at least ``NOISE_FLOOR``."""
    v = values[valid]
    if v.size < 5:
        return np.nan
    return float(max(NOISE_FLOOR, 0.6052697 * np.median(np.abs(2.0 * v[2:-2] - v[:-4] - v[4:]))))


def find_lines(
    wave: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    sigma_pix: float,
    noise: float,
    threshold: float = 3.0,
    min_separation_pix: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Find absorption lines with a matched filter (Gaussian minus its mean).

    The filter output estimates, at every pixel, the amplitude of a Gaussian
    of width ``sigma_pix`` sitting on a local constant level, so it ignores
    continuum offsets. Local maxima above ``threshold`` times its noise are
    returned.

    Returns:
        Centres (wavelength, refined with a parabola) and estimated depths.
    """
    half = max(2, int(np.ceil(2.5 * sigma_pix)))
    j = np.arange(-half, half + 1)
    gauss = np.exp(-0.5 * (j / sigma_pix) ** 2)
    kernel = gauss - gauss.mean()
    kernel /= np.sum(kernel * gauss)
    d = np.where(valid, depth, 0.0)
    amp = np.correlate(d, kernel, mode="same")
    coverage = np.correlate(valid.astype(float), np.ones(j.size) / j.size, mode="same")
    amp_noise = noise * np.sqrt(np.sum(kernel**2))
    peak = (amp[1:-1] > amp[:-2]) & (amp[1:-1] >= amp[2:])
    idx = np.flatnonzero(peak) + 1
    idx = idx[(amp[idx] > threshold * amp_noise) & (coverage[idx] > 0.8) & valid[idx]]
    if idx.size == 0:
        return np.empty(0), np.empty(0)
    sep = min_separation_pix if min_separation_pix is not None else 1.0 * sigma_pix * 2.3548 * 0.5
    keep: list[int] = []
    for i in idx[np.argsort(-amp[idx])]:  # strongest first
        if all(abs(i - k) >= sep for k in keep):
            keep.append(i)
    idx = np.sort(np.asarray(keep))
    a0, a1, a2 = amp[idx - 1], amp[idx], amp[idx + 1]
    denom = a0 - 2.0 * a1 + a2
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(denom < 0, 0.5 * (a0 - a2) / denom, 0.0)
    step = np.gradient(wave)[idx]
    return wave[idx] + np.clip(delta, -1.0, 1.0) * step, np.clip(amp[idx], 0.0, None)


# ----------------------------------------------------------------------------- simultaneous fit
def amplitude_from_depth(depth: np.ndarray, rho: float) -> np.ndarray:
    """Gaussian amplitude ``g`` giving central depth ``depth`` after saturation."""
    f = np.clip(depth, 1e-4, 0.95 * (1.0 - rho))
    return f / (1.0 - f / (1.0 - rho))


@dataclass
class LineFitter:
    """Simultaneous least-squares fit of many saturated Gaussians with a common width.

    The width parameter ``w`` is the FWHM (Angstrom) at ``lam_c``; at other
    wavelengths ``FWHM = w * (1 + slope * (lambda - lam_c) / lam_c)``:
    ``slope = 1`` gives a constant resolving power (FWHM proportional to
    wavelength), ``slope = 0`` the same FWHM everywhere, and any other value a
    linear FWHM(lambda) relation (see :func:`fit_fwhm_relation`).
    """

    wave: np.ndarray
    norm: np.ndarray
    valid: np.ndarray
    lam_c: float
    slope: float
    rho: float
    _rows: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rows = np.full(self.wave.size, -1)
        self._rows[self.valid] = np.arange(np.count_nonzero(self.valid))

    def sigma(self, mu: np.ndarray, w: float) -> np.ndarray:
        """Gaussian sigma (Angstrom) of lines centred at ``mu`` for width parameter ``w``."""
        mu = np.asarray(mu, dtype=float)
        factor = 1.0 + self.slope * (mu - self.lam_c) / self.lam_c
        return np.asarray(w * FWHM_TO_SIGMA * factor)

    def _windows(self, mu0: np.ndarray, w_max: float, dmu: np.ndarray) -> list[slice]:
        reach = 5.0 * self.sigma(mu0, w_max) + dmu
        lo = np.searchsorted(self.wave, mu0 - reach)
        hi = np.searchsorted(self.wave, mu0 + reach, side="right")
        return [slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True)]

    def depth(self, mu: np.ndarray, amp: np.ndarray, w: float, windows: list[slice] | None = None) -> np.ndarray:
        """Total line depth ``sum_k f_k`` on the full pixel grid."""
        if windows is None:
            windows = self._windows(mu, w, np.zeros(mu.size))
        total = np.zeros(self.wave.size)
        sig = self.sigma(mu, w)
        for k, sl in enumerate(windows):
            x = (self.wave[sl] - mu[k]) / sig[k]
            total[sl] += saturated(amp[k] * np.exp(-0.5 * x**2), self.rho)
        return total

    def fit(
        self, mu0: np.ndarray, amp0: np.ndarray, w0: float, *, fit_width: bool, max_nfev: int = 60
    ) -> tuple[np.ndarray, np.ndarray, float, bool]:
        """Fit centres, amplitudes and (optionally) the common width.

        Returns:
            Fitted centres, amplitudes, width and the optimiser success flag.
        """
        n = mu0.size
        if n == 0:
            return mu0, amp0, w0, True
        dmu = 0.75 * self.sigma(mu0, w0) / FWHM_TO_SIGMA
        w_lo, w_hi = (0.5 * w0, 2.0 * w0) if fit_width else (w0, w0)
        windows = self._windows(mu0, w_hi, dmu)
        rows_k = [self._rows[sl] for sl in windows]
        y = self.norm[self.valid]
        n_par = 2 * n + (1 if fit_width else 0)

        def unpack(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
            return p[:n], p[n : 2 * n], (p[-1] if fit_width else w0)

        def fun(p: np.ndarray) -> np.ndarray:
            mu, amp, w = unpack(p)
            return y - 1.0 + self.depth(mu, amp, w, windows)[self.valid]

        def jac(p: np.ndarray) -> sparse.csr_matrix:
            mu, amp, w = unpack(p)
            sig = self.sigma(mu, w)
            rows, cols, vals = [], [], []
            for k, sl in enumerate(windows):
                r = rows_k[k]
                use = r >= 0
                if not use.any():
                    continue
                x = (self.wave[sl][use] - mu[k]) / sig[k]
                e = np.exp(-0.5 * x**2)
                g = amp[k] * e
                dfdg = 1.0 / (1.0 + g / (1.0 - self.rho)) ** 2
                rr = r[use]
                rows += [rr, rr]
                cols += [np.full(rr.size, k), np.full(rr.size, n + k)]
                vals += [dfdg * g * x / sig[k], dfdg * e]
                if fit_width:
                    rows.append(rr)
                    cols.append(np.full(rr.size, n_par - 1))
                    vals.append(dfdg * g * x**2 / w)
            return sparse.csr_matrix(
                (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(y.size, n_par)
            )

        lower = np.concatenate([mu0 - dmu, np.zeros(n), [w_lo] if fit_width else []])
        upper = np.concatenate([mu0 + dmu, np.full(n, 1e3), [w_hi] if fit_width else []])
        p0 = np.concatenate([mu0, np.clip(amp0, 1e-6, 1e3 - 1), [w0] if fit_width else []])
        span = upper - lower
        p0 = np.clip(p0, lower + 1e-9 * span, upper - 1e-9 * span)
        result = least_squares(fun, p0, jac=jac, bounds=(lower, upper), method="trf", x_scale="jac", max_nfev=max_nfev)
        mu, amp, w = unpack(result.x)
        return mu, amp, float(w), bool(result.success)

    def amplitude_errors(
        self, mu: np.ndarray, amp: np.ndarray, w: float, resid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Formal amplitude errors and local residual rms of every line.

        Each line is treated as a linear fit of its own amplitude with known
        shape: ``sigma_A^2 = s^2 / sum (dD/dA)^2`` over the pixels within
        +-1.5 FWHM, where ``s^2`` is the residual variance in the same pixels
        with ``n - 2`` degrees of freedom (centre and amplitude were fitted).
        """
        sig = self.sigma(mu, w)
        err = np.full(mu.size, np.nan)
        local = np.full(mu.size, np.nan)
        for k in range(mu.size):
            reach = 1.5 * sig[k] / FWHM_TO_SIGMA
            sl = slice(int(np.searchsorted(self.wave, mu[k] - reach)), int(np.searchsorted(self.wave, mu[k] + reach)))
            use = self.valid[sl]
            if np.count_nonzero(use) < 3:
                continue
            x = (self.wave[sl][use] - mu[k]) / sig[k]
            e = np.exp(-0.5 * x**2)
            dfdg = 1.0 / (1.0 + amp[k] * e / (1.0 - self.rho)) ** 2
            n_pix = int(np.count_nonzero(use))
            s = float(np.sqrt(np.sum(resid[sl][use] ** 2) / max(n_pix - 2, 1)))  # 2 fitted parameters
            local[k] = s
            err[k] = s / np.sqrt(np.sum((dfdg * e) ** 2))
        return err, local


# ----------------------------------------------------------------------------- settings and results
class EWError(ValueError):
    """The equivalent widths could not be measured (e.g. no line identified)."""


@dataclass(frozen=True, slots=True)
class EWSettings:
    """Parameters of the measurement (4DAO/DAOSPEC names in brackets).

    Attributes:
        resolution: First-guess resolving power [FW]; None estimates the first
            FWHM from the spectrum (:func:`estimate_fwhm`).
        fwhm: Fixed FWHM in Angstrom at the central wavelength (no
            optimisation); it varies with wavelength as set by ``scale_fwhm``.
        fwhm_from: Lines that set the common FWHM: ``"all"`` the lines detected
            in the spectrum (DAOSPEC), ``"list"`` the unblended lines of the list
            (:func:`fit_fwhm_relation`).
        order: Order of the continuum polynomial; -1 keeps the continuum at 1 [ORD].
        continuum: ``"legendre"`` polynomial or cubic ``"spline"``.
        knots: Interior knots of the spline continuum.
        scale_fwhm: FWHM proportional to wavelength (constant R) [SC = 1].
        fix_fwhm: Keep the FWHM at the first guess (no optimisation).
        core: Residual core flux in percent [RE].
        vmin, vmax: Radial-velocity search window (km/s).
        rv: Known radial velocity; the search is restricted to rv +- 20 km/s.
        wmin, wmax: Wavelength range; default: the whole spectrum without
            leading/trailing non-positive pixels [WAVE1, WAVE2].
        masks: Excluded regions.
        threshold: Detection threshold of the line finder (sigma).
        tol: FWHM convergence tolerance in pixels [tol].
        max_iter: Maximum number of 4DAO iterations [nimax].
        fwhm_max: Largest acceptable FWHM in pixels [fmax].
        clip: Sigma clipping of the line velocities [VE].
        inner_iter: Continuum / line refinement iterations per run (DAOSPEC: 5).
        correlated_noise: Scale the EW errors by :func:`correlation_factor`.
    """

    resolution: float | None = None
    fwhm: float | None = None
    fwhm_from: Literal["all", "list"] = "all"
    order: int = 5
    continuum: ContinuumKind = "legendre"
    knots: int = 10
    scale_fwhm: bool = True
    fix_fwhm: bool = False
    core: float = 10.0
    vmin: float = -500.0
    vmax: float = 500.0
    rv: float | None = None
    wmin: float | None = None
    wmax: float | None = None
    masks: tuple[MaskRegion, ...] = DEFAULT_MASKS
    threshold: float = 3.0
    tol: float = 0.1
    max_iter: int = 20
    fwhm_max: float = 50.0
    clip: float = 3.0
    inner_iter: int = 5
    correlated_noise: bool = True

    def __post_init__(self) -> None:
        if self.resolution is not None and not self.resolution > 0:
            raise ValueError(f"resolution must be positive, got {self.resolution}")
        if self.fwhm is not None and not self.fwhm > 0:
            raise ValueError(f"fwhm must be positive, got {self.fwhm}")
        if self.fwhm is not None and self.fwhm_from == "list":
            raise ValueError("a fixed fwhm and fwhm_from='list' exclude each other")
        if self.fwhm_from not in ("all", "list"):
            raise ValueError(f"fwhm_from must be 'all' or 'list', got {self.fwhm_from!r}")
        if not 0.0 <= self.core < 100.0:
            raise ValueError(f"core must be in [0, 100) percent, got {self.core}")
        if self.order < -1:
            raise ValueError(f"order must be >= -1, got {self.order}")
        if self.vmax <= self.vmin:
            raise ValueError("vmax must be larger than vmin")


@dataclass(frozen=True, slots=True)
class MeasuredLine:
    """One identified line. EWs in milli-Angstrom, rest frame."""

    wave: float
    code: float
    ew: float
    ew_err: float
    fwhm: float
    """FWHM of the Gaussian underlying the saturated profile of this line, from
    its own fit (Angstrom, rest frame): the line width without saturation
    broadening, so ``wavelength / fwhm`` measures the resolution of the spectrum."""
    resolution: float
    """``wavelength / fwhm``."""
    q: float
    wave_fit: float
    rv: float
    fwhm_half_depth: float = np.nan
    """Width of the fitted profile at half depth (Angstrom, rest frame); equal to
    ``fwhm`` for weak lines, broader for saturated ones."""
    fit_wave: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    """Rest-frame wavelengths of the window (+-2 FWHM) where the line was fitted on its own."""
    fit_flux: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    """Normalised profile of this line alone, from its own fit, in that window."""
    fit_neighbours: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    """Normalised model of the neighbouring lines (held fixed in the fit) in that window."""
    width_fitted: bool = False
    """True if ``fwhm`` comes from the line's own fit (False: from the common width)."""
    blended: bool = False
    """True if neighbouring lines absorb, within +-1 FWHM, more than ``BLEND_FRACTION``
    of the line's depth (and at least ``BLEND_FLOOR`` of the continuum)."""


@dataclass(frozen=True, slots=True)
class RunResult:
    """Outcome of one DAOSPEC-like run."""

    fwhm: float  # Angstrom at the central wavelength
    rv: float
    rv_sigma: float
    lines: list[MeasuredLine]
    n_detected: int
    residual_pct: float
    noise_factor: float
    """Ratio of the residual noise averaged over one FWHM to its white-noise expectation (>= 1)."""
    wave: np.ndarray = field(repr=False)
    flux: np.ndarray = field(repr=False)
    continuum: np.ndarray = field(repr=False)
    model: np.ndarray = field(repr=False)
    missing: np.ndarray = field(repr=False)
    missing_codes: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    """Species codes of the ``missing`` laboratory lines."""

    @property
    def rv_error(self) -> float:
        """Error of the mean RV: ``rv_sigma / sqrt(N - 1)`` (NaN with fewer than two lines)."""
        n = len(self.lines)
        return self.rv_sigma / float(np.sqrt(n - 1)) if n > 1 else float("nan")


@dataclass(frozen=True, slots=True)
class FwhmModel:
    """The common FWHM used for the equivalent widths.

    ``FWHM(lambda) = fwhm * (1 + slope * (lambda - lam_c) / lam_c)`` in the
    observed frame (see :class:`LineFitter`).
    """

    fwhm: float
    slope: float
    lam_c: float
    source: Literal["all", "list", "fixed"]
    used: tuple[float, ...] = ()
    """Laboratory wavelengths of the lines that defined the relation (source "list")."""
    n_candidates: int = 0
    """Unblended lines with their own width fit, before clipping (source "list")."""

    def at(self, wave_obs: np.ndarray) -> np.ndarray:
        """FWHM (Angstrom) at observed wavelengths ``wave_obs``."""
        return self.fwhm * (1.0 + self.slope * (np.asarray(wave_obs, float) - self.lam_c) / self.lam_c)


@dataclass(frozen=True, slots=True)
class EWResult:
    """Final result of :func:`measure_ew` (4DAO loop + final fixed-FWHM run)."""

    run: RunResult
    convergence: int
    """1 converged (or fixed FWHM), 0 too many iterations, -1 FWHM cycling, -2 FWHM out of range."""
    history: list[tuple[float, float, float, float, float]]
    """Per iteration: FWHM in, FWHM out (pixels), RV window start, RV, RV window end."""
    step: float
    core: float
    fwhm_model: FwhmModel
    """How the common FWHM used for the EWs was set."""

    @property
    def fwhm_pix(self) -> float:
        """Final FWHM in pixels at the central wavelength."""
        return self.run.fwhm / self.step

    @property
    def resolution(self) -> float:
        """Resolving power implied by the final FWHM."""
        return float(np.mean(self.run.wave[[0, -1]]) / self.run.fwhm)

    def normalized_rest(self, name: str = "") -> Spec:
        """Spectrum divided by the effective continuum and shifted to rest frame."""
        norm = Spec(self.run.wave, self.run.flux / self.run.continuum, name=name)
        return to_rest_frame(norm, self.run.rv).with_history(
            f"ew: effective continuum, RV = {self.run.rv:.3f} km/s, FWHM = {self.fwhm_pix:.3f} pix"
        )


# ----------------------------------------------------------------------------- preparation
def prepare(spec: Spec, wmin: float | None, wmax: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Select the range, trim non-positive edges and make the grid uniform."""
    part = spec.select(spec.wave[0] if wmin is None else wmin, spec.wave[-1] if wmax is None else wmax)
    good = np.isfinite(part.flux) & (part.flux > 0)
    if np.count_nonzero(good) < 50:
        raise EWError("fewer than 50 valid pixels in the selected range")
    first, last = np.flatnonzero(good)[[0, -1]]
    part = part.on_grid(part.wave[first : last + 1], part.flux[first : last + 1])
    if not part.is_uniform(rtol=1e-3):
        step = float(np.median(np.diff(part.wave)))
        log.warning("Non-uniform wavelength grid: resampled to a constant step of %.5f A.", step)
        part = resample(part, uniform_grid(part.wave[0], part.wave[-1], step))
    return part.wave, part.flux


# ----------------------------------------------------------------------------- radial velocity
def preliminary_rv(
    mu: np.ndarray, lab: np.ndarray, vmin: float, vmax: float, tol_kms: float
) -> tuple[float, int, float] | None:
    """Shift matching the largest number of detected lines with the laboratory list.

    Returns:
        The velocity, the number of matches and the number expected by chance
        (all line pairs spread uniformly over the window), or None.
    """
    if mu.size == 0 or lab.size == 0:
        return None
    v = (C_KMS * (mu[None, :] / lab[:, None] - 1.0)).ravel()
    v = np.sort(v[(v >= vmin) & (v <= vmax)])
    if v.size == 0:
        return None
    counts = np.searchsorted(v, v + 2.0 * tol_kms, side="right") - np.arange(v.size)
    best = int(np.argmax(counts))
    expected = v.size * 2.0 * tol_kms / (vmax - vmin)
    return float(np.median(v[best : best + counts[best]])), int(counts[best]), float(expected)


def identify(
    mu: np.ndarray, lab: np.ndarray, rv0: float, tol_kms: float, clip: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Match laboratory lines to fitted lines and compute the RV with sigma clipping.

    Returns:
        Indices of matched laboratory lines, indices of the corresponding fitted
        lines, the RV and the line-to-line dispersion.
    """
    order = np.argsort(mu)
    mu_s = mu[order]
    pred = lab * (1.0 + rv0 / C_KMS)
    if mu_s.size == 0 or lab.size == 0:
        return np.empty(0, int), np.empty(0, int), rv0, 0.0
    if mu_s.size == 1:
        near = np.zeros(lab.size, dtype=int)
    else:
        pos = np.clip(np.searchsorted(mu_s, pred), 1, mu_s.size - 1)
        left, right = mu_s[pos - 1], mu_s[pos]
        near = np.where(np.abs(left - pred) < np.abs(right - pred), pos - 1, pos)
    v = C_KMS * (mu_s[near] / lab - 1.0)
    use = np.abs(v - rv0) <= tol_kms
    for target in np.unique(near[use]):  # one fitted line cannot match two laboratory lines
        same = np.flatnonzero(use & (near == target))
        if same.size > 1:
            use[same[np.argsort(np.abs(v[same] - rv0))[1:]]] = False
    rv, sigma = rv0, 0.0
    for _ in range(20):
        if np.count_nonzero(use) == 0:
            break
        rv = float(np.mean(v[use]))
        sigma = float(np.std(v[use], ddof=1)) if np.count_nonzero(use) > 1 else 0.0
        new = use & (np.abs(v - rv) <= max(clip * sigma, 1e-6)) if np.count_nonzero(use) >= 3 else use
        if np.array_equal(new, use):
            break
        use = new
    lab_idx = np.flatnonzero(use)
    return lab_idx, order[near[lab_idx]], rv, sigma


# ----------------------------------------------------------------------------- one run
def _detect(
    fitter: LineFitter,
    depth: np.ndarray,
    valid: np.ndarray,
    w: float,
    noise: float,
    threshold: float,
    step: float,
    segment: float = 300.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Line finder applied in segments, each with the FWHM of its central wavelength."""
    wave = fitter.wave
    edges = np.arange(wave[0], wave[-1] + segment, segment)
    centres, depths = [], []
    for lo, hi in pairwise(edges):
        inside = (wave >= lo) & (wave < hi)
        if np.count_nonzero(inside) < 20:
            continue
        sigma_pix = float(fitter.sigma(np.array([0.5 * (lo + hi)]), w)[0] / step)
        mu, dep = find_lines(wave, depth, valid, sigma_pix, noise, threshold)
        sel = (mu >= lo) & (mu < hi)
        centres.append(mu[sel])
        depths.append(dep[sel])
    if not centres:
        return np.empty(0), np.empty(0)
    return np.concatenate(centres), np.concatenate(depths)


def has_curvature_peak(
    wave: np.ndarray,
    norm: np.ndarray,
    valid: np.ndarray,
    centres: np.ndarray,
    half_width: np.ndarray,
    sigma_pix: float,
) -> np.ndarray:
    """True where the curvature of the line depth has its own positive peak near a centre.

    The curvature is ``-d^2(depth)/d(pixel)^2`` of the spectrum smoothed with a
    Gaussian of half the line sigma (``gaussian_filter1d`` with ``order=2``):
    it peaks at the centre of every line, also of a line seen only as a shoulder
    of a blend (as in DAOSPEC's second-derivative line finder), while on the wing
    of a single line it has a minimum at about 0.74 FWHM from the centre, where a
    model that is too narrow leaves residual dips.
    """
    depth = 1.0 - _fill(norm, valid)
    curvature = -gaussian_filter1d(depth, max(CURVATURE_SMOOTHING * sigma_pix, 0.7), order=2, mode="nearest")
    inner = np.arange(1, wave.size - 1)
    peak = (
        (curvature[inner] > curvature[inner - 1]) & (curvature[inner] >= curvature[inner + 1]) & (curvature[inner] > 0)
    )
    peaks = wave[inner[peak]]
    if peaks.size == 0:
        return np.zeros(np.shape(centres), dtype=bool)
    pos = np.clip(np.searchsorted(peaks, centres), 1, max(peaks.size - 1, 1))
    left = np.abs(peaks[pos - 1] - centres)
    right = np.abs(peaks[np.minimum(pos, peaks.size - 1)] - centres)
    return np.minimum(left, right) <= half_width


def _prune(
    fitter: LineFitter,
    mu: np.ndarray,
    amp: np.ndarray,
    w: float,
    norm: np.ndarray,
    min_significance: float = 2.0,
    kappa: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop insignificant lines and, of two lines closer than half a FWHM, the weaker.

    ``kappa`` (:func:`correlation_factor`) inflates the amplitude errors when
    the noise is correlated, so that noise bumps are not kept as weak lines.
    """
    resid = norm - 1.0 + fitter.depth(mu, amp, w)
    err, _ = fitter.amplitude_errors(mu, amp, w, resid)
    err = err * kappa
    keep = (amp > 1e-4) & ~(amp < min_significance * np.nan_to_num(err, nan=np.inf))
    mu, amp = mu[keep], amp[keep]
    order = np.argsort(mu)
    mu, amp = mu[order], amp[order]
    fwhm = fitter.sigma(mu, w) / FWHM_TO_SIGMA
    drop = np.zeros(mu.size, dtype=bool)
    for k in range(mu.size - 1):
        if not drop[k] and mu[k + 1] - mu[k] < 0.5 * fwhm[k]:
            drop[k if amp[k] < amp[k + 1] else k + 1] = True
    return mu[~drop], amp[~drop]


def run_once(
    wave: np.ndarray,
    flux: np.ndarray,
    lab_wave: np.ndarray,
    lab_code: np.ndarray,
    w: float,
    *,
    fit_width: bool,
    window: tuple[float, float],
    settings: EWSettings,
    measure_widths: bool = False,
    slope: float | None = None,
) -> RunResult:
    """One DAOSPEC-like run with first-guess FWHM ``w`` (Angstrom at the centre).

    With ``measure_widths`` every identified line is also refitted on its own
    to measure its FWHM (:func:`line_width`). ``slope`` sets how the common
    FWHM varies with wavelength (see :class:`LineFitter`); by default 1
    (constant R) or 0 with ``settings.scale_fwhm = False``.
    """
    s = settings
    rho = s.core / 100.0
    if slope is None:
        slope = 1.0 if s.scale_fwhm else 0.0
    step = float(np.median(np.diff(wave)))
    lam_c = float(0.5 * (wave[0] + wave[-1]))
    basis = continuum_basis(wave, s.continuum, s.order, s.knots) if s.order >= 0 else None
    base_valid = np.isfinite(flux) & (flux > 0)
    fwhm_pix = float(w / step)
    grow = max(1, round(0.5 * fwhm_pix))
    first = initial_continuum(flux, basis, base_valid) if basis is not None else np.ones_like(flux)
    base_valid &= ~find_spikes(flux, base_valid, first, 1.5 * fwhm_pix, grow)
    rv_guess = s.rv if s.rv is not None else 0.5 * (window[0] + window[1])
    valid = base_valid & ~mask_pixels(wave, s.masks, rv_guess)

    continuum = initial_continuum(flux, basis, valid) if basis is not None else np.ones_like(flux)
    norm = flux / continuum
    noise = pixel_noise(norm, valid)
    fitter = LineFitter(wave, norm, valid, lam_c, slope, rho)
    mu, depth = _detect(fitter, 1.0 - norm, valid, w, noise, s.threshold, step)
    fwhm_kms = C_KMS * w / lam_c
    tol_kms = max(0.25 * fwhm_kms, 1.0)
    # strongest detections only: noise dips in low-SNR spectra would add chance coincidences
    reach = (1.0 + window[0] / C_KMS, 1.0 + window[1] / C_KMS)
    n_lab = int(np.count_nonzero((lab_wave * reach[1] > wave[0]) & (lab_wave * reach[0] < wave[-1])))
    strongest = np.argsort(-depth)[: max(20, 3 * n_lab)]
    pre = preliminary_rv(mu[strongest], lab_wave, window[0], window[1], tol_kms)
    wide = window[1] - window[0] > 20.0 * tol_kms  # chance coincidences matter only in a wide search
    chance = pre[2] + 3.0 * np.sqrt(pre[2]) if pre is not None and wide else 0.0
    if pre is None or pre[1] < max(MIN_MATCHES, chance):
        raise EWError(
            f"no laboratory line matches the detected lines in the RV window {window[0]:.0f}..{window[1]:.0f} km/s"
            " (too few matches compared with chance coincidences)"
        )
    rv0 = pre[0]
    valid = base_valid & ~mask_pixels(wave, s.masks, rv0)
    amp = amplitude_from_depth(depth, rho)

    for iteration in range(s.inner_iter):
        fitter = LineFitter(wave, norm, valid, lam_c, slope, rho)
        mu, amp, w, _ = fitter.fit(mu, amp, w, fit_width=fit_width)
        resid_now = norm - 1.0 + fitter.depth(mu, amp, w)
        kappa_now = correlation_factor(resid_now[valid], w / step) if s.correlated_noise else 1.0
        mu, amp = _prune(fitter, mu, amp, w, norm, kappa=kappa_now)
        model = 1.0 - fitter.depth(mu, amp, w)
        if basis is not None:
            continuum = effective_continuum(flux, model, basis, valid)
        norm = flux / continuum
        if iteration < s.inner_iter - 2:  # look for lines missed so far
            noise = pixel_noise(norm - model, valid) * kappa_now  # correlated noise: higher detection level
            new_mu, new_depth = _detect(fitter, model - norm, valid, w, noise, s.threshold, step)
            if new_mu.size:
                fwhm_new = fitter.sigma(new_mu, w) / FWHM_TO_SIGMA
                dist = np.min(np.abs(new_mu[:, None] - mu[None, :]), axis=1) if mu.size else np.inf
                near = dist <= RESIDUAL_ISOLATION * fwhm_new  # close to a line already in the model
                sigma_pix = float(np.median(fwhm_new)) * FWHM_TO_SIGMA / step
                shoulder = has_curvature_peak(wave, norm, valid, new_mu, 0.25 * fwhm_new, sigma_pix)
                add = (dist > RESIDUAL_MIN_SEPARATION * fwhm_new) & (~near | shoulder)
                mu = np.concatenate([mu, new_mu[add]])
                amp = np.concatenate([amp, amplitude_from_depth(new_depth[add], rho)])

    fitter = LineFitter(wave, norm, valid, lam_c, slope, rho)
    mu, amp, w, _ = fitter.fit(mu, amp, w, fit_width=fit_width)
    model = 1.0 - fitter.depth(mu, amp, w)
    resid = norm - model
    rms = float(np.sqrt(np.mean(resid[valid] ** 2)))
    amp_err, local = fitter.amplitude_errors(mu, amp, w, resid)
    kappa = correlation_factor(resid[valid], w / step) if s.correlated_noise else 1.0
    amp_err = amp_err * kappa
    sig = fitter.sigma(mu, w)

    in_range = (lab_wave * (1 + rv0 / C_KMS) > wave[0]) & (lab_wave * (1 + rv0 / C_KMS) < wave[-1])
    lab_w, lab_c = lab_wave[in_range], lab_code[in_range]
    lab_idx, fit_idx, rv, rv_sigma = (
        identify(mu, lab_w, rv0, tol_kms, s.clip) if mu.size else (np.empty(0, int), np.empty(0, int), rv0, 0.0)
    )
    to_rest = 1.0 / (1.0 + rv / C_KMS)
    total_depth = 1.0 - model
    lines = []
    for j, k in zip(lab_idx, fit_idx, strict=True):
        centre_k, sigma_k, amp_k = float(mu[k]), float(sig[k]), float(amp[k])  # joint fit (common width)
        width_fitted = False
        if measure_widths:
            own_centre, own_sigma, own_amp = line_width(fitter, mu, amp, w, int(k), total_depth)
            if np.isfinite(own_sigma):
                centre_k, sigma_k, amp_k = own_centre, own_sigma, own_amp
                width_fitted = True
        half_window = 2.0 * float(sig[k]) / FWHM_TO_SIGMA
        win = (wave >= mu[k] - half_window) & (wave <= mu[k] + half_window)
        own_joint = saturated(amp[k] * np.exp(-0.5 * ((wave[win] - mu[k]) / sig[k]) ** 2), rho)
        own_fit = saturated(amp_k * np.exp(-0.5 * ((wave[win] - centre_k) / sigma_k) ** 2), rho)
        fit_flux = 1.0 - own_fit  # the line alone, relative to the continuum
        fit_neighbours = 1.0 - (total_depth[win] - own_joint)  # the other lines, held fixed
        core = np.abs(wave[win] - mu[k]) <= float(sig[k]) / FWHM_TO_SIGMA  # +-1 FWHM
        own_depth = float(np.max(own_joint, initial=0.0))
        neighbour_depth = float(np.max(1.0 - fit_neighbours[core], initial=0.0))
        blended = neighbour_depth > max(BLEND_FRACTION * own_depth, BLEND_FLOOR)
        fwhm_k = sigma_k / FWHM_TO_SIGMA
        half_k = profile_fwhm(amp_k, sigma_k, rho)
        ew = sig[k] * ew_factor(amp[k], rho)[0] * to_rest * 1000.0
        ew_err = sig[k] * ew_factor_derivative(amp[k], rho)[0] * amp_err[k] * to_rest * 1000.0
        lines.append(
            MeasuredLine(
                wave=float(lab_w[j]),
                code=float(lab_c[j]),
                ew=float(ew),
                ew_err=float(ew_err),
                fwhm=float(fwhm_k * to_rest),
                resolution=float(mu[k] / fwhm_k) if fwhm_k > 0 else np.nan,
                fwhm_half_depth=float(half_k * to_rest),
                q=float(local[k] / rms) if rms > 0 else np.nan,
                wave_fit=float(mu[k] * to_rest),
                rv=float(C_KMS * (mu[k] / lab_w[j] - 1.0)),
                fit_wave=wave[win] * to_rest,
                fit_flux=fit_flux,
                fit_neighbours=fit_neighbours,
                width_fitted=width_fitted,
                blended=blended,
            )
        )
    not_found = np.setdiff1d(np.arange(lab_w.size), lab_idx)
    missing, missing_codes = lab_w[not_found], lab_c[not_found]
    return RunResult(
        fwhm=w,
        rv=rv,
        rv_sigma=rv_sigma,
        lines=lines,
        n_detected=int(mu.size),
        residual_pct=100.0 * rms,
        noise_factor=kappa,
        wave=wave,
        flux=flux,
        continuum=continuum,
        model=model,
        missing=missing,
        missing_codes=missing_codes,
    )


# ----------------------------------------------------------------------------- 4DAO loop
def measure_ew(spec: Spec, lab_wave: np.ndarray, lab_code: np.ndarray, settings: EWSettings) -> EWResult:
    """Measure equivalent widths with the 4DAO strategy (FWHM convergence, then a fixed-FWHM run).

    With ``settings.fwhm`` the FWHM is fixed from the start. With
    ``settings.fwhm_from = "list"`` the converged run is followed by a fit of
    FWHM versus wavelength on the unblended lines of the list
    (:func:`fit_fwhm_relation`) and a last run with that relation fixed.

    Raises:
        EWError: If the spectrum has too few valid pixels or no line can be identified.
    """
    s = settings
    wave, flux = prepare(spec, s.wmin, s.wmax)
    step = float(np.median(np.diff(wave)))
    lam_c = float(0.5 * (wave[0] + wave[-1]))
    lab_wave, lab_code = np.asarray(lab_wave, float), np.asarray(lab_code, float)
    reach = (1 + s.vmin / C_KMS, 1 + s.vmax / C_KMS)
    if not np.any((lab_wave * reach[1] > wave[0]) & (lab_wave * reach[0] < wave[-1])):
        raise EWError(f"no line of the list falls in {wave[0]:.1f}-{wave[-1]:.1f} A for the RV window")
    if s.fwhm is not None:
        w = s.fwhm
    elif s.resolution is not None:
        w = lam_c / s.resolution
    else:
        try:
            w, n_used = estimate_fwhm(wave, flux, np.isfinite(flux) & (flux > 0) & ~mask_pixels(wave, s.masks, 0.0))
            log.info("First-guess FWHM %.3f pix (R %.0f) from %d lines.", w / step, lam_c / w, n_used)
        except EWError as exc:
            w = lam_c / DEFAULT_RESOLUTION
            log.warning("%s: starting from R = %.0f (use --resol to choose).", exc, DEFAULT_RESOLUTION)
    window = (s.rv - 20.0, s.rv + 20.0) if s.rv is not None else (s.vmin, s.vmax)
    history: list[tuple[float, float, float, float, float]] = []
    convergence = 1
    fixed = s.fix_fwhm or s.fwhm is not None
    first_guess = w
    if s.fwhm_from == "list":
        return _measure_from_list(wave, flux, lab_wave, lab_code, w, window, s, step, lam_c)
    if not fixed:
        convergence = 0
        seen: list[float] = []
        for _ in range(s.max_iter):
            result = run_once(wave, flux, lab_wave, lab_code, w, fit_width=True, window=window, settings=s)
            w_in_pix, w_out_pix = w / step, result.fwhm / step
            history.append((w_in_pix, w_out_pix, window[0], result.rv, window[1]))
            log.debug(
                "FWHM %7.3f -> %7.3f pix   RV %8.2f km/s (window %.1f, %.1f)", w_in_pix, w_out_pix, result.rv, *window
            )
            if not 0 < w_out_pix <= s.fwhm_max:
                convergence = -2
                break
            if abs(w_out_pix - w_in_pix) < s.tol:
                convergence = 1
                w = result.fwhm
                break
            if any(abs(w_out_pix - old) < 1e-3 for old in seen):
                convergence = -1
                w = result.fwhm
                break
            seen.append(w_out_pix)
            w = result.fwhm
            if result.rv_sigma > 0 and len(result.lines) >= 3:
                half = max(5.0 * result.rv_sigma, MIN_RV_HALF_WINDOW)
                window = (result.rv - half, result.rv + half)
    final = run_once(wave, flux, lab_wave, lab_code, w, fit_width=False, window=window, settings=s, measure_widths=True)
    history.append((w / step, final.fwhm / step, window[0], final.rv, window[1]))
    default_slope = 1.0 if s.scale_fwhm else 0.0
    model = FwhmModel(w, default_slope, lam_c, "fixed" if fixed else "all")
    if not fixed and w < COLLAPSE_RATIO * first_guess:
        log.warning(
            "The common FWHM (%.3f A) is much narrower than the first guess (%.3f A): narrow features such as "
            "telluric lines may dominate it. Consider --fwhm-from list.",
            w,
            first_guess,
        )
    return EWResult(run=final, convergence=convergence, history=history, step=step, core=s.core, fwhm_model=model)


# ----------------------------------------------------------------------------- files
def read_kurucz_linelist(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read wavelength (A) and species code (Kurucz format, e.g. 26.00) from the first two columns."""
    data = np.loadtxt(path, comments="#", usecols=(0, 1), ndmin=2)
    if data.size == 0:
        raise EWError(f"{path}: empty line list")
    return data[:, 0], data[:, 1]


def missing_reasons(run: RunResult, masks: Sequence[MaskRegion]) -> list[str]:
    """Why each line of ``run.missing`` was not identified.

    ``"masked"`` if its centre falls in a mask, ``"partly masked"`` if a mask
    covers part of its profile (+-1 FWHM), otherwise ``"not identified"`` (no
    line detected at the expected position, or its RV deviates from the others).
    """
    fwhm = float(np.median([ln.fwhm for ln in run.lines])) if run.lines else 0.0
    reasons = []
    for wave in run.missing:
        centre = wave * (1.0 + run.rv / C_KMS)
        hit = mask_pixels(np.linspace(centre - fwhm, centre + fwhm, 21), masks, run.rv)
        reasons.append("masked" if hit[10] else "partly masked" if hit.any() else "not identified")
    return reasons


def format_rejected(run: RunResult, masks: Sequence[MaskRegion]) -> str:
    """Table of the lines of the list that fall in the spectrum but were not identified.

    Columns: wavelength [A], species code, reason (see :func:`missing_reasons`).
    """
    rows = ["#     wave    elem  reason\n"]
    order = np.argsort(run.missing)
    reasons = missing_reasons(run, masks)
    rows += [f"{run.missing[i]:9.3f} {run.missing_codes[i]:7.2f}  {reasons[i]}\n" for i in order]
    return "".join(rows)


TABLE_COLUMNS = ("wave", "EW", "error_Ew", "elem", "FWHM", "Resol")
"""Column names written in the header of the EW table."""


def format_table(lines: Sequence[MeasuredLine], half_depth: bool = False) -> str:
    """Table of identified lines, as the 4DAO ``.in`` file plus two columns.

    Columns: wavelength [A], EW [mA], EW error [mA], species code, FWHM [A]
    and resolving power ``wavelength / FWHM``. The FWHM is the saturation-free
    width (:attr:`MeasuredLine.fwhm`), or the half-depth width with
    ``half_depth``. The first line is a header starting with ``#`` (skipped by
    ``numpy.loadtxt`` and similar readers).
    """
    widths = (8, 10, 10, 7, 9, 9)
    rows = ["#" + " ".join(f"{name:>{width}s}" for name, width in zip(TABLE_COLUMNS, widths, strict=True)) + "\n"]
    for ln in sorted(lines, key=lambda x: x.wave):
        fwhm = ln.fwhm_half_depth if half_depth else ln.fwhm
        rows.append(
            f"{ln.wave:9.3f} {ln.ew:10.3f} {ln.ew_err:10.3f} {ln.code:7.2f} {fwhm:9.4f} {ln.wave / fwhm:9.0f}\n"
        )
    return "".join(rows)


# ----------------------------------------------------------------------------- widths
def estimate_fwhm(
    wave: np.ndarray, flux: np.ndarray, valid: np.ndarray, min_lines: int = 3, smooth_pix: float = 1.0
) -> tuple[float, int]:
    """First-guess FWHM (Angstrom) from the half-depth width of clean lines.

    Spikes are removed and the spectrum is roughly normalised and smoothed
    with a Gaussian of ``smooth_pix`` pixels (its broadening is removed in
    quadrature), so that noisy spectra can be measured too. Local minima
    deeper than ``max(5 sigma, 0.08)`` (sigma of the smoothed spectrum) and
    shallower than 0.7 are measured at half depth by linear interpolation;
    asymmetric profiles (likely blends) are discarded and the median width,
    scaled to the central wavelength, is returned.

    Raises:
        EWError: If fewer than ``min_lines`` lines can be measured.
    """
    basis = continuum_basis(wave, "legendre", 5, 0)
    valid = valid & ~find_spikes(flux, valid, initial_continuum(flux, basis, valid), 15, 3)
    norm = gaussian_filter1d(_fill(flux / initial_continuum(flux, basis, valid), valid), smooth_pix, mode="nearest")
    # noise of the *smoothed* spectrum, measured directly (GIRAFFE-like noise is correlated)
    resid = norm - median_filter(norm, size=31, mode="nearest")
    noise = MAD_TO_SIGMA * float(np.median(np.abs(resid[valid] - np.median(resid[valid]))))
    step = float(np.median(np.diff(wave)))
    smoothing = 2.0 * np.sqrt(2.0 * np.log(2.0)) * smooth_pix * step
    lam_c = 0.5 * (wave[0] + wave[-1])
    inner = np.arange(2, wave.size - 2)
    minima = inner[(norm[inner] < norm[inner - 1]) & (norm[inner] <= norm[inner + 1]) & valid[inner]]
    depth = 1.0 - norm[minima]
    minima = minima[(depth > max(5.0 * noise, 0.08)) & (depth < 0.7)]
    widths = []
    for i in minima:
        half = 1.0 - 0.5 * (1.0 - norm[i])
        sides = []
        for direction in (-1, 1):
            j = i
            while 0 < j < wave.size - 1 and abs(j - i) < 40 and norm[j] < half:
                j += direction
            if norm[j] < half or not valid[j]:
                break
            k = j - direction  # last pixel below half depth
            x0, x1, y0, y1 = wave[k], wave[j], norm[k], norm[j]
            sides.append(abs(x0 + (half - y0) * (x1 - x0) / (y1 - y0) - wave[i]))
        if len(sides) == 2 and max(sides) < 1.5 * min(sides):
            measured = sum(sides)
            if measured > max(smoothing, MIN_LINE_FWHM_PIX * step):  # narrower features are noise
                widths.append(np.sqrt(measured**2 - smoothing**2) * lam_c / wave[i])
    if len(widths) < min_lines:
        raise EWError(f"only {len(widths)} clean lines to estimate the FWHM")
    return float(np.median(widths)), len(widths)


def line_width(
    fitter: LineFitter, mu: np.ndarray, amp: np.ndarray, w: float, k: int, total_depth: np.ndarray
) -> tuple[float, float, float]:
    """Centre, Gaussian sigma and amplitude (observed frame) of line ``k`` fitted on its own.

    All other lines are kept fixed at their model; centre, amplitude and
    Gaussian width of line ``k`` are free, over +-2 FWHM. Returns NaNs if the
    fit fails or the width reaches its limits (0.3-3 times the common width).
    """
    sig0 = float(fitter.sigma(mu[k : k + 1], w)[0])
    fwhm0 = sig0 / FWHM_TO_SIGMA
    sl = slice(
        int(np.searchsorted(fitter.wave, mu[k] - 2 * fwhm0)), int(np.searchsorted(fitter.wave, mu[k] + 2 * fwhm0))
    )
    use = fitter.valid[sl]
    if np.count_nonzero(use) < 6:
        return np.nan, np.nan, np.nan
    x = fitter.wave[sl][use]
    own = saturated(amp[k] * np.exp(-0.5 * ((fitter.wave[sl] - mu[k]) / sig0) ** 2), fitter.rho)
    target = (1.0 - (total_depth[sl] - own))[use]
    y = fitter.norm[sl][use]

    def resid(p: np.ndarray) -> np.ndarray:
        return y - target + saturated(p[1] * np.exp(-0.5 * ((x - p[0]) / p[2]) ** 2), fitter.rho)

    lower = [mu[k] - 0.5 * fwhm0, 0.0, 0.3 * sig0]
    upper = [mu[k] + 0.5 * fwhm0, 1e3, 3.0 * sig0]
    p0 = np.clip([mu[k], max(amp[k], 1e-6), sig0], np.add(lower, 1e-9), np.subtract(upper, 1e-9))
    try:
        result = least_squares(resid, p0, bounds=(lower, upper), method="trf", x_scale="jac", max_nfev=200)
    except ValueError:
        return np.nan, np.nan, np.nan
    sigma = float(result.x[2])
    if not result.success or not 0.31 * sig0 < sigma < 2.99 * sig0:
        return np.nan, np.nan, np.nan
    return float(result.x[0]), sigma, float(result.x[1])


def profile_fwhm(amp: float, sigma: float, rho: float) -> float:
    """Full width at half depth of the saturated Gaussian ``saturated(amp * exp(-x^2 / 2 sigma^2))``.

    Equal to ``2.355 sigma`` for weak lines, broader for saturated ones.
    """
    if amp <= 0:
        return 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    half = 0.5 * float(saturated(np.array(amp), rho))
    g_half = half / (1.0 - half / (1.0 - rho))
    return float(2.0 * sigma * np.sqrt(2.0 * np.log(amp / g_half)))


def fit_fwhm_relation(
    lines: Sequence[MeasuredLine],
    rv: float,
    lam_c: float,
    *,
    constant: bool = False,
    clip: float = 3.0,
    min_linear: int = 5,
    min_lines: int = 3,
) -> FwhmModel:
    """FWHM(lambda) from the unblended lines of the list.

    Lines whose own width fit succeeded and that are not blended
    (:attr:`MeasuredLine.blended`) are fitted with a straight line
    ``FWHM = a + b (lambda - lam_c)`` (a constant with ``constant`` or fewer
    than ``min_linear`` lines), rejecting iteratively the lines more than
    ``clip`` robust sigmas away (remaining blends, narrow features). Works in
    the observed frame, like the fitter.

    Raises:
        EWError: If fewer than ``min_lines`` lines are left.
    """
    z = 1.0 + rv / C_KMS
    candidates = [ln for ln in lines if ln.width_fitted and not ln.blended and np.isfinite(ln.fwhm)]
    if len(candidates) < min_lines:
        raise EWError(f"only {len(candidates)} unblended lines of the list with a width fit")
    x = np.array([ln.wave_fit * z for ln in candidates]) - lam_c
    y = np.array([ln.fwhm * z for ln in candidates])
    use = np.ones(y.size, dtype=bool)
    a, b = float(np.median(y)), 0.0
    for _ in range(20):
        if constant or np.count_nonzero(use) < min_linear:
            a, b = float(np.median(y[use])), 0.0
        else:
            b, a = (float(v) for v in np.polyfit(x[use], y[use], 1))
        resid = y - (a + b * x)
        scale = MAD_TO_SIGMA * float(np.median(np.abs(resid[use] - np.median(resid[use]))))
        new = np.abs(resid) <= clip * max(scale, 1e-6 * a)
        if np.array_equal(new, use) or np.count_nonzero(new) < min_lines:
            break
        use = new
    if not a > 0:
        raise EWError("the FWHM relation from the list lines is not positive")
    return FwhmModel(
        fwhm=a,
        slope=b * lam_c / a,
        lam_c=lam_c,
        source="list",
        used=tuple(ln.wave for ln, u in zip(candidates, use, strict=True) if u),
        n_candidates=len(candidates),
    )


def _measure_from_list(
    wave: np.ndarray,
    flux: np.ndarray,
    lab_wave: np.ndarray,
    lab_code: np.ndarray,
    w: float,
    window: tuple[float, float],
    s: EWSettings,
    step: float,
    lam_c: float,
) -> EWResult:
    """FWHM relation from the unblended lines of the list, iterated with the width never free.

    Every run keeps the FWHM fixed to the current relation (so lines cannot be
    split into narrower components), measures the identified lines on their
    own, and refits the relation (:func:`fit_fwhm_relation`), until it changes
    by less than ``s.tol`` pixels anywhere in the spectrum.
    """
    slope = 1.0 if s.scale_fwhm else 0.0
    model = FwhmModel(w, slope, lam_c, "fixed")
    history: list[tuple[float, float, float, float, float]] = []
    edges = np.array([wave[0], wave[-1]])
    convergence = 0
    run: RunResult | None = None
    for _ in range(LIST_MAX_ITER):
        run = run_once(
            wave,
            flux,
            lab_wave,
            lab_code,
            model.fwhm,
            fit_width=False,
            window=window,
            settings=s,
            measure_widths=True,
            slope=model.slope,
        )
        try:
            new = fit_fwhm_relation(run.lines, run.rv, lam_c, constant=not s.scale_fwhm)
        except EWError as exc:
            log.warning("%s: no FWHM relation from the list, the first-guess FWHM is kept.", exc)
            convergence = -2
            break
        history.append((model.fwhm / step, new.fwhm / step, window[0], run.rv, window[1]))
        change = float(np.max(np.abs(new.at(edges) - model.at(edges)))) / step
        model = new
        if run.rv_sigma > 0 and len(run.lines) >= 3:
            half = max(5.0 * run.rv_sigma, MIN_RV_HALF_WINDOW)
            window = (run.rv - half, run.rv + half)
        if change < s.tol:
            convergence = 1
            break
    final = run_once(
        wave,
        flux,
        lab_wave,
        lab_code,
        model.fwhm,
        fit_width=False,
        window=window,
        settings=s,
        measure_widths=True,
        slope=model.slope,
    )
    history.append((model.fwhm / step, final.fwhm / step, window[0], final.rv, window[1]))
    if model.source == "list":  # keep the relation used, record the supporting lines of the final run
        try:
            check = fit_fwhm_relation(final.lines, final.rv, lam_c, constant=not s.scale_fwhm)
            model = replace(model, used=check.used, n_candidates=check.n_candidates)
        except EWError:
            pass
    return EWResult(run=final, convergence=convergence, history=history, step=step, core=s.core, fwhm_model=model)
