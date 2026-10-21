"""Radial velocities by cross-correlation (Tonry & Davis 1979, AJ 84, 1511).

Both spectra are resampled onto a common grid uniform in ``ln(lambda)``,
high-pass filtered, standardised and apodised; the CCF is computed by FFT in
``O(N log N)``. A lag of ``k`` pixels corresponds to
``v = c * (exp(k * dln) - 1)``, which is the exact inverse of the shift
``lambda_obs = lambda_rest * (1 + v/c)`` used by :func:`bobspec.spectrum.to_rest_frame`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import correlate, correlation_lags

from bobspec.constants import C_KMS, FWHM_TO_SIGMA, MAD_TO_SIGMA, Region
from bobspec.io import Spec
from bobspec.spectrum import add_noise, doppler_factor, exclusion_mask, interpolate, log_grid, resample

log = logging.getLogger(__name__)

MIN_POINTS = 50
TD_RANGE_KMS = 1000.0
"""Lag range (km/s) over which the Tonry & Davis antisymmetric noise is measured.

Fixed, so that the error does not depend on the velocity search window."""


@dataclass(frozen=True, slots=True)
class CCFResult:
    """Outcome of :func:`cross_correlate`.

    Attributes:
        rv: Radial velocity of ``spec`` relative to ``template`` (km/s).
        rv_err_td: Tonry & Davis error ``3 w / (8 (1 + r))`` (km/s).
        fwhm: FWHM of the CCF peak from the Gaussian fit (km/s).
        r_td: Tonry & Davis ``r = h / (sqrt(2) sigma_a)``.
        fit_ok: False if the Gaussian fit failed and a parabolic estimate was used.
        velocity: Velocity lags inside the search window (km/s).
        ccf: Normalised CCF at ``velocity``.
        fit_params: Gaussian parameters ``(amplitude, centre, sigma, offset)``.
        wave_range: Wavelength interval actually correlated (Angstrom).
        dv: Velocity step of the ln(lambda) grid (km/s).
    """

    rv: float
    rv_err_td: float
    fwhm: float
    r_td: float
    fit_ok: bool
    velocity: np.ndarray = field(repr=False)
    ccf: np.ndarray = field(repr=False)
    fit_params: tuple[float, float, float, float] = field(repr=False)
    wave_range: tuple[float, float] = (np.nan, np.nan)
    dv: float = np.nan

    def model(self, velocity: np.ndarray) -> np.ndarray:
        """Evaluate the fitted Gaussian at ``velocity``."""
        return _gauss(velocity, *self.fit_params)


@dataclass(frozen=True, slots=True)
class MCResult:
    """Velocities measured on noisy mock spectra by :func:`monte_carlo`."""

    rvs: np.ndarray = field(repr=False)
    n_failed: int = 0

    @property
    def std(self) -> float:
        """Sample standard deviation of the MC velocities (km/s)."""
        return float(np.std(self.rvs, ddof=1)) if self.rvs.size > 1 else np.nan

    @property
    def robust_std(self) -> float:
        """MAD-based sigma, insensitive to occasional jumps to a wrong CCF peak."""
        if self.rvs.size < 2:
            return np.nan
        return float(MAD_TO_SIGMA * np.median(np.abs(self.rvs - np.median(self.rvs))))


def _gauss(v: np.ndarray, amp: float, mu: float, sigma: float, offset: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((v - mu) / sigma) ** 2) + offset


def _prepare(flux: np.ndarray, valid: np.ndarray, sigma_pix: float, window: np.ndarray) -> np.ndarray:
    """High-pass filter (NaN-aware), standardise and apodise one spectrum."""
    weights = valid.astype(float)
    filled = np.where(valid, flux, 0.0)
    smooth = gaussian_filter1d(filled, sigma_pix, mode="nearest")
    norm = gaussian_filter1d(weights, sigma_pix, mode="nearest")
    high = np.where(valid, filled - smooth / np.maximum(norm, 1e-12), 0.0)
    std = high[valid].std()
    if not std > 0:
        raise ValueError("spectrum has no features in the correlated range")
    high[valid] = (high[valid] - high[valid].mean()) / std
    return high * window


def _parabolic_vertex(x: np.ndarray, y: np.ndarray, i: int) -> float:
    if not 0 < i < y.size - 1:
        return float(x[i])
    denom = y[i - 1] - 2.0 * y[i] + y[i + 1]
    if denom >= 0:  # not a maximum
        return float(x[i])
    delta = 0.5 * (y[i - 1] - y[i + 1]) / denom
    return float(x[i] + delta * 0.5 * (x[i + 1] - x[i - 1]))


def _half_max_width(v: np.ndarray, c: np.ndarray, ipk: int, base: float) -> float:
    half = base + 0.5 * (c[ipk] - base)
    left = np.flatnonzero(c[:ipk] < half)
    right = np.flatnonzero(c[ipk:] < half)
    v_lo = v[left[-1]] if left.size else v[0]
    v_hi = v[ipk + right[0]] if right.size else v[-1]
    return float(v_hi - v_lo)


def cross_correlate(
    spec: Spec,
    template: Spec,
    *,
    v_range: tuple[float, float] = (-600.0, 600.0),
    wave_range: tuple[float, float] | None = None,
    exclude_regions: Iterable[Region] = (),
    dv_kms: float | None = None,
    hp_sigma_kms: float = 150.0,
    apodize_frac: float = 0.0,
) -> CCFResult:
    """Measure the radial velocity of ``spec`` against ``template``.

    Args:
        spec: Observed spectrum (any grid, NaN allowed).
        template: Reference spectrum, assumed at rest.
        v_range: Velocity search window (km/s).
        wave_range: Optional wavelength interval to correlate (Angstrom).
        exclude_regions: Observed-frame intervals of ``spec`` ignored in the CCF
            (tellurics); the template is never masked.
        dv_kms: Step of the ln(lambda) grid; default half the finest pixel of ``spec``.
        hp_sigma_kms: Width of the Gaussian high-pass filter removing the continuum.
        apodize_frac: Fraction of the observed window tapered by a Hann window at
            each end. Off by default: since the template extends beyond the window
            and the correlation is linear (not circular), the truncation creates no
            spurious peak, while a taper skews lines on its slopes and biases the RV
            of short windows by a few 10 m/s.

    Returns:
        A :class:`CCFResult`.

    Raises:
        ValueError: On missing overlap, too few usable pixels, a velocity
            window narrower than a few grid steps, or a featureless spectrum.
    """
    dv = float(dv_kms) if dv_kms is not None else spec.velocity_step / 2.0
    dln = dv / C_KMS
    vmin, vmax = sorted(map(float, v_range))
    if not (vmin > -C_KMS and vmax < C_KMS):
        raise ValueError("velocity window must lie within (-c, c)")
    lo, hi = spec.wave[0], spec.wave[-1]
    if wave_range is not None:
        w1, w2 = sorted(wave_range)
        lo, hi = max(lo, w1), min(hi, w2)
    # The observed window is correlated against a template that extends beyond it by
    # the largest searched shift plus the high-pass filter width, so no line enters or
    # leaves the window with the lag and the template edges (where the filter is
    # one-sided) never touch the correlated pixels.
    margin = np.exp(hp_sigma_kms / C_KMS)
    lo = max(lo, template.wave[0] * (1 + max(vmax, 0.0) / C_KMS) * margin)
    hi = min(hi, template.wave[-1] * (1 + min(vmin, 0.0) / C_KMS) / margin)
    if not hi > lo:
        raise ValueError("spectrum and template do not overlap in the requested range")

    grid = log_grid(lo, hi, dv)
    n = grid.size
    pad_lo = int(np.ceil((np.log1p(max(vmax, 0.0) / C_KMS) + hp_sigma_kms / C_KMS) / dln)) + 1
    pad_hi = int(np.ceil((-np.log1p(min(vmin, 0.0) / C_KMS) + hp_sigma_kms / C_KMS) / dln)) + 1
    grid_tpl = np.exp(np.log(grid[0]) + dln * np.arange(-pad_lo, n + pad_hi))
    f_obs, f_tpl = interpolate(spec, grid), interpolate(template, grid_tpl)
    valid_obs = np.isfinite(f_obs) & exclusion_mask(grid, exclude_regions)  # tellurics: observed frame only
    valid_tpl = np.isfinite(f_tpl)
    n_valid = int(np.count_nonzero(valid_obs))
    if n_valid < MIN_POINTS:
        raise ValueError(f"only {n_valid} usable pixels in the overlap (< {MIN_POINTS})")

    window = np.ones(n)
    m = int(np.clip(apodize_frac * n, 1, n // 4))
    if m > 1:
        taper = np.hanning(2 * m)
        window[:m], window[-m:] = taper[:m], taper[m:]
    sigma_pix = max(5.0, hp_sigma_kms / dv)
    a = _prepare(f_obs, valid_obs, sigma_pix, window)
    b = _prepare(f_tpl, valid_tpl, sigma_pix, np.ones(grid_tpl.size))

    ccf_full = correlate(a, b, mode="full", method="fft") / n_valid
    shift_pix = correlation_lags(a.size, b.size, mode="full") + pad_lo
    v_full = C_KMS * np.expm1(shift_pix * dln)
    sel = (v_full >= vmin) & (v_full <= vmax)
    if np.count_nonzero(sel) < 7:
        raise ValueError(f"velocity window {vmin}..{vmax} km/s is narrower than 7 grid steps ({dv:.3g} km/s)")
    v, c = v_full[sel], ccf_full[sel]

    ipk = int(np.argmax(c))
    if ipk in (0, c.size - 1):
        log.warning("CCF maximum at the edge of the velocity window: widen vmin/vmax.")
    base = float(np.median(c))
    v_peak = _parabolic_vertex(v, c, ipk)
    fwhm_guess = max(_half_max_width(v, c, ipk, base), 3.0 * dv)

    half_width = max(fwhm_guess, 3.0 * dv)
    in_fit = np.abs(v - v_peak) <= half_width
    p0 = [max(c[ipk] - base, 1e-12), v_peak, fwhm_guess * FWHM_TO_SIGMA, base]
    lower = [0.0, v_peak - half_width, dv / 4.0, -np.inf]
    upper = [np.inf, v_peak + half_width, 2.0 * half_width, np.inf]
    p0 = list(np.clip(p0, lower, upper))
    try:
        popt, _ = curve_fit(_gauss, v[in_fit], c[in_fit], p0=p0, bounds=(lower, upper), maxfev=10_000)
        fit_ok = True
    except (RuntimeError, ValueError, TypeError) as exc:  # TypeError: fewer points than parameters
        log.warning("Gaussian fit of the CCF peak failed (%s); using the parabolic estimate.", exc)
        popt, fit_ok = np.asarray(p0), False
    amp, mu, sigma, offset = map(float, popt)
    fwhm = sigma / FWHM_TO_SIGMA

    # Tonry & Davis r: peak height over the rms of the CCF antisymmetric part.
    ipk_full = int(np.flatnonzero(sel)[0]) + ipk
    k = np.arange(1, min(ipk_full, ccf_full.size - 1 - ipk_full) + 1)
    k = k[np.abs(v_full[ipk_full + k] - v_full[ipk_full]) <= TD_RANGE_KMS]  # independent of v_range
    sigma_a = float(np.sqrt(np.mean((0.5 * (ccf_full[ipk_full + k] - ccf_full[ipk_full - k])) ** 2))) if k.size else 0.0
    height = amp + offset  # peak of the normalised CCF (T&D h), independent of the local baseline
    r_td = height / (np.sqrt(2.0) * sigma_a) if sigma_a > 0 else np.inf
    rv_err_td = 3.0 * fwhm / (8.0 * (1.0 + r_td))

    return CCFResult(
        rv=mu,
        rv_err_td=float(rv_err_td),
        fwhm=float(fwhm),
        r_td=float(r_td),
        fit_ok=fit_ok,
        velocity=v,
        ccf=c,
        fit_params=(amp, mu, sigma, float(popt[3])),
        wave_range=(float(lo), float(hi)),
        dv=dv,
    )


def monte_carlo(
    template: Spec,
    obs_wave: np.ndarray,
    rv: float,
    snr: float,
    n_sim: int,
    rng: np.random.Generator,
    **ccf_kwargs: Any,
) -> MCResult:
    """Estimate the photon-noise RV error with a parametric bootstrap.

    The template is shifted to ``rv``, resampled onto the observed pixels,
    perturbed with Gaussian noise at the given continuum ``snr`` and
    re-measured ``n_sim`` times. The scatter captures photon noise only:
    template mismatch is *not* included (compare with ``CCFResult.rv_err_td``).

    Args:
        template: Rest-frame template.
        obs_wave: Wavelengths of the observed pixels to simulate.
        rv: Velocity of the mock spectra (km/s).
        snr: Continuum signal-to-noise ratio per pixel (> 0).
        n_sim: Number of realisations (>= 2).
        rng: Random generator; seed it for reproducible errors.
        **ccf_kwargs: Forwarded to :func:`cross_correlate`.

    Returns:
        An :class:`MCResult`; failed realisations are counted, not raised.
    """
    if n_sim < 2:
        raise ValueError(f"n_sim must be >= 2, got {n_sim}")
    shifted = template.with_wave(template.wave * doppler_factor(rv))
    mock = resample(shifted, np.asarray(obs_wave, float))
    reference = float(np.nanpercentile(mock.flux, 95))
    rvs: list[float] = []
    for _ in range(n_sim):
        noisy = add_noise(mock, snr, rng, reference=reference)
        try:
            rvs.append(cross_correlate(noisy, template, **ccf_kwargs).rv)
        except ValueError as exc:
            log.debug("MC realisation failed: %s", exc)
    return MCResult(rvs=np.asarray(rvs), n_failed=n_sim - len(rvs))
