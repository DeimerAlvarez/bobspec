"""Pure operations on :class:`~bobspec.io.Spec` objects.

Doppler convention used everywhere in the package: ``z = v / c`` and
``lambda_obs = lambda_rest * (1 + z)``. The cross-correlation in
:mod:`bobspec.rv` measures velocities with the same convention, so measuring
and correcting are exact inverses of each other.

Broadening kernels (instrumental Gaussian and rotation) are applied on a grid
uniform in ``ln(lambda)``, where a kernel of fixed velocity width is the same
at every wavelength. This makes constant-R and constant-vsini broadening exact
over arbitrarily wide ranges.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Literal

import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter1d

from bobspec.constants import C_KMS, FWHM_TO_SIGMA, Region
from bobspec.io import Spec


# ----------------------------------------------------------------------------- grids
def uniform_grid(start: float, stop: float, step: float) -> np.ndarray:
    """Return ``start, start+step, ...`` up to and including ``stop`` (within rounding).

    Raises:
        ValueError: If ``step`` is not positive or ``stop < start``.
    """
    if not step > 0:
        raise ValueError(f"step must be positive, got {step}")
    if stop < start:
        raise ValueError(f"stop ({stop}) < start ({start})")
    n = int(np.floor((stop - start) / step + 1e-9)) + 1
    return start + step * np.arange(n)


def log_grid(start: float, stop: float, dv_kms: float) -> np.ndarray:
    """Return a wavelength grid with a constant velocity step ``dv_kms``."""
    if not dv_kms > 0:
        raise ValueError(f"dv_kms must be positive, got {dv_kms}")
    return np.exp(uniform_grid(np.log(start), np.log(stop), dv_kms / C_KMS))


def interpolate(spec: Spec, new_wave: np.ndarray) -> np.ndarray:
    """Linearly interpolate the flux onto ``new_wave`` (NaN outside the range).

    NaN pixels in the input propagate only to their immediate neighbours.
    """
    good = spec.finite
    out = np.interp(new_wave, spec.wave[good], spec.flux[good], left=np.nan, right=np.nan)
    if not good.all():  # re-insert gaps wider than one pixel
        bad_frac = np.interp(new_wave, spec.wave, (~good).astype(float))
        out[bad_frac > 0.5] = np.nan
    return out


def _bin_edges(wave: np.ndarray) -> np.ndarray:
    mid = 0.5 * (wave[1:] + wave[:-1])
    return np.concatenate([[wave[0] - (mid[0] - wave[0])], mid, [wave[-1] + (wave[-1] - mid[-1])]])


def rebin(spec: Spec, new_wave: np.ndarray) -> Spec:
    """Flux-conserving rebinning (mean flux density in each new pixel).

    Each input pixel is treated as a box of constant flux density; output
    pixels average the boxes they overlap, weighted by the overlap. NaN pixels
    are ignored; output pixels with no valid coverage are NaN. Use this
    instead of point interpolation when the new pixels are coarser than the
    old ones (otherwise the noise and line profiles are aliased).

    Args:
        spec: Input spectrum.
        new_wave: Centres of the output pixels (strictly increasing).

    Returns:
        A new spectrum on ``new_wave``. Complexity ``O(N + M)``.
    """
    old_edges, new_edges = _bin_edges(spec.wave), _bin_edges(np.asarray(new_wave, float))
    widths = np.diff(old_edges)
    good = spec.finite
    cum_flux = np.concatenate([[0.0], np.cumsum(np.where(good, spec.flux, 0.0) * widths)])
    cum_cover = np.concatenate([[0.0], np.cumsum(good * widths)])
    flux_in = np.diff(np.interp(new_edges, old_edges, cum_flux))
    cover = np.diff(np.interp(new_edges, old_edges, cum_cover))
    inside = (new_edges[:-1] >= old_edges[0]) & (new_edges[1:] <= old_edges[-1])
    with np.errstate(invalid="ignore", divide="ignore"):
        flux = np.where(inside & (cover > 0.5 * np.diff(new_edges)), flux_in / cover, np.nan)
    return spec.on_grid(np.asarray(new_wave, float), flux)


def resample(spec: Spec, new_wave: np.ndarray) -> Spec:
    """Resample onto ``new_wave`` choosing the right method.

    Flux-conserving :func:`rebin` is used when the new pixels are coarser than
    the native ones, linear interpolation otherwise.
    """
    new_wave = np.asarray(new_wave, float)
    if np.median(np.diff(new_wave)) > np.median(np.diff(spec.wave)):
        return rebin(spec, new_wave)
    return spec.on_grid(new_wave, interpolate(spec, new_wave))


# ----------------------------------------------------------------------------- Doppler
def doppler_factor(rv_kms: float) -> float:
    """Return ``1 + v/c``, rejecting unphysical velocities."""
    if not abs(rv_kms) < C_KMS:
        raise ValueError(f"|rv| must be smaller than c, got {rv_kms} km/s")
    return 1.0 + rv_kms / C_KMS


def to_rest_frame(spec: Spec, rv_kms: float) -> Spec:
    """Remove a radial velocity: ``lambda_rest = lambda_obs / (1 + v/c)``."""
    out = spec.with_wave(spec.wave / doppler_factor(rv_kms))
    return out.with_history(f"shifted to rest frame, RV = {rv_kms:.4f} km/s")


def exclusion_mask(wave: np.ndarray, regions: Iterable[Region]) -> np.ndarray:
    """Return a boolean mask that is False inside any of ``regions``."""
    keep = np.ones(wave.shape, dtype=bool)
    for start, end in regions:
        lo, hi = sorted((start, end))
        keep &= ~((wave >= lo) & (wave <= hi))
    return keep


# ----------------------------------------------------------------------------- broadening
def _velocity_step(spec: Spec, kernel_width_kms: float, oversample: float) -> float:
    return min(spec.velocity_step / oversample, kernel_width_kms / 10.0)


def _apply_velocity_kernel(spec: Spec, dv: float, convolve: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Resample to ln(lambda), apply ``convolve`` and resample back."""
    good = spec.finite
    if np.count_nonzero(good) < 2:
        raise ValueError("spectrum has fewer than two finite pixels")
    grid = log_grid(spec.wave[good][0], spec.wave[good][-1], dv)
    flux_log = np.interp(grid, spec.wave[good], spec.flux[good])
    out = np.interp(spec.wave, grid, convolve(flux_log))  # last pixel may lie < dv beyond the grid
    first, last = spec.wave[good][0], spec.wave[good][-1]
    out[~good | (spec.wave < first) | (spec.wave > last)] = np.nan
    return out


def gaussian_broaden(spec: Spec, resolution: float, *, oversample: float = 2.0) -> Spec:
    """Convolve with a Gaussian of constant resolving power ``R = lambda / FWHM``.

    Args:
        spec: Input spectrum (any grid).
        resolution: Target resolving power ``R`` (> 0).
        oversample: Sampling of the internal ln(lambda) grid relative to the
            finest native pixel.

    Returns:
        The broadened spectrum on the input grid.
    """
    if not resolution > 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    fwhm_kms = C_KMS / resolution
    dv = _velocity_step(spec, fwhm_kms, oversample)
    sigma_pix = fwhm_kms * FWHM_TO_SIGMA / dv
    flux = _apply_velocity_kernel(spec, dv, lambda f: gaussian_filter1d(f, sigma_pix, mode="nearest"))
    return spec.with_flux(flux).with_history(f"Gaussian broadening to R = {resolution:g}")


def rotation_kernel(vsini: float, epsilon: float, dv: float) -> np.ndarray:
    """Gray's rotational broadening kernel with linear limb darkening.

    Args:
        vsini: Projected rotational velocity (km/s, > 0).
        epsilon: Linear limb-darkening coefficient in ``[0, 1]``.
        dv: Velocity step of the kernel samples (km/s).

    Returns:
        Kernel samples normalised to unit sum (odd length, centred).
    """
    half = int(np.ceil(vsini / dv))
    x = np.arange(-half, half + 1) * dv / vsini
    inside = np.abs(x) < 1.0
    one_minus_x2 = np.where(inside, 1.0 - x**2, 0.0)
    kernel = 2.0 * (1.0 - epsilon) * np.sqrt(one_minus_x2) + 0.5 * np.pi * epsilon * one_minus_x2
    return kernel / kernel.sum()


def rotational_broaden(spec: Spec, vsini: float, epsilon: float, *, oversample: float = 2.0) -> Spec:
    """Apply rotational broadening with a linear limb-darkening law.

    The kernel is that of Gray, *The Observation and Analysis of Stellar
    Photospheres*. Unlike a single kernel computed at the central wavelength, the kernel is
    applied in velocity space, so ``vsini`` is exact at every wavelength.

    Raises:
        ValueError: If ``vsini < 0`` or ``epsilon`` is outside ``[0, 1]``.
    """
    if vsini < 0:
        raise ValueError(f"vsini must be >= 0, got {vsini}")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
    if vsini == 0:
        return spec
    dv = _velocity_step(spec, vsini, oversample)
    kernel = rotation_kernel(vsini, epsilon, dv)
    flux = _apply_velocity_kernel(spec, dv, lambda f: convolve1d(f, kernel, mode="nearest"))
    return spec.with_flux(flux).with_history(f"rotational broadening vsini = {vsini:g} km/s, eps = {epsilon:g}")


# ----------------------------------------------------------------------------- noise
NoiseModel = Literal["constant", "photon"]


def add_noise(
    spec: Spec,
    snr: float,
    rng: np.random.Generator,
    *,
    model: NoiseModel = "constant",
    reference: float | None = None,
) -> Spec:
    """Add Gaussian noise so that the continuum has signal-to-noise ``snr``.

    Args:
        spec: Noiseless spectrum (normalised or not).
        snr: Target SNR per pixel at the continuum level (> 0).
        rng: Random generator (pass a seeded one for reproducibility).
        model: ``"constant"`` uses ``sigma = reference / snr`` everywhere
            (read-noise/sky-dominated regime); ``"photon"`` uses
            ``sigma = sqrt(flux * reference) / snr`` (photon-noise regime,
            lower noise in line cores).
        reference: Continuum level; defaults to the 95th percentile of the
            flux, which is ~1 for normalised spectra.

    Raises:
        ValueError: For non-positive ``snr`` or unknown ``model``.
    """
    if not snr > 0:
        raise ValueError(f"snr must be positive, got {snr}")
    ref = float(np.nanpercentile(spec.flux, 95)) if reference is None else float(reference)
    match model:
        case "constant":
            sigma = np.full(spec.flux.shape, ref / snr)
        case "photon":
            sigma = np.sqrt(np.clip(spec.flux, 0.0, None) * ref) / snr
        case _:
            raise ValueError(f"unknown noise model {model!r}")
    noisy = spec.flux + rng.standard_normal(spec.flux.size) * sigma
    return spec.with_flux(noisy).with_history(f"Gaussian noise added, SNR = {snr:g} ({model})")
