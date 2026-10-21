"""Co-addition of several exposures of the same star with cosmic-ray rejection.

Algorithm (generalising the original ``combine.py``):

1. **Alignment.** Every spectrum is put on the wavelength grid of the first
   one. Grids that agree within ``tolerance_pix`` are used as they are (no
   interpolation, so the noise stays uncorrelated); otherwise the flux is
   linearly interpolated. Optionally each exposure is first shifted to the
   barycentric frame, which matters when exposures come from different nights.
2. **Scaling.** Each spectrum is divided by the median of its ratio to the
   median spectrum, removing differences in exposure time, seeing or slit losses.
3. **Rejection.** Each spectrum is compared with a reference and the residual
   with its local robust noise (running ``1.4826 * MAD``). Only *positive*
   outliers above ``sigma`` are rejected (cosmic rays), and the mask can be
   grown by a few pixels. The default reference for spectrum *i* is the median
   of the *other* spectra. The original code used the median of *all* spectra,
   which contains spectrum *i* itself: its residuals are then exactly zero
   wherever it is the median (1 pixel in N), the MAD underestimates the noise
   (by a factor ~0.45 for N = 3, ~0.73 for N = 5) and ~2.7 % (N = 3) or ~0.3 %
   (N = 5) of pure-noise pixels are rejected, always on the positive side,
   biasing the combined flux low. ``reference="median"`` reproduces the
   original behaviour exactly.
4. **Combination.** Mean (optionally inverse-variance weighted), median, or
   sum (= mean x number of exposures, so pixels with rejected values are
   compensated). The empirical error is ``std / sqrt(n_used)``.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from astropy.io import fits

from bobspec.constants import C_KMS, MAD_TO_SIGMA
from bobspec.io import Spec
from bobspec.spectrum import interpolate

log = logging.getLogger(__name__)

Method = Literal["mean", "median", "sum"]
Reference = Literal["others", "median"]
MIN_FOR_REJECTION = 3


@dataclass(frozen=True, slots=True)
class CombineResult:
    """Outcome of :func:`combine_spectra`.

    Attributes:
        spectrum: Combined spectrum on the reference grid (header from the first input).
        error: Empirical standard error per pixel (NaN where fewer than 2 values).
        n_used: Number of accepted values per pixel.
        rejected: Boolean mask of rejected values, shape ``(n_spectra, n_pixels)``.
        scale: Scale factor that each input was divided by.
        noise: Global robust noise of each scaled input (median of the local noise).
        significance: Residual over local noise, same shape as ``rejected``.
        scaled: Aligned and rescaled input fluxes, same shape as ``rejected``.
        reference: Reference spectrum used for each input, same shape as ``rejected``.
        resampled: Which inputs had to be interpolated onto the reference grid.
    """

    spectrum: Spec
    error: np.ndarray = field(repr=False)
    n_used: np.ndarray = field(repr=False)
    rejected: np.ndarray = field(repr=False)
    scale: np.ndarray
    noise: np.ndarray
    significance: np.ndarray = field(repr=False)
    scaled: np.ndarray = field(repr=False)
    reference: np.ndarray = field(repr=False)
    resampled: tuple[bool, ...] = ()


def running_nanmedian(values: np.ndarray, window: int) -> np.ndarray:
    """Running median ignoring NaN; the window shrinks at the edges.

    Vectorised with a sliding window view: ``O(N * window)`` time and memory.

    Raises:
        ValueError: If ``window`` is not a positive odd integer.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd integer, got {window}")
    half = window // 2
    padded = np.pad(np.asarray(values, float), half, constant_values=np.nan)
    with _quiet_nan_warnings():
        return np.nanmedian(np.lib.stride_tricks.sliding_window_view(padded, window), axis=1)


@contextmanager
def _quiet_nan_warnings() -> Iterator[None]:
    """Silence 'All-NaN slice' and empty-slice warnings (NaN results are intended)."""
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def local_robust_sigma(residual: np.ndarray, window: int) -> np.ndarray:
    """Local noise ``1.4826 * running MAD``, floored at a quarter of its median.

    Raises:
        ValueError: If the noise cannot be estimated anywhere.
    """
    centre = running_nanmedian(residual, window)
    sigma = MAD_TO_SIGMA * running_nanmedian(np.abs(residual - centre), window)
    good = np.isfinite(sigma) & (sigma > 0)
    if not good.any():
        raise ValueError("unable to estimate the local noise (constant or empty spectrum)")
    typical = float(np.median(sigma[good]))
    sigma[~good] = typical
    return np.maximum(sigma, 0.25 * typical)


def _noise_or_nan(residual: np.ndarray, window: int) -> np.ndarray:
    try:
        return local_robust_sigma(residual, window)
    except ValueError:
        return np.full(residual.shape, np.nan)


def grow_mask(mask: np.ndarray, npix: int) -> np.ndarray:
    """Extend True pixels by ``npix`` neighbours on each side (along the last axis)."""
    out = mask.copy()
    for shift in range(1, npix + 1):
        out[..., shift:] |= mask[..., :-shift]
        out[..., :-shift] |= mask[..., shift:]
    return out


def align(spectra: Sequence[Spec], tolerance_pix: float = 0.01) -> tuple[np.ndarray, np.ndarray, tuple[bool, ...]]:
    """Put all spectra on the grid of the first one.

    Returns:
        The reference wavelength grid, an ``(n, npix)`` flux array (NaN where a
        spectrum does not cover the grid) and, per spectrum, whether it was
        interpolated.
    """
    ref = spectra[0].wave
    pixel = float(np.median(np.diff(ref)))
    fluxes, resampled = [], []
    for spec in spectra:
        same = spec.wave.size == ref.size and np.max(np.abs(spec.wave - ref)) <= tolerance_pix * pixel
        if same:
            fluxes.append(spec.flux)
        else:
            log.info("%s: wavelength grid differs from the first spectrum: interpolated.", spec.name or "spectrum")
            fluxes.append(interpolate(spec, ref))
        resampled.append(not same)
    return ref, np.vstack(fluxes), tuple(resampled)


def combine_spectra(
    spectra: Sequence[Spec],
    *,
    method: Method = "mean",
    sigma: float | None = 5.0,
    window: int = 51,
    grow: int = 1,
    rescale: bool = True,
    weighted: bool = False,
    tolerance_pix: float = 0.01,
    reference: Reference = "others",
) -> CombineResult:
    """Combine exposures of the same star with positive-outlier (cosmic-ray) rejection.

    Args:
        spectra: Input spectra (already shifted to a common frame if needed).
        method: ``"mean"``, ``"median"`` or ``"sum"`` (mean times the number of inputs).
        sigma: Rejection threshold in local-noise units; ``None`` disables rejection.
        window: Odd window (pixels) of the local noise estimate.
        grow: Pixels added on each side of a rejected pixel.
        rescale: Bring all inputs to the flux level of the median spectrum first.
        weighted: Inverse-variance weights from each input's global noise
            (useful when exposure times differ); ignored by ``"median"``.
        tolerance_pix: Largest grid difference (in pixels) accepted without interpolation.
        reference: ``"others"`` (median of the other spectra, default) or
            ``"median"`` (median of all spectra, as the original code).

    Returns:
        A :class:`CombineResult`.

    Raises:
        ValueError: For fewer than 2 inputs, fewer than 3 with rejection
            enabled, a non-positive threshold or an even/invalid window.
    """
    n = len(spectra)
    if n < 2:
        raise ValueError(f"at least 2 spectra are needed, got {n}")
    if sigma is not None and sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if sigma is not None and n < MIN_FOR_REJECTION:
        raise ValueError(f"cosmic-ray rejection needs at least {MIN_FOR_REJECTION} spectra (got {n}); disable it")
    if grow < 0:
        raise ValueError(f"grow must be >= 0, got {grow}")
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd integer, got {window}")
    if reference not in ("others", "median"):
        raise ValueError(f"unknown reference {reference!r}")
    if method not in ("mean", "median", "sum"):
        raise ValueError(f"unknown method {method!r}")

    wave, flux, resampled = align(spectra, tolerance_pix)
    with _quiet_nan_warnings():
        scale = np.ones(n)
        if rescale:
            preliminary = np.nanmedian(flux, axis=0)
            for i in range(n):
                good = np.isfinite(flux[i]) & np.isfinite(preliminary) & (preliminary != 0)
                if not good.any():
                    raise ValueError(f"{spectra[i].name or f'spectrum {i + 1}'} does not overlap the others")
                scale[i] = np.median(flux[i, good] / preliminary[good])
            if np.any(~(scale > 0)):
                raise ValueError(f"non-positive scale factor(s): {scale}")
            flux = flux / scale[:, None]

        if reference == "median":
            references = np.broadcast_to(np.nanmedian(flux, axis=0), flux.shape)
        else:
            references = np.vstack([np.nanmedian(np.delete(flux, i, axis=0), axis=0) for i in range(n)])
        residual = flux - references
        local_noise = np.vstack([_noise_or_nan(r, window) for r in residual])
        is_reference = np.all(np.isnan(local_noise), axis=1)
        if is_reference.all():
            raise ValueError("unable to estimate the noise: are the inputs identical?")
        # A spectrum that *is* the median everywhere has zero residuals: give it the
        # typical noise of the others (it cannot contain outliers relative to itself).
        local_noise[is_reference] = np.nanmedian(local_noise[~is_reference], axis=0)
        significance = residual / local_noise
        rejected = np.zeros(flux.shape, dtype=bool)
        if sigma is not None:
            rejected = grow_mask((significance > sigma) & np.isfinite(flux), grow)

        clean = np.where(rejected, np.nan, flux)
        n_used = np.sum(np.isfinite(clean), axis=0)
        noise = np.median(local_noise, axis=1)
        if method == "median":
            combined = np.nanmedian(clean, axis=0)
        else:
            weights = (1.0 / noise**2 if weighted else np.ones(n))[:, None] * np.isfinite(clean)
            combined = np.nansum(np.nan_to_num(clean) * weights, axis=0) / weights.sum(axis=0)
            if method == "sum":
                combined = combined * n
        error = np.nanstd(clean, axis=0, ddof=1) / np.sqrt(n_used)
    error[n_used < 2] = np.nan
    combined[n_used == 0] = np.nan
    if method == "sum":
        error = error * n

    header = _combined_header(spectra, method, sigma, window, scale)
    out = Spec(wave, combined, header, name="combined")
    return CombineResult(
        out, error, n_used, rejected, scale, noise, significance, flux, np.asarray(references), resampled
    )


def _combined_header(
    spectra: Sequence[Spec], method: str, sigma: float | None, window: int, scale: np.ndarray
) -> fits.Header:
    header = spectra[0].header.copy()
    header["NCOMBINE"] = (len(spectra), "number of combined spectra")
    header["COMBMETH"] = (method, "combination method")
    header["CRSIGMA"] = (sigma if sigma is not None else "none", "cosmic-ray rejection threshold")
    header["CRWIN"] = (window, "local noise window [pix]")
    mjd = _effective_mjd(spectra, scale)
    if mjd is not None:
        header["MJD-MID"] = (mjd, "flux-weighted mid-exposure MJD")
    exptimes = [s.header.get("EXPTIME") for s in spectra]
    if all(isinstance(t, int | float) for t in exptimes):
        header["EXPTOT"] = (float(sum(exptimes)), "total exposure time [s]")
    for spec in spectra:
        header.add_history(f"bobspec: combined input {spec.name}")
    return header


def _effective_mjd(spectra: Sequence[Spec], scale: np.ndarray) -> float | None:
    """Flux-weighted mean of the mid-exposure MJDs, if every input has MJD-OBS."""
    mids = []
    for spec in spectra:
        if "MJD-OBS" not in spec.header:
            return None
        mids.append(float(spec.header["MJD-OBS"]) + 0.5 * float(spec.header.get("EXPTIME", 0.0)) / 86_400.0)
    return float(np.average(mids, weights=scale))


def velocity_spread(corrections: Sequence[float], wave: np.ndarray) -> tuple[float, float]:
    """Spread of barycentric corrections in km/s and in pixels of ``wave``."""
    spread = float(np.ptp(corrections))
    pixel_kms = float(C_KMS * np.median(np.diff(wave) / wave[:-1]))
    return spread, spread / pixel_kms
