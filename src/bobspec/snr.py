"""Signal-to-noise estimators."""

from __future__ import annotations

import numpy as np

_DER_SNR_CONST = 0.6052697
"""Normalisation of the DER_SNR noise estimator (Stoehr et al. 2008)."""


def der_snr(flux: np.ndarray) -> float:
    """DER_SNR estimate (Stoehr et al. 2008, ASPC 394, 505).

    ``noise = 0.6052697 * median(|2 f_i - f_{i-2} - f_{i+2}|)``, robust to
    absorption lines and unbiased for white noise sampled at >= 2.5 pixels per
    resolution element. In very crowded spectra at high SNR the line curvature
    inflates the noise term, so DER_SNR becomes a lower limit. Non-finite
    values are dropped first.

    Raises:
        ValueError: If fewer than 5 finite pixels are available.
    """
    f = np.asarray(flux, float)
    f = f[np.isfinite(f)]
    if f.size < 5:
        raise ValueError("DER_SNR needs at least 5 finite pixels")
    noise = _DER_SNR_CONST * np.median(np.abs(2.0 * f[2:-2] - f[:-4] - f[4:]))
    return float(np.median(f) / noise) if noise > 0 else np.inf


def region_snr(flux: np.ndarray) -> float:
    """Mean over sample standard deviation; only meaningful on line-free continuum.

    Raises:
        ValueError: If fewer than 3 finite pixels are available.
    """
    f = np.asarray(flux, float)
    f = f[np.isfinite(f)]
    if f.size < 3:
        raise ValueError("need at least 3 finite pixels")
    std = np.std(f, ddof=1)
    return float(np.mean(f) / std) if std > 0 else np.inf


def snr_profile(
    wave: np.ndarray, flux: np.ndarray, mask: np.ndarray | None = None, chunk: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """DER_SNR in consecutive chunks of ``chunk`` contiguous good pixels.

    Chunks never straddle masked or non-finite pixels, so telluric gaps do not
    bias the noise estimate. Run it on a normalised spectrum.

    Returns:
        Mean wavelength and DER_SNR of each chunk (possibly empty arrays).
    """
    if chunk < 5:
        raise ValueError(f"chunk must be >= 5, got {chunk}")
    good = np.isfinite(flux) if mask is None else (np.isfinite(flux) & mask)
    idx = np.flatnonzero(good)
    centres: list[float] = []
    values: list[float] = []
    for segment in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        for start in range(0, segment.size - chunk + 1, chunk):
            block = segment[start : start + chunk]
            centres.append(float(wave[block].mean()))
            values.append(der_snr(flux[block]))
    return np.asarray(centres), np.asarray(values)
