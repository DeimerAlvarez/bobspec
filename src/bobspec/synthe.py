"""SYNTHE synthetic spectra and line lists (``.syn`` / ``.lin`` files).

Used by the ``cows`` and ``sline`` commands.

* A SYNTHE spectrum file has four columns: wavelength, normalised flux, line
  flux and continuum flux. When a resolving power is given, line and continuum
  fluxes are both broadened to that resolution and divided, as in ``convolve``.
* A ``.lin`` file lists one line per row; the columns used are 0 wavelength
  [A], 1 species code (e.g. 26.01), 2 log gf, 3 excitation potential [eV],
  8 reference and 9 residual intensity (RI: smaller means stronger line).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from bobspec.constants import FWHM_TO_SIGMA
from bobspec.io import PathLike, Spec, SpectrumReadError, read_spectrum
from bobspec.spectrum import gaussian_broaden

MIN_LIN_COLUMNS = 10
_RESOL_PATTERN = re.compile(r"^#\s*RESOL\s+([0-9.]+(?:[eE][+-]?[0-9]+)?)", re.IGNORECASE)


def synthetic_resolution(path: PathLike) -> float | None:
    """Resolving power written in the ``# RESOL <R>`` header line of a SYNTHE file.

    Only the comment lines at the top of the file are read. Returns None if
    the line is missing or the file cannot be read.
    """
    try:
        with Path(path).open() as handle:
            for line in handle:
                if not line.lstrip().startswith("#"):
                    break
                if match := _RESOL_PATTERN.match(line.strip()):
                    return float(match.group(1))
    except (OSError, UnicodeDecodeError):
        return None
    return None


def read_synthetic(path: PathLike, resolution: float | None = None) -> Spec:
    """Read a synthetic spectrum, optionally degraded to ``resolution``.

    Args:
        path: Synthetic file. With ``resolution``, a 4-column SYNTHE file;
            without, any spectrum bob can read (2-column ASCII, FITS).
        resolution: Resolving power R (> 0) to degrade to, or None to use the
            normalised flux as it is.

    Raises:
        SpectrumReadError: If the file is missing or, with ``resolution``,
            has fewer than four columns.
    """
    if resolution is None:
        return read_spectrum(path)
    if not resolution > 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    path = Path(path)
    if not path.is_file():
        raise SpectrumReadError(f"{path}: file not found")
    try:
        data = np.loadtxt(path, comments="#", ndmin=2)
    except ValueError as exc:
        raise SpectrumReadError(f"{path}: {exc}") from exc
    if data.shape[1] < 4:
        raise SpectrumReadError(f"{path}: degrading needs a 4-column SYNTHE file, found {data.shape[1]} columns")
    line = gaussian_broaden(Spec(data[:, 0], data[:, 2]), resolution)
    continuum = gaussian_broaden(Spec(data[:, 0], data[:, 3]), resolution)
    with np.errstate(divide="ignore", invalid="ignore"):
        flux = line.flux / continuum.flux
    return Spec(line.wave, flux, name=path.name).with_history(f"SYNTHE spectrum degraded to R = {resolution:g}")


@dataclass(frozen=True, slots=True)
class LineList:
    """Atomic line list (one entry per line, sorted by wavelength)."""

    wave: np.ndarray
    code: np.ndarray
    loggf: np.ndarray
    ep: np.ndarray
    ref: np.ndarray
    ri: np.ndarray
    split: np.ndarray
    """1 if the entry stands for several split components (see :func:`merge_split_lines`)."""

    def __len__(self) -> int:
        return self.wave.size

    def subset(self, mask: np.ndarray) -> LineList:
        """Entries where ``mask`` is True (boolean mask or index array)."""
        return LineList(
            wave=self.wave[mask],
            code=self.code[mask],
            loggf=self.loggf[mask],
            ep=self.ep[mask],
            ref=self.ref[mask],
            ri=self.ri[mask],
            split=self.split[mask],
        )


def read_linelist(path: PathLike) -> LineList:
    """Read a SYNTHE ``.lin`` file (columns 0, 1, 2, 3, 8, 9; see module docstring).

    Raises:
        SpectrumReadError: If the file is missing, empty or has fewer than 10 columns.
    """
    path = Path(path)
    if not path.is_file():
        raise SpectrumReadError(f"{path}: file not found")
    try:
        data = np.atleast_1d(np.genfromtxt(path, comments="#", dtype=None, encoding="utf-8"))
    except ValueError as exc:  # rows with different numbers of columns
        raise SpectrumReadError(f"{path}: {exc}") from exc
    names = data.dtype.names
    if data.size == 0 or names is None or len(names) < MIN_LIN_COLUMNS:
        found = 0 if names is None else len(names)
        raise SpectrumReadError(f"{path}: expected at least {MIN_LIN_COLUMNS} columns per line, found {found}")
    try:
        columns = [np.asarray(data[names[i]], dtype=float) for i in (0, 1, 2, 3)]
        ri = np.asarray(data[names[9]], dtype=float)
    except ValueError as exc:
        raise SpectrumReadError(f"{path}: non-numeric value in columns 1-4 or 10 ({exc})") from exc
    order = np.argsort(columns[0], kind="stable")
    ref = np.asarray(data[names[8]]).astype(str)
    wave, code, loggf, ep = (c[order] for c in columns)
    split = np.zeros(order.size, dtype=int)
    return LineList(wave=wave, code=code, loggf=loggf, ep=ep, ref=ref[order], ri=ri[order], split=split)


def merge_split_lines(lines: LineList, tolerance: float = 0.1) -> LineList:
    """Merge split components (e.g. isotopic) into one entry each.

    Entries with the same species code, log gf and excitation potential whose
    wavelengths follow each other within ``tolerance`` Angstrom form a group.
    A group with more than one member is replaced by its strongest member
    (smallest RI) with ``split = 1``. Same logic as the original ``sline.py``.

    Returns:
        The merged list, sorted by wavelength.
    """
    if len(lines) == 0:
        return lines
    order = np.lexsort((lines.wave, lines.ep, lines.loggf, lines.code))
    s = lines.subset(order)
    new_group = np.concatenate(
        [
            [True],
            (np.diff(s.code) != 0)
            | ~np.isclose(np.diff(s.loggf), 0.0)
            | ~np.isclose(np.diff(s.ep), 0.0)
            | (np.diff(s.wave) > tolerance),
        ]
    )
    group_id = np.cumsum(new_group)
    keep, split = [], []
    for gid in np.unique(group_id):
        members = np.flatnonzero(group_id == gid)
        keep.append(members[np.argmin(s.ri[members])])
        split.append(int(members.size > 1))
    merged = replace(s.subset(np.asarray(keep)), split=np.asarray(split, dtype=int))
    return merged.subset(np.argsort(merged.wave, kind="stable"))


def blend_flags(wave: np.ndarray, neighbours: np.ndarray, resolution: float, nsig: float = 1.5) -> np.ndarray:
    """``'clean'`` if no other line of ``neighbours`` lies within ``nsig`` sigma, else ``'/'``.

    ``sigma = wave / R / 2.355`` is the instrumental Gaussian sigma. Each line
    of ``wave`` must also be in ``neighbours`` (it is not counted as its own
    blend). With ``resolution <= 0`` the test is impossible and ``'-'`` is
    returned.
    """
    wave = np.asarray(wave, float)
    if resolution <= 0:
        return np.full(wave.size, "-", dtype=object)
    half = nsig * wave / resolution * FWHM_TO_SIGMA
    nb = np.sort(np.asarray(neighbours, float))
    count = np.searchsorted(nb, wave + half, side="right") - np.searchsorted(nb, wave - half, side="left")
    return np.where(count <= 1, "clean", "/").astype(object)
