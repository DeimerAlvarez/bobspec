"""Reading and writing 1-D spectra.

Supported inputs:

* FITS images with a linear (``CRVAL1``/``CDELT1`` or ``CD1_1``), FITS-standard
  logarithmic (``CTYPE1 = '...-LOG'``) or IRAF log10 (``DC-FLAG = 1``)
  wavelength solution, in any unit given by ``CUNIT1``;
* FITS binary tables such as ESO Phase-3 products (``WAVE``/``FLUX`` columns,
  either one row of arrays or one row per pixel);
* whitespace-separated ASCII files (wavelength and flux in the first two
  columns, ``#`` comments allowed).

Wavelengths are always returned in Angstrom and sorted in increasing order.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final, Self

import numpy as np
from astropy import units as u
from astropy.io import fits

from bobspec.constants import C_KMS

log = logging.getLogger(__name__)

PathLike = str | os.PathLike[str]

FITS_SUFFIXES: Final[tuple[str, ...]] = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fts.gz")
_WAVE_COLUMNS: Final[tuple[str, ...]] = ("WAVE", "WAVELENGTH", "LAMBDA", "WAVE_AIR", "WAVE_VAC")
_FLUX_COLUMNS: Final[tuple[str, ...]] = ("FLUX", "FLUX_REDUCED", "FLUX_NORM", "NORM_FLUX", "SPEC")
_WCS_KEYS: Final[tuple[str, ...]] = (
    "CRVAL1",
    "CDELT1",
    "CD1_1",
    "CRPIX1",
    "CTYPE1",
    "CUNIT1",
    "DC-FLAG",
    "LTV1",
    "LTM1_1",
    "WAT0_001",
    "WAT1_001",
    "WAT2_001",
    "BUNIT",
)


class SpectrumReadError(Exception):
    """Raised when a file cannot be interpreted as a 1-D spectrum."""


@dataclass(frozen=True, slots=True, eq=False)
class Spec:
    """Immutable 1-D spectrum on a strictly increasing wavelength grid.

    Attributes:
        wave: Wavelengths in Angstrom (strictly increasing, finite).
        flux: Flux values (may contain NaN for bad pixels).
        header: FITS header carried along for provenance.
        name: Human-readable label, usually the source file name.

    Raises:
        ValueError: If the arrays are not 1-D, differ in length, contain
            fewer than two points, non-finite or duplicated wavelengths.
    """

    wave: np.ndarray
    flux: np.ndarray
    header: fits.Header = field(default_factory=fits.Header, repr=False)
    name: str = ""

    def __post_init__(self) -> None:
        wave = np.asarray(self.wave, dtype=np.float64)
        flux = np.asarray(self.flux, dtype=np.float64)
        if wave.ndim != 1 or flux.ndim != 1:
            raise ValueError(f"wave and flux must be 1-D, got {wave.shape} and {flux.shape}")
        if wave.shape != flux.shape:
            raise ValueError(f"wave and flux lengths differ: {wave.size} != {flux.size}")
        if wave.size < 2:
            raise ValueError("a spectrum needs at least two points")
        if not np.all(np.isfinite(wave)):
            raise ValueError("wavelength array contains non-finite values")
        order = np.argsort(wave, kind="stable")
        wave, flux = wave[order], flux[order]
        if np.any(np.diff(wave) <= 0):
            raise ValueError("wavelength array contains duplicated values")
        wave.flags.writeable = False
        flux.flags.writeable = False
        object.__setattr__(self, "wave", wave)
        object.__setattr__(self, "flux", flux)

    def __len__(self) -> int:
        return self.wave.size

    @property
    def finite(self) -> np.ndarray:
        """Boolean mask of pixels with finite flux."""
        return np.isfinite(self.flux)

    @property
    def velocity_step(self) -> float:
        """Finest pixel size expressed as a velocity (km/s)."""
        return float(C_KMS * np.min(np.diff(self.wave) / self.wave[:-1]))

    def is_uniform(self, rtol: float = 1e-5) -> bool:
        """Return True if the wavelength step is constant within ``rtol``."""
        steps = np.diff(self.wave)
        return bool(np.allclose(steps, steps[0], rtol=rtol, atol=0.0))

    def with_flux(self, flux: np.ndarray) -> Self:
        """Return a copy with new flux values on the same grid."""
        return replace(self, flux=flux)

    def with_wave(self, wave: np.ndarray) -> Self:
        """Return a copy with a new wavelength grid and the same flux."""
        return replace(self, wave=wave)

    def on_grid(self, wave: np.ndarray, flux: np.ndarray) -> Self:
        """Return a copy with both a new wavelength grid and new flux values."""
        return replace(self, wave=wave, flux=flux)

    def with_history(self, text: str) -> Self:
        """Return a copy whose header has an extra ``HISTORY`` card."""
        header = self.header.copy()
        header.add_history(f"bobspec: {text}")
        return replace(self, header=header)

    def select(self, w1: float, w2: float) -> Self:
        """Return the part of the spectrum with ``min(w1, w2) <= wave <= max(w1, w2)``.

        Raises:
            ValueError: If fewer than two pixels fall inside the interval.
        """
        lo, hi = sorted((float(w1), float(w2)))
        keep = (self.wave >= lo) & (self.wave <= hi)
        if np.count_nonzero(keep) < 2:
            raise ValueError(
                f"range {lo:.3f}-{hi:.3f} A contains fewer than 2 pixels "
                f"(spectrum covers {self.wave[0]:.3f}-{self.wave[-1]:.3f} A)"
            )
        return replace(self, wave=self.wave[keep], flux=self.flux[keep])


# ----------------------------------------------------------------------------- reading
def is_fits(path: PathLike) -> bool:
    """Return True if ``path`` has a FITS-like suffix (case-insensitive, gzip allowed)."""
    return str(path).lower().endswith(FITS_SUFFIXES)


def read_spectrum(path: PathLike) -> Spec:
    """Read a 1-D spectrum from a FITS or ASCII file.

    Args:
        path: File to read. The format is chosen from the suffix.

    Returns:
        The spectrum, with wavelengths in Angstrom sorted in increasing order.

    Raises:
        SpectrumReadError: If the file is missing or cannot be interpreted.
    """
    path = Path(path)
    if not path.is_file():
        raise SpectrumReadError(f"{path}: file not found")
    try:
        if is_fits(path):
            wave, flux, header = _read_fits(path)
        else:
            wave, flux = _read_ascii(path)
            header = fits.Header()
        return Spec(wave, flux, header, name=path.name)
    except SpectrumReadError:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise SpectrumReadError(f"{path}: {exc}") from exc


def read_header(path: PathLike) -> fits.Header:
    """Return a copy of the primary header of a FITS file.

    Raises:
        SpectrumReadError: If the file is missing, not FITS, or unreadable.
    """
    path = Path(path)
    if not path.is_file():
        raise SpectrumReadError(f"{path}: file not found")
    if not is_fits(path):
        raise SpectrumReadError(f"{path}: not a FITS file")
    try:
        with fits.open(path, memmap=False) as hdul:
            return hdul[0].header.copy()
    except OSError as exc:
        raise SpectrumReadError(f"{path}: {exc}") from exc


def wavelength_from_wcs(header: fits.Header, npix: int) -> np.ndarray:
    """Build the wavelength array (Angstrom) of a 1-D FITS image.

    Args:
        header: Header containing the axis-1 world coordinate keywords.
        npix: Number of pixels along axis 1.

    Raises:
        SpectrumReadError: If ``CRVAL1`` or both ``CDELT1``/``CD1_1`` are missing.
    """
    if "CRVAL1" not in header:
        raise SpectrumReadError("missing CRVAL1 keyword")
    step = header.get("CDELT1", header.get("CD1_1"))
    if step is None:
        raise SpectrumReadError("missing CDELT1 and CD1_1 keywords")
    crval, step = float(header["CRVAL1"]), float(step)
    offset = (np.arange(1, npix + 1, dtype=np.float64) - float(header.get("CRPIX1", 1.0))) * step

    ctype = str(header.get("CTYPE1", "")).strip().upper()
    if int(header.get("DC-FLAG", 0)) == 1:  # IRAF: log10(lambda) is linear in pixels
        wave = 10.0 ** (crval + offset)
    elif ctype.endswith("-LOG"):  # FITS WCS paper III logarithmic axis
        wave = crval * np.exp(offset / crval)
    else:
        wave = crval + offset
    return wave * _to_angstrom(header.get("CUNIT1"))


def length_factor(unit: str | u.UnitBase | None) -> float | None:
    """Factor converting ``unit`` to Angstrom, or None if it is not a length unit."""
    if unit in (None, ""):
        return None
    for candidate in (str(unit), str(unit).lower()):
        try:
            return float(u.Unit(candidate).to(u.AA))
        except (ValueError, u.UnitConversionError):
            continue
    return None


def _to_angstrom(unit: str | u.UnitBase | None) -> float:
    """Return the factor converting ``unit`` to Angstrom (1 if unknown or empty)."""
    if unit in (None, ""):
        return 1.0
    for candidate in (str(unit), str(unit).lower()):
        try:
            return float(u.Unit(candidate).to(u.AA))
        except (ValueError, u.UnitConversionError):
            continue
    log.warning("Unknown wavelength unit %r: assuming Angstrom.", unit)
    return 1.0


def _read_ascii(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#", usecols=(0, 1), ndmin=2, dtype=np.float64)
    return data[:, 0], data[:, 1]


def _squeeze_1d(data: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(data, dtype=np.float64))
    if arr.ndim != 1:
        raise SpectrumReadError(
            f"data array has shape {np.shape(data)}; multi-dimensional files "
            "(echelle orders, IRAF multispec bands) are not supported"
        )
    return arr


def _read_fits(path: Path) -> tuple[np.ndarray, np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        primary = hdul[0].header
        for hdu in hdul:
            if isinstance(hdu, fits.PrimaryHDU | fits.ImageHDU) and hdu.data is not None:
                flux = _squeeze_1d(hdu.data)
                wave = wavelength_from_wcs(hdu.header, flux.size)
                header = primary.copy()
                if hdu is not hdul[0]:
                    header.extend(hdu.header, unique=True, update=True)
                return wave, flux, header
            if isinstance(hdu, fits.BinTableHDU) and (table := _read_table(hdu)) is not None:
                return *table, primary.copy()
    raise SpectrumReadError("no 1-D image or WAVE/FLUX table found")


def _read_table(hdu: fits.BinTableHDU) -> tuple[np.ndarray, np.ndarray] | None:
    names = {name.upper(): name for name in hdu.columns.names}
    wcol = next((names[c] for c in _WAVE_COLUMNS if c in names), None)
    fcol = next((names[c] for c in _FLUX_COLUMNS if c in names), None)
    if wcol is None or fcol is None or hdu.data is None or len(hdu.data) == 0:
        return None

    def column(name: str) -> np.ndarray:
        arr = np.asarray(hdu.data[name], dtype=np.float64)
        if arr.ndim == 2:  # ESO Phase-3: a single row holding the whole spectrum
            if arr.shape[0] != 1:
                raise SpectrumReadError(f"column {name} holds {arr.shape[0]} spectra; expected one")
            arr = arr[0]
        return arr

    return column(wcol) * _to_angstrom(hdu.columns[wcol].unit), column(fcol)


# ----------------------------------------------------------------------------- tables
@dataclass(frozen=True, slots=True)
class TableColumn:
    """Description of one column of a FITS binary table."""

    index: int
    name: str
    unit: str
    format: str
    length: int
    """Number of values per row (the spectrum length for ESO Phase-3 tables)."""


def _open_table(hdul: fits.HDUList, ext: int | None) -> tuple[int, fits.BinTableHDU]:
    if ext is not None:
        if not 0 <= ext < len(hdul):
            raise SpectrumReadError(f"extension {ext} does not exist (the file has {len(hdul)})")
        hdu = hdul[ext]
        if not isinstance(hdu, fits.BinTableHDU):
            raise SpectrumReadError(f"extension {ext} is a {type(hdu).__name__}, not a binary table")
        return ext, hdu
    for i, hdu in enumerate(hdul):
        if isinstance(hdu, fits.BinTableHDU):
            return i, hdu
    raise SpectrumReadError("the file contains no binary table")


def list_table_columns(path: PathLike, ext: int | None = None) -> tuple[int, int, list[TableColumn]]:
    """Describe the columns of a FITS binary table.

    Args:
        path: FITS file.
        ext: Extension number; default: the first binary table (1 for ESO products).

    Returns:
        The extension number, the number of rows and the columns.

    Raises:
        SpectrumReadError: If the file is missing or has no binary table.
    """
    path = Path(path)
    if not path.is_file():
        raise SpectrumReadError(f"{path}: file not found")
    try:
        with fits.open(path, memmap=False) as hdul:
            ext, hdu = _open_table(hdul, ext)
            nrows = 0 if hdu.data is None else len(hdu.data)
            columns = [
                TableColumn(
                    index=i,
                    name=col.name,
                    unit=str(col.unit or ""),
                    format=str(col.format),
                    length=int(np.size(hdu.data.field(i)[0])) if nrows else 0,
                )
                for i, col in enumerate(hdu.columns)
            ]
    except OSError as exc:
        raise SpectrumReadError(f"{path}: {exc}") from exc
    return ext, nrows, columns


def _column_index(spec: str | int, columns: list[TableColumn]) -> int:
    """Resolve a 0-based index or a (case-insensitive) column name."""
    names = ", ".join(f"{c.index}={c.name}" for c in columns)
    if isinstance(spec, int) or str(spec).lstrip("-").isdigit():
        index = int(spec)
        if not 0 <= index < len(columns):
            raise SpectrumReadError(f"column {index} does not exist; columns are: {names}")
        return index
    matches = [c.index for c in columns if c.name.upper() == str(spec).upper()]
    if not matches:
        raise SpectrumReadError(f"no column named {spec!r}; columns are: {names}")
    return matches[0]


def read_table_columns(
    path: PathLike,
    xcol: str | int,
    ycol: str | int,
    *,
    ext: int | None = None,
    row: int = 0,
    convert_wave: bool = True,
) -> tuple[Spec, TableColumn, TableColumn]:
    """Extract two columns of a FITS binary table (e.g. a reduced ESO spectrum) as a spectrum.

    Works for both table layouts: one row holding whole arrays (ESO Phase 3,
    ``data[row][column]``) and one row per pixel.

    Args:
        path: FITS file.
        xcol: x (wavelength) column: 0-based index or name.
        ycol: y column (flux, error, ...): 0-based index or name.
        ext: Extension number; default: the first binary table.
        row: Row to read when each row holds a whole spectrum.
        convert_wave: Convert the x column to Angstrom when its unit is a length.

    Returns:
        The spectrum (with the primary header plus a ``HISTORY`` card) and the
        descriptions of the two columns.

    Raises:
        SpectrumReadError: For missing files, columns or rows, or mismatched lengths.
    """
    path = Path(path)
    ext, nrows, columns = list_table_columns(path, ext)
    ix, iy = _column_index(xcol, columns), _column_index(ycol, columns)
    if nrows == 0:
        raise SpectrumReadError(f"{path}: the table is empty")
    with fits.open(path, memmap=False) as hdul:
        data = hdul[ext].data
        header = hdul[0].header.copy()

        def values(index: int) -> np.ndarray:
            column = data.field(index)
            if np.ndim(column[0]) >= 1:  # each row holds a whole spectrum
                if not 0 <= row < nrows:
                    raise SpectrumReadError(f"row {row} does not exist (the table has {nrows})")
                return np.asarray(column[row], dtype=np.float64).ravel()
            return np.asarray(column, dtype=np.float64)  # one row per pixel

        x, y = values(ix), values(iy)
    cx, cy = columns[ix], columns[iy]
    if x.size != y.size:
        raise SpectrumReadError(f"columns {cx.name} ({x.size} values) and {cy.name} ({y.size}) differ in length")
    if convert_wave:
        factor = length_factor(cx.unit)
        if factor is None:
            if cx.unit:
                log.warning("Column %s has unit %r, not a length: values kept as they are.", cx.name, cx.unit)
        elif factor != 1.0:
            x = x * factor
            log.info("Column %s converted from %s to Angstrom.", cx.name, cx.unit)
    finite = np.isfinite(x)
    if not finite.all():
        log.warning("%d rows with non-finite %s dropped.", int((~finite).sum()), cx.name)
        x, y = x[finite], y[finite]
    header.add_history(f"bobspec: columns {cx.name} ({ix}) and {cy.name} ({iy}) of ext {ext} of {path.name}")
    try:
        spec = Spec(x, y, header, name=path.name)
    except ValueError as exc:
        raise SpectrumReadError(f"{path}: {exc}") from exc
    return spec, cx, cy


# ----------------------------------------------------------------------------- writing
def write_spectrum(path: PathLike, spec: Spec, *, overwrite: bool = True, ascii_header: str = "wave[A] flux") -> Path:
    """Write a spectrum; FITS if the suffix is FITS-like, ASCII otherwise.

    FITS output needs a uniform grid: non-uniform spectra are linearly
    resampled onto the finest native step (a warning is logged). The input
    header is propagated, with stale WCS keywords removed.

    Args:
        path: Output file.
        spec: Spectrum to save.
        overwrite: Overwrite an existing file.
        ascii_header: Column description written as the first line of ASCII files.

    Returns:
        The path written.

    Raises:
        FileExistsError: If the file exists and ``overwrite`` is False.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    if is_fits(path):
        _write_fits(path, spec)
    else:
        _write_ascii(path, spec, ascii_header)
    log.debug("Spectrum saved as %s", path)
    return path


def _clean_header(header: fits.Header) -> fits.Header:
    clean = header.copy(strip=True)
    for key in _WCS_KEYS:
        clean.remove(key, ignore_missing=True, remove_all=True)
    return clean


def _write_fits(path: Path, spec: Spec) -> None:
    wave, flux = spec.wave, spec.flux
    if not spec.is_uniform():
        step = float(np.min(np.diff(wave)))
        new_wave = wave[0] + step * np.arange(int(np.floor((wave[-1] - wave[0]) / step)) + 1)
        log.warning("Non-uniform grid resampled to a constant step of %.6g A for FITS output.", step)
        flux = np.interp(new_wave, wave, flux)
        wave = new_wave
    header = _clean_header(spec.header)
    header["CRVAL1"] = (float(wave[0]), "wavelength of reference pixel")
    header["CDELT1"] = (float(wave[1] - wave[0]), "wavelength step")
    header["CRPIX1"] = (1.0, "reference pixel")
    header["CTYPE1"] = "WAVE"
    header["CUNIT1"] = "Angstrom"
    fits.PrimaryHDU(flux, header=header).writeto(path, overwrite=True, output_verify="silentfix")


def _write_ascii(path: Path, spec: Spec, header: str) -> None:
    np.savetxt(
        path,
        np.column_stack([spec.wave, spec.flux]),
        fmt=("%.6f", "%.8g"),
        header=header,
    )
