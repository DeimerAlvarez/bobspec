"""User-facing commands.

Handlers parse nothing themselves (arguments arrive validated by
:mod:`bobspec.registry`), call the pure functions of the core modules, write
files and print results. Commands that produce numbers return result rows,
which batch mode (``<command> @FILE``) collects into a table.
"""

from __future__ import annotations

import argparse
import logging
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from bobspec import continuum as cont
from bobspec import spectrum as sp
from bobspec.barycentric import Observation, barycentric_correction, mid_exposure_mjd, observation_from_eso_header
from bobspec.constants import MAD_TO_SIGMA
from bobspec.coords import parse_coordinates
from bobspec.io import (
    FITS_SUFFIXES,
    PathLike,
    Spec,
    SpectrumReadError,
    is_fits,
    list_table_columns,
    read_header,
    read_spectrum,
    read_table_columns,
    write_spectrum,
)
from bobspec.lines import LineMeasurement, measure_line
from bobspec.registry import SESSION, Result, Row, UsageError, arg, command, default_table, write_table
from bobspec.rv import cross_correlate, monte_carlo
from bobspec.snr import der_snr, region_snr, snr_profile

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

log = logging.getLogger(__name__)

_OUTPUT_SUFFIXES = (*FITS_SUFFIXES, ".txt", ".dat", ".csv", ".asc")


# ----------------------------------------------------------------------------- argument types
def positive_float(text: str) -> float:
    """Argparse type: float > 0."""
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text}")
    return value


def nonneg_float(text: str) -> float:
    """Argparse type: float >= 0."""
    value = float(text)
    if not value >= 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {text}")
    return value


def positive_int(text: str) -> int:
    """Argparse type: int > 0."""
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text}")
    return value


def nonneg_int(text: str) -> int:
    """Argparse type: int >= 0."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {text}")
    return value


def unit_interval(text: str) -> float:
    """Argparse type: float in [0, 1]."""
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be in [0, 1], got {text}")
    return value


def spline_degree(text: str) -> int:
    """Argparse type: int in 1..5."""
    value = int(text)
    if not 1 <= value <= 5:
        raise argparse.ArgumentTypeError(f"must be in 1..5, got {text}")
    return value


# ----------------------------------------------------------------------------- helpers
def companion(output: PathLike, suffix: str) -> Path:
    """Path next to ``output`` with its spectral suffix replaced by ``suffix``.

    ``companion("../out/star.fits", ".snr")`` -> ``../out/star.snr``.
    """
    path = Path(output)
    name = path.name
    for known in sorted(_OUTPUT_SUFFIXES, key=len, reverse=True):
        if name.lower().endswith(known):
            name = name[: -len(known)]
            break
    return path.with_name(name + suffix)


def _pyplot() -> Any:
    """Import pyplot lazily so that non-graphical commands never need a display."""
    import matplotlib.pyplot as plt

    return plt


def _telluric_mask(spec: Spec) -> np.ndarray:
    return sp.exclusion_mask(spec.wave, SESSION.telluric_regions)


def _save_figure(fig: Figure, path: Path) -> None:
    fig.savefig(path, dpi=150)
    log.info("Figure saved as %s", path)


def _write_snr_file(normalized: Spec, mask: np.ndarray, output: PathLike) -> None:
    centres, values = snr_profile(normalized.wave, normalized.flux, mask)
    if centres.size == 0:
        log.warning("Not enough contiguous continuum pixels for an SNR profile.")
        return
    path = companion(output, ".snr")
    np.savetxt(path, np.column_stack([centres, values]), fmt=("%.3f", "%.1f"), header="wave[A] DER_SNR")
    log.info("SNR profile saved as %s (median DER_SNR %.0f)", path, np.median(values))


def _show_comparison(before: Spec, after: Spec, labels: tuple[str, str], title: str) -> None:
    """Overplot a spectrum before and after an operation and wait until the window is closed."""
    from bobspec.plotting import comparison_figure

    comparison_figure(before, after, labels, title)
    _pyplot().show(block=True)


def _ask(prompt: str, choices: set[str]) -> str:
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            return "q"
        if answer in choices:
            return answer
        print(f"Please answer one of: {', '.join(sorted(c or 'ENTER' for c in choices))}")


def _ask_float(name: str, current: float) -> float:
    while True:
        try:
            text = input(f"New {name} [{current:g}]: ").strip()
        except EOFError:
            return current
        if not text:
            return current
        try:
            return float(text)
        except ValueError:
            print(f"'{text}' is not a number.")


# ----------------------------------------------------------------------------- format conversion
@command("tofits", arg("input", help="spectrum (FITS or ASCII)"), arg("output", help="FITS file"))
def cmd_tofits(ns: argparse.Namespace) -> Result:
    """Convert a spectrum to FITS (resampled to a constant step if needed)."""
    output = ns.output if is_fits(ns.output) else f"{ns.output}.fits"
    write_spectrum(output, read_spectrum(ns.input))
    print(f"Saved {output}")
    return None


@command("totxt", arg("input", help="spectrum (FITS or ASCII)"), arg("output", help="ASCII file"))
def cmd_totxt(ns: argparse.Namespace) -> Result:
    """Convert a spectrum to a two-column ASCII file."""
    output = f"{ns.output}.txt" if is_fits(ns.output) else ns.output
    write_spectrum(output, read_spectrum(ns.input))
    print(f"Saved {output}")
    return None


_ASCII_SUFFIXES = (".txt", ".dat")


@command(
    "read_eso_fits",
    arg("file", help="FITS file with a binary table (e.g. a reduced ESO spectrum)"),
    arg("xcol", nargs="?", help="x (wavelength) column: 0-based index or name"),
    arg("ycol", nargs="?", help="y column (flux, error, ...): 0-based index or name"),
    arg("output", nargs="?", help="output: .txt or .dat (ASCII) or .fits"),
    arg("--ext", type=nonneg_int, help="table extension (default: first binary table, 1 for ESO)"),
    arg("--row", type=nonneg_int, default=0, help="row to read when a row holds a whole spectrum (default 0)"),
    arg("--keep-units", action="store_true", help="do not convert the wavelength column to Angstrom"),
)
def cmd_read_eso_fits(ns: argparse.Namespace) -> Result:
    """Extract two columns of a FITS table spectrum (e.g. ESO Phase 3) to ASCII or FITS.

    With only the file name, the columns are listed with their index, name,
    unit and length. Columns can be given by 0-based index (as data[0][i] in
    Python) or by name:
      read_eso_fits ADP.2017-01-01.fits
      read_eso_fits ADP.2017-01-01.fits 0 1 star.txt
      read_eso_fits ADP.2017-01-01.fits WAVE FLUX_REDUCED star.fits
    A wavelength column in nm or um is converted to Angstrom (bob's unit)
    unless --keep-units is given. FITS outputs keep the primary header, so
    helio_eso works on them; non-uniform grids are resampled for FITS output.
    """
    given = [ns.xcol, ns.ycol, ns.output]
    if not any(given):
        ext, nrows, columns = list_table_columns(ns.file, ns.ext)
        print(f"{ns.file}: extension {ext}, {nrows} row(s)")
        print(f"{'index':>5s}  {'name':20s} {'unit':12s} {'format':10s} {'values/row':>10s}")
        for c in columns:
            print(f"{c.index:5d}  {c.name:20s} {c.unit:12s} {c.format:10s} {c.length:10d}")
        return None
    if not all(given):
        raise UsageError("read_eso_fits: give the file alone (to list the columns) or file, xcol, ycol and output")
    if not (is_fits(ns.output) or ns.output.lower().endswith(_ASCII_SUFFIXES)):
        raise UsageError(f"read_eso_fits: output must end in .txt, .dat or .fits, got {ns.output}")
    spec, cx, cy = read_table_columns(ns.file, ns.xcol, ns.ycol, ext=ns.ext, row=ns.row, convert_wave=not ns.keep_units)
    x_unit = cx.unit if ns.keep_units or not cx.unit else "Angstrom"
    write_spectrum(ns.output, spec, ascii_header=f"{cx.name}[{x_unit}] {cy.name}[{cy.unit}]")
    print(
        f"Saved {ns.output}: {cx.name} vs {cy.name}, {len(spec)} points, "
        f"{spec.wave[0]:.4f}-{spec.wave[-1]:.4f} {x_unit or '(no unit)'}"
    )
    return {"file": ns.file, "output": ns.output, "x": cx.name, "y": cy.name, "n_points": len(spec)}


@command(
    "cut",
    arg("spectrum"),
    arg("w1", type=float),
    arg("w2", type=float),
    arg("output"),
    arg("--auto", action="store_true", help="do not show the plot (useful with @lists)"),
)
def cmd_cut(ns: argparse.Namespace) -> Result:
    """Cut a spectrum to the wavelength range [w1, w2].

    The cut spectrum is saved, then shown over the original one (close the
    window to continue); --auto skips the plot.
    """
    spec = read_spectrum(ns.spectrum)
    part = spec.select(ns.w1, ns.w2).with_history(f"cut to {min(ns.w1, ns.w2)}-{max(ns.w1, ns.w2)} A")
    write_spectrum(ns.output, part)
    print(f"Saved {ns.output} ({part.wave[0]:.3f}-{part.wave[-1]:.3f} A, {len(part)} pixels)")
    if not ns.auto:
        _show_comparison(spec, part, ("Original", f"Cut: {ns.output}"), spec.name)
    return None


@command(
    "header",
    arg("file", help="FITS file"),
    arg("what", nargs="?", default="minimum", help="'minimum' (default), 'all', or a keyword"),
)
def cmd_header(ns: argparse.Namespace) -> Result:
    """Print a FITS header summary, the whole header, or one keyword.

    Keywords with spaces (HIERARCH) must be quoted: header f.fits "ESO TEL GEOLON".
    In batch mode (header @list, lines 'file KEYWORD') the values are collected in a table.
    """
    header = read_header(ns.file)
    match ns.what.lower():
        case "all":
            print(repr(header))
            return None
        case "minimum":
            for key in (
                "OBJECT",
                "INSTRUME",
                "RA",
                "DEC",
                "MJD-OBS",
                "EXPTIME",
                "NAXIS1",
                "CRVAL1",
                "CDELT1",
                "CD1_1",
                "CRPIX1",
                "CTYPE1",
                "CUNIT1",
            ):
                if key in header:
                    print(f"{key:10s} = {header[key]}")
            return None
    value = header.get(ns.what, header.get(ns.what.upper()))
    if value is None:
        log.warning("%s: keyword %r not found", ns.file, ns.what)
    print(f"{ns.what} = {value}")
    return {"file": ns.file, ns.what: "" if value is None else value}


# ----------------------------------------------------------------------------- velocities
@command("shift_rv", arg("spectrum"), arg("rv", type=float, help="km/s"), arg("output"))
def cmd_shift_rv(ns: argparse.Namespace) -> Result:
    """Shift a spectrum to the rest frame: lambda / (1 + rv/c)."""
    write_spectrum(ns.output, sp.to_rest_frame(read_spectrum(ns.spectrum), ns.rv))
    print(f"Saved {ns.output}")
    return None


@command(
    "rv",
    arg("spectrum", help="observed spectrum"),
    arg("template", help="rest-frame template"),
    arg("output", help="RV-corrected (rest-frame) spectrum"),
    arg(
        "--snr",
        type=positive_float,
        help="SNR per pixel for the Monte Carlo (default: DER_SNR measured on the spectrum)",
    ),
    arg("--nsim", type=nonneg_int, default=200, help="Monte Carlo realisations, 0 = skip (default 200)"),
    arg("--vmin", type=float, default=-600.0, help="km/s (default -600)"),
    arg("--vmax", type=float, default=600.0, help="km/s (default 600)"),
    arg("--wmin", type=float, help="first wavelength used in the CCF"),
    arg("--wmax", type=float, help="last wavelength used in the CCF"),
    arg("--dv", type=positive_float, help="CCF velocity step (default: half the finest pixel)"),
    arg("--seed", type=int, help="random seed for reproducible MC errors"),
    arg("--auto", action="store_true", help="no plots or questions"),
)
def cmd_rv(ns: argparse.Namespace) -> Result:
    """Measure the radial velocity by cross-correlation (Tonry & Davis 1979).

    Reports the Tonry & Davis error and, if --nsim > 0, the Monte Carlo
    (photon-noise) error. Saves the rest-frame spectrum, <output>_ccf.png and
    <output>_mc.png. The velocity is topocentric: add the barycentric
    correction (helio_eso / helio_manual).
    """
    spec, template = read_spectrum(ns.spectrum), read_spectrum(ns.template)
    v_range = (ns.vmin, ns.vmax)
    wave_range = (spec.wave[0] if ns.wmin is None else ns.wmin, spec.wave[-1] if ns.wmax is None else ns.wmax)
    while True:
        options = {
            "v_range": v_range,
            "wave_range": wave_range,
            "exclude_regions": SESSION.telluric_regions,
            "dv_kms": ns.dv,
        }
        result = cross_correlate(spec, template, **options)
        print(
            f"RV = {result.rv:.3f} km/s  (T&D err {result.rv_err_td:.3f}, FWHM {result.fwhm:.2f}, r {result.r_td:.1f})"
        )
        if ns.auto:
            break
        from bobspec.plotting import rv_figure

        plt = _pyplot()
        rv_figure(spec, template, result, fig=plt.figure(figsize=(14, 8), layout="constrained"))
        plt.show(block=True)
        answer = _ask("Accept? [y/ENTER] yes, [n] change ranges, [q] quit without saving: ", {"", "y", "n", "q"})
        if answer in ("", "y"):
            break
        if answer == "q":
            print("Nothing saved.")
            return None
        v_range = (_ask_float("vmin", v_range[0]), _ask_float("vmax", v_range[1]))
        wave_range = (_ask_float("wmin", wave_range[0]), _ask_float("wmax", wave_range[1]))

    from bobspec.plotting import mc_histogram, rv_figure

    _save_figure(rv_figure(spec, template, result), companion(ns.output, "_ccf.png"))
    row: Row = {
        "spectrum": ns.spectrum,
        "template": ns.template,
        "rv": result.rv,
        "rv_err_td": result.rv_err_td,
        "fwhm": result.fwhm,
        "r_td": result.r_td,
        "fit_ok": result.fit_ok,
    }
    if ns.nsim > 0:
        in_range = (spec.wave >= min(wave_range)) & (spec.wave <= max(wave_range))
        snr = ns.snr if ns.snr is not None else der_snr(spec.flux[in_range & (spec.flux > 0)])
        source = "given" if ns.snr is not None else "DER_SNR of the spectrum"
        print(f"Monte Carlo: {ns.nsim} realisations at SNR {snr:.0f} ({source}) ...")
        mc = monte_carlo(
            template, spec.wave[in_range], result.rv, snr, ns.nsim, np.random.default_rng(ns.seed), **options
        )
        if mc.n_failed:
            log.warning("%d MC realisations failed.", mc.n_failed)
        if mc.rvs.size > 1:
            _save_figure(mc_histogram(mc.rvs, f"MC: std {mc.std:.3f} km/s"), companion(ns.output, "_mc.png"))
            if mc.robust_std > 0 and mc.std > 2.0 * mc.robust_std:
                log.warning(
                    "MC distribution has outliers (std %.3f vs robust %.3f): CCF peak may be ambiguous.",
                    mc.std,
                    mc.robust_std,
                )
        row |= {
            "snr_mc": snr,
            "rv_err_mc": mc.std,
            "rv_err_mc_robust": mc.robust_std,
            "n_mc": mc.rvs.size,
            "seed": ns.seed,
        }
        print(f"RV = {result.rv:.3f} +/- {mc.std:.3f} km/s (MC), +/- {result.rv_err_td:.3f} km/s (T&D)")
    write_spectrum(ns.output, sp.to_rest_frame(spec, result.rv))
    print(f"Saved {ns.output}")
    return row


def _print_bary(label: str, obs: Observation, v_corr: float) -> None:
    print(f"{label}: MJD(mid) {obs.mjd_mid:.6f}  barycentric correction {v_corr:+.3f} km/s")


@command("helio_eso", arg("file", help="ESO FITS file"))
def cmd_helio_eso(ns: argparse.Namespace) -> Result:
    """Barycentric velocity correction from an ESO header (at mid-exposure).

    v_bary = v_measured + correction (+ v*correction/c).
    """
    obs = observation_from_eso_header(read_header(ns.file))
    v_corr = barycentric_correction(obs)
    _print_bary(ns.file, obs, v_corr)
    return {"file": ns.file, "mjd_mid": obs.mjd_mid, "v_bary_corr": v_corr}


@command(
    "helio_manual",
    arg("ra", help="deg or hh:mm:ss"),
    arg("dec", help="deg or dd:mm:ss"),
    arg("mjd", type=float, help="MJD at exposure start"),
    arg("lon", type=float, help="site longitude, deg, east positive"),
    arg("lat", type=float, help="site latitude, deg, north positive"),
    arg("alt", type=float, help="site altitude, m"),
    arg("--exptime", type=nonneg_float, default=0.0, help="exposure time in s (default 0)"),
)
def cmd_helio_manual(ns: argparse.Namespace) -> Result:
    """Barycentric velocity correction from explicit parameters."""
    coords = parse_coordinates(ns.ra, ns.dec)
    obs = Observation(coords.ra_deg, coords.dec_deg, mid_exposure_mjd(ns.mjd, ns.exptime), ns.lon, ns.lat, ns.alt)
    v_corr = barycentric_correction(obs)
    _print_bary("Target", obs, v_corr)
    return {"ra_deg": obs.ra_deg, "dec_deg": obs.dec_deg, "mjd_mid": obs.mjd_mid, "v_bary_corr": v_corr}


@command("coords", arg("ra", help="deg or hh:mm:ss"), arg("dec", help="deg or dd:mm:ss"))
def cmd_coords(ns: argparse.Namespace) -> Result:
    """Convert RA/Dec between decimal degrees and sexagesimal."""
    c = parse_coordinates(ns.ra, ns.dec)
    print(f"Degrees    : {c.ra_deg:.6f} {c.dec_deg:+.6f}")
    print(f"Sexagesimal: {c.ra_hms} {c.dec_dms}")
    return {"ra_in": ns.ra, "dec_in": ns.dec, **c.as_row()}


# ----------------------------------------------------------------------------- synthesis
@command(
    "add_noise",
    arg("spectrum", help="noiseless spectrum"),
    arg("snr", type=positive_float),
    arg("n", type=positive_int, help="number of realisations"),
    arg("rootname"),
    arg("--step", type=positive_float, help="output pixel size in A (default: native grid)"),
    arg("--format", choices=("fits", "ascii"), default="fits"),
    arg(
        "--noise",
        choices=("constant", "photon"),
        default="constant",
        help="constant sigma, or photon-like sigma ~ sqrt(flux)",
    ),
    arg("--outdir", default="noisy_spec"),
    arg("--seed", type=int),
)
def cmd_add_noise(ns: argparse.Namespace) -> Result:
    """Create N noisy copies of a spectrum: <outdir>/<rootname>_<snr>_<i>.

    With --step the spectrum is first rebinned (flux-conserving) to that pixel
    size. Apply the instrumental resolution first (convolve).
    """
    spec = read_spectrum(ns.spectrum)
    if ns.step:
        spec = sp.resample(spec, sp.uniform_grid(spec.wave[0], spec.wave[-1], ns.step))
    outdir = Path(ns.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(ns.seed)
    suffix = ".fits" if ns.format == "fits" else ".txt"
    for i in range(1, ns.n + 1):
        write_spectrum(
            outdir / f"{ns.rootname}_{ns.snr:g}_{i}{suffix}", sp.add_noise(spec, ns.snr, rng, model=ns.noise)
        )
    print(f"{ns.n} spectra saved in {outdir}/")
    return None


@command(
    "convolve",
    arg("input"),
    arg("resolution", type=positive_float),
    arg("output"),
    arg(
        "--format",
        choices=("auto", "synthe", "spectrum"),
        default="auto",
        help="synthe: ASCII with wave, -, line flux, continuum (cols 1,3,4); auto: synthe if >= 4 columns",
    ),
)
def cmd_convolve(ns: argparse.Namespace) -> Result:
    """Degrade a spectrum to resolving power R with a constant-R Gaussian kernel.

    SYNTHE files: line and continuum fluxes are both convolved and the output
    has 4 columns: wave, normalised flux, line flux, continuum flux.
    """
    fmt = ns.format
    data = None
    if fmt != "spectrum" and not is_fits(ns.input):
        data = np.loadtxt(ns.input, comments="#", ndmin=2)
        if fmt == "auto":
            fmt = "synthe" if data.shape[1] >= 4 else "spectrum"
        elif data.shape[1] < 4:
            raise UsageError(f"{ns.input} has {data.shape[1]} columns; SYNTHE format needs 4")
    if fmt == "synthe" and data is not None:
        line = sp.gaussian_broaden(Spec(data[:, 0], data[:, 2]), ns.resolution)
        cont_b = sp.gaussian_broaden(Spec(data[:, 0], data[:, 3]), ns.resolution)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = line.flux / cont_b.flux
        np.savetxt(
            ns.output,
            np.column_stack([line.wave, norm, line.flux, cont_b.flux]),
            fmt=("%.4f", "%.6f", "%.8e", "%.8e"),
            header=f"wave[A] flux_norm flux_line flux_cont  R={ns.resolution:g}",
        )
    else:
        write_spectrum(ns.output, sp.gaussian_broaden(read_spectrum(ns.input), ns.resolution))
    print(f"Saved {ns.output} (R = {ns.resolution:g}, {fmt} format)")
    return None


@command(
    "rotate",
    arg("spectrum"),
    arg("epsilon", type=unit_interval, help="linear limb darkening, 0..1"),
    arg("vsini", type=nonneg_float, help="km/s"),
    arg("output"),
    arg("--auto", action="store_true", help="do not show the plot (useful with @lists)"),
)
def cmd_rotate(ns: argparse.Namespace) -> Result:
    """Apply rotational broadening (Gray's kernel, linear limb darkening).

    The broadened spectrum is saved, then shown over the original one (close
    the window to continue); --auto skips the plot.
    """
    spec = read_spectrum(ns.spectrum)
    out = sp.rotational_broaden(spec, ns.vsini, ns.epsilon)
    write_spectrum(ns.output, out)
    print(f"Saved {ns.output}")
    if not ns.auto:
        label = f"vsini = {ns.vsini:g} km/s, \u03b5 = {ns.epsilon:g}"
        _show_comparison(spec, out, ("Original", label), spec.name)
    return None


# ----------------------------------------------------------------------------- normalisation
def _normalize_command(
    ns: argparse.Namespace,
    make_model: Callable[[int], cont.ContinuumModel],
    degree: int,
    bounds: tuple[int, int],
    label: str,
    method: str,
) -> Result:
    spec = read_spectrum(ns.spectrum)
    mask = _telluric_mask(spec)
    fit: cont.ContinuumFit | None
    if ns.auto:
        fit = cont.iterative_fit(
            spec.wave, spec.flux, make_model(degree), sigma_low=ns.sigma_low, sigma_up=ns.sigma_up, mask=mask
        )
        from bobspec.plotting import continuum_figure

        _save_figure(
            continuum_figure(spec, fit.continuum, fit.used, f"{spec.name}: {method}"),
            companion(ns.output, "_continuum.pdf"),
        )
    else:
        from bobspec.interactive import ContinuumEditor

        editor = ContinuumEditor(spec, make_model, degree, bounds, ns.sigma_low, ns.sigma_up, mask, label=label)
        fit = editor.run()
        if fit is None:
            print("Cancelled: nothing saved.")
            return None
        degree = editor.degree
    if not fit.converged:
        log.warning("Sigma clipping did not converge in %d iterations.", fit.n_iter)
    if fit.used.mean() < 0.1:
        log.warning(
            "Only %.0f%% of the pixels define the continuum: sigma_low may be too small.", 100 * fit.used.mean()
        )
    normalized = cont.normalize(
        spec, fit.continuum, f"{method}, {label}={degree}, sigma {ns.sigma_low:g}/{ns.sigma_up:g}"
    )
    write_spectrum(ns.output, normalized)
    _write_snr_file(normalized, mask, ns.output)
    print(f"Saved {ns.output}")
    return {"spectrum": ns.spectrum, label: degree, "n_used": int(fit.used.sum())}


_NORM_ARGS = (
    arg("sigma_low", type=positive_float, help="rejection threshold below the continuum"),
    arg("sigma_up", type=positive_float, help="rejection threshold above the continuum"),
    arg("output"),
    arg("--auto", action="store_true", help="no interaction; save <output>_continuum.pdf"),
)


@command("ipoly", arg("spectrum"), arg("deg", type=positive_int, help="Chebyshev degree"), *_NORM_ARGS)
def cmd_ipoly(ns: argparse.Namespace) -> Result:
    """Normalise with a sigma-clipped Chebyshev polynomial (interactive).

    Keys: j/k degree, l/o sigma_low, h/y sigma_up, right click masks, u undo,
    ENTER accepts, closing the window cancels. Also writes <output>.snr.
    """
    return _normalize_command(ns, cont.chebyshev_model, ns.deg, (1, 60), "deg", "Chebyshev")


@command(
    "ispline",
    arg("spectrum"),
    arg("k", type=spline_degree, help="spline degree 1..5"),
    *_NORM_ARGS,
    arg("--knots", type=positive_int, default=20, help="number of interior knots (default 20)"),
)
def cmd_ispline(ns: argparse.Namespace) -> Result:
    """Normalise with a sigma-clipped least-squares spline (interactive).

    Keys as in ipoly (j/k change the spline degree). Also writes <output>.snr.
    """
    return _normalize_command(ns, lambda k: cont.spline_model(k, ns.knots), ns.k, (1, 5), "k", "spline")


@command("manual_spline", arg("spectrum"), arg("k", type=spline_degree, help="spline degree 1..5"), arg("output"))
def cmd_manual_spline(ns: argparse.Namespace) -> Result:
    """Normalise with a spline through anchor points placed by hand.

    Right click adds, left click removes the last anchor, ENTER fits, ENTER
    again saves; closing the window cancels.
    """
    from bobspec.interactive import anchor_editor

    spec = read_spectrum(ns.spectrum)
    curve = anchor_editor(spec, ns.k)
    if curve is None:
        print("Cancelled: nothing saved.")
        return None
    normalized = cont.normalize(spec, curve(spec.wave), f"manual spline k={ns.k}")
    write_spectrum(ns.output, normalized)
    _write_snr_file(normalized, _telluric_mask(spec), ns.output)
    print(f"Saved {ns.output}")
    return None


@command(
    "smooth",
    arg("spectrum"),
    arg("window", type=positive_float, help="running window in A"),
    arg("output"),
    arg("--percentile", type=float, default=90.0, help="upper-envelope percentile (default 90)"),
    arg("--auto", action="store_true", help="no plot window; save <output>_continuum.pdf"),
)
def cmd_smooth(ns: argparse.Namespace) -> Result:
    """Normalise with a running upper-envelope (percentile + Gaussian) continuum.

    The envelope follows the upper part of the noise: with the default 90th
    percentile the continuum sits about 1.3 noise sigma too high (normalised
    flux ~1% low at SNR 150). ipoly/ispline are preferable for precise work.
    """
    spec = read_spectrum(ns.spectrum)
    mask = _telluric_mask(spec)
    continuum = cont.smooth_continuum(spec, ns.window, percentile=ns.percentile, mask=mask)
    from bobspec.plotting import continuum_figure

    title = f"{spec.name}: smoothing, window {ns.window:g} A"
    if ns.auto:
        _save_figure(continuum_figure(spec, continuum, None, title), companion(ns.output, "_continuum.pdf"))
    else:
        plt = _pyplot()
        _, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
        continuum_figure(spec, continuum, None, title, ax=ax)
        plt.show(block=True)
    normalized = cont.normalize(spec, continuum, f"smoothing window {ns.window:g} A")
    write_spectrum(ns.output, normalized)
    _write_snr_file(normalized, mask, ns.output)
    print(f"Saved {ns.output}")
    return None


# ----------------------------------------------------------------------------- inspection
_PLOT_EXIT_WORDS = frozenset({"e", "q", "exit", "quit"})


@command(
    "plot",
    arg("spectra", nargs="+", help="one or more FITS/ASCII files (quoted wildcards and @list allowed)"),
    arg("--no-add", action="store_true", help="only show the given spectra, without asking for more"),
    batchable=False,
)
def cmd_plot(ns: argparse.Namespace) -> Result:
    """Overplot spectra, then add more to the same window from the terminal.

    After the first plot, type the name of another spectrum (or several, or a
    quoted wildcard) and press ENTER to add it to the window; type 'e' (or
    close the window) to finish.
    """
    from bobspec.plotting import new_figure, overlay_axes

    plt = _pyplot()
    files = expand_inputs(ns.spectra)
    if not files:
        raise ValueError("no spectrum to plot")
    fig, ax = new_figure()
    overlay_axes(ax, [read_spectrum(path) for path in files])
    if ns.no_add:
        plt.show(block=True)
        return None

    was_interactive = plt.isinteractive()
    plt.ion()
    plt.show(block=False)
    plt.pause(0.1)
    try:
        while plt.fignum_exists(fig.number):
            try:
                answer = input("Spectrum to add (or 'e' to finish): ").strip()
            except EOFError:
                break
            if answer.lower() in _PLOT_EXIT_WORDS:
                break
            if not answer:
                continue
            for path in expand_inputs(shlex.split(answer)):
                try:
                    overlay_axes(ax, [read_spectrum(path)])
                except SpectrumReadError as exc:
                    log.error("%s", exc)
                    continue
                print(f"Added: {path}")
            fig.canvas.draw_idle()
            plt.pause(0.1)
    finally:
        plt.close(fig)
        if not was_interactive:
            plt.ioff()
    return None


@command(
    "snr",
    arg("spectrum"),
    arg("--range", nargs=2, type=float, metavar=("W1", "W2"), help="non-interactive: measure in [W1, W2]"),
)
def cmd_snr(ns: argparse.Namespace) -> Result:
    """Signal-to-noise in selected regions: DER_SNR and mean/std.

    Interactive by default (two right clicks per region); with --range it
    runs without plots, so it can be batched (snr @list).
    """
    spec = read_spectrum(ns.spectrum)

    def measure(w1: float, w2: float) -> Row:
        flux = spec.select(w1, w2).flux
        row: Row = {
            "spectrum": ns.spectrum,
            "w1": w1,
            "w2": w2,
            "der_snr": der_snr(flux),
            "mean_std_snr": region_snr(flux),
        }
        print(f"{w1:.2f}-{w2:.2f} A: DER_SNR {row['der_snr']:.1f}  mean/std {row['mean_std_snr']:.1f}")
        return row

    if ns.range:
        return measure(*sorted(ns.range))

    from bobspec.interactive import pick_ranges

    rows: list[Row] = []

    def on_range(w1: float, w2: float, ax: Axes) -> tuple[str, list[Artist]]:
        try:
            rows.append(measure(w1, w2))
        except ValueError as exc:
            return str(exc), []
        return f"DER_SNR {rows[-1]['der_snr']:.0f}  mean/std {rows[-1]['mean_std_snr']:.0f}", []

    pick_ranges(spec, on_range, f"SNR: {spec.name}")
    return rows


@command(
    "fitline",
    arg("spectrum"),
    arg("waves", nargs="*", type=float, help="central wavelength(s) of the line(s)"),
    arg("--lines", type=Path, help="file with one central wavelength per line"),
    arg("--resol", type=positive_float, help="resolving power (initial FWHM guess)"),
    arg("--halfwidth", type=positive_float, default=2.0, help="half width of the view in A (default 2)"),
    arg("--table", type=Path, help="results table (default <spectrum>_lines.txt)"),
    batchable=False,
)
def cmd_fitline(ns: argparse.Namespace) -> Result:
    """Measure centre, FWHM and EW of one or more lines interactively.

    For each line select the integration window with two right clicks (repeat
    to refine), ENTER moves to the next line, closing the window stops.
    """
    waves = list(ns.waves)
    if ns.lines:
        waves += np.atleast_1d(np.loadtxt(ns.lines, comments="#", usecols=0, ndmin=1)).tolist()
    if not waves:
        raise UsageError("fitline: give at least one wavelength or --lines FILE")

    from bobspec.interactive import pick_ranges

    spec = read_spectrum(ns.spectrum)
    rows: list[Row] = []
    for i, wave in enumerate(waves, start=1):
        last: list[LineMeasurement] = []

        def on_range(
            w1: float, w2: float, ax: Axes, wave: float = wave, last: list[LineMeasurement] = last
        ) -> tuple[str, list[Artist]]:
            try:
                m = measure_line(spec, w1, w2, resolution=ns.resol)
            except ValueError as exc:
                return str(exc), []
            last[:] = [m]
            print(
                f"{wave:.3f}: centre {m.center:.3f} A  FWHM {m.fwhm:.3f} A  "
                f"EW int {m.ew_int:.1f} mA  EW gauss {m.ew_gauss:.1f} mA"
            )
            dense = np.linspace(m.w1, m.w2, 400)
            artists: list[Artist] = [*ax.plot(dense, m.profile(dense), "tab:red", lw=2)]
            artists += ax.plot([m.w1, m.w2], m.continuum, "tab:blue", ls="--")
            return f"EW int {m.ew_int:.1f} mA | EW gauss {m.ew_gauss:.1f} mA | centre {m.center:.3f}", artists

        finished = pick_ranges(
            spec,
            on_range,
            f"[{i}/{len(waves)}] {spec.name}: line {wave:.3f} A",
            xlim=(wave - ns.halfwidth, wave + ns.halfwidth),
        )
        if last:
            rows.append({"spectrum": ns.spectrum, "line": wave, **last[0].as_row()})
        if not finished:
            log.info("Stopped by the user.")
            break
    if rows:
        table = ns.table or companion(ns.spectrum, "_lines.txt")
        write_table(table, rows)
        print(f"Results saved in {table}")
    return rows


# ----------------------------------------------------------------------------- output files
def append_target(output: str | None, append: bool | str, default: str) -> tuple[Path, bool]:
    """Output file and append flag from ``-o`` and ``--append [FILE]``.

    ``--append FILE`` appends to FILE; ``--append`` alone appends to the ``-o``
    file (or ``default``). Giving two different files is an error.

    Raises:
        UsageError: If ``-o`` and ``--append FILE`` name different files.
    """
    if isinstance(append, str):
        if output is not None and Path(output) != Path(append):
            raise UsageError(f"-o {output} and --append {append} name two different files: give only one")
        return Path(append), True
    return Path(output or default), bool(append)


# ----------------------------------------------------------------------------- combination
_GLOB_CHARS = frozenset("*?[")


def expand_inputs(patterns: list[str]) -> list[str]:
    """Expand ``@list`` files and shell-style wildcards; drop duplicates and missing files.

    ``@exposures.txt`` is replaced by the first word of each non-empty,
    non-comment line of that file (wildcards allowed there too), as in IRAF.
    """
    import glob

    entries: list[str] = []
    for pattern in patterns:
        if pattern.startswith("@"):
            path = Path(pattern[1:])
            if not path.is_file():
                log.warning("List file not found, skipped: %s", path)
                continue
            entries += [
                line.split()[0]
                for line in path.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        else:
            entries.append(pattern)
    files: list[str] = []
    for entry in entries:
        if _GLOB_CHARS & set(entry):
            matches = sorted(glob.glob(entry))  # noqa: PTH207 - absolute patterns allowed
            if not matches:
                log.warning("No file matches %s", entry)
            files += matches
        elif Path(entry).is_file():
            files.append(entry)
        else:
            log.warning("File not found, skipped: %s", entry)
    return list(dict.fromkeys(files))


def _to_barycentric(spectra: list[Spec], apply: bool) -> list[Spec]:
    """Shift to the barycentric frame (if ``apply``), or warn when it would matter."""
    from bobspec.barycentric import MissingKeywordError
    from bobspec.combine import velocity_spread

    try:
        corrections = [barycentric_correction(observation_from_eso_header(s.header)) for s in spectra]
    except MissingKeywordError as exc:
        if apply:
            raise ValueError(f"--barycentric needs ESO headers: {exc}") from exc
        return spectra
    spread_kms, spread_pix = velocity_spread(corrections, spectra[0].wave)
    if not apply:
        if spread_pix > 0.2:
            log.warning(
                "Barycentric corrections of the inputs differ by %.2f km/s (%.2f pixels): "
                "lines will be smeared; consider --barycentric.",
                spread_kms,
                spread_pix,
            )
        return spectra
    shifted = []
    for spec, v in zip(spectra, corrections, strict=True):
        moved = spec.with_wave(spec.wave * sp.doppler_factor(v)).with_history(f"barycentric shift {v:+.4f} km/s")
        moved.header["BARYFRM"] = (True, "wavelengths are barycentric")
        shifted.append(moved)
    log.info("Inputs shifted to the barycentric frame (corrections span %.3f km/s).", spread_kms)
    return shifted


def _combine_one(output: str, patterns: list[str], ns: argparse.Namespace) -> Row:
    from bobspec.combine import combine_spectra

    files = expand_inputs(patterns)
    if len(files) < 2:
        raise ValueError(f"{output}: only {len(files)} input spectra found")
    spectra = _to_barycentric([read_spectrum(f) for f in files], ns.barycentric)
    result = combine_spectra(
        spectra,
        method=ns.method,
        sigma=None if ns.no_reject else ns.sigma,
        window=ns.window,
        grow=ns.grow,
        rescale=not ns.no_rescale,
        weighted=ns.weighted,
        tolerance_pix=ns.tolerance,
        reference=ns.reference,
    )
    write_spectrum(output, result.spectrum)

    wave = result.spectrum.wave
    lines = ["# spectrum pixel wavelength_A flux reference significance file"]
    for i, name in enumerate(files):
        lines += [
            f"{i + 1:3d} {pix:7d} {wave[pix]:13.6f} {result.scaled[i, pix]:16.6f} "
            f"{result.reference[i, pix]:16.6f} {result.significance[i, pix]:12.4f} {name}"
            for pix in np.flatnonzero(result.rejected[i])
        ]
    cosmics = companion(output, "_cosmics.txt")
    cosmics.write_text("\n".join(lines) + "\n")
    if ns.save_error:
        write_spectrum(
            companion(output, "_error.fits" if is_fits(output) else "_error.txt"),
            result.spectrum.with_flux(result.error),
        )

    print(f"\n{'input':40s} {'scale':>8s} {'noise':>10s} {'rejected':>9s}")
    for i, name in enumerate(files):
        flag = "  (interpolated)" if result.resampled[i] else ""
        print(
            f"{Path(name).name:40s} {result.scale[i]:8.4f} {result.noise[i]:10.4g} "
            f"{int(result.rejected[i].sum()):9d}{flag}"
        )
    snr_in = np.median([der_snr(s.flux) for s in spectra])
    snr_out = der_snr(result.spectrum.flux)
    print(
        f"DER_SNR: median input {snr_in:.0f} -> combined {snr_out:.0f} "
        f"(ideal sqrt(N) gain: {snr_in * np.sqrt(len(files)):.0f})"
    )
    print(f"Saved {output} ({len(files)} spectra, {ns.method}); rejected pixels listed in {cosmics}")
    return {
        "output": output,
        "n_input": len(files),
        "n_rejected": int(result.rejected.sum()),
        "der_snr_input": snr_in,
        "der_snr_combined": snr_out,
    }


@command(
    "combine",
    arg("output", help="combined spectrum; may contain {id} (see --ids)"),
    arg("inputs", nargs="+", help="spectra to combine: names, quoted wildcards or @list; may contain {id}"),
    arg("--method", choices=("mean", "median", "sum"), default="mean", help="default: mean"),
    arg("--sigma", type=positive_float, default=5.0, help="cosmic-ray threshold in local sigmas (default 5)"),
    arg("--no-reject", action="store_true", help="no cosmic-ray rejection (allows 2 spectra)"),
    arg("--window", type=positive_int, default=51, help="odd window of the local noise [pix] (default 51)"),
    arg("--grow", type=nonneg_int, default=1, help="pixels masked around each cosmic (default 1)"),
    arg("--no-rescale", action="store_true", help="do not rescale the inputs to a common flux level"),
    arg("--weighted", action="store_true", help="inverse-variance weighted mean (different exposure times)"),
    arg(
        "--tolerance",
        type=nonneg_float,
        default=0.01,
        help="grid difference [pix] accepted without interpolation (default 0.01)",
    ),
    arg(
        "--reference",
        choices=("others", "median"),
        default="others",
        help="rejection reference: median of the OTHER spectra (default) or of all spectra (original code)",
    ),
    arg("--barycentric", action="store_true", help="shift each input to the barycentric frame first (ESO headers)"),
    arg("--ids", type=Path, help="file with one star id per line: repeat for each id, replacing {id}"),
    arg("--save-error", action="store_true", help="also write <output>_error with the empirical error"),
)
def cmd_combine(ns: argparse.Namespace) -> Result:
    """Combine several exposures of the same star, rejecting cosmic rays.

    Examples:
      combine star.fits exp1.fits exp2.fits exp3.fits
      combine star.fits "M54_h15_*_star_SH.fits" --weighted
      combine star.fits @exposures.txt
      combine "m54_hr15_{id}.fits" "M54_h15_*_{id}_SH.fits" --ids lista.id

    Writes the combined spectrum (header of the first input plus NCOMBINE,
    MJD-MID, EXPTOT and HISTORY) and <output>_cosmics.txt with the rejected pixels.
    Inputs from different nights should be combined with --barycentric.
    Note: "combine @stars.txt" (an @ list as FIRST argument) is batch mode:
    one combination per line of stars.txt.
    """
    if ns.window % 2 == 0:
        raise UsageError(f"combine: --window must be odd, got {ns.window}")
    if not ns.ids:
        return _combine_one(ns.output, ns.inputs, ns)
    if "{id}" not in ns.output:
        raise UsageError("combine: with --ids the output name must contain {id}")
    ids = [
        line.split()[0]
        for line in ns.ids.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    rows: list[Row] = []
    for star in ids:
        print(f"\n=== {star} ===")
        try:
            rows.append(_combine_one(ns.output.replace("{id}", star), [p.replace("{id}", star) for p in ns.inputs], ns))
        except (SpectrumReadError, ValueError, OSError, KeyError) as exc:  # one bad star must not stop the rest
            log.error("%s: %s", star, exc)
    if rows:
        table = default_table(ns.ids, "combine")
        write_table(table, rows)
        print(f"\n{len(rows)}/{len(ids)} stars combined; summary in {table}")
    return rows


# ----------------------------------------------------------------------------- cows and sline
def odd_kernel(text: str) -> int:
    """Argparse type: odd integer >= 3."""
    value = int(text)
    if value < 3 or value % 2 == 0:
        raise argparse.ArgumentTypeError(f"must be an odd integer >= 3, got {text}")
    return value


@command(
    "cows",
    arg("spectrum", help="observed spectrum (FITS or ASCII)"),
    arg("synthetic", help="synthetic spectrum (2-column ASCII or FITS; 4-column SYNTHE file with --resol)"),
    arg("l1", type=float, help="start of the region [A]"),
    arg("l2", type=float, help="end of the region [A]"),
    arg("-o", "--output", help="continuum windows file (default wind_continuum.txt)"),
    arg("--resol", type=positive_float, help="degrade the SYNTHE synthetic to this resolving power"),
    arg("--rv", default="0", help="RV of the observed spectrum [km/s], or 'auto' to estimate it (default 0)"),
    arg("--kernel", type=odd_kernel, default=1001, help="initial running-median kernel [pix] (default 1001)"),
    arg("--step", type=positive_float, default=15.0, help="width of each selection window [A] (default 15)"),
    arg(
        "--append",
        nargs="?",
        const=True,
        default=False,
        metavar="FILE",
        help="add to FILE (or to the -o file) instead of overwriting it",
    ),
    batchable=False,
)
def cmd_cows(ns: argparse.Namespace) -> Result:
    """Select continuum windows by clicking on the normalised spectrum.

    1. The observed spectrum is shifted to rest frame (--rv) and cut to [l1, l2].
    2. A running-median continuum is shown: answer y to accept it, or k to try
       another kernel (odd number of pixels).
    3. The region is shown in windows of --step A with the synthetic in blue:
       left click marks a point, right click removes the last one; close the
       window to go on. The points are sorted and written in pairs
       (start  end of a continuum window) to the output file.
    4. After each window: 'end the code? (y/n)'.
    """
    from bobspec.cows import choose_kernel, select_windows
    from bobspec.synthe import read_synthetic

    spec = read_spectrum(ns.spectrum)
    synthetic = read_synthetic(ns.synthetic, ns.resol)
    if ns.rv.strip().lower() == "auto":
        rv = cross_correlate(spec, synthetic, v_range=(-400.0, 400.0), exclude_regions=SESSION.telluric_regions).rv
        print(f"Spectrum shifted by: {rv:.2f} [km/s]")
    else:
        try:
            rv = float(ns.rv)
        except ValueError as exc:
            raise UsageError(f"cows: --rv must be a number or 'auto', got {ns.rv!r}") from exc
    rest = sp.to_rest_frame(spec, rv) if rv else spec
    try:
        part = rest.select(ns.l1, ns.l2)
    except ValueError as exc:
        raise ValueError(f"No spectrum is present in the region {ns.l1:.0f} - {ns.l2:.0f} A") from exc
    try:
        syn_part: Spec | None = synthetic.select(min(ns.l1, ns.l2) - 100, max(ns.l1, ns.l2) + 100)
    except ValueError:
        log.warning("The synthetic does not cover %.0f-%.0f A: it will not be shown.", ns.l1, ns.l2)
        syn_part = None
    print(f"starting wavelength: {part.wave[0]:.2f}\nlast wavelength: {part.wave[-1]:.2f}\n")

    print(f"First kernel size used: {ns.kernel}\n")
    continuum, kernel = choose_kernel(part.wave, part.flux, ns.kernel)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = part.with_flux(np.where(continuum > 0, part.flux / continuum, np.nan))

    output, append = append_target(ns.output, ns.append, "wind_continuum.txt")
    if not (append and output.is_file()):
        output.write_text("")
    written = select_windows(norm, syn_part, output, ns.step)
    print(f"cows ended: {written} continuum window(s) written to {output}")
    return {"spectrum": ns.spectrum, "output": str(output), "rv": rv, "kernel": kernel, "n_windows": written}


@command(
    "sline",
    arg("spectrum", help="observed spectrum, normalised (FITS or ASCII)"),
    arg("synthetic", help="root name of the SYNTHE files, without .syn and .lin"),
    arg("wave1", type=float, help="starting wavelength [A]"),
    arg("min", type=float, help="show the lines with residual intensity RI < MIN"),
    arg("sel", type=float, help="species code drawn in red, e.g. 20.00"),
    arg("--step", type=positive_float, default=8.0, help="width of the window [A] (default 8)"),
    arg("--snr", type=nonneg_float, default=0.0, help="noise added to the synthetic, 0 = none (default 0)"),
    arg("--wsh", type=float, default=0.0, help="velocity shift [km/s] removed from the observed spectrum (default 0)"),
    arg(
        "--res",
        type=positive_float,
        help="resolving power of the synthetic, only for the blend flags (default: '# RESOL' line of the .syn)",
    ),
    arg("--ysta", type=float, default=0.5, help="lower limit of the y axis (default 0.5)"),
    arg("--yend", type=float, default=1.2, help="upper limit of the y axis (default 1.2)"),
    arg("-o", "--output", help="selected lines file (default linelist.txt)"),
    arg(
        "--append",
        nargs="?",
        const=True,
        default=False,
        metavar="FILE",
        help="add to FILE (or to the -o file) instead of overwriting it",
    ),
    arg("--seed", type=int, help="random seed for the synthetic noise (--snr)"),
    batchable=False,
)
def cmd_sline(ns: argparse.Namespace) -> Result:
    """Select lines from a SYNTHE line list, window by window.

    The synthetic <root>.syn must already be convolved to the resolution of
    the observations: its normalised flux is used as it is. That resolution,
    read from the '# RESOL' header line (or given with --res), only sets the
    blend criterion.
    The plot shows the observed spectrum (black), the synthetic (red) and the
    lines with RI < MIN, numbered: red = species SEL, green = merged split
    components, blue = other lines. The terminal lists them with a blend flag
    ('clean' if no other line with RI < MIN is within 1.5 sigma). Commands:
      y  next window          b  previous window
      w  write a line: its number is asked (or type 'w 3')
      up change settings, e.g. 'up step=10 ysta=0.3 min=0.9'
      e  exit
    """
    from bobspec.sline import SlineSettings, start_session

    root = ns.synthetic
    for suffix in (".syn", ".lin"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            log.info("Using the root name %s (without %s).", root, suffix)
    from bobspec.synthe import synthetic_resolution

    resolution = ns.res or synthetic_resolution(f"{root}.syn") or 0.0
    if resolution:
        print(f"Resolution used for the blend flags: R = {resolution:g}")
    settings = SlineSettings(
        wave1=ns.wave1,
        min=ns.min,
        sel=ns.sel,
        res=resolution,
        step=ns.step,
        snr=ns.snr,
        wsh=ns.wsh,
        ysta=ns.ysta,
        yend=ns.yend,
    )
    output, append = append_target(ns.output, ns.append, "linelist.txt")
    session = start_session(ns.spectrum, root, settings, output, append=append, rng=np.random.default_rng(ns.seed))
    session.run()
    return {"spectrum": ns.spectrum, "output": str(output), "n_lines": len(session.written)}


# ----------------------------------------------------------------------------- equivalent widths
def describe_fwhm_model(result: Any) -> str:
    """One line on how the common FWHM used for the EWs was set (an ``EWResult``)."""
    model = result.fwhm_model
    fwhm, resolution = model.fwhm, model.lam_c / model.fwhm
    shape = {1.0: "constant R", 0.0: "constant FWHM"}.get(round(model.slope, 9), f"slope {model.slope:.2f}")
    head = f"FWHM {fwhm:.4f} A = {fwhm / result.step:.3f} pix at {model.lam_c:.1f} A (R {resolution:.0f}), {shape}"
    if model.source == "list":
        return (
            f"FWHM relation from {len(model.used)} of {model.n_candidates} unblended lines of the list: {head} "
            "(slope: 1 = constant R, 0 = constant FWHM)"
        )
    if model.source == "fixed":
        return f"Fixed FWHM: {head}"
    return f"Common FWHM from all the detected lines: {head}"


def nrv_name(spectrum: str) -> Path:
    """``<name>_nrv`` with the same extension (and so the same format) as ``spectrum``."""
    path = Path(spectrum)
    if is_fits(path):
        return companion(path, "_nrv.fits")
    suffix = path.suffix or ".txt"
    return companion(path, f"_nrv{suffix}")


@command(
    "ew",
    arg("spectrum", help="spectrum (FITS or ASCII), normalised or not, RV-corrected or not"),
    arg("linelist", help="line list: wavelength [A] and species code in Kurucz format (26.00), first two columns"),
    arg("--resol", type=positive_float, help="first-guess resolving power (default: estimated from the spectrum)"),
    arg("--fwhm", type=positive_float, help="fixed FWHM [A] at the central wavelength (no optimisation)"),
    arg(
        "--fwhm-from",
        choices=("all", "list"),
        default="all",
        help="lines that set the common FWHM: all the detected lines (default) or the unblended lines of the list",
    ),
    arg("--order", type=int, default=5, help="continuum polynomial order; -1 = continuum fixed at 1 (default 5)"),
    arg(
        "--continuum", choices=("legendre", "spline"), default="legendre", help="continuum function (default legendre)"
    ),
    arg("--knots", type=positive_int, default=10, help="interior knots of the spline continuum (default 10)"),
    arg("--fwhm-const", action="store_true", help="same FWHM in A at all wavelengths (default: constant R)"),
    arg("--fix-fwhm", action="store_true", help="keep the first-guess FWHM (needs --resol)"),
    arg("--core", type=nonneg_float, default=10.0, help="residual core flux of strong lines, percent (default 10)"),
    arg("--vmin", type=float, default=-500.0, help="RV search window, km/s (default -500)"),
    arg("--vmax", type=float, default=500.0, help="RV search window, km/s (default 500)"),
    arg("--rv", type=float, help="known RV: search only within +-20 km/s of it"),
    arg("--wmin", type=float, help="first wavelength used (default: start of the spectrum)"),
    arg("--wmax", type=float, help="last wavelength used (default: end of the spectrum)"),
    arg("--mask", help="file with extra regions to mask (two columns, observed frame)"),
    arg("--no-mask", action="store_true", help="do not mask Na D, H alpha and the O2 B and A bands"),
    arg("--threshold", type=positive_float, default=3.0, help="line detection threshold in sigma (default 3)"),
    arg("--tol", type=positive_float, default=0.1, help="FWHM convergence tolerance in pixels (default 0.1)"),
    arg("--max-iter", type=positive_int, default=20, help="maximum FWHM iterations (default 20)"),
    arg("--fwhm-max", type=positive_float, default=50.0, help="largest accepted FWHM in pixels (default 50)"),
    arg("--white-noise-errors", action="store_true", help="do not correct the EW errors for correlated noise"),
    arg("--half-depth-fwhm", action="store_true", help="FWHM column: width at half depth (includes saturation)"),
    arg("-o", "--output", help="EW table (default <spectrum>_ew.txt)"),
    arg("--nrv", help="normalised, rest-frame spectrum (default <spectrum>_nrv, same format as the input)"),
    arg("--pdf", help="PDF report with the plots of the fits (default <spectrum>_ew.pdf)"),
    arg("--rejected", help="lines of the list not identified (default <spectrum>_ew_rejected.txt)"),
    arg("--no-pdf", action="store_true", help="do not write the PDF report (faster for long lists)"),
    batch_summary="bob_ew.log",
)
def cmd_ew(ns: argparse.Namespace) -> Result:
    """Measure equivalent widths and line FWHMs automatically (DAOSPEC/4DAO-like).

    Only the spectrum and a line list (wavelength, Kurucz code) are needed.
    Lines are found in the whole spectrum and fitted together with saturated
    Gaussians against an effective continuum; the RV comes from the list; the
    common FWHM is iterated until it converges (4DAO), then every identified
    line is refitted on its own to measure its FWHM.

    The common FWHM can instead be fixed (--fwhm, in A) or derived from the
    unblended lines of the list only (--fwhm-from list): each line is fitted
    with a free width, blends are excluded, FWHM versus wavelength is fitted
    with outlier rejection, and the EWs are measured with that relation fixed.

    Writes <spectrum>_ew.txt with a header and six columns: wavelength [A],
    EW [mA], EW error [mA], code, FWHM [A], R = wavelength / FWHM. The FWHM is
    that of the Gaussian under the saturated profile, so strong and weak lines
    give the same R; --half-depth-fwhm gives the width at half depth instead
    (larger for strong lines). The lines of the list that fall in the spectrum
    but were not identified go to <spectrum>_ew_rejected.txt (wavelength, code,
    reason). Unless --no-pdf, <spectrum>_ew.pdf shows the
    spectrum with its continuum, every line with its fit (lines not identified
    in blue at the end), and the RV and R of the lines. Also writes
    <spectrum>_nrv.fits (or .txt/.dat, as the input): normalised, rest frame.
    With @list, the summary of all spectra goes to bob_ew.log.
    """
    from bobspec.ew import (
        DEFAULT_MASKS,
        EWSettings,
        format_rejected,
        format_table,
        measure_ew,
        read_kurucz_linelist,
        read_masks,
    )

    if ns.fix_fwhm and ns.resol is None:
        raise UsageError("ew: --fix-fwhm needs --resol")
    if ns.fwhm is not None and (ns.resol is not None or ns.fix_fwhm):
        raise UsageError("ew: --fwhm already fixes the FWHM; do not combine it with --resol or --fix-fwhm")
    if ns.fwhm is not None and ns.fwhm_from == "list":
        raise UsageError("ew: --fwhm (fixed) and --fwhm-from list exclude each other")
    masks = () if ns.no_mask else DEFAULT_MASKS
    if ns.mask:
        masks = (*masks, *read_masks(ns.mask))
    settings = EWSettings(
        resolution=ns.resol,
        fwhm=ns.fwhm,
        fwhm_from=ns.fwhm_from,
        order=ns.order,
        continuum=ns.continuum,
        knots=ns.knots,
        scale_fwhm=not ns.fwhm_const,
        fix_fwhm=ns.fix_fwhm,
        core=ns.core,
        vmin=ns.vmin,
        vmax=ns.vmax,
        rv=ns.rv,
        wmin=ns.wmin,
        wmax=ns.wmax,
        masks=masks,
        threshold=ns.threshold,
        tol=ns.tol,
        max_iter=ns.max_iter,
        fwhm_max=ns.fwhm_max,
        correlated_noise=not ns.white_noise_errors,
    )
    spec = read_spectrum(ns.spectrum)
    lab_wave, lab_code = read_kurucz_linelist(ns.linelist)
    result = measure_ew(spec, lab_wave, lab_code, settings)
    run = result.run

    print(f"\n{'FWHM':>8s} {'RE':>4s} {'RV1':>9s} {'RV':>9s} {'RV2':>9s}")
    from_list = ns.fwhm_from == "list"
    for i, (w_in, _w_out, v1, rv, v2) in enumerate(result.history):
        last = i == len(result.history) - 1
        label = "(FIXED)" if last else "(LIST)" if from_list else ""
        print(f"{w_in:8.3f} {result.core:4.0f} {v1:9.1f} {rv:9.2f} {v2:9.1f}  {label}")
    status = {1: "FWHM CONVERGED", 0: "MAX ITERATIONS REACHED", -1: "FWHM CYCLING", -2: "FWHM OUT OF RANGE"}
    if from_list and result.convergence == -2:
        status[-2] = "NO FWHM RELATION FROM THE LIST (first guess kept)"
    print(status[result.convergence])

    table = Path(ns.output) if ns.output else companion(ns.spectrum, "_ew.txt")
    table.write_text(format_table(run.lines, half_depth=ns.half_depth_fwhm))
    nrv = Path(ns.nrv) if ns.nrv else nrv_name(ns.spectrum)
    write_spectrum(nrv, result.normalized_rest(Path(ns.spectrum).name))
    resolutions = np.array([line.resolution for line in run.lines])
    resolutions = resolutions[np.isfinite(resolutions)]
    r_lines = (
        f"R from the line FWHMs: median {np.median(resolutions):.0f}, "
        f"scatter {MAD_TO_SIGMA * np.median(np.abs(resolutions - np.median(resolutions))):.0f} "
        f"({resolutions.size} lines)"
        if resolutions.size
        else "no line FWHM measured"
    )
    print(
        f"\nRV {run.rv:.3f} +- {run.rv_error:.3f} km/s (error of the mean; line-to-line sigma {run.rv_sigma:.3f}, "
        f"{len(run.lines)} lines)\nresiduals {run.residual_pct:.3f}%  noise factor {run.noise_factor:.2f}\n{r_lines}\n"
        f"{describe_fwhm_model(result)}"
    )
    rejected = Path(ns.rejected) if ns.rejected else companion(ns.spectrum, "_ew_rejected.txt")
    rejected.write_text(format_rejected(run, masks))
    print(f"Lines of the list not identified: {run.missing.size} (listed in {rejected})")
    saved = [str(table), str(nrv), str(rejected)]
    if not ns.no_pdf:
        from bobspec.ew_report import write_ew_report

        pdf = Path(ns.pdf) if ns.pdf else companion(ns.spectrum, "_ew.pdf")
        write_ew_report(pdf, result, Path(ns.spectrum).name, masks)
        saved.append(str(pdf))
    print("Saved " + ", ".join(saved))
    return {
        "star": Path(ns.spectrum).name,
        "R_lines": float(np.median(resolutions)) if resolutions.size else np.nan,
        "FWHM": result.fwhm_pix,
        "RV": run.rv,
        "eRV": run.rv_sigma,
        "errRV": run.rv_error,
        "FWHM_from": result.fwhm_model.source,
        "N_FWHM": len(result.fwhm_model.used),
        "FWHM_slope": result.fwhm_model.slope,
        "Nlin": len(run.lines),
        "RES(%)": run.residual_pct,
        "CONV": result.convergence,
        "WAVE1": float(run.wave[0]),
        "WAVE2": float(run.wave[-1]),
        "RE": result.core,
        "noise_factor": run.noise_factor,
    }
