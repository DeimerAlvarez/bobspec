"""PDF report of the ``ew`` command (in the spirit of the 4DAO plots).

Pages:

1. the input spectrum with the effective continuum (top) and the normalised,
   rest-frame spectrum (bottom);
2. one 4x4 page per 16 lines: every identified line with its own fit drawn
   over the region where its FWHM is measured, followed by the lines of the
   list that were not identified (in blue);
3. radial velocity of every line versus wavelength and versus EW;
4. resolving power of every line versus wavelength and versus EW.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from bobspec.constants import C_KMS, MAD_TO_SIGMA
from bobspec.ew import EWResult, MaskRegion, MeasuredLine, mask_pixels, missing_reasons
from bobspec.plotting import style_axes

A4 = (8.27, 11.69)
PANELS_PER_PAGE = 16
COLOR_SPECTRUM = "black"
COLOR_CONTINUUM = "tab:red"
COLOR_MODEL = "0.6"
COLOR_FIT = "tab:red"
COLOR_NEIGHBOURS = "tab:orange"
COLOR_LINE = "tab:green"
COLOR_MISSING = "tab:blue"

_ELEMENT_TABLE = """
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca
Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr
Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd
Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg
Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es
"""
_ELEMENTS = tuple(_ELEMENT_TABLE.split())  # atomic number - 1 -> symbol
_ROMAN = ("I", "II", "III", "IV", "V", "VI")


MOLECULES: dict[int, str] = {
    101: "H2",
    106: "CH",
    107: "NH",
    108: "OH",
    112: "MgH",
    114: "SiH",
    120: "CaH",
    126: "FeH",
    606: "C2",
    607: "CN",
    608: "CO",
    708: "NO",
    808: "O2",
    814: "SiO",
    822: "TiO",
    823: "VO",
}
"""Usual names of common diatomic molecules, by Kurucz/MOOG code (atomic numbers of the two atoms)."""


def molecule_name(code: int) -> str | None:
    """Name of a diatomic molecule from its code ``Z1 * 100 + Z2`` (e.g. 607 -> 'CN').

    Codes in :data:`MOLECULES` get their usual name; any other valid code is
    written as the two symbols in the order of the code. Returns None if the
    code is not a valid molecule.
    """
    if code in MOLECULES:
        return MOLECULES[code]
    z1, z2 = divmod(code, 100)
    if not (1 <= z1 <= len(_ELEMENTS) and 1 <= z2 <= len(_ELEMENTS)):
        return None
    return f"{_ELEMENTS[z1 - 1]}2" if z1 == z2 else f"{_ELEMENTS[z1 - 1]}{_ELEMENTS[z2 - 1]}"


def species_name(code: float) -> str:
    """Species from a Kurucz code: 26.00 -> 'Fe I', 26.01 -> 'Fe II', 607.00 -> 'CN', 106.01 -> 'CH+'."""
    z = int(np.floor(code + 1e-6))
    ion = round((code - z) * 100)
    if 1 <= z <= len(_ELEMENTS) and 0 <= ion < len(_ROMAN):
        return f"{_ELEMENTS[z - 1]} {_ROMAN[ion]}"
    molecule = molecule_name(z) if z >= 100 else None
    if molecule is not None and 0 <= ion <= 2:
        return molecule + "+" * ion
    return f"{code:.2f}"


@dataclass(frozen=True, slots=True)
class _Panel:
    """One small plot: an identified line or a line of the list that was not identified."""

    wave: float
    code: float
    line: MeasuredLine | None
    reason: str = ""


def _summary(result: EWResult, name: str) -> str:
    run = result.run
    flags = {1: "converged", 0: "max iterations", -1: "FWHM cycling", -2: "FWHM out of range"}
    return (
        f"{name}    FWHM {result.fwhm_pix:.3f} pix (R {result.resolution:.0f})    "
        f"RV {run.rv:.2f} ± {run.rv_error:.2f} km/s (\u03c3 {run.rv_sigma:.2f})    {len(run.lines)} lines    "
        f"residuals {run.residual_pct:.2f}%    {flags.get(result.convergence, result.convergence)}"
    )


def _rest(result: EWResult) -> tuple[np.ndarray, np.ndarray]:
    run = result.run
    return run.wave / (1.0 + run.rv / C_KMS), run.flux / run.continuum


def _overview(result: EWResult, name: str, masks: Sequence[MaskRegion]) -> Figure:
    run = result.run
    fig = Figure(figsize=A4, layout="constrained")
    fig.suptitle(_summary(result, name), fontsize=8.5)
    top, bottom = fig.subplots(2, 1)
    masked = mask_pixels(run.wave, masks, run.rv)
    top.plot(run.wave, run.flux, color=COLOR_SPECTRUM, lw=0.5, label=name)
    top.plot(run.wave, run.continuum, color=COLOR_CONTINUUM, lw=1.5, label="effective continuum")
    _shade(top, run.wave, masked, label="masked")
    clean = ~masked & np.isfinite(run.flux)
    f_lo, f_hi = np.nanpercentile(run.flux[clean], [0.2, 99.9]) if clean.any() else (0.0, 1.0)
    pad = 0.08 * (f_hi - f_lo)
    top.set(
        xlabel=r"Observed wavelength [$\AA$]",
        ylabel="Flux",
        xlim=(run.wave[0], run.wave[-1]),
        ylim=(max(0.0, f_lo - pad), f_hi + pad),
    )
    top.legend(loc="lower right", fontsize=8)

    rest, norm = _rest(result)
    lo = max(0.0, float(np.nanpercentile(norm[clean], 0.2)) - 0.05) if clean.any() else 0.0
    bottom.plot(rest, norm, color=COLOR_SPECTRUM, lw=0.5, label=f"{name}, RV = {run.rv:.2f} km/s")
    bottom.axhline(1.0, color=COLOR_CONTINUUM, lw=0.8, ls="--")
    _shade(bottom, rest, masked)
    ticks = [ln.wave for ln in run.lines]
    bottom.vlines(ticks, 1.04, 1.09, color=COLOR_LINE, lw=1.0, label=f"identified lines ({len(ticks)})")
    if run.missing.size:
        bottom.vlines(
            run.missing, 1.04, 1.09, color=COLOR_MISSING, lw=1.0, label=f"lines not identified ({run.missing.size})"
        )
    bottom.set(
        xlabel=r"Rest-frame wavelength [$\AA$]", ylabel="Normalised flux", xlim=(rest[0], rest[-1]), ylim=(lo, 1.15)
    )
    bottom.legend(loc="lower right", fontsize=8)
    for ax in (top, bottom):
        style_axes(ax)
        ax.tick_params(labelsize=9)
        _plain_ticks(ax)
        ax.xaxis.label.set_fontsize(11)
        ax.yaxis.label.set_fontsize(11)
    return fig


def _plain_ticks(ax: Axes) -> None:
    """Tick labels as plain numbers (14628, not 2 with an offset +1.4626e4)."""
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)


def _shade(ax: Axes, wave: np.ndarray, masked: np.ndarray, label: str | None = None) -> None:
    edges = np.flatnonzero(np.diff(np.concatenate([[0], masked.astype(int), [0]])))
    for i, (start, stop) in enumerate(zip(edges[0::2], edges[1::2], strict=True)):
        ax.axvspan(wave[start], wave[stop - 1], color="0.85", zorder=0, label=label if i == 0 else None)


def _panels(result: EWResult, masks: Sequence[MaskRegion]) -> list[_Panel]:
    run = result.run
    found = [_Panel(ln.wave, ln.code, ln) for ln in sorted(run.lines, key=lambda x: x.wave)]
    reasons = missing_reasons(run, masks)
    missing = [
        _Panel(float(wave), float(code), None, reason)
        for wave, code, reason in sorted(zip(run.missing, run.missing_codes, reasons, strict=True))
    ]
    return found + missing


def _chunks(items: Sequence[_Panel], size: int) -> Iterator[Sequence[_Panel]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _line_page(result: EWResult, panels: Sequence[_Panel], page_title: str) -> Figure:
    fig = Figure(figsize=A4, layout="constrained")
    fig.suptitle(page_title, fontsize=10)
    axes = fig.subplots(4, 4).ravel()
    rest, norm = _rest(result)
    run = result.run
    model = run.model
    half = max(1.0, 4.0 * float(np.median([ln.fwhm for ln in run.lines])) if run.lines else 1.5)
    for ax, panel in zip(axes, panels, strict=False):
        sel = (rest > panel.wave - half) & (rest < panel.wave + half)
        if not sel.any():
            ax.set_axis_off()
            continue
        ax.step(rest[sel], norm[sel], where="mid", color=COLOR_SPECTRUM, lw=0.7)
        ax.plot(rest[sel], model[sel], color=COLOR_MODEL, lw=0.8, ls="--")
        low = float(np.nanmin(norm[sel]))
        ymin = max(0.0, min(low, 0.9) - 0.08)
        ax.set_ylim(ymin, 1.12)
        ax.set_xlim(panel.wave - half, panel.wave + half)
        name = f"{species_name(panel.code)}  {panel.wave:.3f}"
        if panel.line is not None:
            line = panel.line
            if np.min(line.fit_neighbours, initial=1.0) < 0.995:  # neighbours inside the window
                ax.plot(line.fit_wave, line.fit_neighbours, color=COLOR_NEIGHBOURS, lw=1.2, ls="--")
            ax.plot(line.fit_wave, line.fit_flux, color=COLOR_FIT, lw=1.8)
            ax.vlines(panel.wave, 1.03, 1.10, color=COLOR_LINE, lw=1.5)
            ax.set_title(name, fontsize=7.5, fontweight="bold", pad=2)
            text = f"EW = {line.ew:.1f} ± {line.ew_err:.1f} mÅ\nRV = {line.rv:.2f} km/s\nR = {line.resolution:.0f}"
            ax.text(
                0.03,
                0.04,
                text,
                transform=ax.transAxes,
                fontsize=6,
                va="bottom",
                ha="left",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1},
            )
        else:
            ax.vlines(panel.wave, 1.03, 1.10, color=COLOR_MISSING, lw=1.5)
            ax.set_title(name, fontsize=7.5, fontweight="bold", color=COLOR_MISSING, pad=2)
            ax.text(
                0.03,
                0.04,
                panel.reason,
                transform=ax.transAxes,
                fontsize=6.5,
                color=COLOR_MISSING,
                va="bottom",
                ha="left",
            )
        ax.tick_params(labelsize=5.5, direction="in", top=True, right=True, length=3)
        ax.locator_params(axis="x", nbins=3)
        _plain_ticks(ax)
    for ax in axes[len(panels) :]:
        ax.set_axis_off()
    fig.legend(handles=_line_legend(), loc="outside lower center", ncol=3, fontsize=7, frameon=False)
    return fig


def _line_legend() -> list[Line2D]:
    """Explanation of the colours of the line panels."""
    return [
        Line2D([], [], color=COLOR_SPECTRUM, lw=0.8, label="normalised spectrum"),
        Line2D([], [], color=COLOR_MODEL, lw=0.8, ls="--", label="model of all lines"),
        Line2D([], [], color=COLOR_FIT, lw=1.8, label="the line alone (its own fit, FWHM region)"),
        Line2D([], [], color=COLOR_NEIGHBOURS, lw=1.2, ls="--", label="neighbouring lines (held fixed)"),
        Line2D([], [], color=COLOR_LINE, lw=1.5, label="laboratory wavelength (identified)"),
        Line2D([], [], color=COLOR_MISSING, lw=1.5, label="laboratory wavelength (not identified)"),
    ]


MAX_ELEMENT_COLOURS = 20
"""Above this number of elements the points are coloured by nucleosynthetic family."""

_MARKERS = ("o", "^", "s", "D", "P")  # ionisation stage I, II, III, IV, V
_MOLECULE_MARKER = "*"

FAMILIES: tuple[tuple[str, frozenset[int], str], ...] = (
    ("light (Li-N)", frozenset(range(1, 8)), "tab:brown"),
    ("odd-Z (Na, Al, P, Cl, K)", frozenset({9, 11, 13, 15, 17, 19}), "tab:blue"),
    ("\u03b1 (O, Mg, Si, S, Ca, Ti)", frozenset({8, 10, 12, 14, 16, 18, 20, 22}), "tab:green"),
    ("iron peak (Sc, V-Zn)", frozenset({21, *range(23, 31)}), "tab:red"),
    ("neutron capture (Z > 30)", frozenset(range(31, 100)), "tab:purple"),
)
"""Groups used when there are too many elements to give each a colour."""


def split_code(code: float) -> tuple[int, int]:
    """Atomic number (or molecular code) and ionisation stage (0 = neutral) of a Kurucz code."""
    z = int(np.floor(code + 1e-6))
    return z, round((code - z) * 100)


def family_of(z: int) -> int:
    """Index in :data:`FAMILIES` of atomic number ``z`` (len(FAMILIES) for molecules)."""
    return next((i for i, (_, members, _) in enumerate(FAMILIES) if z in members), len(FAMILIES))


@dataclass(frozen=True, slots=True)
class _Style:
    """Colour and marker of every species, and the legend entries explaining them."""

    colour: dict[float, tuple[float, float, float, float] | str]
    marker: dict[float, str]
    handles: list[Line2D]


def species_styles(codes: Sequence[float]) -> _Style:
    """Colour per element (or per family beyond ``MAX_ELEMENT_COLOURS``), marker per ionisation stage."""
    from matplotlib import colormaps

    unique = sorted(set(codes))
    elements = sorted({split_code(c)[0] for c in unique if split_code(c)[0] < 100})
    molecules = sorted({split_code(c)[0] for c in unique if split_code(c)[0] >= 100})
    has_molecules = bool(molecules)
    colour: dict[float, tuple[float, float, float, float] | str] = {}
    handles: list[Line2D] = []
    if len(elements) + len(molecules) <= MAX_ELEMENT_COLOURS:  # every element and molecule its own colour
        palette = colormaps["tab20"]
        order = [*range(0, 20, 2), *range(1, 20, 2)]  # the 10 distinct hues first, then their pale shades
        index = {z: palette(order[i % 20]) for i, z in enumerate([*elements, *molecules])}
        for code in unique:
            colour[code] = index[split_code(code)[0]]
        for z in elements:
            label = _ELEMENTS[z - 1] if z <= len(_ELEMENTS) else str(z)
            handles.append(Line2D([], [], ls="", marker="o", ms=6, mfc=index[z], mec="k", mew=0.4, label=label))
        for z in molecules:
            label = molecule_name(z) or str(z)
            handles.append(
                Line2D([], [], ls="", marker=_MOLECULE_MARKER, ms=8, mfc=index[z], mec="k", mew=0.4, label=label)
            )
        has_molecules = False  # already in the legend, one entry each
    else:
        used = sorted({family_of(split_code(c)[0]) for c in unique if split_code(c)[0] < 100})
        for code in unique:
            z = split_code(code)[0]
            colour[code] = FAMILIES[family_of(z)][2] if z < 100 else "0.5"
        for f in used:
            name, _, col = FAMILIES[f]
            handles.append(Line2D([], [], ls="", marker="o", ms=6, mfc=col, mec="k", mew=0.4, label=name))
    if has_molecules:
        handles.append(
            Line2D([], [], ls="", marker=_MOLECULE_MARKER, ms=8, mfc="0.5", mec="k", mew=0.4, label="molecules")
        )
    marker = {
        code: _MOLECULE_MARKER if split_code(code)[0] >= 100 else _MARKERS[min(split_code(code)[1], len(_MARKERS) - 1)]
        for code in unique
    }
    stages = sorted({split_code(c)[1] for c in unique if split_code(c)[0] < 100})
    if len(stages) > 1 or (stages and stages[0] > 0):
        for stage in stages:
            m = _MARKERS[min(stage, len(_MARKERS) - 1)]
            label = _ROMAN[stage] if stage < len(_ROMAN) else str(stage + 1)
            handles.append(Line2D([], [], ls="", marker=m, ms=6, mfc="white", mec="k", mew=0.8, label=label))
    return _Style(colour, marker, handles)


def _blank() -> Line2D:
    return Line2D([], [], ls="", label=" ")


def row_major(handles: Sequence[Line2D], ncol: int) -> list[Line2D]:
    """Reorder legend entries so that they read by rows (matplotlib fills legends by columns)."""
    rows = -(-len(handles) // ncol)
    padded = [*handles, *[_blank()] * (rows * ncol - len(handles))]
    return [padded[r * ncol + c] for c in range(ncol) for r in range(rows)]


def legend_columns(n_entries: int, max_rows: int = 3, max_cols: int = 8) -> int:
    """Columns so that the legend uses at most ``max_rows`` rows (and at most ``max_cols`` columns)."""
    return int(min(max_cols, max(4, -(-n_entries // max_rows))))


def _scatter_page(
    result: EWResult, values: np.ndarray, centre: float, sigma: float, ylabel: str, title: str, legend: str
) -> Figure:
    """Two panels: ``values`` versus wavelength and versus EW, with +-1, 2, 3 sigma bands.

    Points are coloured by element (by nucleosynthetic family when there are
    more than ``MAX_ELEMENT_COLOURS`` elements) with one marker per
    ionisation stage; one compact legend below the panels explains both.
    """
    lines = result.run.lines
    waves = np.array([ln.wave for ln in lines])
    ews = np.array([ln.ew for ln in lines])
    codes = [ln.code for ln in lines]
    style = species_styles(codes)
    fig = Figure(figsize=A4, layout="constrained")
    fig.suptitle(title, fontsize=10)
    axes = fig.subplots(2, 1)
    for ax, x, xlabel in zip(axes, (waves, ews), (r"Rest-frame wavelength [$\AA$]", "EW [mÅ]"), strict=True):
        for k, alpha in ((3, 0.08), (2, 0.12), (1, 0.18)):
            if sigma > 0:
                ax.axhspan(
                    centre - k * sigma, centre + k * sigma, color="tab:blue", alpha=alpha, lw=0, label=f"±{k}\u03c3"
                )
        ax.axhline(centre, color="tab:blue", lw=1.2, label=legend)
        for code in style.colour:
            sel = np.array([c == code for c in codes])
            ax.scatter(
                x[sel],
                values[sel],
                s=26,
                color=style.colour[code],
                marker=style.marker[code],
                edgecolor="k",
                lw=0.4,
                zorder=3,
            )
        ax.set(xlabel=xlabel, ylabel=ylabel)
        if sigma > 0:
            span = max(4 * sigma, float(np.nanmax(np.abs(values - centre))) * 1.15)
            ax.set_ylim(centre - span, centre + span)
        style_axes(ax)
        ax.tick_params(labelsize=9)
        _plain_ticks(ax)
        ax.xaxis.label.set_fontsize(11)
        ax.yaxis.label.set_fontsize(11)
    reference, _ = axes[0].get_legend_handles_labels()
    reference = [reference[-1], *reference[-2::-1]]  # mean first, then +-1, 2, 3 sigma
    ncol = max(len(reference), legend_columns(len(style.handles)))
    handles = row_major([*reference, *[_blank()] * (ncol - len(reference)), *style.handles], ncol)
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=ncol,
        fontsize=8,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    return fig


def write_ew_report(path: Path, result: EWResult, name: str, masks: Sequence[MaskRegion]) -> Path:
    """Write the PDF report of an ``ew`` measurement and return its path."""
    run = result.run
    panels = _panels(result, masks)
    n_found = len(run.lines)
    with PdfPages(path, metadata={"Title": f"bob ew: {name}", "Creator": "bobspec"}) as pdf:
        pdf.savefig(_overview(result, name, masks))
        for start, page in zip(range(0, len(panels), PANELS_PER_PAGE), _chunks(panels, PANELS_PER_PAGE), strict=True):
            stop = start + len(page)
            parts = []
            if start < n_found:
                parts.append(f"identified lines {start + 1}-{min(stop, n_found)} of {n_found}")
            if stop > n_found:
                n_missing = len(panels) - n_found
                first = max(start, n_found) - n_found + 1
                parts.append(f"lines not identified {first}-{stop - n_found} of {n_missing}")
            pdf.savefig(_line_page(result, page, f"{name}: " + ", ".join(parts)))
        if run.lines:
            rvs = np.array([ln.rv for ln in run.lines])
            n = rvs.size
            err_mean = run.rv_sigma / np.sqrt(n - 1) if n > 1 else np.nan
            legend = f"⟨RV⟩ = {run.rv:.2f} ± {err_mean:.2f} km/s  (\u03c3 = {run.rv_sigma:.2f}, N = {n})"
            pdf.savefig(
                _scatter_page(
                    result, rvs, run.rv, run.rv_sigma, "RV [km/s]", f"{name}: radial velocity of the lines", legend
                )
            )
            res = np.array([ln.resolution for ln in run.lines])
            finite = res[np.isfinite(res)]
            median = float(np.median(finite)) if finite.size else np.nan
            scatter = float(MAD_TO_SIGMA * np.median(np.abs(finite - median))) if finite.size else 0.0
            legend = f"median R = {median:.0f}  (scatter {scatter:.0f}, N = {finite.size})"
            pdf.savefig(
                _scatter_page(
                    result, res, median, scatter, "R = λ / FWHM", f"{name}: resolving power from the line FWHMs", legend
                )
            )
    return path
