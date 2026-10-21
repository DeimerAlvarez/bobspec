"""Static figures and matplotlib helpers (no blocking calls here)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import cast

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bobspec.constants import C_KMS
from bobspec.io import Spec
from bobspec.rv import CCFResult

WAVE_LABEL = r"Wavelength [$\AA$]"

# ----------------------------------------------------------------------------- font sizes
# Font sizes (in points) used by every bob figure. Change them here.
TICK_LABEL_SIZE = 12  # numbers along the axes
AXIS_LABEL_SIZE = 16  # axis names, e.g. "Wavelength", "Flux", "Radial velocity"
TITLE_SIZE = 15  # text above each panel
LEGEND_SIZE = 12  # legend entries


def style_axes(ax: Axes) -> Axes:
    """Apply the package style (ticks, spines and font sizes) to ``ax``.

    Call it after the labels and the legend have been added; it has no
    figure-wide side effects.
    """
    ax.tick_params(which="major", direction="in", length=8, top=True, right=True, labelsize=TICK_LABEL_SIZE)
    ax.tick_params(which="minor", direction="in", length=4, top=True, right=True)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.xaxis.label.set_fontsize(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_fontsize(AXIS_LABEL_SIZE)
    ax.title.set_fontsize(TITLE_SIZE)
    if (legend := ax.get_legend()) is not None:
        for text in legend.get_texts():
            text.set_fontsize(LEGEND_SIZE)
    return ax


def set_title(ax: Axes, text: str) -> None:
    """Set a panel title with the package font size (``Axes.set_title`` resets it)."""
    ax.set_title(text, fontsize=TITLE_SIZE)


@contextmanager
def suppress_keymaps(keys: Iterable[str]) -> Iterator[None]:
    """Temporarily remove ``keys`` from matplotlib's default key bindings.

    Without this, e.g. ``k``/``l`` would also toggle log axes and ``o`` the
    zoom tool while an interactive editor uses them.
    """
    keys = set(keys)
    saved = {name: list(value) for name, value in mpl.rcParams.items() if name.startswith("keymap.")}
    try:
        for name, value in saved.items():
            mpl.rcParams[name] = [k for k in value if k not in keys]
        yield
    finally:
        mpl.rcParams.update(saved)


def new_figure(figsize: tuple[float, float] = (12.0, 5.5)) -> tuple[Figure, Axes]:
    """Create a pyplot-managed figure with one styled axes."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.set_xlabel(WAVE_LABEL)
    ax.set_ylabel("Flux")
    style_axes(ax)
    return fig, ax


def _robust_norm(flux: np.ndarray) -> np.ndarray:
    level = np.nanpercentile(flux, 95)
    return flux / level if level > 0 else flux


def rv_figure(spec: Spec, template: Spec, result: CCFResult, fig: Figure | None = None) -> Figure:
    """Diagnostic figure: full CCF, zoom on the fitted peak, rest-frame overplot.

    Pass a pyplot-managed ``fig`` to show it interactively; by default an
    unmanaged :class:`~matplotlib.figure.Figure` is built (save-only, no leaks).
    """
    fig = fig if fig is not None else Figure(figsize=(14, 8), layout="constrained")
    grid = fig.add_gridspec(2, 2)
    ax_ccf, ax_zoom, ax_spec = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])

    ax_ccf.plot(result.velocity, result.ccf, color="k", lw=1)
    ax_ccf.axvline(result.rv, color="tab:red", ls="--")
    ax_ccf.set(xlabel="Radial velocity [km/s]", ylabel="CCF", title="CCF")

    half = 3.0 * max(result.fwhm, result.dv) if np.isfinite(result.fwhm) else 50.0
    near = np.abs(result.velocity - result.rv) <= half
    v_dense = np.linspace(result.rv - half, result.rv + half, 500)
    ax_zoom.plot(result.velocity[near], result.ccf[near], "k.-", lw=1, label="CCF")
    ax_zoom.plot(v_dense, result.model(v_dense), color="tab:orange", label="Gaussian fit")
    ax_zoom.axvline(result.rv, color="tab:red", ls=":")
    ax_zoom.set(
        xlabel="Radial velocity [km/s]",
        title=f"RV = {result.rv:.2f} km/s  FWHM = {result.fwhm:.1f}  r = {result.r_td:.1f}",
    )
    ax_zoom.legend()

    lo, hi = result.wave_range
    rest = spec.wave / (1.0 + result.rv / C_KMS)
    ax_spec.plot(rest, _robust_norm(spec.flux), color="k", lw=0.8, alpha=0.7, label="Observed (rest frame)")
    ax_spec.plot(template.wave, _robust_norm(template.flux), color="tab:red", lw=0.8, alpha=0.7, label="Template")
    ax_spec.set(xlim=(lo, hi), xlabel=WAVE_LABEL, ylabel="Flux / 95th percentile")
    ax_spec.legend(loc="lower left")
    for ax in (ax_ccf, ax_zoom, ax_spec):
        style_axes(ax)
    return fig


def mc_histogram(rvs: np.ndarray, title: str) -> Figure:
    """Histogram of Monte Carlo velocities (automatic binning)."""
    fig = Figure(figsize=(7, 5), layout="constrained")
    ax = fig.add_subplot()
    ax.hist(rvs, bins="auto", color="skyblue", edgecolor="k")
    ax.set(xlabel="RV [km/s]", ylabel="N", title=title)
    style_axes(ax)
    return fig


def continuum_figure(
    spec: Spec, continuum: np.ndarray, used: np.ndarray | None, title: str, ax: Axes | None = None
) -> Figure:
    """Spectrum, fitted continuum and the pixels used in the fit."""
    if ax is None:
        fig = Figure(figsize=(12, 5.5), layout="constrained")
        ax = fig.add_subplot()
    else:
        fig = cast(Figure, ax.figure)
        ax.clear()
    ax.plot(spec.wave, spec.flux, color="gray", lw=0.8, alpha=0.6, label="Spectrum")
    if used is not None:
        ax.plot(spec.wave[used], spec.flux[used], "r.", ms=2, alpha=0.4, label="Used points")
    ax.plot(spec.wave, continuum, color="tab:blue", lw=2, label="Continuum")
    ax.set(xlabel=WAVE_LABEL, ylabel="Flux", title=title)
    ax.legend(loc="best")
    style_axes(ax)
    return fig


def overlay_axes(ax: Axes, spectra: Sequence[Spec], labels: Sequence[str] | None = None) -> None:
    """Plot several spectra on ``ax`` with a legend."""
    for spec, label in zip(spectra, labels or [s.name for s in spectra], strict=True):
        ax.plot(spec.wave, spec.flux, lw=0.8, alpha=0.8, label=label)
    ax.relim()
    ax.autoscale_view()
    ax.legend(loc="best")
    style_axes(ax)


def comparison_figure(before: Spec, after: Spec, labels: tuple[str, str], title: str) -> tuple[Figure, Axes]:
    """Pyplot figure overplotting a spectrum before and after an operation."""
    fig, ax = new_figure()
    ax.plot(before.wave, before.flux, color="gray", lw=0.8, alpha=0.8, label=labels[0])
    ax.plot(after.wave, after.flux, color="tab:red", lw=0.8, alpha=0.9, label=labels[1])
    ax.set_title(title)
    ax.legend(loc="best")
    style_axes(ax)
    return fig, ax
