"""Interactive matplotlib tools.

These are thin wrappers that collect user choices and delegate all numerics
to :mod:`bobspec.continuum` and friends. Every tool blocks until the window is
closed and returns ``None`` when the user cancels (closes the window without
confirming with ENTER), so callers never save a half-finished result.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event, KeyEvent, MouseButton, MouseEvent
from matplotlib.lines import Line2D

from bobspec.continuum import ContinuumFit, ContinuumModel, Curve, anchor_curve, iterative_fit
from bobspec.io import Spec
from bobspec.plotting import continuum_figure, new_figure, set_title, suppress_keymaps

log = logging.getLogger(__name__)


def _toolbar_busy(fig: object) -> bool:
    """True while the pan/zoom tool is active (clicks belong to the toolbar)."""
    toolbar = getattr(getattr(fig, "canvas", None), "toolbar", None)
    return bool(getattr(toolbar, "mode", ""))


@dataclass
class ContinuumEditor:
    """Keyboard/mouse editor for sigma-clipped continuum fits (``ipoly``, ``ispline``).

    Keys: ``j``/``k`` degree -/+, ``l``/``o`` sigma_low -/+, ``h``/``y``
    sigma_up -/+, ``u`` undo last masked region, ENTER accept.
    Right click masks ``mask_halfwidth`` pixels around the cursor.
    """

    spec: Spec
    make_model: Callable[[int], ContinuumModel]
    degree: int
    degree_bounds: tuple[int, int]
    sigma_low: float
    sigma_up: float
    base_mask: np.ndarray
    label: str = "deg"
    mask_halfwidth: int = 25
    step: float = 0.2
    _masked: list[slice] = field(default_factory=list, init=False)
    _result: ContinuumFit | None = field(default=None, init=False)
    _accepted: bool = field(default=False, init=False)

    KEYS = ("j", "k", "l", "o", "h", "y", "u", "enter")

    def run(self) -> ContinuumFit | None:
        """Show the editor and block; return the accepted fit or None if cancelled."""
        self.fig, self.ax = new_figure()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._refresh()
        with suppress_keymaps(self.KEYS):
            plt.show(block=True)
        return self._result if self._accepted else None

    def _mask(self) -> np.ndarray:
        mask = self.base_mask.copy()
        for region in self._masked:
            mask[region] = False
        return mask

    def _on_key(self, event: Event) -> None:
        if not isinstance(event, KeyEvent):
            return
        lo, hi = self.degree_bounds
        match event.key:
            case "k":
                self.degree = min(hi, self.degree + 1)
            case "j":
                self.degree = max(lo, self.degree - 1)
            case "o":
                self.sigma_low = round(self.sigma_low + self.step, 2)
            case "l":
                self.sigma_low = max(self.step, round(self.sigma_low - self.step, 2))
            case "y":
                self.sigma_up = round(self.sigma_up + self.step, 2)
            case "h":
                self.sigma_up = max(self.step, round(self.sigma_up - self.step, 2))
            case "u" if self._masked:
                self._masked.pop()
            case "enter":
                if self._result is None:
                    log.warning("No valid fit to accept: change the parameters first.")
                    return
                self._accepted = True
                plt.close(self.fig)
                return
            case _:
                return
        self._refresh()

    def _on_click(self, event: Event) -> None:
        if not isinstance(event, MouseEvent) or event.xdata is None:
            return
        if event.inaxes is not self.ax or event.button is not MouseButton.RIGHT or _toolbar_busy(self.fig):
            return
        idx = int(np.argmin(np.abs(self.spec.wave - event.xdata)))
        self._masked.append(slice(max(0, idx - self.mask_halfwidth), idx + self.mask_halfwidth + 1))
        self._refresh()

    def _refresh(self) -> None:
        try:
            self._result = iterative_fit(
                self.spec.wave,
                self.spec.flux,
                self.make_model(self.degree),
                sigma_low=self.sigma_low,
                sigma_up=self.sigma_up,
                mask=self._mask(),
            )
            continuum, used = self._result.continuum, self._result.used
        except ValueError as exc:
            log.warning("Fit failed: %s", exc)
            self._result = None
            continuum, used = np.full(self.spec.wave.shape, np.nan), None
        title = (
            f"{self.spec.name}: {self.label}={self.degree}  \u03c3_low={self.sigma_low:.1f}  "
            f"\u03c3_up={self.sigma_up:.1f}\n[j/k] {self.label}  [l/o] \u03c3_low  [h/y] \u03c3_up  "
            "[right click] mask  [u] undo  [ENTER] accept  (close = cancel)"
        )
        continuum_figure(self.spec, continuum, used, title, ax=self.ax)
        self.fig.canvas.draw_idle()
        log.info("%s=%d sigma_low=%.2f sigma_up=%.2f", self.label, self.degree, self.sigma_low, self.sigma_up)


@dataclass
class _AnchorState:
    points: list[tuple[float, float]] = field(default_factory=list)
    markers: list[Line2D] = field(default_factory=list)
    curve: Curve | None = None
    line: Line2D | None = None
    dirty: bool = True
    accepted: bool = False


def anchor_editor(spec: Spec, k: int) -> Curve | None:
    """Place continuum anchors by hand (``manual_spline``).

    Right click adds an anchor, left click removes the last one, ENTER fits
    the spline; ENTER again without changes accepts it. Closing the window
    cancels.
    """
    fig, ax = new_figure()
    ax.plot(spec.wave, spec.flux, color="gray", lw=0.8, alpha=0.6)
    set_title(
        ax,
        f"{spec.name}: manual spline k={k}\n[right] add  [left] remove last  "
        "[ENTER] fit, ENTER again to accept  (close = cancel)",
    )
    state = _AnchorState()

    def on_click(event: Event) -> None:
        if not isinstance(event, MouseEvent) or event.inaxes is not ax or _toolbar_busy(fig):
            return
        if event.button is MouseButton.RIGHT and event.xdata is not None and event.ydata is not None:
            state.points.append((event.xdata, event.ydata))
            state.markers.append(ax.plot(event.xdata, event.ydata, "rx", ms=8, mew=2)[0])
        elif event.button is MouseButton.LEFT and state.points:
            state.points.pop()
            state.markers.pop().remove()
        else:
            return
        state.dirty = True
        fig.canvas.draw_idle()

    def on_key(event: Event) -> None:
        if not isinstance(event, KeyEvent) or event.key != "enter":
            return
        if not state.dirty and state.curve is not None:
            state.accepted = True
            plt.close(fig)
            return
        try:
            curve = anchor_curve([p[0] for p in state.points], [p[1] for p in state.points], k=k)
        except ValueError as exc:
            log.warning("%s", exc)
            return
        if state.line is not None:
            state.line.remove()
        state.curve, state.dirty = curve, False
        state.line = ax.plot(spec.wave, curve(spec.wave), "tab:green", lw=2)[0]
        fig.canvas.draw_idle()
        log.info("Continuum updated: ENTER again to accept, or keep editing.")

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    with suppress_keymaps(("enter",)):
        plt.show(block=True)
    return state.curve if state.accepted else None


def pick_ranges(
    spec: Spec,
    on_range: Callable[[float, float, Axes], tuple[str, list[Artist]]],
    title: str,
    xlim: tuple[float, float] | None = None,
) -> bool:
    """Repeatedly let the user select a wavelength range with two right clicks.

    ``on_range(w1, w2, ax)`` is called for each selection; it may draw on
    ``ax`` and returns a status line for the title plus the artists to remove
    at the next selection.

    Returns:
        True if the user finished with ENTER, False if the window was closed
        (callers processing several items treat this as "stop").
    """
    fig, ax = new_figure()
    ax.plot(spec.wave, spec.flux, color="k", lw=0.8)
    if xlim is not None:
        ax.set_xlim(*xlim)
        inside = (spec.wave >= xlim[0]) & (spec.wave <= xlim[1]) & spec.finite
        if inside.any():
            lo, hi = spec.flux[inside].min(), spec.flux[inside].max()
            pad = 0.1 * (hi - lo) or 0.1
            ax.set_ylim(lo - pad, hi + pad)
    set_title(ax, f"{title}\n2 right clicks: select range | ENTER: done | close window: stop")
    plt.show(block=False)
    plt.pause(0.1)
    artists: list[Artist] = []
    while plt.fignum_exists(fig.number):
        pts = fig.ginput(
            n=2,
            timeout=0,
            show_clicks=True,
            mouse_add=MouseButton.RIGHT,
            mouse_pop=MouseButton.LEFT,
            mouse_stop=MouseButton.MIDDLE,
        )
        if not plt.fignum_exists(fig.number):
            return False
        if len(pts) < 2:
            plt.close(fig)
            return True
        w1, w2 = sorted(p[0] for p in pts)
        for artist in artists:
            artist.remove()
        status, artists = on_range(w1, w2, ax)
        artists.append(ax.axvspan(w1, w2, color="tab:green", alpha=0.15))
        set_title(ax, f"{status}\n2 right clicks: new range | ENTER: done | close window: stop")
        fig.canvas.draw_idle()
    return False
