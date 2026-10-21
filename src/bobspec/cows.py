"""``cows``: semi-automatic selection of continuum windows.

Workflow (same as the original ``cows.py``):

1. read the observed spectrum and a synthetic one (optionally degraded to R);
2. shift the observed spectrum to rest frame (given RV, or estimated by CCF);
3. keep the region [l1, l2] and normalise it with a running median, whose
   kernel can be changed from the terminal until the continuum looks right;
4. show the normalised spectrum and the synthetic in consecutive windows of
   ``step`` Angstrom: left click marks a point, right click removes the last;
5. after each window the points are sorted and written in pairs
   (start, end of a continuum window) to the output file.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from scipy.ndimage import median_filter

from bobspec.io import Spec

log = logging.getLogger(__name__)

InputFunc = Callable[[str], str]


def running_median_continuum(flux: np.ndarray, kernel: int) -> np.ndarray:
    """First-guess continuum: running median over ``kernel`` pixels.

    Edges are mirrored, as ``np.pad(mode="reflect")`` + ``medfilt`` in the
    original script. NaN pixels are filled by linear interpolation first.
    A kernel longer than the spectrum is shortened (with a warning).

    Raises:
        ValueError: If ``kernel`` is not an odd integer >= 3 or there are
            fewer than 3 finite pixels.
    """
    if kernel < 3 or kernel % 2 == 0:
        raise ValueError(f"the kernel must be an odd integer >= 3, got {kernel}")
    flux = np.asarray(flux, float)
    good = np.isfinite(flux)
    if np.count_nonzero(good) < 3:
        raise ValueError("fewer than 3 finite pixels")
    if not good.all():
        idx = np.arange(flux.size)
        flux = np.interp(idx, idx[good], flux[good])
    largest = flux.size if flux.size % 2 else flux.size - 1
    if kernel > largest:
        log.warning("Kernel %d longer than the spectrum (%d pixels): using %d.", kernel, flux.size, largest)
        kernel = largest
    return median_filter(flux, size=kernel, mode="mirror")


def pair_points(xs: list[float]) -> tuple[list[tuple[float, float]], float | None]:
    """Sort the clicked wavelengths and group them in consecutive pairs.

    Returns:
        The pairs and the leftover point when the number of clicks is odd.
    """
    ordered = sorted(xs)
    pairs = list(zip(ordered[0::2], ordered[1::2], strict=False))
    leftover = ordered[-1] if len(ordered) % 2 else None
    return pairs, leftover


def plot_windows(start: float, stop: float, step: float) -> Iterator[tuple[float, float]]:
    """Consecutive windows ``(x0, x0 + step)`` from ``start - 2`` until ``stop`` is covered."""
    if not step > 0:
        raise ValueError(f"step must be positive, got {step}")
    x0 = start - 2.0
    while x0 < stop:
        yield x0, x0 + step
        x0 += step


def format_pair(a: float, b: float) -> str:
    """One output line, in the format of the original script."""
    return f"{a:.2f} \t {b:.2f}\n"


@dataclass
class ClickCollector:
    """Left click adds a point (red cross + number), right click removes the last one.

    Clicks made while the zoom or pan tool is active are ignored.
    """

    ax: Axes
    offset: float = 0.1
    points: list[float] = field(default_factory=list)
    _artists: list[tuple[Artist, Artist]] = field(default_factory=list)

    def __call__(self, event: object) -> None:
        """Handle one mouse click (connected to ``button_press_event``)."""
        from matplotlib.backend_bases import MouseButton, MouseEvent

        from bobspec.interactive import _toolbar_busy

        if not isinstance(event, MouseEvent) or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        fig = self.ax.get_figure()
        if fig is None:
            return
        if _toolbar_busy(fig):
            return
        if event.button is MouseButton.LEFT:
            x, y = float(event.xdata), float(event.ydata)
            print(f"wvl selected: {x:.3f}")
            self.points.append(x)
            marker = self.ax.plot(x, y, c="r", marker="x", markersize=12, zorder=3, mew=3)[0]
            label = self.ax.text(
                x, y + self.offset, str(len(self.points)), color="black", fontsize=12, va="bottom", ha="center"
            )
            self._artists.append((marker, label))
        elif event.button is MouseButton.RIGHT and self.points:
            self.points.pop()
            for artist in self._artists.pop():
                artist.remove()
        else:
            return
        fig.canvas.draw_idle()


def choose_kernel(
    wave: np.ndarray, flux: np.ndarray, kernel: int, input_func: InputFunc | None = None
) -> tuple[np.ndarray, int]:
    """Show spectrum and running-median continuum; ``k`` changes the kernel, ``y`` accepts.

    Returns:
        The accepted continuum and kernel.
    """
    input_func = input_func or input
    import matplotlib.pyplot as plt

    from bobspec.plotting import new_figure, style_axes

    continuum = running_median_continuum(flux, kernel)
    was_interactive = plt.isinteractive()
    plt.ion()
    fig, ax = new_figure()
    try:
        while True:
            ax.clear()
            ax.plot(wave, flux, c="k")
            ax.plot(wave, continuum, c="r", zorder=5)
            ax.set(xlabel=r"Wavelength [$\AA$]", ylabel="Flux", title=f"Running median, kernel = {kernel} pixels")
            style_axes(ax)
            fig.canvas.draw_idle()
            plt.pause(0.001)
            print("# ok (y)\n# new kernel value (k)\n")
            answer = input_func("insert: ").strip().lower()
            if answer == "y":
                break
            if not answer.startswith("k"):
                print("Wrong option")
                continue
            new = _ask_kernel(answer[1:].strip(), input_func)
            try:
                continuum, kernel = running_median_continuum(flux, new), new
            except ValueError as exc:
                print(exc)
        ax.clear()
        ax.plot(wave, flux / continuum, c="k")
        ax.axhline(1.0, c="r", ls="--")
        ax.set(xlabel=r"Wavelength [$\AA$]", ylabel="Normalised flux", title="First-guess normalisation")
        style_axes(ax)
        fig.canvas.draw_idle()
        plt.pause(3)
    finally:
        plt.close(fig)
        if not was_interactive:
            plt.ioff()
    return continuum, kernel


def _ask_kernel(text: str, input_func: InputFunc) -> int:
    while True:
        if not text:
            text = input_func("new value: ").strip()
        try:
            value = int(text)
        except ValueError:
            print("insert a correct value")
        else:
            if value >= 3 and value % 2:
                return value
            print("insert an odd number >= 3!!!")
        text = ""


def select_windows(
    norm: Spec,
    synthetic: Spec | None,
    output: Path,
    step: float,
    input_func: InputFunc | None = None,
) -> int:
    """Show consecutive windows and append the selected continuum windows to ``output``.

    Returns:
        The number of windows written.
    """
    input_func = input_func or input
    import matplotlib.pyplot as plt

    from bobspec.plotting import new_figure, style_axes

    written = 0
    windows = list(plot_windows(norm.wave[0], norm.wave[-1], step))
    for i, (x0, x1) in enumerate(windows, start=1):
        inside = (norm.wave >= x0) & (norm.wave <= x1) & norm.finite
        y0, y1 = (norm.flux[inside].min() - 0.2, norm.flux[inside].max() + 0.2) if inside.any() else (0.0, 1.2)
        fig, ax = new_figure(figsize=(14, 5))
        ax.plot(norm.wave, norm.flux, c="k", zorder=2, label="Observed")
        if synthetic is not None:
            ax.plot(synthetic.wave, synthetic.flux, c="b", label="Synthetic")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_ylabel("Normalized Flux")
        ax.set_title(f"window {i}/{len(windows)}: [left] add point  [right] remove last  (close to continue)")
        ax.legend(loc=4, frameon=True, edgecolor="k")
        style_axes(ax)
        collector = ClickCollector(ax)
        fig.canvas.mpl_connect("button_press_event", collector)
        plt.show(block=True)
        plt.close(fig)

        pairs, leftover = pair_points(collector.points)
        if leftover is not None:
            print(f"Odd number of points: the last one ({leftover:.2f}) has no partner and is ignored.")
        if pairs:
            with output.open("a") as handle:
                handle.writelines(format_pair(a, b) for a, b in pairs)
            written += len(pairs)
            print(f"{len(pairs)} window(s) written to {output}")
        if i == len(windows):
            break
        if _ask_yes_no("\nend the code? (y/n): ", input_func):
            break
    return written


def _ask_yes_no(prompt: str, input_func: InputFunc) -> bool:
    while True:
        try:
            answer = input_func(prompt).strip().lower()
        except EOFError:
            return True
        print("")
        if answer in ("y", "n"):
            return answer == "y"
        print("Wrong option")
