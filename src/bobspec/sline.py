"""``sline``: interactive selection of spectral lines from a SYNTHE line list.

Workflow (same as the original ``sline.py``):

1. read the observed spectrum (shifted by ``wsh`` km/s), the synthetic
   ``<root>.syn`` (already convolved: its normalised flux is used as it is)
   and the line list ``<root>.lin``;
2. merge split components (same species, log gf and EP within 0.1 A);
3. show windows of ``step`` A from ``wave1``: observed (black), synthetic
   (red) and the lines with RI < ``min`` as numbered vertical marks: red for
   the selected species, green for merged split lines, blue otherwise;
4. print the lines of the window with a blend flag ('clean' if no other line
   with RI < ``min`` lies within 1.5 sigma of the instrumental profile);
5. terminal commands: ``y`` next window, ``b`` previous, ``w`` write a line
   (by its number), ``up`` change settings, ``e`` exit.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from bobspec.io import Spec, read_spectrum
from bobspec.spectrum import add_noise, resample, to_rest_frame, uniform_grid
from bobspec.synthe import LineList, blend_flags, merge_split_lines, read_linelist, read_synthetic

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

log = logging.getLogger(__name__)

InputFunc = Callable[[str], str]

FIXED_SETTINGS = frozenset({"res"})
"""Settings that cannot be changed during a session (the synthetic is already convolved)."""

HEADER = "#wave      code   log_gf      EP     RI   flag  split \n"
MENU = "# ok and next window (y)\n# previous window (b)\n# update (up)\n# write line (w)\n# exit (e)\n"


@dataclass(frozen=True, slots=True)
class SlineSettings:
    """Parameters of a session (names as in the original ``input_sline.txt``)."""

    wave1: float
    min: float
    sel: float
    res: float = 0.0
    """Resolving power of the synthetic, used only to flag blends (0 = unknown)."""
    step: float = 8.0
    snr: float = 0.0
    wsh: float = 0.0
    ysta: float = 0.5
    yend: float = 1.2

    def __post_init__(self) -> None:
        if not self.step > 0:
            raise ValueError(f"step must be positive, got {self.step}")
        if self.res < 0 or self.snr < 0:
            raise ValueError("res and snr must be >= 0")
        if not self.yend > self.ysta:
            raise ValueError(f"yend ({self.yend}) must be larger than ysta ({self.ysta})")


def format_line(lines: LineList, i: int, flag: str) -> str:
    """One output line, in the format of the original script."""
    return (
        f"{lines.wave[i]:>8.3f}  {lines.code[i]:>5.2f}  {lines.loggf[i]:>7.3f}  {lines.ep[i]:>6.3f}  "
        f"{lines.ri[i]:>5.3f}  {flag:>5s}  {int(lines.split[i]):>5d}  \n"
    )


def parse_updates(text: str) -> dict[str, float]:
    """Parse ``"step=10 ysta=0.3"`` into ``{"step": 10.0, "ysta": 0.3}``.

    Raises:
        ValueError: For unknown names or non-numeric values.
    """
    allowed = {f.name for f in fields(SlineSettings)} - FIXED_SETTINGS
    updates: dict[str, float] = {}
    for item in text.replace(",", " ").split():
        name, sep, value = item.partition("=")
        name = name.strip().lower()
        if not sep or name not in allowed:
            raise ValueError(f"'{item}': use name=value with name in {', '.join(sorted(allowed))}")
        updates[name] = float(value)
    return updates


class SlineSession:
    """State of an ``sline`` session; ``run`` drives it from the terminal."""

    def __init__(
        self,
        observed: Spec,
        synthetic_root: str,
        settings: SlineSettings,
        output: Path,
        input_func: InputFunc | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.observed_raw = observed
        self.root = synthetic_root
        self.settings = settings
        self.output = output
        self.input: InputFunc = input_func or input
        self.rng = rng if rng is not None else np.random.default_rng()
        self.written: set[tuple[float, float]] = set()
        self.fig: Figure | None = None
        self.ax: Axes | None = None
        self.all_lines = merge_split_lines(read_linelist(f"{synthetic_root}.lin"))
        self._load_synthetic()
        self._prepare()

    # ------------------------------------------------------------------ data
    def _load_synthetic(self) -> None:
        """The synthetic is already convolved: its normalised flux is used as it is."""
        self.synthetic_raw = read_synthetic(f"{self.root}.syn")

    def _prepare(self) -> None:
        """Apply shift, noise and the RI limit after any change of the settings."""
        s = self.settings
        self.observed = to_rest_frame(self.observed_raw, s.wsh) if s.wsh else self.observed_raw
        pixel = float(np.min(np.diff(self.observed.wave)))
        syn = self.synthetic_raw
        syn = resample(syn, uniform_grid(syn.wave[0], syn.wave[-1], pixel))
        self.synthetic = add_noise(syn, s.snr, self.rng, reference=1.0) if s.snr > 0 else syn
        self.lines = self.all_lines.subset(self.all_lines.ri < s.min)
        self.flags = blend_flags(self.lines.wave, self.lines.wave, s.res)
        if s.res <= 0:
            log.warning("Resolution unknown (no '# RESOL' line in the .syn file and no --res): blends flagged '-'.")

    def window(self) -> np.ndarray:
        """Indices (in ``self.lines``) of the lines inside the current window."""
        s = self.settings
        return np.flatnonzero((self.lines.wave >= s.wave1) & (self.lines.wave < s.wave1 + s.step))

    # ------------------------------------------------------------------ output
    def print_table(self, idx: np.ndarray) -> None:
        """Print the lines of the window, numbered as in the plot."""
        if idx.size == 0:
            print(
                f"No lines with RI < {self.settings.min} in {self.settings.wave1:.2f}-"
                f"{self.settings.wave1 + self.settings.step:.2f} A\n"
            )
            return
        print(" N       wave    code   log_gf      EP      RI    ref    flag   split")
        for n, i in enumerate(idx, start=1):
            split = "y" if self.lines.split[i] else "n"
            print(
                f"{n:2d}   {self.lines.wave[i]:8.3f}  {self.lines.code[i]:6.2f}  {self.lines.loggf[i]:7.3f}   "
                f"{self.lines.ep[i]:6.3f}   {self.lines.ri[i]:.3f}   {self.lines.ref[i]:>4s}   "
                f"{self.flags[i]:>5s}       {split}"
            )
        print("")

    def write_line(self, idx: np.ndarray, number: int) -> bool:
        """Append line ``number`` (1-based, as in the plot) to the output file."""
        if not 1 <= number <= idx.size:
            print(f"Not valid line number!!! (1-{idx.size})")
            return False
        i = idx[number - 1]
        key = (round(float(self.lines.wave[i]), 4), round(float(self.lines.code[i]), 2))
        if key in self.written:
            print(f"Line {self.lines.wave[i]:.3f} already written: skipped.")
            return False
        with self.output.open("a") as handle:
            handle.write(format_line(self.lines, i, str(self.flags[i])))
        self.written.add(key)
        print(f"Written: {self.lines.wave[i]:.3f}  {self.lines.code[i]:.2f}")
        return True

    # ------------------------------------------------------------------ plot
    def draw(self, idx: np.ndarray) -> None:
        """Draw the current window, reusing the figure unless the user closed it."""
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter

        from bobspec.plotting import new_figure, style_axes

        if self.fig is None or self.ax is None or not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = new_figure(figsize=(12.5, 5))
        fig, ax = self.fig, self.ax
        s = self.settings
        ax.clear()
        ax.plot(self.observed.wave, self.observed.flux, c="k")
        ax.plot(self.synthetic.wave, self.synthetic.flux, c="r")
        for n, i in enumerate(idx, start=1):
            x = self.lines.wave[i]
            if np.isclose(self.lines.code[i], s.sel, atol=1e-3):
                color, width, z = "r", 2.5, 5
            elif self.lines.split[i]:
                color, width, z = "g", 2.5, 5
            else:
                color, width, z = "b", 1.0, 1
            ax.vlines(x, 0.95, s.yend, color=color, linewidth=width, zorder=z)
            ax.text(x, s.yend - 0.1, str(n), weight=700)
        ax.set_xlim(s.wave1, s.wave1 + s.step)
        ax.set_ylim(s.ysta, s.yend)
        ax.set(xlabel=r"Wavelength [$\AA$]", ylabel="Normalized Flux")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        style_axes(ax)
        fig.canvas.draw_idle()
        plt.pause(0.001)

    def show(self) -> np.ndarray:
        """Draw the window and print its table; returns the window indices."""
        idx = self.window()
        self.draw(idx)
        self.print_table(idx)
        return idx

    # ------------------------------------------------------------------ loop
    def update(self, text: str) -> None:
        """Apply ``name=value`` changes (the resolution cannot be changed)."""
        self.settings = replace(self.settings, **parse_updates(text))
        self._prepare()

    def run(self) -> None:
        """Terminal loop: y next, b previous, up update, w write, e exit."""
        import matplotlib.pyplot as plt

        was_interactive = plt.isinteractive()
        plt.ion()
        print(MENU)
        idx = self.show()
        try:
            while True:
                try:
                    answer = self.input("insert: ").strip()
                except EOFError:
                    break
                command, _, rest = answer.partition(" ")
                match command.lower():
                    case "y":
                        self.settings = replace(self.settings, wave1=self.settings.wave1 + self.settings.step)
                        idx = self.show()
                    case "b":
                        self.settings = replace(self.settings, wave1=self.settings.wave1 - self.settings.step)
                        idx = self.show()
                    case "up":
                        self._ask_update(rest)
                        idx = self.show()
                    case "w":
                        self._ask_write(idx, rest)
                    case "e":
                        break
                    case "h" | "?":
                        print(MENU)
                    case _:
                        print("Wrong option!!!  (y, b, up, w, e)")
        finally:
            if self.fig is not None:
                plt.close(self.fig)
            if not was_interactive:
                plt.ioff()
        print(f"\nLines written in {self.output}")

    def _ask_update(self, text: str) -> None:
        s = self.settings
        print(
            f"Current: wave1={s.wave1:g} step={s.step:g} min={s.min:g} sel={s.sel:g} "
            f"snr={s.snr:g} wsh={s.wsh:g} ysta={s.ysta:g} yend={s.yend:g}"
        )
        while True:
            if not text:
                text = self.input("new values (e.g. step=10 ysta=0.3), ENTER to keep: ").strip()
            if not text:
                return
            try:
                self.update(text)
                return
            except ValueError as exc:
                print(exc)
                text = ""

    def _ask_write(self, idx: np.ndarray, text: str) -> None:
        if idx.size == 0:
            print("No lines in this window.")
            return
        while True:
            if not text:
                text = self.input("line number (ENTER to cancel): ").strip()
            if not text:
                return
            try:
                number = int(text)
            except ValueError:
                print("insert a number!!!")
                text = ""
                continue
            if 1 <= number <= idx.size:
                self.write_line(idx, number)
                return
            print(f"Not valid line number!!! (1-{idx.size})")
            text = ""


def start_session(
    spectrum: str,
    synthetic_root: str,
    settings: SlineSettings,
    output: Path,
    *,
    append: bool = False,
    input_func: InputFunc | None = None,
    rng: np.random.Generator | None = None,
) -> SlineSession:
    """Read the inputs, prepare the output file and return the session.

    The output file is created with its header line; with ``append`` an
    existing file is kept and the lines already in it are not written twice.
    """
    observed = read_spectrum(spectrum)
    session = SlineSession(observed, synthetic_root, settings, output, input_func, rng)
    if append and output.is_file():
        previous = np.loadtxt(output, comments="#", usecols=(0, 1), ndmin=2)
        session.written = {(round(float(w), 4), round(float(c), 2)) for w, c in previous}
        print(f"Appending to {output} ({len(session.written)} lines already there).")
    else:
        output.write_text(HEADER)
    return session
