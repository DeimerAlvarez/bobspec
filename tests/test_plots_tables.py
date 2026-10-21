"""Plot windows (headless) and plain-text result tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.table import Table

import bobspec.cli  # noqa: F401  (registers the commands)
from bobspec import plotting
from bobspec.io import Spec, write_spectrum
from bobspec.registry import default_table, execute, resolve, write_table


@pytest.fixture
def two_spectra(tmp_path: Path) -> tuple[Path, Path]:
    wave = np.linspace(5000.0, 5100.0, 500)
    a = write_spectrum(tmp_path / "a.txt", Spec(wave, np.ones(500)))
    b = write_spectrum(tmp_path / "b.fits", Spec(wave, 2 * np.ones(500)))
    return a, b


@pytest.fixture
def shows(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record blocking plt.show() calls instead of opening windows."""
    calls: list[int] = []
    monkeypatch.setattr(plt, "show", lambda *a, block=None, **k: calls.append(len(plt.gcf().axes[0].lines)))
    return calls


def test_plot_adds_spectra_from_terminal(two_spectra, monkeypatch, capsys) -> None:
    a, b = two_spectra
    answers = iter(["", "missing.fits", str(b), "e"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    lines: list[int] = []
    monkeypatch.setattr(plt, "close", lambda fig=None: lines.append(len(fig.axes[0].lines)))
    execute(resolve("plot"), [str(a)])
    out = capsys.readouterr().out
    assert f"Added: {b}" in out
    assert lines == [2]  # first spectrum + the one added from the terminal


def test_plot_end_of_input_finishes(two_spectra, monkeypatch) -> None:
    def no_more_input(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", no_more_input)
    execute(resolve("plot"), [str(two_spectra[0])])


def test_plot_no_add_and_list(two_spectra, tmp_path, shows) -> None:
    lst = tmp_path / "spectra.txt"
    lst.write_text("\n".join(map(str, two_spectra)) + "\n")
    execute(resolve("plot"), [f"@{lst}", "--no-add"])
    assert shows == [2]


@pytest.mark.parametrize(("command", "args"), [("cut", ["5020", "5050"]), ("rotate", ["0.6", "20"])])
def test_cut_and_rotate_show_before_and_after(two_spectra, tmp_path, shows, command, args) -> None:
    spec = str(two_spectra[0])
    execute(resolve(command), [spec, *args, str(tmp_path / "out.txt")])
    assert shows == [2]  # original + result in the same window
    execute(resolve(command), [spec, *args, str(tmp_path / "out2.txt"), "--auto"])
    assert shows == [2]  # --auto: no window
    assert (tmp_path / "out.txt").exists() and (tmp_path / "out2.txt").exists()


def test_font_sizes_applied() -> None:
    fig, ax = plotting.new_figure()
    plotting.overlay_axes(ax, [Spec([1.0, 2.0], [1.0, 2.0], name="x")])
    plotting.set_title(ax, "title")
    assert ax.xaxis.label.get_fontsize() == plotting.AXIS_LABEL_SIZE
    assert ax.yaxis.label.get_fontsize() == plotting.AXIS_LABEL_SIZE
    assert ax.title.get_fontsize() == plotting.TITLE_SIZE
    assert ax.get_legend().get_texts()[0].get_fontsize() == plotting.LEGEND_SIZE
    assert ax.xaxis.get_ticklabels()[0].get_fontsize() == plotting.TICK_LABEL_SIZE
    plt.close(fig)


def test_text_table_format(tmp_path: Path) -> None:
    rows = [
        {"line": 2, "spectrum": "a b.fits", "rv": 134.294123},
        {"line": 3, "spectrum": "c.fits", "ESO TEL GEOLAT": -24.6},
    ]
    path = write_table(tmp_path / "t.txt", rows)
    lines = path.read_text().splitlines()
    assert lines[0].split() == ["#", "line", "spectrum", "rv", "ESO_TEL_GEOLAT"]
    table = Table.read(path, format="ascii.commented_header")
    assert list(table["spectrum"]) == ["a b.fits", "c.fits"]
    assert table["rv"][0] == pytest.approx(134.294) and np.isnan(table["rv"][1])


def test_default_table_name() -> None:
    assert default_table(Path("dir/input_rv.txt"), "rv") == Path("dir/input_rv_rv.txt")
    assert default_table(Path("lista.id"), "combine") == Path("lista_combine.txt")
