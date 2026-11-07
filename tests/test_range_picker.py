"""RangePicker: range selection by mouse events (replaces ginput, which fails on macOS)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent

from bobspec import interactive
from bobspec.interactive import RangePicker, pick_ranges
from bobspec.io import Spec


@pytest.fixture
def picker():
    fig, ax = plt.subplots()
    ax.plot([5000, 5010], [0.5, 1.0])
    ax.set_xlim(5000, 5010)
    ax.set_ylim(0, 1.2)
    fig.canvas.draw()
    calls: list[tuple[float, float]] = []

    def on_range(w1, w2, axes):
        calls.append((w1, w2))
        return f"range {w1:.1f}-{w2:.1f}", [*axes.plot([w1, w2], [1, 1])]

    p = RangePicker(fig, ax, on_range)
    yield p, calls
    plt.close("all")


def click(p: RangePicker, x: float, button: MouseButton) -> None:
    xd, yd = p.ax.transData.transform((x, 0.6))
    p.on_click(MouseEvent("button_press_event", p.fig.canvas, xd, yd, button))


def key(p: RangePicker, name: str) -> None:
    p.on_key(KeyEvent("key_press_event", p.fig.canvas, name))


def test_two_right_clicks_measure_the_sorted_range(picker) -> None:
    p, calls = picker
    click(p, 5007.0, MouseButton.RIGHT)
    assert len(p.marks) == 1 and not calls
    click(p, 5003.0, MouseButton.RIGHT)
    assert calls == [pytest.approx((5003.0, 5007.0), abs=0.05)]
    assert not p.points and not p.marks  # edges cleared, range shaded
    assert len(p.artists) == 2  # the line from on_range and the shaded span
    assert p.ax.get_title().startswith("range 5003.0-5007.0")


def test_new_range_replaces_the_previous_drawing(picker) -> None:
    p, calls = picker
    for x in (5001.0, 5002.0, 5004.0, 5006.0):
        click(p, x, MouseButton.RIGHT)
    assert len(calls) == 2 and len(p.artists) == 2
    assert len(p.ax.lines) == 2  # the spectrum and the newest on_range line only


def test_left_click_and_backspace_remove_the_last_edge(picker) -> None:
    p, calls = picker
    click(p, 5005.0, MouseButton.RIGHT)
    click(p, 5005.0, MouseButton.LEFT)
    assert not p.points and not p.marks
    click(p, 5005.0, MouseButton.RIGHT)
    key(p, "backspace")
    assert not p.points
    click(p, 5005.0, MouseButton.LEFT)  # nothing to remove: no error
    assert not calls


@pytest.mark.parametrize("how", ["enter", "middle"])
def test_enter_or_middle_click_finish_and_close(picker, how: str) -> None:
    p, _ = picker
    number = p.fig.number
    if how == "enter":
        key(p, "enter")
    else:
        click(p, 5005.0, MouseButton.MIDDLE)
    assert p.finished and not plt.fignum_exists(number)


def test_clicks_ignored_while_zoom_or_pan_is_active(picker, monkeypatch: pytest.MonkeyPatch) -> None:
    p, calls = picker
    monkeypatch.setattr(interactive, "_toolbar_busy", lambda fig: True)
    click(p, 5003.0, MouseButton.RIGHT)
    click(p, 5007.0, MouseButton.RIGHT)
    assert not p.points and not calls


def test_other_keys_do_nothing(picker) -> None:
    p, calls = picker
    key(p, "a")
    key(p, "x")
    assert not p.points and not p.finished and not calls


def test_pick_ranges_closing_without_enter_means_stop() -> None:
    spec = Spec(np.linspace(5000, 5010, 50), np.ones(50))
    assert pick_ranges(spec, lambda w1, w2, ax: ("", []), "test", xlim=(5002, 5008)) is False
    assert not plt.get_fignums()  # no window left behind


# ----------------------------------------------------------------------------- --append [FILE] (sline, cows)
import bobspec.cli  # noqa: E402,F401  (registers the commands)
from bobspec.commands import append_target  # noqa: E402
from bobspec.registry import UsageError, resolve  # noqa: E402


@pytest.mark.parametrize(
    ("command", "positional", "default"),
    [
        ("sline", ["obs.fits", "syn", "5580", "0.96", "20.00"], "linelist.txt"),
        ("cows", ["obs.fits", "syn.txt", "5500", "5600"], "wind_continuum.txt"),
    ],
)
def test_append_accepts_an_optional_file(command: str, positional: list[str], default: str) -> None:
    parser = resolve(command).parser
    ns = parser.parse_args([*positional, "--append", "mylines.txt"])
    assert append_target(ns.output, ns.append, default) == (Path("mylines.txt"), True)
    ns = parser.parse_args([*positional, "--append"])
    assert append_target(ns.output, ns.append, default) == (Path(default), True)
    ns = parser.parse_args([*positional, "-o", "other.txt", "--append"])
    assert append_target(ns.output, ns.append, default) == (Path("other.txt"), True)
    ns = parser.parse_args([*positional, "-o", "other.txt"])
    assert append_target(ns.output, ns.append, default) == (Path("other.txt"), False)
    ns = parser.parse_args(positional)
    assert append_target(ns.output, ns.append, default) == (Path(default), False)


def test_append_and_output_naming_different_files_is_an_error() -> None:
    with pytest.raises(UsageError, match="two different files"):
        append_target("a.txt", "b.txt", "linelist.txt")
    assert append_target("a.txt", "a.txt", "linelist.txt") == (Path("a.txt"), True)
