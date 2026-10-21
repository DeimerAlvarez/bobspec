"""Interactive tools driven headlessly: plt.show is replaced by a scripted event sequence."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent

from bobspec import continuum as cont
from bobspec import interactive
from bobspec.io import Spec

Script = Callable[[plt.Figure], None]


def key(fig: plt.Figure, k: str) -> None:
    fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", fig.canvas, key=k))


def click(fig: plt.Figure, xdata: float, ydata: float, button: MouseButton) -> None:
    ax = fig.axes[0]
    x, y = ax.transData.transform((xdata, ydata))
    fig.canvas.callbacks.process("button_press_event", MouseEvent("button_press_event", fig.canvas, x, y, button))


@pytest.fixture
def scripted_show(monkeypatch: pytest.MonkeyPatch) -> Callable[[Script], None]:
    def install(script: Script) -> None:
        monkeypatch.setattr(interactive.plt, "show", lambda *a, **k: script(plt.gcf()))

    return install


@pytest.fixture
def spec() -> Spec:
    wave = np.linspace(5000.0, 5100.0, 2000)
    rng = np.random.default_rng(0)
    return Spec(wave, 100.0 + 0.1 * (wave - 5000.0) + rng.normal(0, 0.5, wave.size), name="test")


def _editor(spec: Spec) -> interactive.ContinuumEditor:
    return interactive.ContinuumEditor(spec, cont.chebyshev_model, 2, (1, 10), 2.0, 3.0, np.ones(len(spec), bool))


def test_editor_keys_and_accept(spec: Spec, scripted_show) -> None:
    editor = _editor(spec)
    lowest: list[float] = []

    def script(fig: plt.Figure) -> None:
        for k in ("k", "k", "j", "y"):
            key(fig, k)
        for _ in range(12):
            key(fig, "l")
        lowest.append(editor.sigma_low)
        for _ in range(9):
            key(fig, "o")
        key(fig, "enter")

    scripted_show(script)
    fit = editor.run()
    assert fit is not None
    assert lowest == [pytest.approx(0.2)]  # never below one step
    assert (editor.degree, editor.sigma_low, editor.sigma_up) == (3, pytest.approx(2.0), pytest.approx(3.2))


def test_editor_matplotlib_keymaps_restored(spec: Spec, scripted_show) -> None:
    scripted_show(lambda fig: key(fig, "enter"))
    _editor(spec).run()
    assert "k" in plt.rcParams["keymap.xscale"]


def test_editor_close_cancels(spec: Spec, scripted_show) -> None:
    scripted_show(lambda fig: plt.close(fig))
    assert _editor(spec).run() is None


def test_editor_right_click_masks_and_undo(spec: Spec, scripted_show) -> None:
    editor = _editor(spec)
    used: list[int] = []

    def script(fig: plt.Figure) -> None:
        click(fig, 5050.0, 105.0, MouseButton.RIGHT)
        used.append(int(editor._result.used.sum()))
        key(fig, "u")
        used.append(int(editor._result.used.sum()))
        key(fig, "enter")

    scripted_show(script)
    editor.run()
    assert used[0] < used[1]


def test_anchor_editor_fit_then_accept(spec: Spec, scripted_show) -> None:
    def script(fig: plt.Figure) -> None:
        for x in (5005.0, 5050.0, 5095.0):
            click(fig, x, 100.0 + 0.1 * (x - 5000.0), MouseButton.RIGHT)
        click(fig, 5060.0, 0.0, MouseButton.RIGHT)
        click(fig, 5060.0, 0.0, MouseButton.LEFT)  # removes the last (bad) anchor
        key(fig, "enter")  # fit
        key(fig, "enter")  # accept

    scripted_show(script)
    curve = interactive.anchor_editor(spec, k=1)
    assert curve is not None
    assert curve(np.array([5050.0]))[0] == pytest.approx(105.0, abs=0.05)


def test_anchor_editor_too_few_points_is_cancel(spec: Spec, scripted_show) -> None:
    def script(fig: plt.Figure) -> None:
        click(fig, 5050.0, 100.0, MouseButton.RIGHT)
        key(fig, "enter")  # not enough anchors: warning only
        plt.close(fig)

    scripted_show(script)
    assert interactive.anchor_editor(spec, k=3) is None
