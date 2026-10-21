"""cows and sline: pure functions and headless interactive sessions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from scipy.signal import medfilt

import bobspec.cli  # noqa: F401  (registers the commands)
from bobspec import cows, sline
from bobspec.io import Spec, SpectrumReadError, write_spectrum
from bobspec.registry import UsageError, execute, resolve
from bobspec.synthe import (
    LineList,
    blend_flags,
    merge_split_lines,
    read_linelist,
    read_synthetic,
    synthetic_resolution,
)

WAVE = np.arange(5580.0, 5640.0, 0.01)
LINES = [  # wave, code, loggf, EP, ref, RI
    (5588.749, 20.00, 0.358, 2.526, "K07", 0.60),
    (5590.114, 20.00, -0.571, 2.521, "K07", 0.85),
    (5594.462, 20.00, 0.097, 2.523, "K07", 0.70),
    (5598.480, 20.00, -0.087, 2.521, "K07", 0.75),
    (5598.520, 26.00, -1.500, 3.000, "K14", 0.90),  # blend with the previous line
    (5601.277, 20.00, -0.523, 2.526, "K07", 0.80),
    (5602.842, 26.00, -0.950, 3.430, "K14", 0.92),
    (5615.000, 56.01, 0.100, 0.600, "B99", 0.70),  # two split components ...
    (5615.040, 56.01, 0.100, 0.600, "B99", 0.65),  # ... merged into one entry
    (5630.000, 26.00, -2.000, 4.000, "K14", 0.995),  # too weak for MIN = 0.96
]


def profile(wave: np.ndarray, rv: float = 0.0) -> np.ndarray:
    flux = np.ones_like(wave)
    for w, *_, ri in LINES:
        flux *= 1 - (1 - ri) * np.exp(-0.5 * ((wave - w * (1 + rv / 299792.458)) / 0.08) ** 2)
    return flux


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "syn_ca"
    cont = np.full(WAVE.size, 2.0e6)
    np.savetxt(f"{root}.syn", np.column_stack([WAVE, profile(WAVE), profile(WAVE) * cont, cont]), header="RESOL 20000")
    rows = [f"{w:.3f} {c:.2f} {g:.3f} {e:.3f} 0 0 0 0 {r} {ri:.3f}" for w, c, g, e, r, ri in LINES]
    Path(f"{root}.lin").write_text("# test line list\n" + "\n".join(rows) + "\n")
    rng = np.random.default_rng(1)
    observed = 800 * (1 + 0.001 * (WAVE - 5600)) * profile(WAVE, 50.0) * (1 + rng.normal(0, 0.003, WAVE.size))
    obs = write_spectrum(tmp_path / "obs.txt", Spec(WAVE, observed))
    norm = write_spectrum(tmp_path / "norm.txt", Spec(WAVE, profile(WAVE) + rng.normal(0, 0.003, WAVE.size)))
    return {"root": root, "obs": obs, "norm": norm, "dir": tmp_path}


# ----------------------------------------------------------------------------- synthe
def test_read_synthetic(files) -> None:
    raw = read_synthetic(f"{files['root']}.syn")  # 2 columns used: normalised flux
    degraded = read_synthetic(f"{files['root']}.syn", 20000)
    assert raw.flux.min() < 0.6  # the two lines at 5598.5 A overlap
    assert degraded.flux.min() > raw.flux.min()  # broadened lines are shallower
    np.testing.assert_allclose(degraded.flux[np.abs(WAVE - 5635) < 2], 1.0, atol=1e-3)


def test_read_synthetic_needs_four_columns(tmp_path) -> None:
    path = tmp_path / "two.syn"
    np.savetxt(path, np.column_stack([WAVE, profile(WAVE)]))
    with pytest.raises(SpectrumReadError, match="4-column"):
        read_synthetic(path, 20000)


def test_read_linelist_and_merge(files) -> None:
    lines = read_linelist(f"{files['root']}.lin")
    assert len(lines) == len(LINES) and lines.ref[0] == "K07"
    merged = merge_split_lines(lines)
    assert len(merged) == len(LINES) - 1
    ba = np.flatnonzero(merged.code == 56.01)
    assert ba.size == 1 and merged.split[ba[0]] == 1 and merged.ri[ba[0]] == pytest.approx(0.65)
    assert merged.split.sum() == 1
    assert np.all(np.diff(merged.wave) >= 0)


def test_merge_requires_same_gf_and_ep() -> None:
    wave = np.array([5000.0, 5000.05])
    lines = LineList(
        wave=wave,
        code=np.array([26.0, 26.0]),
        loggf=np.array([-1.0, -1.2]),
        ep=np.array([2.0, 2.0]),
        ref=np.array(["a", "b"]),
        ri=np.array([0.8, 0.9]),
        split=np.zeros(2, int),
    )
    assert len(merge_split_lines(lines)) == 2


def test_read_linelist_errors(tmp_path) -> None:
    bad = tmp_path / "bad.lin"
    bad.write_text("5000.0 26.0 -1.0 2.0\n")
    with pytest.raises(SpectrumReadError, match="10 columns"):
        read_linelist(bad)
    with pytest.raises(SpectrumReadError, match="not found"):
        read_linelist(tmp_path / "missing.lin")


def test_single_line_file(tmp_path) -> None:
    one = tmp_path / "one.lin"
    one.write_text("5000.0 26.0 -1.0 2.0 0 0 0 0 K14 0.8\n")
    assert len(read_linelist(one)) == 1


def test_blend_flags() -> None:
    wave = np.array([5000.0, 5000.05, 5010.0])
    np.testing.assert_array_equal(blend_flags(wave, wave, 20000), ["/", "/", "clean"])
    np.testing.assert_array_equal(blend_flags(wave, wave, 0), ["-", "-", "-"])
    # a neighbour outside the shown subset still counts (lines across a window edge)
    np.testing.assert_array_equal(blend_flags(wave[2:], np.append(wave, 5010.08), 20000), ["/"])


# ----------------------------------------------------------------------------- cows: pure parts
def old_first_norm(flux: np.ndarray, ksize: int) -> np.ndarray:
    padded = np.pad(flux, (ksize // 2, ksize // 2), mode="reflect")
    return medfilt(padded, kernel_size=ksize)[ksize // 2 : -ksize // 2 + 1]


@pytest.mark.parametrize("kernel", [3, 11, 101])
def test_running_median_matches_original(kernel: int) -> None:
    flux = np.random.default_rng(kernel).normal(1, 0.1, 500)
    np.testing.assert_allclose(cows.running_median_continuum(flux, kernel), old_first_norm(flux, kernel))


def test_running_median_validation_and_nan() -> None:
    with pytest.raises(ValueError):
        cows.running_median_continuum(np.ones(50), 10)
    flux = np.ones(50)
    flux[10] = np.nan
    assert np.isfinite(cows.running_median_continuum(flux, 5)).all()
    assert cows.running_median_continuum(np.ones(20), 101).size == 20  # kernel shortened


def test_pairs_and_windows() -> None:
    pairs, leftover = cows.pair_points([5.0, 1.0, 3.0, 2.0, 4.0])
    assert pairs == [(1.0, 2.0), (3.0, 4.0)] and leftover == 5.0
    assert cows.pair_points([]) == ([], None)
    windows = list(cows.plot_windows(100.0, 130.0, 15.0))
    assert windows[0] == (98.0, 113.0) and windows[-1][1] >= 130.0
    assert cows.format_pair(5600.123, 5601.5) == "5600.12 \t 5601.50\n"


# ----------------------------------------------------------------------------- cows: session
def click(fig, x, y, button) -> None:
    ax = fig.axes[0]
    px, py = ax.transData.transform((x, y))
    fig.canvas.callbacks.process("button_press_event", MouseEvent("button_press_event", fig.canvas, px, py, button))


def test_cows_command(files, monkeypatch) -> None:
    monkeypatch.chdir(files["dir"])
    answers = iter(["x", "k", "10", "51", "y", "n", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    windows: list[int] = []

    def fake_show(*args, block=None, **kwargs):
        if not block:
            return
        fig = plt.gcf()
        windows.append(1)
        if len(windows) == 1:  # first window: 3 points (one leftover), then remove one and add two
            for x in (5588.0, 5586.0, 5590.0, 5592.0):
                click(fig, x, 1.0, MouseButton.LEFT)
            click(fig, 0, 0, MouseButton.MIDDLE)
            click(fig, 5592.0, 1.0, MouseButton.RIGHT)  # removes 5592
            click(fig, 5584.0, 1.0, MouseButton.LEFT)
        else:
            for x in (5596.0, 5597.0, 5599.5):
                click(fig, x, 1.0, MouseButton.LEFT)

    monkeypatch.setattr(plt, "show", fake_show)
    monkeypatch.setattr(plt, "pause", lambda *a: None)
    row = execute(
        resolve("cows"),
        [
            str(files["obs"]),
            f"{files['root']}.syn",
            "5582",
            "5638",
            "-o",
            "cont.txt",
            "--resol",
            "20000",
            "--rv",
            "auto",
            "--kernel",
            "101",
        ],
    )
    assert row["rv"] == pytest.approx(50.0, abs=0.5)
    assert row["kernel"] == 51 and len(windows) == 2
    text = Path("cont.txt").read_text()
    assert text == "5584.00 \t 5586.00\n5588.00 \t 5590.00\n5596.00 \t 5597.00\n"


def test_cows_region_without_data(files, monkeypatch) -> None:
    with pytest.raises(ValueError, match="No spectrum"):
        execute(resolve("cows"), [str(files["obs"]), f"{files['root']}.syn", "7000", "7100"])


def test_cows_bad_rv_and_kernel(files) -> None:
    with pytest.raises(UsageError):
        execute(resolve("cows"), [str(files["obs"]), f"{files['root']}.syn", "5582", "5638", "--rv", "fast"])
    with pytest.raises(UsageError):
        execute(resolve("cows"), [str(files["obs"]), f"{files['root']}.syn", "5582", "5638", "--kernel", "100"])


# ----------------------------------------------------------------------------- sline
def test_parse_updates() -> None:
    assert sline.parse_updates("step=10, ysta=0.3") == {"step": 10.0, "ysta": 0.3}
    with pytest.raises(ValueError):
        sline.parse_updates("colour=3")
    with pytest.raises(ValueError):
        sline.parse_updates("step")


def test_settings_validation() -> None:
    with pytest.raises(ValueError):
        sline.SlineSettings(wave1=5000, min=0.9, sel=26.0, res=20000, ysta=1.2, yend=0.5)


def run_sline(files, monkeypatch, answers, extra=()):
    monkeypatch.chdir(files["dir"])
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))
    monkeypatch.setattr(plt, "pause", lambda *a: None)
    return execute(resolve("sline"), [str(files["norm"]), "syn_ca", "5585", "0.96", "20.00", *extra])


def test_sline_session(files, monkeypatch, capsys) -> None:
    answers = [
        "w",
        "0",
        "9",
        "abc",
        "1",  # invalid numbers are refused, then line 1 (5588.749)
        "w 1",  # same line again: skipped
        "y",  # next window 5593-5601
        "w 2",  # 5598.480 (blended with the Fe line)
        "b",
        "b",  # back before the start: still works (no data cut)
        "up step=60 wave1=5585",
        "w 8",  # one big window: line 8 is the merged Ba line
        "zz",
        "e",
    ]
    row = run_sline(files, monkeypatch, answers, extra=["-o", "sel.txt"])
    lines = Path(files["dir"] / "sel.txt").read_text().splitlines()
    assert lines[0] == sline.HEADER.rstrip("\n")
    assert [line.split()[0] for line in lines[1:]] == ["5588.749", "5598.480", "5615.040"]
    fields = lines[2].split()
    assert fields[5] == "/" and lines[1].split()[5] == "clean"
    assert lines[3].split()[-1] == "1"  # split flag of the merged Ba line
    assert row["n_lines"] == 3
    out = capsys.readouterr().out
    assert "already written" in out and "Wrong option" in out and "Not valid line number" in out


def test_sline_window_without_lines_and_append(files, monkeypatch) -> None:
    run_sline(files, monkeypatch, ["w 1", "e"], extra=["-o", "sel.txt"])
    run_sline(files, monkeypatch, ["w 1", "y", "w 1", "e"], extra=["-o", "sel.txt", "--append"])
    lines = Path(files["dir"] / "sel.txt").read_text().splitlines()
    assert [line.split()[0] for line in lines[1:]] == ["5588.749", "5594.462"]
    run_sline(files, monkeypatch, ["up wave1=5700", "w", "e"])  # empty window: nothing to write


def test_sline_root_with_extension_and_snr(files, monkeypatch) -> None:
    monkeypatch.chdir(files["dir"])
    answers = iter(["up res=0", "snr=50", "e"])  # the resolution cannot be changed
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(plt, "pause", lambda *a: None)
    execute(resolve("sline"), [str(files["norm"]), "syn_ca.syn", "5585", "0.96", "20", "--seed", "3"])


# ----------------------------------------------------------------------------- real file format
DATA = Path(__file__).parent / "data"


def test_real_format_files() -> None:
    syn = read_synthetic(DATA / "format_example.syn", 20000)
    assert syn.wave[0] == pytest.approx(5550.009) and np.isfinite(syn.flux).all()
    lines = read_linelist(DATA / "format_example.lin")
    assert len(lines) == 69 and lines.ref[0] == "L09" and lines.ri[0] == pytest.approx(0.98)
    merged = merge_split_lines(lines)
    assert len(merged) == 35 and merged.split.sum() == 6
    nb = np.flatnonzero(np.isclose(merged.wave, 5550.165))
    assert merged.code[nb[0]] == 41.0 and merged.split[nb[0]] == 1  # hyperfine components share log gf
    strong = merged.subset(merged.ri < 0.96)
    assert strong.wave.tolist() == [5552.215]
    assert blend_flags(strong.wave, strong.wave, 20000).tolist() == ["clean"]


def test_sline_on_real_format(tmp_path, monkeypatch) -> None:
    for suffix in (".syn", ".lin"):
        (tmp_path / f"example{suffix}").write_bytes((DATA / f"format_example{suffix}").read_bytes())
    syn = read_synthetic(DATA / "format_example.syn", 20000)
    write_spectrum(tmp_path / "obs.fits", syn)
    monkeypatch.chdir(tmp_path)
    answers = iter(["w 1", "up min=1.01", "w 11", "e"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(plt, "pause", lambda *a: None)
    execute(resolve("sline"), ["obs.fits", "example", "5550", "0.96", "41.00"])
    rows = Path("linelist.txt").read_text().splitlines()
    assert rows[1].split() == ["5552.215", "21.01", "-2.090", "1.455", "0.940", "clean", "1"]
    assert rows[2].split()[:2] == ["5550.651", "24.00"]


def test_resolution_from_header(files, tmp_path) -> None:
    assert synthetic_resolution(f"{files['root']}.syn") == 20000
    assert synthetic_resolution(DATA / "format_example.syn") == 20000
    bare = tmp_path / "bare.syn"
    np.savetxt(bare, np.column_stack([WAVE, profile(WAVE)]))
    assert synthetic_resolution(bare) is None
    assert synthetic_resolution(tmp_path / "missing.syn") is None


def test_sline_resolution_override_and_missing(files, monkeypatch, capsys) -> None:
    def table_flags(extra):
        run_sline(files, monkeypatch, ["e"], extra=extra)
        out = capsys.readouterr().out
        return [line.split()[7] for line in out.splitlines() if line.strip()[:2].strip().isdigit()]

    assert table_flags([]) == ["clean", "clean"]  # R = 20000 from the header
    assert table_flags(["--res", "500"]) == ["/", "/"]  # much lower R: the two lines blend
    header_free = files["dir"] / "syn_ca.syn"
    text = header_free.read_text().splitlines()
    header_free.write_text("\n".join(line for line in text if "RESOL" not in line) + "\n")
    assert table_flags([]) == ["-", "-"]  # unknown resolution
