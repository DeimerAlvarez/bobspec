"""PDF report of the ew command."""

import re
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from test_ew import CENTRES, CODES, make_spectrum

import bobspec.cli  # noqa: F401  (registers the commands)
from bobspec import ew
from bobspec.ew_report import PANELS_PER_PAGE, _panels, species_name, write_ew_report
from bobspec.io import write_spectrum
from bobspec.registry import execute, resolve


@pytest.mark.parametrize(
    ("code", "name"),
    [(26.00, "Fe I"), (26.01, "Fe II"), (20.0, "Ca I"), (56.01, "Ba II"), (1.0, "H I"), (106.0, "106.00")],
)
def test_species_name(code: float, name: str) -> None:
    assert species_name(code) == name


@pytest.fixture(scope="module")
def result() -> ew.EWResult:
    extra = np.array([5650.0, 5750.0])  # list lines with no absorption: not identified
    lab = np.concatenate([CENTRES, extra])
    codes = np.concatenate([CODES, [22.0, 22.01]])
    return ew.measure_ew(make_spectrum(), lab, codes, ew.EWSettings(masks=()))


def test_panels_identified_first_then_missing(result: ew.EWResult) -> None:
    panels = _panels(result, ())
    n_found = len(result.run.lines)
    assert all(p.line is not None for p in panels[:n_found])
    assert all(p.line is None and p.reason == "not identified" for p in panels[n_found:])
    assert {p.wave for p in panels[n_found:]} >= {5650.0, 5750.0}
    waves = [p.wave for p in panels[:n_found]]
    assert waves == sorted(waves)


def test_partly_masked_reason(result: ew.EWResult) -> None:
    target = 5650.0 * (1 + result.run.rv / 299792.458)
    masks = (ew.MaskRegion(target + 0.1, target + 5.0),)
    reasons = {p.wave: p.reason for p in _panels(result, masks) if p.line is None}
    assert reasons[5650.0] == "partly masked"


def test_line_fit_arrays(result: ew.EWResult) -> None:
    for line in result.run.lines:
        assert line.fit_wave.size >= 6 and line.fit_wave.size == line.fit_flux.size == line.fit_neighbours.size
        assert line.fit_wave[0] < line.wave_fit < line.fit_wave[-1]
        assert np.min(line.fit_flux) < 1.0 and np.all(line.fit_flux <= 1.0)
        assert np.all(line.fit_neighbours <= 1.0 + 1e-12)


def test_line_alone_excludes_its_neighbour(result: ew.EWResult) -> None:
    """In a close pair, the red profile is the line alone; the neighbour is in fit_neighbours."""
    lines = sorted(result.run.lines, key=lambda ln: ln.wave)
    pairs = [(a, b) for a, b in pairwise(lines) if b.wave - a.wave < 2 * a.fwhm]
    assert pairs, "the synthetic spectrum should contain a close pair"
    first, second = pairs[0]
    inside = np.abs(first.fit_wave - second.wave) < 0.3 * first.fwhm
    if inside.any():
        assert np.min(first.fit_neighbours[inside]) < 0.99  # the neighbour is there ...
        assert np.min(first.fit_flux[inside]) > np.min(first.fit_neighbours[inside])  # ... not in the line alone


def test_report_pages(result: ew.EWResult, tmp_path: Path) -> None:
    path = write_ew_report(tmp_path / "report.pdf", result, "synthetic", ())
    n_panels = len(result.run.lines) + result.run.missing.size
    expected = 1 + -(-n_panels // PANELS_PER_PAGE) + 2  # overview + line pages + RV + R
    pages = len(re.findall(rb"/Type /Page[^s]", path.read_bytes()))
    assert pages == expected


def test_command_pdf_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_spectrum("star.fits", make_spectrum(seed=4))
    np.savetxt("lines.txt", np.column_stack([CENTRES, CODES]), fmt=("%.3f", "%.2f"))
    execute(resolve("ew"), ["star.fits", "lines.txt", "--no-mask"])
    assert Path("star_ew.pdf").stat().st_size > 10_000
    execute(resolve("ew"), ["star.fits", "lines.txt", "--no-mask", "--pdf", "other.pdf", "-o", "t2.txt"])
    assert Path("other.pdf").exists()
    Path("other.pdf").unlink()
    execute(resolve("ew"), ["star.fits", "lines.txt", "--no-mask", "--no-pdf", "-o", "t3.txt", "--pdf", "x.pdf"])
    assert not Path("x.pdf").exists()


# ----------------------------------------------------------------------------- species styles
def test_split_code_and_families() -> None:
    from bobspec.ew_report import FAMILIES, family_of, split_code

    assert split_code(26.01) == (26, 1) and split_code(20.0) == (20, 0) and split_code(106.0) == (106, 0)
    assert FAMILIES[family_of(11)][0].startswith("odd-Z")
    assert FAMILIES[family_of(12)][0].startswith("\u03b1")
    assert FAMILIES[family_of(26)][0].startswith("iron peak")
    assert FAMILIES[family_of(63)][0].startswith("neutron capture")
    assert family_of(106) == len(FAMILIES)  # molecules


def test_styles_colour_per_element_and_marker_per_ion() -> None:
    from bobspec.ew_report import species_styles

    style = species_styles([26.0, 26.01, 22.0, 22.01, 20.0])
    assert style.colour[26.0] == style.colour[26.01] != style.colour[22.0]
    assert style.marker[26.0] == style.marker[22.0] == "o" and style.marker[26.01] == "^"
    labels = [h.get_label() for h in style.handles]
    assert labels == ["Ca", "Ti", "Fe", "I", "II"]


def test_styles_families_beyond_twenty_elements() -> None:
    from bobspec.ew_report import MAX_ELEMENT_COLOURS, species_styles

    codes = [float(z) for z in range(3, 4 + MAX_ELEMENT_COLOURS + 5)] + [106.0]
    style = species_styles(codes)
    labels = [h.get_label() for h in style.handles]
    assert any(label.startswith("iron peak") for label in labels) and "molecules" in labels
    assert len(labels) <= 8  # families + molecules + one ionisation stage: compact
    assert style.colour[26.0] == style.colour[28.0]  # same family, same colour
    assert style.marker[106.0] == "*"


def test_legend_layout_helpers() -> None:
    from matplotlib.lines import Line2D

    from bobspec.ew_report import legend_columns, row_major

    assert legend_columns(3) == 4 and legend_columns(22) == 8 and legend_columns(15) == 5
    handles = [Line2D([], [], label=str(i)) for i in range(6)]
    ordered = [h.get_label() for h in row_major(handles, 3)]
    assert ordered == ["0", "3", "1", "4", "2", "5"]  # column-major fill that reads row by row
