from pathlib import Path

import numpy as np
import pytest

from bobspec.cli import dispatch, main
from bobspec.io import Spec, read_spectrum, write_spectrum
from bobspec.registry import COMMANDS, UsageError, all_names, execute, resolve


def test_unmatched_quote_does_not_crash(capsys) -> None:
    assert dispatch('plot "abc') is True
    assert "Syntax error" in capsys.readouterr().out


def test_exit_words() -> None:
    assert dispatch("exit") is False
    assert dispatch("  ") is True


def test_unknown_command_suggests(capsys) -> None:
    dispatch("rotat")
    assert "rotate" in capsys.readouterr().out


def test_usage_error_is_reported_not_raised(capsys) -> None:
    dispatch("cut only_one_arg")
    assert "required" in capsys.readouterr().out


def test_help_does_not_exit(capsys) -> None:
    dispatch("rv -h")
    assert "--nsim" in capsys.readouterr().out


def test_negative_values_are_not_options() -> None:
    ns = COMMANDS["coords"].parser.parse_args(["12:30:00", "-12:18:00"])
    assert ns.dec == "-12:18:00"
    ns = COMMANDS["rv"].parser.parse_args(["a", "b", "c", "--vmin", "-300"])
    assert ns.vmin == -300.0


def test_type_validation() -> None:
    with pytest.raises(UsageError):
        COMMANDS["rotate"].parser.parse_args(["s", "1.5", "10", "o"])


def test_no_separate_list_commands() -> None:
    assert resolve("rv_list") is None and resolve("key_list") is None
    assert "rv" in all_names() and not any(name.endswith("_list") for name in all_names())


def test_batch_not_available_for_interactive_commands(tmp_path: Path) -> None:
    from bobspec.registry import run_batch

    with pytest.raises(UsageError, match="not available"):
        run_batch(resolve("fitline"), [f"@{tmp_path / 'x.txt'}"])


def test_batch_missing_list_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        execute(resolve("coords"), [f"@{tmp_path / 'missing.txt'}"])
    with pytest.raises(UsageError, match="right after"):
        execute(resolve("coords"), ["@"])


def test_help_mentions_batch_mode(capsys) -> None:
    dispatch("help rv")
    assert "rv @FILE" in capsys.readouterr().out


def test_coords_list_variable_columns_and_bad_line(tmp_path: Path) -> None:
    lst = tmp_path / "c.txt"
    lst.write_text("# ra dec\n187.5 -12.3\n12:30:00 -12:18:00\nnot valid\n\n10 +5  # comment\n")
    rows = execute(resolve("coords"), [f"@{lst}"])
    assert [r["line"] for r in rows] == [2, 3, 6]
    table = (tmp_path / "c_coords.txt").read_text().splitlines()
    assert table[0].split()[:3] == ["#", "line", "ra_in"] and len(table) == 4


def test_one_shot_cli_exit_codes(tmp_path: Path, capsys) -> None:
    assert main(["coords", "187.5", "-12.3"]) == 0
    assert main(["coords", "187.5"]) == 2
    assert main(["totxt", str(tmp_path / "missing.fits"), "x.txt"]) == 1
    assert main(["nonsense"]) == 2
    assert main(["help", "rv"]) == 0


@pytest.fixture
def spectrum_file(tmp_path: Path, line_model) -> Path:
    wave = np.arange(5010.0, 5290.0, 0.03)
    rng = np.random.default_rng(4)
    flux = 800.0 * (1 + 1e-3 * (wave - 5150)) * line_model(wave, 120.0) * (1 + rng.normal(0, 0.01, wave.size))
    return write_spectrum(tmp_path / "obs.fits", Spec(wave, flux))


def test_rv_command_end_to_end(tmp_path: Path, spectrum_file: Path, template) -> None:
    tpl = write_spectrum(tmp_path / "tpl.txt", template)
    out = tmp_path / "rest.fits"
    row = execute(resolve("rv"), [str(spectrum_file), str(tpl), str(out), "--nsim", "10", "--seed", "1", "--auto"])
    assert row["rv"] == pytest.approx(120.0, abs=0.1)
    assert row["n_mc"] == 10 and row["rv_err_mc"] > 0
    assert (tmp_path / "rest_ccf.png").exists() and (tmp_path / "rest_mc.png").exists()
    assert any("rest frame" in str(h) for h in read_spectrum(out).header["HISTORY"])


def test_rv_list_with_optional_columns(tmp_path: Path, spectrum_file: Path, template) -> None:
    tpl = write_spectrum(tmp_path / "tpl.fits", template)
    lst = tmp_path / "rv.txt"
    lst.write_text(
        f"{spectrum_file} {tpl} {tmp_path / 'a.fits'} --nsim 0\n"
        f"{spectrum_file} {tpl} {tmp_path / 'b.fits'} --nsim 0 --vmin -300 --vmax 300 --wmin 5050 --wmax 5250\n"
        f"missing.fits {tpl} {tmp_path / 'c.fits'}\n"
    )
    rows = execute(resolve("rv"), [f"@{lst}", "--auto", "--summary", str(tmp_path / "res.txt")])
    assert len(rows) == 2
    assert all(r["rv"] == pytest.approx(120.0, abs=0.1) for r in rows)
    assert (tmp_path / "res.txt").read_text().count("\n") == 3


def test_common_options_apply_to_every_line_and_lines_override(tmp_path: Path, spectrum_file: Path, template) -> None:
    tpl = write_spectrum(tmp_path / "tpl.fits", template)
    lst = tmp_path / "rv.txt"
    lst.write_text(
        f"{spectrum_file} {tpl} {tmp_path / 'a.fits'}\n{spectrum_file} {tpl} {tmp_path / 'b.fits'} --nsim 5\n"
    )
    rows = execute(resolve("rv"), [f"@{lst}", "--auto", "--nsim", "0", "--seed", "3"])
    assert "n_mc" not in rows[0] and rows[1]["n_mc"] == 5


def test_invalid_common_options_abort_before_running(tmp_path: Path, spectrum_file: Path, template) -> None:
    tpl = write_spectrum(tmp_path / "tpl.fits", template)
    lst = tmp_path / "rv.txt"
    lst.write_text(f"{spectrum_file} {tpl} {tmp_path / 'a.fits'}\n")
    with pytest.raises(UsageError, match="after @FILE"):
        execute(resolve("rv"), [f"@{lst}", "--nonsense"])
    assert not (tmp_path / "a.fits").exists()


def test_ipoly_auto_writes_outputs(tmp_path: Path, spectrum_file: Path) -> None:
    out = tmp_path / "norm.fits"
    execute(resolve("ipoly"), [str(spectrum_file), "3", "1.5", "3", str(out), "--auto"])
    norm = read_spectrum(out)
    assert np.nanpercentile(norm.flux, 90) == pytest.approx(1.0, abs=0.02)
    assert (tmp_path / "norm.snr").exists() and (tmp_path / "norm_continuum.pdf").exists()


def test_convolve_synthe_format(tmp_path: Path) -> None:
    wave = np.arange(5000.0, 5010.0, 0.002)
    line = 1.0 - 0.5 * np.exp(-0.5 * ((wave - 5005) / 0.01) ** 2)
    synthe = tmp_path / "syn.txt"
    np.savetxt(synthe, np.column_stack([wave, line, line * 1e6, np.full(wave.size, 1e6)]))
    execute(resolve("convolve"), [str(synthe), "20000", str(tmp_path / "out.txt")])
    data = np.loadtxt(tmp_path / "out.txt")
    assert data.shape[1] == 4
    assert data[:, 1].min() > 0.5  # broadened line is shallower


def test_add_noise_command(tmp_path: Path, spectrum_file: Path) -> None:
    outdir = tmp_path / "noisy"
    execute(
        resolve("add_noise"),
        [str(spectrum_file), "50", "3", "star", "--step", "0.1", "--outdir", str(outdir), "--seed", "1"],
    )
    files = sorted(outdir.glob("star_50_*.fits"))
    assert len(files) == 3
    assert read_spectrum(files[0]).wave[1] - read_spectrum(files[0]).wave[0] == pytest.approx(0.1)


def test_batch_separator_after_every_line(tmp_path: Path, capsys) -> None:
    """A line of dashes follows every entry of a batch list, also the ones that fail."""
    from bobspec.registry import BATCH_SEPARATOR, run_batch

    lst = tmp_path / "coords.txt"
    lst.write_text("10.5 -20.25\nnot-a-number 5\n180 45\n")
    run_batch(resolve("coords"), [f"@{lst}"])
    out = capsys.readouterr().out
    assert BATCH_SEPARATOR == "-" * 70
    assert out.count(BATCH_SEPARATOR) == 3  # two successes and one failure
