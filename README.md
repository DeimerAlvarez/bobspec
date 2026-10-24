# bobspec

[![tests](https://github.com/DeimerAlvarez/bobspec/actions/workflows/tests.yml/badge.svg)](https://github.com/DeimerAlvarez/bobspec/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Interactive and batch tools for 1-D stellar spectra: co-addition of exposures
with cosmic-ray rejection, extraction of ESO Phase-3 table spectra, continuum
normalisation, radial velocities by cross-correlation, barycentric corrections,
instrumental and rotational broadening, synthetic noise, line measurements,
selection of continuum windows and spectral lines by comparison with SYNTHE
synthetic spectra, and format conversion.

> New users: start from [GETTING_STARTED.md](GETTING_STARTED.md).
>
> **Status:** version 0.3.0, first public release. Released under the [MIT license](LICENSE).

## Who is it for

bobspec is meant for astronomers and students who prepare one-dimensional
stellar spectra, observed or synthetic, for chemical-abundance or kinematic
analyses, from a few stars to the hundreds or thousands observed with
multi-object spectrographs. The steps that need judgement (placing a continuum,
checking a cross-correlation peak, choosing continuum windows and lines) are
interactive; every non-interactive command can also run in batch mode on a list
of files, with the syntax familiar to IRAF users (`command @list`), and every
output records how it was obtained.

## Installation

Python ≥ 3.11 is required. The dependencies (NumPy, SciPy, Astropy and
Matplotlib) are installed automatically by `pip`. Installing in a virtual
environment (`python3 -m venv`) is recommended.

```bash
pip install git+https://github.com/DeimerAlvarez/bobspec.git    # users

git clone https://github.com/DeimerAlvarez/bobspec.git           # developers
cd bobspec && pip install -e ".[dev]"
```

This installs the `bob` command; no shell configuration files are edited.
`python -m bobspec` works as well.

## Quick start

```text
$ bob                                     # interactive shell (TAB completion, history)
bob> help                                 # list of commands
bob> help rv                              # full help of one command
bob> combine star.fits "star_exp*.fits"    # co-add exposures, reject cosmic rays
bob> read_eso_fits ADP.fits                # list the columns of an ESO table spectrum
bob> read_eso_fits ADP.fits 0 1 star.txt   # extract wavelength (in A) and flux
bob> rv star.fits template.txt star_rest.fits --snr 50 --seed 1
bob> ipoly star_rest.fits 10 1.8 2.3 star_norm.fits
bob> cows star.fits syn.syn 5580 5850 --resol 20000      # click continuum windows
bob> sline star_norm.fits syn 5580 0.96 20.00             # choose lines from syn.lin
bob> rv @my_rv_list.txt --auto            # batch (IRAF style): one set of arguments per line

$ bob helio_eso star.fits                 # one-shot use from the system shell
```

A list file for `rv @list` might look like this (optional arguments may differ
from line to line; options written after `@list` apply to every line, and a line
can override them; failing lines are reported and skipped):

```text
# spectrum      template        output           options
s1.fits         syn_m54.txt     s1_rest.fits     --nsim 500
s2.fits         syn_m54.txt     s2_rest.fits     --snr 25 --vmin -300 --vmax 300 --wmin 7470 --wmax 7880
```

Results of batch runs are collected in a plain-text table
(`<list name>_<command>.txt`, or `--summary FILE`). The complete command reference
is in [docs/commands.md](docs/commands.md).

## Conventions

* Wavelengths are in Ångström; FITS `CUNIT1` and table units are converted on input.
* Doppler shifts use `λ_obs = λ_rest (1 + v/c)` everywhere; the CCF converts
  lags with the exact inverse, so measured RVs carry no `v²/2c` bias.
* `rv` returns *topocentric* velocities. `helio_eso`/`helio_manual` give the
  barycentric correction (astropy/ERFA) at **mid-exposure**:
  `v_bary = v + v_corr + v·v_corr/c`.
* Telluric regions (default: O₂ B and A bands) are excluded from continuum fits
  and from the observed spectrum in the CCF; change them with
  `tellurics none | default | <file>`.
* Instrumental (`convolve`) and rotational (`rotate`) kernels are applied on a
  ln λ grid, so R and v sin i are exact at every wavelength.
* Random numbers (`add_noise`, Monte Carlo errors) accept `--seed` for
  reproducible results.
* Output FITS files keep the input header and gain `HISTORY` cards describing
  each processing step.

## Changes with respect to the previous `bob.py`

| Previous usage | Now |
|---|---|
| `add_noise spec pix snr n root F` | `add_noise spec snr n root --step pix --format fits [--noise photon] [--seed S]` |
| `rv spec tpl out SNR NSIM vmin vmax w1 w2` | `rv spec tpl out --snr SNR --nsim NSIM --vmin .. --vmax .. --wmin .. --wmax ..` |
| `fitline spec resol wave` / `fitline_list spec resol file` | `fitline spec wave [wave ...] [--lines file] --resol R` |
| `<command>_list file` (e.g. `rv_list`, `key_list`, `coords_list`) | `<command> @file [options] [--summary out]` (e.g. `rv @file`, `header @file`, `coords @file`) |
| `plot spec` + prompt loop | unchanged; several files or `@list` also accepted at start |
| `combine.py` script with a loop over `lista.id` | `combine "m54_hr15_{id}.fits" "M54_h15_*_{id}_SH.fits" --ids lista.id` (`--reference median` reproduces the script exactly) |
| `ipoly`/`ispline`/`smooth` lists asked "interactive?" | add `--auto` to the list command or to single lines |
| `residual.dat`, `rv_plot.jpg` in the working directory | `<output>_continuum.pdf`, `<output>_ccf.png`, `<output>_mc.png` next to the output |

## Testing

```bash
pip install -e ".[dev]"
pytest                # headless (matplotlib Agg backend)
ruff check src tests && ruff format --check src tests
mypy src --ignore-missing-imports
```

The same checks run by continuous integration on Linux and macOS with
Python 3.11-3.13 for every push and pull request.

## Getting help and contributing

Questions, bug reports and feature requests go to the
[issue tracker](https://github.com/DeimerAlvarez/bobspec/issues); run the failing
command with `bob --debug` and include the full output. Contributions are welcome:
see [CONTRIBUTING.md](CONTRIBUTING.md), which also describes how the project is
maintained and supported. Changes between versions are listed in
[CHANGELOG.md](CHANGELOG.md).

## Citation

If you use bobspec, please cite it using the information in [CITATION.cff](CITATION.cff)
(GitHub shows it under "Cite this repository"). Please also cite the methods
you use:

* Tonry, J., & Davis, M. 1979, AJ, 84, 1511, doi:10.1086/112569 (cross-correlation method);
* Stoehr, F. et al. 2008, ASP Conf. Ser. 394, 505 (DER_SNR);
* Gray, D. F. 2005, *The Observation and Analysis of Stellar Photospheres*, 3rd ed.,
  Cambridge University Press (rotation kernel).
