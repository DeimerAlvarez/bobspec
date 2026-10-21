# Changelog

## 0.3.0 (unreleased)

First public release, rewriting the original `bob.py`/`utilities_bob.py` scripts as an
installable, tested Python package. Numerical results differ from the scripts where
bugs were fixed; see "Changed" and "Fixed".

### Added

* `cows`: selection of continuum windows by clicking on the normalised spectrum
  (from the author's `cows.py`), with all parameters on the command line.
* `sline`: line-by-line selection from a SYNTHE line list (from the author's
  `sline.py`), with optional `--step`, `--snr`, `--wsh`, `--ysta`, `--yend`.
  The synthetic is used as already convolved (no second convolution); its
  resolution, read from the `# RESOL` header line or `--res`, only sets the
  blend criterion.
* `combine`: co-addition of exposures with cosmic-ray rejection, wildcards and
  `{id}` templates for many stars, optional barycentric alignment and weighting;
  writes `MJD-MID`, used by `helio_eso` for combined spectra.
  `--reference median` reproduces the original `combine.py` exactly.
* `read_eso_fits`: lists the columns of FITS binary tables (ESO Phase 3) and
  extracts any two of them, by index or name, to ASCII or FITS.
* All readers accept ESO Phase-3 tables, FITS images with linear, logarithmic
  or IRAF log10 axes, and any `CUNIT1`/`TUNIT` length unit.
* IRAF-style batch mode `<command> @list` for every non-interactive command (replacing the
  separate `_list` commands); each line takes the same arguments as the command,
  options after `@list` apply to every line, failures are skipped, results go to a plain-text table (`<list>_<command>.txt`).
  `combine` also accepts `@list` as the list of its input exposures.
* One-shot use from the system shell (`bob <command> ...`), `--debug`, `tellurics`,
  `cd`, typo suggestions, `--seed` for reproducible random numbers.
* Output FITS files keep the input header and record each step in `HISTORY`.
* 176 automatic tests and continuous integration on Linux and macOS.

### Changed

* `plot` accepts several files, wildcards or `@list`, then still asks for more
  spectra to add to the same window (as before; `--no-add` to skip).
* `cut` and `rotate` show the result over the original spectrum (`--auto` to skip).
* Font sizes of all figures are defined once at the top of `plotting.py`;
  axis labels are larger.

* Command syntax: optional parameters are now `--options`
  (see the table in `README.md`).
* `rv`: the Monte Carlo SNR defaults to the DER_SNR of the spectrum; lags converted with the exact inverse of the Doppler shift
  (removes a bias of -v^2/2c, 0.27 km/s at 400 km/s); the template extends beyond
  the observed window, telluric masks apply only to the observed spectrum, no
  apodisation; FWHM and Tonry & Davis r now refer to the observed spectrum, with r
  measured from the CCF peak height over a fixed +-1000 km/s lag range (independent of
  the search window).
* `helio_eso`/`helio_manual`: astropy instead of PyAstronomy, evaluated at
  mid-exposure; the value is a *barycentric* correction.
* `convolve`, `rotate`: kernels applied on a ln(lambda) grid, exact R and vsini
  at every wavelength.
* `add_noise`: flux-conserving rebinning; noise relative to the continuum level.
* `.snr` files: DER_SNR (Stoehr et al. 2008) on line-free chunks.
* `ipoly`/`ispline`: clipping iterated to convergence; closing the window cancels.
* `manual_spline`: honours the spline degree `k`.

### Fixed

* `fitline` crashed with NumPy >= 2.4 (`np.trapz` removed).
* FITS templates could not be read by `rv`.
* `coords` silently misread `+12.3` and exponent notation.
* An unmatched quote closed the interactive session.
* Batch RV lists failed when lines had different numbers of columns.
* Stale `rv_plot.jpg` files were renamed to the wrong star.
* Matplotlib shortcuts (`k`, `l`, `o`, `h`) interfered with the editors.

### Removed

* `residual.dat`, `fitline_plot<wave>.jpg`, the HJD printout of `helio_manual`,
  and the `Makefile` installer.
