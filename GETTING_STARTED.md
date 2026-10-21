# Getting started with bob

This guide explains, step by step, how to install the new `bob` and use it.
You do not need to know anything about Python packaging.

## 1. What is inside the folder

Only three files are meant for you to read:

| File | What it is |
|---|---|
| `GETTING_STARTED.md` | this guide |
| `README.md` | short overview and the table of syntax changes from the old `bob.py` |
| `docs/commands.md` | the full description of every command (the same text `help <command>` prints) |

Everything else is used by Python, by GitHub or by developers, and you can ignore it
(`LICENSE`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CITATION.cff` and `paper/` are for
the public repository and the JOSS paper):

| Item | What it does |
|---|---|
| `src/bobspec/` | the code: one file per topic (`rv.py` radial velocities, `continuum.py` normalisation, `combine.py` co-addition, ...) |
| `tests/` | about 180 automatic checks that verify the code gives correct results |
| `pyproject.toml` | the "recipe" that tells `pip` how to install bob and which packages it needs |
| `.github/` | makes GitHub run the tests automatically (only matters once the code is on GitHub) |
| `docs/generate_commands.py` | regenerates `docs/commands.md` after a code change |

## 2. Installation (once)

bob needs Python 3.11 or newer (`python3 --version`). It can be installed into
the Python you already use; a dedicated environment (`python3 -m venv`, conda)
is optional.

### 2.1 Put the code in a permanent place

From GitHub (makes updating easy):

```bash
mkdir -p ~/softw && cd ~/softw
git clone https://github.com/DeimerAlvarez/bobspec.git     # creates ~/softw/bobspec
```

or from a zip file:

```bash
mkdir -p ~/softw && cd ~/softw
unzip ~/Downloads/bobspec.zip                                # creates ~/softw/bobspec
```

Do not delete or move this folder later: the installation points to it (see 2.5).

### 2.2 Install bob

```bash
cd ~/softw/bobspec
python3 -m pip install -e .
```

`pip` reads `pyproject.toml`, installs numpy, scipy, astropy and matplotlib if
they are missing, and creates the command `bob`. Writing `python3 -m pip`
(instead of `pip`) guarantees that bob goes into the Python you just checked.

* If pip warns that the script `bob` was installed in a folder *not on PATH*,
  add that folder to your PATH in `~/.bashrc` (`export PATH="<folder>:$PATH"`)
  or `~/.tcshrc` (`setenv PATH <folder>:$PATH`) and open a new terminal.
* If pip refuses with *externally-managed-environment*, `python3` is the
  operating system's own Python: use
  `python3 -m pip install --user --break-system-packages -e .`, which installs
  only inside your home folder.

### 2.3 Check

```bash
bob --version          # -> bobspec 0.3.0
```

Optionally run the automatic checks (about 30 s):

```bash
python3 -m pip install -e ".[test]"
python3 -m pytest
```

### 2.4 Changing the code

Thanks to the `-e` ("editable") of step 2.2, Python reads the files in
`~/softw/bobspec/src/bobspec` every time `bob` starts. After you edit and save
a `.py` file, the change is active the next time you launch `bob`: if a bob
session is already open, type `exit` and start `bob` again (no new terminal
and no reinstallation needed). Run `python3 -m pip install -e .` again only if
you change `pyproject.toml` (version, dependencies, command name).

### 2.5 Where bob is installed

* **With `git clone` or a zip + `pip install -e .`** (the way above) the code
  stays in the folder you cloned or unzipped; Python only stores a small
  pointer to it. That is why the folder must not be moved or deleted.
* **With `python3 -m pip install git+https://github.com/DeimerAlvarez/bobspec.git`**
  pip downloads the code into a temporary folder, copies it into your Python's
  `site-packages` folder and deletes the temporary copy. Nothing is left in
  the folder you were in when you typed the command. This is the simplest way
  for people who only *use* bob; edits are not possible there.

`python3 -m pip show bobspec` prints the installed version and its *Location*.

### 2.6 Uninstalling

```bash
python3 -m pip uninstall bobspec
```

This removes the `bob` command and the package; numpy, scipy, astropy and
matplotlib stay, because other programs may use them. The folder
`~/softw/bobspec` stays too: delete it by hand if you want to remove everything.

The old program was called `bob.py`; the new one is called `bob`, so both can
coexist. To remove the old one from your PATH, go to the old folder and run
`make uninstall`, or delete the line that adds it to `PATH` in your shell file.

### 2.7 Updating to a new version

If you installed with `git clone`:

```bash
cd ~/softw/bobspec && git pull
```

If you installed from a zip, replace the content of `~/softw/bobspec` with the
new version (same path). In both cases run `python3 -m pip install -e .` there
if `pyproject.toml` changed.

## 3. Two ways to use bob

**Interactive shell**, as before:

```text
$ bob
bob> help                 # list of all commands
bob> help rv              # full help of one command
bob> ls                   # also: cd, pwd
bob> exit
```

TAB completes command and file names, and the arrow keys recall previous
commands (also from previous sessions).

**One command from the normal terminal**, useful in shell scripts:

```bash
bob cut star.fits 6500 6600 star_cut.fits
```

## 4. How to read a command

Every command has **required arguments**, given in a fixed order, and
**options**, which start with `--`, can be given in any order after them, and
have default values. Example from `help rv`:

```text
usage: rv [-h] [--snr SNR] [--nsim NSIM] [--vmin VMIN] [--vmax VMAX] ...
          spectrum template output
```

```text
bob> rv star.fits syn_m54.txt star_rest.fits                          # all defaults
bob> rv star.fits syn_m54.txt star_rest.fits --snr 40 --vmin -200 --vmax 400
```

Rules worth remembering:

* negative numbers need nothing special: `--vmin -300`, `coords 12:30:00 -12:18:00`;
* use quotes for wildcards and for names with spaces:
  `combine out.fits "M54_h15_*_1500138_SH.fits"`,
  `header file.fits "ESO TEL GEOLON"`;
* a mistake in the arguments prints the correct usage and nothing is done.

## 5. A complete workflow (GIRAFFE spectra of M54)

### Step 1: combine the exposures of each star

One star:

```text
bob> combine m54_hr15_1500138.fits "M54_h15_*_1500138_SH.fits"
```

All stars of `lista.id` at once (this replaces your loop in `combine.py`):

```text
bob> combine "m54_hr15_{id}.fits" "M54_h15_*_{id}_SH.fits" --ids lista.id
```

`{id}` is replaced by each line of `lista.id`. For every star bob prints the
scale factor, noise and number of rejected pixels of each exposure, and the
SNR before and after. It writes `m54_hr15_<id>.fits`,
`m54_hr15_<id>_cosmics.txt` (the rejected pixels) and a summary table
`lista_combine.txt`. Useful options: `--weighted` (exposures of different
length), `--barycentric` (exposures from different nights), `--save-error`,
`--reference median` (exactly the rejection of the original script).

If you start from reduced ESO archive products (FITS tables), first list their
columns and extract the ones you need (index from 0, or column name):

```text
bob> read_eso_fits ADP.2017-01-01T00:00:00.000.fits
bob> read_eso_fits ADP.2017-01-01T00:00:00.000.fits 0 1 star.fits
bob> read_eso_fits ADP.2017-01-01T00:00:00.000.fits WAVE ERR star_err.txt
```

The wavelength is converted to Angstrom if it is in nm (`--keep-units` to
avoid it). FITS outputs keep the ESO header, so `helio_eso` works on them.
All bob commands can also read ESO tables directly (they use the `WAVE` and
`FLUX` columns); `read_eso_fits` is for choosing other columns or converting.

### Step 2: barycentric correction

```text
bob> helio_eso m54_hr15_1500138.fits
```

For many files, first create a list from the normal terminal, then run the
batch version:

```bash
ls m54_hr15_*[0-9].fits > combined.txt
bob helio_eso @combined.txt             # -> combined_helio_eso.txt
```

For combined spectra bob uses the flux-weighted mid-exposure time of the
inputs (keyword `MJD-MID`), not the start of the first exposure.

### Step 3: normalisation

```text
bob> ipoly m54_hr15_1500138.fits 10 1.8 2.3 m54_hr15_1500138_norm.fits
```

The window opens; keys: `j`/`k` degree, `l`/`o` sigma_low, `h`/`y` sigma_up,
right click masks a region, `u` undoes it, **ENTER saves**. Closing the window
without ENTER cancels (nothing is written). A `.snr` file is also written.
`ispline` works the same way; `manual_spline` and `smooth` are the other methods.

### Step 4: radial velocity

```text
bob> rv m54_hr15_1500138_norm.fits syn_m54.txt m54_hr15_1500138_rest.fits --seed 1
```

Look at the plot, close it, then answer `y` (accept), `n` (change velocity
and wavelength ranges) or `q` (quit without saving). bob prints the RV with
both errors and saves the rest-frame spectrum plus `_ccf.png` and `_mc.png`.

The Monte Carlo uses the SNR measured on the spectrum (DER_SNR) unless you
give `--snr`. It includes photon noise only, so it is a lower limit: on real
spectra the Tonry & Davis error, which also reflects the template mismatch,
is usually the more realistic one.

The RV is **topocentric**. The final value is
`RV_bary = RV + v_corr` (+ `RV·v_corr/c`, a few m/s), with `v_corr` from step 2.

### Step 5: line measurements

```text
bob> fitline m54_hr15_1500138_rest.fits 7698.96 7664.90 --resol 20000
bob> fitline m54_hr15_1500138_rest.fits --lines mylines.txt --resol 20000
```

For each line: two right clicks select the window (repeat to refine), ENTER
goes to the next line, closing the window stops. Results go to
`m54_hr15_1500138_rest_lines.txt`.

### Step 6: continuum windows and line lists for abundances

`cows` lets you mark the continuum windows by clicking:

```text
bob> cows star.fits am154t4248g077k195_cl_ap04_5550_5850_R20000.syn 5580 5850 --resol 20000 -o cont_5580.txt
```

The spectrum is shifted (`--rv 134.3`, or `--rv auto` to estimate it), cut to
5580-5850 A and divided by a running median. Answer `y` if the red continuum
looks right, or `k` and a new odd kernel. Then the region is shown in 15 A
windows (`--step`) with the synthetic in blue: left click marks a point, right
click removes the last one, close the window to go on. The points of each window
are sorted and saved in pairs in `cont_5580.txt`.

`sline` lets you choose the lines to measure, window by window:

```text
bob> sline star_norm.fits am154t4248g077k195_cl_ap04_5550_5850_R20000 5580 0.96 20.00 -o ca_lines.txt
```

Arguments: normalised spectrum, root name of the `.syn`/`.lin` files, first
wavelength, RI limit, species code shown in red. Optional:
`--step 8 --snr 0 --wsh 0 --ysta 0.5 --yend 1.2`. The synthetic must already be
convolved to the resolution of your observations; it is never convolved again.
Its resolution, read from the `# RESOL` line of the `.syn` header (or given with
`--res`), is used only to flag blended lines.
In the terminal: `y` next window, `b` previous, `w 3` writes line 3,
`up step=10 ysta=0.3` changes settings, `e` exits.

## 6. Batch lists

As in IRAF, every command marked `[@]` in `help` accepts a list file written
with `@` as first argument: `rv @rv.txt`. The list file contains, on each line,
the arguments of one call exactly as you would type them after the command
name. Lines may differ in their options, and `#` starts a comment. Options
written after `@file` apply to every line (a line can override them), and
`--summary FILE` chooses the name of the results table.

`rv.txt`:

```text
# spectrum                      template      output                          options
m54_hr15_1500138_norm.fits      syn_m54.txt   m54_hr15_1500138_rest.fits      --nsim 500
m54_hr15_1500200_norm.fits      syn_m54.txt   m54_hr15_1500200_rest.fits      --snr 25 --vmin -100 --vmax 400
```

```text
bob> rv @rv.txt --auto             # --auto: no windows and no questions, for every line
```

A line that fails is reported and skipped; the others continue. Numerical
results are collected in the text file `rv_rv.txt` (choose another name with
`--summary`): one line per spectrum, aligned columns, column names in the
first line after `#`. It opens in any text editor or spreadsheet, and in Python
with `astropy.table.Table.read("rv_rv.txt", format="ascii.commented_header")`.

## 7. Files written by each command

| Command | Writes |
|---|---|
| `combine` | `<out>`, `<out>_cosmics.txt`, optionally `<out>_error.fits` |
| `rv` | `<out>` (rest frame), `<out>_ccf.png`, `<out>_mc.png` |
| `ipoly`, `ispline`, `smooth`, `manual_spline` | `<out>`, `<out>.snr`; with `--auto` also `<out>_continuum.pdf` |
| `fitline` | `<spectrum>_lines.txt` |
| `cut`, `rotate` | `<out>` (and they show the result over the original; `--auto` to skip) |
| `add_noise` | `noisy_spec/<root>_<snr>_<i>.fits` |
| any `<command> @list` | `<list name>_<command>.txt` with the results |

Every FITS output keeps the header of the input and gets `HISTORY` lines
describing what was done, so you can always reconstruct its processing.

## 8. Plots

`plot` opens a window with the spectra you give, then asks in the terminal for
more: type a file name (or several, or a quoted wildcard) and press ENTER to add
it to the same window; type `e` to finish. `plot a.fits b.fits --no-add` only
shows the given spectra.

The font sizes of all bob figures are set at the top of
`src/bobspec/plotting.py`:

```python
TICK_LABEL_SIZE = 12  # numbers along the axes
AXIS_LABEL_SIZE = 16  # axis names, e.g. "Wavelength", "Flux", "Radial velocity"
TITLE_SIZE = 15  # text above each panel
LEGEND_SIZE = 12  # legend entries
```

Change the numbers, save the file and restart `bob` (see 2.4).

## 9. Settings

```text
bob> tellurics                 # show the excluded telluric regions
bob> tellurics none            # disable the mask
bob> tellurics mybands.txt     # two columns: start end [A]
bob> tellurics default         # O2 B and A bands
```

## 10. When something goes wrong

| Symptom | Solution |
|---|---|
| `bob: command not found` | the folder where pip put `bob` is not on your PATH (see 2.2), or bob is installed in another Python |
| `No module named bobspec` | bob is installed in another Python: reinstall with `python3 -m pip install -e .` (2.2) |
| plot windows do not appear | try `export MPLBACKEND=TkAgg` (or `MacOSX` on a Mac); on remote machines use `ssh -X` |
| a change to the code has no effect | restart `bob` (2.4) |
| an error message you do not understand | rerun with `bob --debug` and open an issue on GitHub with the full output |

An error never closes the interactive shell, and a cancelled or failed
command never writes a partial output file.
