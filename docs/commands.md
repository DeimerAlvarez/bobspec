# Command reference

Generated from the argument parsers (`python docs/generate_commands.py`); inside `bob` the same text is shown by `help <command>`.

Every command marked *batch* also runs as `<command> @FILE [options] [--summary TABLE]` (IRAF style): each line of FILE holds the arguments of one call, exactly as typed on the command line; options after @FILE apply to every line.


## `add_noise` (batch)

```text
usage: add_noise [-h] [--step STEP] [--format {fits,ascii}] [--noise {constant,photon}]
                 [--outdir OUTDIR] [--seed SEED]
                 spectrum snr n rootname

Create N noisy copies of a spectrum: <outdir>/<rootname>_<snr>_<i>.

With --step the spectrum is first rebinned (flux-conserving) to that pixel
size. Apply the instrumental resolution first (convolve).

positional arguments:
  spectrum              noiseless spectrum
  snr
  n                     number of realisations
  rootname

options:
  -h, --help            show this help message and exit
  --step STEP           output pixel size in A (default: native grid)
  --format {fits,ascii}
  --noise {constant,photon}
                        constant sigma, or photon-like sigma ~ sqrt(flux)
  --outdir OUTDIR
  --seed SEED

batch mode (IRAF style):
  add_noise @FILE [options] [--summary TABLE]
  Runs 'add_noise' once for every non-empty line of FILE. Each line holds the
  arguments exactly as they would be typed after 'add_noise' (quotes and '#'
  comments allowed). Options written after @FILE apply to every line, and a
  line can override them. A failing line is reported and skipped; numerical
  results are collected in the text table TABLE (default: <FILE name>_add_noise.txt).
```


## `combine` (batch)

```text
usage: combine [-h] [--method {mean,median,sum}] [--sigma SIGMA] [--no-reject] [--window WINDOW]
               [--grow GROW] [--no-rescale] [--weighted] [--tolerance TOLERANCE]
               [--reference {others,median}] [--barycentric] [--ids IDS] [--save-error]
               output inputs [inputs ...]

Combine several exposures of the same star, rejecting cosmic rays.

Examples:
  combine star.fits exp1.fits exp2.fits exp3.fits
  combine star.fits "M54_h15_*_star_SH.fits" --weighted
  combine star.fits @exposures.txt
  combine "m54_hr15_{id}.fits" "M54_h15_*_{id}_SH.fits" --ids lista.id

Writes the combined spectrum (header of the first input plus NCOMBINE,
MJD-MID, EXPTOT and HISTORY) and <output>_cosmics.txt with the rejected pixels.
Inputs from different nights should be combined with --barycentric.
Note: "combine @stars.txt" (an @ list as FIRST argument) is batch mode:
one combination per line of stars.txt.

positional arguments:
  output                combined spectrum; may contain {id} (see --ids)
  inputs                spectra to combine: names, quoted wildcards or @list; may contain {id}

options:
  -h, --help            show this help message and exit
  --method {mean,median,sum}
                        default: mean
  --sigma SIGMA         cosmic-ray threshold in local sigmas (default 5)
  --no-reject           no cosmic-ray rejection (allows 2 spectra)
  --window WINDOW       odd window of the local noise [pix] (default 51)
  --grow GROW           pixels masked around each cosmic (default 1)
  --no-rescale          do not rescale the inputs to a common flux level
  --weighted            inverse-variance weighted mean (different exposure times)
  --tolerance TOLERANCE
                        grid difference [pix] accepted without interpolation (default 0.01)
  --reference {others,median}
                        rejection reference: median of the OTHER spectra (default) or of all
                        spectra (original code)
  --barycentric         shift each input to the barycentric frame first (ESO headers)
  --ids IDS             file with one star id per line: repeat for each id, replacing {id}
  --save-error          also write <output>_error with the empirical error

batch mode (IRAF style):
  combine @FILE [options] [--summary TABLE]
  Runs 'combine' once for every non-empty line of FILE. Each line holds the
  arguments exactly as they would be typed after 'combine' (quotes and '#'
  comments allowed). Options written after @FILE apply to every line, and a
  line can override them. A failing line is reported and skipped; numerical
  results are collected in the text table TABLE (default: <FILE name>_combine.txt).
```


## `convolve` (batch)

```text
usage: convolve [-h] [--format {auto,synthe,spectrum}] input resolution output

Degrade a spectrum to resolving power R with a constant-R Gaussian kernel.

SYNTHE files: line and continuum fluxes are both convolved and the output
has 4 columns: wave, normalised flux, line flux, continuum flux.

positional arguments:
  input
  resolution
  output

options:
  -h, --help            show this help message and exit
  --format {auto,synthe,spectrum}
                        synthe: ASCII with wave, -, line flux, continuum (cols 1,3,4); auto:
                        synthe if >= 4 columns

batch mode (IRAF style):
  convolve @FILE [options] [--summary TABLE]
  Runs 'convolve' once for every non-empty line of FILE. Each line holds the
  arguments exactly as they would be typed after 'convolve' (quotes and '#'
  comments allowed). Options written after @FILE apply to every line, and a
  line can override them. A failing line is reported and skipped; numerical
  results are collected in the text table TABLE (default: <FILE name>_convolve.txt).
```


## `coords` (batch)

```text
usage: coords [-h] ra dec

Convert RA/Dec between decimal degrees and sexagesimal.

positional arguments:
  ra          deg or hh:mm:ss
  dec         deg or dd:mm:ss

options:
  -h, --help  show this help message and exit

batch mode (IRAF style):
  coords @FILE [options] [--summary TABLE]
  Runs 'coords' once for every non-empty line of FILE. Each line holds the
  arguments exactly as they would be typed after 'coords' (quotes and '#'
  comments allowed). Options written after @FILE apply to every line, and a
  line can override them. A failing line is reported and skipped; numerical
  results are collected in the text table TABLE (default: <FILE name>_coords.txt).
```


## `cows`

```text
usage: cows [-h] [-o OUTPUT] [--resol RESOL] [--rv RV] [--kernel KERNEL] [--step STEP] [--append]
            spectrum synthetic l1 l2

Select continuum windows by clicking on the normalised spectrum.

1. The observed spectrum is shifted to rest frame (--rv) and cut to [l1, l2].
2. A running-median continuum is shown: answer y to accept it, or k to try
   another kernel (odd number of pixels).
3. The region is shown in windows of --step A with the synthetic in blue:
   left click marks a point, right click removes the last one; close the
   window to go on. The points are sorted and written in pairs
   (start  end of a continuum window) to the output file.
4. After each window: 'end the code? (y/n)'.

positional arguments:
  spectrum              observed spectrum (FITS or ASCII)
  synthetic             synthetic spectrum (2-column ASCII or FITS; 4-column SYNTHE file with
                        --resol)
  l1                    start of the region [A]
  l2                    end of the region [A]

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        continuum windows file (default wind_continuum.txt)
  --resol RESOL         degrade the SYNTHE synthetic to this resolving power
  --rv RV               RV of the observed spectrum [km/s], or 'auto' to estimate it (default 0)
  --kernel KERNEL       initial running-median kernel [pix] (default 1001)
  --step STEP           width of each selection window [A] (default 15)
  --append              add to an existing output file instead of overwriting it
```


## `cut` (batch)

```text
usage: cut [-h] [--auto] spectrum w1 w2 output

Cut a spectrum to the wavelength range [w1, w2].

The cut spectrum is saved, then shown over the original one (close the
window to continue); --auto skips the plot.

positional arguments:
  spectrum
  w1
  w2
  output

options:
  -h, --help  show this help message and exit
  --auto      do not show the plot (useful with @lists)

batch mode (IRAF style):
  cut @FILE [options] [--summary TABLE]
  Runs 'cut' once for every non-empty line of FILE. Each line holds the
  arguments exactly as they would be typed after 'cut' (quotes and '#'
  comments allowed). Options written after @FILE apply to every line, and a
  line can override them. A failing line is reported and skipped; numerical
  results are collected in the text table TABLE (default: <FILE name>_cut.txt).
```


## `fitline`

```text
usage: fitline [-h] [--lines LINES] [--resol RESOL] [--halfwidth HALFWIDTH] [--table TABLE]
               spectrum [waves ...]

Measure centre, FWHM and EW of one or more lines interactively.

For each line select the integration window with two right clicks (repeat
to refine), ENTER moves to the next line, closing the window stops.

positional arguments:
  spectrum
  waves                 central wavelength(s) of the line(s)

options:
  -h, --help            show this help message and exit
  --lines LINES         file with one central wavelength per line
  --resol RESOL         resolving power (initial FWHM guess)
  --halfwidth HALFWIDTH
                        half width of the view in A (default 2)
  --table TABLE         results table (default <spectrum>_lines.txt)
```


## `header` (batch)

```text
usage: header [-h] file [what]

Print a FITS header summary, the whole header, or one keyword.

Keywords with spaces (HIERARCH) must be quoted: header f.fits "ESO TEL GEOLON".
In batch mode (header @list, lines 'file KEYWORD') the values are collected in a table.

positional arguments:
  file        FITS file
  what        'minimum' (default), 'all', or a keyword

options:
  -h, --help  show this help message and exit

batch mode (IRAF style):
  header @FILE [options] [--summary TABLE]
  Runs 'header' once for every non-empty line of FILE. Each line holds the
  arguments exactly as they would be typed after 'header' (quotes and '#'
  comments allowed). Options written after @FILE apply to every line, and a
  line can override them. A failing line is reported and skipped; numerical
  results are collected in the text table TABLE (default: <FILE name>_header.txt).
```


## `helio_eso` (batch)

```text
usage: helio_eso [-h] file

Barycentric velocity correction from an ESO header (at mid-exposure).

v_bary = v_measured + correction (+ v*correction/c).

positional arguments:
  file        ESO FITS file

options:
  -h, --help  show this help message and exit

batch mode (IRAF style):
  helio_eso @FILE [options] [--summary TABLE]
  Runs 'helio_eso' once for every non-empty line of FILE. Each line holds the
  arguments exactly as they would be typed after 'helio_eso' (quotes and '#'
  comments allowed). Options written after @FILE apply to every line, and a
  line can override them. A failing line is reported and skipped; numerical
  results are collected in the text table TABLE (default: <FILE name>_helio_eso.txt).
```


## `helio_manual` (batch)

```text
usage: helio_manual [-h] [--exptime EXPTIME] ra dec mjd lon lat alt

Barycentric velocity correction from explicit parameters.

positional arguments:
  ra                 deg or hh:mm:ss
  dec                deg or dd:mm:ss
  mjd                MJD at exposure start
  lon                site longitude, deg, east positive
  lat                site latitude, deg, north positive
  alt                site altitude, m

options:
  -h, --help         show this help message and exit
  --exptime EXPTIME  exposure time in s (default 0)

batch mode (IRAF style):
  helio_manual @FILE [options] [--summary TABLE]
  Runs 'helio_manual' once for every non-empty line of FILE. Each line holds the
  arguments exactly as they would be typed after 'helio_manual' (quotes and '#'
  comments allowed). Options written after @FILE apply to every line, and a
  line can override them. A failing line is reported and skipped; numerical
  results are collected in the text table TABLE (default: <FILE name>_helio_manual.txt).
```


## `ipoly` (batch)

```text
usage: ipoly [-h] [--auto] spectrum deg sigma_low sigma_up output

Normalise with a sigma-clipped Chebyshev polynomial (interactive).

Keys: j/k degree, l/o sigma_low, h/y sigma_up, right click masks, u undo,
ENTER accepts, closing the window cancels. Also writes <output>.snr.

positional arguments:
  spectrum
  deg         Chebyshev degree
  sigma_low   rejection threshold below the continuum
  sigma_up    rejection threshold above the continuum
  output

options:
  -h, --help  show this help message and exit
  --auto      no interaction; save <output>_continuum.pdf

batch mode (IRAF style):
  ipoly @FILE [options] [--summary TABLE]
  Runs 'ipoly' once for every non-empty line of FILE. Each line holds the
  arguments exactly as they would be typed after 'ipoly' (quotes and '#'
  comments allowed). Options written after @FILE apply to every line, and a
  line can override them. A failing line is reported and skipped; numerical
  results are collected in the text table TABLE (default: <FILE name>_ipoly.txt).
```


## `ispline` (batch)

```text
usage: ispline [-h] [--auto] [--knots KNOTS] spectrum k sigma_low sigma_up output

Normalise with a sigma-clipped least-squares spline (interactive).

Keys as in ipoly (j/k change the spline degree). Also writes <output>.snr.

positional arguments:
  spectrum
  k              spline degree 1..5
  sigma_low      rejection threshold below the continuum
  sigma_up       rejection threshold above the continuum
  output

options:
  -h, --help     show this help message and exit
  --auto         no interaction; save <output>_continuum.pdf
  --knots KNOTS  number of interior knots (default 20)

batch mode (IRAF style):
  ispline @FILE [options] [--summary TABLE]
  Runs 'ispline' once for every non-empty line of FILE. Each line holds the
  arguments exactly as they would be typed after 'ispline' (quotes and '#'
  comments allowed). Options written after @FILE apply to every line, and a
  line can override them. A failing line is reported and skipped; numerical
  results are collected in the text table TABLE (default: <FILE name>_ispline.txt).
```


## `manual_spline` (batch)

```text
usage: manual_spline [-h] spectrum k output

Normalise with a spline through anchor points placed by hand.

Right click adds, left click removes the last anchor, ENTER fits, ENTER
again saves; closing the window cancels.

positional arguments:
  spectrum
  k           spline degree 1..5
  output

options:
  -h, --help  show this help message and exit

batch mode (IRAF style):
  manual_spline @FILE [options] [--summary TABLE]
  Runs 'manual_spline' once for every non-empty line of FILE. Each line holds the
  arguments exactly as they would be typed after 'manual_spline' (quotes and '#'
  comments allowed). Options written after @FILE apply to every line, and a
  line can override them. A failing line is reported and skipped; numerical
  results are collected in the text table TABLE (default: <FILE name>_manual_spline.txt).
```


## `plot`

```text
usage: plot [-h] [--no-add] spectra [spectra ...]

Overplot spectra, then add more to the same window from the terminal.

After the first plot, type the name of another spectrum (or several, or a
quoted wildcard) and press ENTER to add it to the window; type 'e' (or
close the window) to finish.

positional arguments:
  spectra     one or more FITS/ASCII files (quoted wildcards and @list allowed)

options:
  -h, --help  show this help message and exit
  --no-add    only show the given spectra, without asking for more
```


## `read_eso_fits` (batch)

```text
usage: read_eso_fits [-h] [--ext EXT] [--row ROW] [--keep-units] file [xcol] [ycol] [output]

Extract two columns of a FITS table spectrum (e.g. ESO Phase 3) to ASCII or FITS.

With only the file name, the columns are listed with their index, name,
unit and length. Columns can be given by 0-based index (as data[0][i] in
Python) or by name:
  read_eso_fits ADP.2017-01-01.fits
  read_eso_fits ADP.2017-01-01.fits 0 1 star.txt
  read_eso_fits ADP.2017-01-01.fits WAVE FLUX_REDUCED star.fits
A wavelength column in nm or um is converted to Angstrom (bob's unit)
unless --keep-units is given. FITS outputs keep the primary header, so
helio_eso works on them; non-uniform grids are resampled for FITS output.

positional arguments:
  file          FITS file with a binary table (e.g. a reduced ESO spectrum)
  xcol          x (wavelength) column: 0-based index or name
  ycol          y column (flux, error, ...): 0-based index or name
  output        output: .txt or .dat (ASCII) or .fits

options:
  -h, --help    show this help message and exit
  --ext EXT     table extension (default: first binary table, 1 for ESO)
  --row ROW     row to read when a row holds a whole spectrum (default 0)
  --keep-units  do not convert the wavelength column to Angstrom

batch mode (IRAF style):
  read_eso_fits @FILE [options] [--summary TABLE]
  Runs 'read_eso_fits' once for every non-empty line of FILE. Each line holds the
  arguments exactly as they would be typed after 'read_eso_fits' (quotes and '#'
  comments allowed). Options written after @FILE apply to every line, and a
  line can override them. A failing line is reported and skipped; numerical
  results are collected in the text table TABLE (default: <FILE name>_read_eso_fits.txt).
```


## `rotate` (batch)

```text
usage: rotate [-h] [--auto] spectrum epsilon vsini output

Apply rotational broadening (Gray's kernel, linear limb darkening).

The broadened spectrum is saved, then shown over the original one (close
the window to continue); --auto skips the plot.

positional arguments:
  spectrum
  epsilon     linear limb darkening, 0..1
  vsini       km/s
  output

options:
  -h, --help  show this help message and exit
  --auto      do not show the plot (useful with @lists)

batch mode (IRAF style):
  rotate @FILE [options] [--summary TABLE]
  Runs 'rotate' once for every non-empty line of FILE. Each line holds the
  arguments exactly as they would be typed after 'rotate' (quotes and '#'
  comments allowed). Options written after @FILE apply to every line, and a
  line can override them. A failing line is reported and skipped; numerical
  results are collected in the text table TABLE (default: <FILE name>_rotate.txt).
```


## `rv` (batch)

```text
usage: rv [-h] [--snr SNR] [--nsim NSIM] [--vmin VMIN] [--vmax VMAX] [--wmin WMIN] [--wmax WMAX]
          [--dv DV] [--seed SEED] [--auto]
          spectrum template output

Measure the radial velocity by cross-correlation (Tonry & Davis 1979).

Reports the Tonry & Davis error and, if --nsim > 0, the Monte Carlo
(photon-noise) error. Saves the rest-frame spectrum, <output>_ccf.png and
<output>_mc.png. The velocity is topocentric: add the barycentric
correction (helio_eso / helio_manual).

positional arguments:
  spectrum     observed spectrum
  template     rest-frame template
  output       RV-corrected (rest-frame) spectrum

options:
  -h, --help   show this help message and exit
  --snr SNR    SNR per pixel for the Monte Carlo (default: DER_SNR measured on the spectrum)
  --nsim NSIM  Monte Carlo realisations, 0 = skip (default 200)
  --vmin VMIN  km/s (default -600)
  --vmax VMAX  km/s (default 600)
  --wmin WMIN  first wavelength used in the CCF
  --wmax WMAX  last wavelength used in the CCF
  --dv DV      CCF velocity step (default: half the finest pixel)
  --seed SEED  random seed for reproducible MC errors
  --auto       no plots or questions

batch mode (IRAF style):
  rv @FILE [options] [--summary TABLE]
  Runs 'rv' once for every non-empty line of FILE. Each line holds the
  arguments exactly as they would be typed after 'rv' (quotes and '#'
  comments allowed). Options written after @FILE apply to every line, and a
  line can override them. A failing line is reported and skipped; numerical
  results are collected in the text table TABLE (default: <FILE name>_rv.txt).
```


## `shift_rv` (batch)

```text
usage: shift_rv [-h] spectrum rv output

Shift a spectrum to the rest frame: lambda / (1 + rv/c).

positional arguments:
  spectrum
  rv          km/s
  output

options:
  -h, --help  show this help message and exit

batch mode (IRAF style):
  shift_rv @FILE [options] [--summary TABLE]
  Runs 'shift_rv' once for every non-empty line of FILE. Each line holds the
  arguments exactly as they would be typed after 'shift_rv' (quotes and '#'
  comments allowed). Options written after @FILE apply to every line, and a
  line can override them. A failing line is reported and skipped; numerical
  results are collected in the text table TABLE (default: <FILE name>_shift_rv.txt).
```


## `sline`

```text
usage: sline [-h] [--step STEP] [--snr SNR] [--wsh WSH] [--res RES] [--ysta YSTA] [--yend YEND]
             [-o OUTPUT] [--append] [--seed SEED]
             spectrum synthetic wave1 min sel

Select lines from a SYNTHE line list, window by window.

The synthetic <root>.syn must already be convolved to the resolution of
the observations: its normalised flux is used as it is. That resolution,
read from the '# RESOL' header line (or given with --res), only sets the
blend criterion.
The plot shows the observed spectrum (black), the synthetic (red) and the
lines with RI < MIN, numbered: red = species SEL, green = merged split
components, blue = other lines. The terminal lists them with a blend flag
('clean' if no other line with RI < MIN is within 1.5 sigma). Commands:
  y  next window          b  previous window
  w  write a line: its number is asked (or type 'w 3')
  up change settings, e.g. 'up step=10 ysta=0.3 min=0.9'
  e  exit

positional arguments:
  spectrum              observed spectrum, normalised (FITS or ASCII)
  synthetic             root name of the SYNTHE files, without .syn and .lin
  wave1                 starting wavelength [A]
  min                   show the lines with residual intensity RI < MIN
  sel                   species code drawn in red, e.g. 20.00

options:
  -h, --help            show this help message and exit
  --step STEP           width of the window [A] (default 8)
  --snr SNR             noise added to the synthetic, 0 = none (default 0)
  --wsh WSH             velocity shift [km/s] removed from the observed spectrum (default 0)
  --res RES             resolving power of the synthetic, only for the blend flags (default: '#
                        RESOL' line of the .syn)
  --ysta YSTA           lower limit of the y axis (default 0.5)
  --yend YEND           upper limit of the y axis (default 1.2)
  -o OUTPUT, --output OUTPUT
                        selected lines file (default linelist.txt)
  --append              add to an existing output file instead of overwriting it
  --seed SEED           random seed for the synthetic noise (--snr)
```


## `smooth` (batch)

```text
usage: smooth [-h] [--percentile PERCENTILE] [--auto] spectrum window output

Normalise with a running upper-envelope (percentile + Gaussian) continuum.

The envelope follows the upper part of the noise: with the default 90th
percentile the continuum sits about 1.3 noise sigma too high (normalised
flux ~1% low at SNR 150). ipoly/ispline are preferable for precise work.

positional arguments:
  spectrum
  window                running window in A
  output

options:
  -h, --help            show this help message and exit
  --percentile PERCENTILE
                        upper-envelope percentile (default 90)
  --auto                no plot window; save <output>_continuum.pdf

batch mode (IRAF style):
  smooth @FILE [options] [--summary TABLE]
  Runs 'smooth' once for every non-empty line of FILE. Each line holds the
  arguments exactly as they would be typed after 'smooth' (quotes and '#'
  comments allowed). Options written after @FILE apply to every line, and a
  line can override them. A failing line is reported and skipped; numerical
  results are collected in the text table TABLE (default: <FILE name>_smooth.txt).
```


## `snr` (batch)

```text
usage: snr [-h] [--range W1 W2] spectrum

Signal-to-noise in selected regions: DER_SNR and mean/std.

Interactive by default (two right clicks per region); with --range it
runs without plots, so it can be batched (snr @list).

positional arguments:
  spectrum

options:
  -h, --help     show this help message and exit
  --range W1 W2  non-interactive: measure in [W1, W2]

batch mode (IRAF style):
  snr @FILE [options] [--summary TABLE]
  Runs 'snr' once for every non-empty line of FILE. Each line holds the
  arguments exactly as they would be typed after 'snr' (quotes and '#'
  comments allowed). Options written after @FILE apply to every line, and a
  line can override them. A failing line is reported and skipped; numerical
  results are collected in the text table TABLE (default: <FILE name>_snr.txt).
```


## `tofits` (batch)

```text
usage: tofits [-h] input output

Convert a spectrum to FITS (resampled to a constant step if needed).

positional arguments:
  input       spectrum (FITS or ASCII)
  output      FITS file

options:
  -h, --help  show this help message and exit

batch mode (IRAF style):
  tofits @FILE [options] [--summary TABLE]
  Runs 'tofits' once for every non-empty line of FILE. Each line holds the
  arguments exactly as they would be typed after 'tofits' (quotes and '#'
  comments allowed). Options written after @FILE apply to every line, and a
  line can override them. A failing line is reported and skipped; numerical
  results are collected in the text table TABLE (default: <FILE name>_tofits.txt).
```


## `totxt` (batch)

```text
usage: totxt [-h] input output

Convert a spectrum to a two-column ASCII file.

positional arguments:
  input       spectrum (FITS or ASCII)
  output      ASCII file

options:
  -h, --help  show this help message and exit

batch mode (IRAF style):
  totxt @FILE [options] [--summary TABLE]
  Runs 'totxt' once for every non-empty line of FILE. Each line holds the
  arguments exactly as they would be typed after 'totxt' (quotes and '#'
  comments allowed). Options written after @FILE apply to every line, and a
  line can override them. A failing line is reported and skipped; numerical
  results are collected in the text table TABLE (default: <FILE name>_totxt.txt).
```
