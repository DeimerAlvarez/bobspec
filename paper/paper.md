---
title: 'bobspec: interactive and batch tools for one-dimensional stellar spectra'
tags:
  - Python
  - astronomy
  - stellar spectroscopy
  - radial velocities
  - continuum normalisation
  - chemical abundances
authors:
  - name: Deimer Antonio Alvarez Garay
    orcid: 0000-0002-9156-0729
    affiliation: 1
    corresponding: true
affiliations:
  - name: INAF -- Osservatorio Astrofisico di Arcetri, Largo E. Fermi 5, 50125 Firenze, Italy
    index: 1
    # ror: [AUTHOR: ROR ID of the top-level organisation (INAF), from https://ror.org]
date: 24 October 2026
bibliography: paper.bib
---

<!--
DRAFT. Items marked [AUTHOR: ...] need your input; claims about your own use,
publications and verification must be written by you. Remove this comment
before submission. Target length: 750-1750 words (JOSS).
-->

# Summary

The spectrum of a star records both its chemical composition and its motion
along the line of sight. Before either can be measured, the observed spectrum
has to be prepared: separate exposures of the same star are combined, the flux
is divided by the continuum (the smooth emission on which the absorption lines
are imprinted), and the Doppler shift due to the motion of the star and of the
observer is measured and removed. Chemical abundances obtained by spectral
synthesis also require choosing the wavelength intervals that define the
continuum and the absorption lines free from blends, by comparing the observed
spectrum with a synthetic one; synthetic spectra, in turn, are degraded to the
instrumental resolution, broadened by stellar rotation and perturbed with noise
to test the analysis. `bobspec` is a Python package that collects these steps.
It can be used from an interactive shell (the `bob` command), with single
commands from the system shell, or in batch mode on long lists of stars, and it
writes standard FITS or text files that record each processing step, so that
the history of every spectrum can be traced.

# Statement of need

Spectroscopic surveys of star clusters routinely deliver hundreds to thousands
of one-dimensional spectra per programme, for example from multi-object
spectrographs such as FLAMES/GIRAFFE [@pasquini2002]. The preparatory steps
listed above are conceptually simple but repeated for every star, and small
choices in their implementation (how the continuum is clipped, how a
cross-correlation lag is converted into a velocity, which time is used for the
barycentric correction) propagate into abundances and kinematics. The choice
of continuum windows and of unblended lines for each spectral setup is equally
repetitive and is usually done by visual inspection supported by ad hoc
scripts, so that it is seldom documented or reproducible. For many years these
steps were performed with IRAF [@tody1986] tasks such as `continuum`, `fxcor`
and `scombine`; since IRAF is no longer developed by its original maintainers,
these workflows have largely been replaced by collections of personal scripts
that are rarely tested, documented or shared.

`bobspec` is intended for researchers and students who analyse one-dimensional
stellar spectra, observed or synthetic, and in particular for those who prepare
large samples for chemical-abundance or kinematic studies with spectral-synthesis
codes such as SYNTHE [@kurucz2005]. It provides a lightweight, scriptable and
tested replacement for this workflow: interactive tools for the steps that
require judgement (placing a continuum, checking a cross-correlation peak,
selecting continuum windows and lines), and batch execution of the same
commands, with the same arguments, for large samples. Its command syntax,
including IRAF-style `@list` files, is meant to be familiar to former IRAF
users. [AUTHOR: if you can quantify it, add one sentence on the practical gain
over your previous workflow, e.g. the time needed to prepare a typical
FLAMES/GIRAFFE sample, or the number of manual steps removed. Do not add a
number that you have not measured.]

# State of the field

Several open-source packages address parts of this problem. iSpec
[@blancocuaresma2014] is a comprehensive framework that includes continuum
normalisation and radial velocities together with spectral synthesis and the
determination of atmospheric parameters; its scope and installation, which
bundles radiative-transfer codes, are much larger than needed for data
preparation. PyAstronomy [@czesla2019] provides a broad collection of routines
(e.g. cross-correlation, rotational broadening, barycentric corrections) but no
workflow or file-level tools. The Astropy-affiliated package specutils
[@specutils] offers data containers and analysis building blocks, including
template cross-correlation, on which users still have to build their own
procedures.

`bobspec` was built as a separate package, rather than as a contribution to
these projects, because what is missing is mainly a workflow layer
(interactive editors, batch execution of every command, provenance recorded in
the output files) and not a new algorithm: such a layer lies outside the scope
of a library of building blocks such as specutils, and would add little to a
full analysis framework such as iSpec. Where community libraries cover a task,
`bobspec` uses them, e.g. Astropy for FITS input/output, units, coordinates and
barycentric corrections. It covers steps not found together elsewhere, such as
the co-addition of exposures with cosmic-ray rejection, the extraction of
columns from ESO Phase-3 data products, and interactive tools to choose
continuum windows and spectral lines by comparison with synthetic spectra and
line lists computed with SYNTHE.

# Software design

The package separates a numerical core from the user interface. The core
modules (input/output, resampling and broadening, continuum fitting,
co-addition, cross-correlation, signal-to-noise and barycentric corrections) are
pure functions acting on an immutable spectrum object; they never plot or
prompt, which makes them directly testable and reusable from other Python code.
Interactive editors and figures are confined to two modules. The interface is a
terminal shell with Matplotlib [@hunter2007] windows rather than a graphical
application or a notebook: it works on remote machines over SSH and can be
driven from shell scripts, and the normalisation and measurement commands also
run without interaction in batch mode. The cost is a less discoverable
interface, mitigated by built-in help and suggestions for mistyped commands.

Each user command is declared once, with typed arguments, through a small
registry; from this single declaration `bobspec` derives the help text,
argument validation, the interactive shell, one-shot execution from the system
shell and the batch mode (`command @list`), in which each line of the list file
is parsed exactly like a command line. Failures in batch mode are reported and
skipped, and numerical results are collected in plain-text tables. All readers
accept ASCII tables, FITS images with linear, logarithmic or IRAF log10
wavelength axes, and ESO Phase-3 binary tables, converting any wavelength unit
to Ångström on input; FITS outputs keep the input header and record each
processing step in `HISTORY` cards.

The dependencies are limited to NumPy [@harris2020], SciPy [@virtanen2020],
Astropy [@astropy2022] and Matplotlib. Earlier, unpackaged versions of the code
also relied on specutils and PyAstronomy; implementing cross-correlation and
broadening directly made installation simpler and gave full control over the
conventions shared by measurement and correction, at the cost of maintaining
and testing these routines within `bobspec`. Radial velocities are measured by
cross-correlation following @tonry1979, on a logarithmic wavelength grid,
converting lags with the exact inverse of the Doppler shift applied to the
spectra, so that measurement and correction are consistent; the template is
extended beyond the observed window so that no feature enters or leaves the
correlation as the lag changes. Uncertainties are given both by the
@tonry1979 estimator and by a seeded Monte Carlo simulation. Instrumental and
rotational broadening [@gray2005] are applied in velocity space, so that the
resolving power and the projected rotational velocity are exact at every
wavelength. Signal-to-noise ratios use the DER_SNR estimator [@stoehr2008].
Barycentric corrections are evaluated at mid-exposure, and combined spectra
store the flux-weighted mid-exposure time of their inputs. In the co-addition
of exposures, each spectrum is compared with the median of the *other*
spectra, which avoids underestimating the noise, and hence over-rejecting
positive fluctuations, as happens when the median includes the spectrum itself.

Two interactive commands prepare abundance analyses with SYNTHE products.
`cows` shows the observed spectrum, shifted to the rest frame and divided by a
running median whose kernel can be tuned, over the synthetic spectrum, and
writes the continuum windows marked by the user. `sline` steps through the
spectrum together with the lines of the SYNTHE line list deeper than a chosen
threshold, merges split components of the same transition, flags lines blended
within the instrumental profile, and writes the selected lines with their
atomic data.

The test suite (214 tests, run by continuous integration on Linux and macOS
with Python 3.11 to 3.13, together with linting and static type checking)
verifies the numerical results against synthetic spectra with known properties,
for instance the recovery of injected velocities to better than 0.01 km/s, the
conservation of equivalent widths under broadening, and the agreement between
Monte Carlo and empirical velocity scatter. The interactive sessions of `cows`
and `sline` are tested with scripted user input on files in the SYNTHE format.

# Research impact statement

[AUTHOR: list the publications, theses or ongoing projects in which bob/bobspec
has been used (e.g. the analysis of FLAMES/GIRAFFE spectra of globular-cluster
stars), with references added to paper.bib, and any external users or groups.
State explicitly whether related publications exist (published, in review or
in preparation), as required by JOSS.]

Beyond its use by the author, `bobspec` is prepared for use by others: it is
installable with `pip`, documented by a getting-started guide and a reference
of all commands [AUTHOR: add the user manual (bob_manual.tex/PDF) to the
repository and mention it here], and released under the MIT licence with
citation metadata (`CITATION.cff`) and contribution guidelines.

# AI usage disclosure

Generative AI was used in the preparation of this version of the software and
of this paper. Claude Opus 5.5 (Anthropic) was used in October 2026 to review
the original scripts for bugs, to refactor them into the present package, to
implement the `combine` and `read_eso_fits` commands from the author's
specifications and prototype code, to integrate the author's `cows` and `sline`
scripts into the package, to write the automatic tests and the documentation,
and to draft and revise the text of this paper.

[AUTHOR: describe, truthfully and specifically, how you reviewed, modified and
validated the AI-generated code, tests, documentation and text (for example:
line-by-line review of the numerical modules, comparison with your previous
results on real spectra, checks of the test cases), and which architectural and
scientific decisions were made by you. JOSS requires an explicit confirmation
that the author reviewed, edited and validated all AI-assisted outputs and made
the core design decisions; this statement must be accurate and complete.]

# Acknowledgements

[AUTHOR: funding (grant numbers, and whether the funder had any role in the
work), colleagues, data sources. JOSS also asks for a statement on conflicts of
interest, e.g. "The author declares no conflicts of interest."]

# References
