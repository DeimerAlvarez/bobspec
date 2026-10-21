---
title: 'bobspec: interactive and batch tools for one-dimensional stellar spectra'
tags:
  - Python
  - astronomy
  - stellar spectroscopy
  - radial velocities
  - continuum normalisation
authors:
  - name: Deimer Antonio Alvarez Garay
    orcid: 0000-0002-9156-0729
    affiliation: 1
affiliations:
  - name: INAF - Osservatorio Astrofisico di Arcetri, Largo E. Fermi 5, 50125 Firenze, Italy
    index: 1
date: 16 October 2026
bibliography: paper.bib
---

<!--
DRAFT. Items marked [AUTHOR: ...] need your input; claims about your own use,
publications and verification must be written by you. Remove this comment
before submission. Target length: 750-1750 words (JOSS).
-->

# Summary

Before the chemical composition of a star can be measured, its spectrum has to
be prepared: the exposures of the same star are combined, the spectrum is
divided by its continuum, the Doppler shift caused by the motion of the star and
of the observer is measured and removed, and the absorption lines are measured.
Synthetic spectra used for comparison undergo the complementary operations:
they are degraded to the instrumental resolution, broadened by rotation and
perturbed with noise to test the analysis. `bobspec` collects these steps in a
single Python package, usable from an interactive shell (the `bob` command),
from the system command line, or in batch mode on long lists of stars. Every
step writes its results to standard FITS or text files and records what was done
in the file header, so that the provenance of each spectrum can be traced.

# Statement of need

Spectroscopic surveys of star clusters routinely deliver hundreds to thousands
of one-dimensional spectra per programme, for example from multi-object
spectrographs such as FLAMES/GIRAFFE [@pasquini2002]. The preparatory steps
listed above are conceptually simple but repeated for every star, and small
choices in their implementation (how the continuum is clipped, how a
cross-correlation lag is converted into a velocity, which time is used for the
barycentric correction) propagate into abundances and kinematics. For many
years these steps were performed with IRAF [@tody1986] tasks such as
`continuum`, `fxcor` and `scombine`; IRAF is no longer developed by its original
maintainers and is increasingly difficult to install on modern systems, and
research groups have replaced it with collections of personal scripts that are
rarely tested or shared.

`bobspec` targets astronomers who analyse one-dimensional stellar spectra,
observed or synthetic, and need a lightweight, scriptable and tested replacement
for this workflow. It combines interactive tools, needed for the judgement-based
steps (placing a continuum, checking a cross-correlation peak), with batch
execution of the same commands for large samples, and it depends only on
NumPy [@harris2020], SciPy [@virtanen2020], Astropy [@astropy2022] and
Matplotlib [@hunter2007].

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
procedures. `bobspec` is complementary: it focuses on the end-to-end preparation
of spectra, adds interactive editors and batch processing, and covers steps not
found together elsewhere, such as the co-addition of exposures with cosmic-ray
rejection, the extraction of columns from ESO Phase-3 data products, and
interactive tools to choose continuum windows and spectral lines for abundance
analyses by comparison with SYNTHE synthetic spectra.

# Software design

The package separates a numerical core from the user interface. The core
modules (input/output, resampling and broadening, continuum fitting,
co-addition, cross-correlation, signal-to-noise and barycentric corrections) are
pure functions acting on an immutable spectrum object; they never plot or
prompt, which makes them directly testable and reusable from other Python code.
Interactive editors and figures are confined to two modules. Each user command
is declared once, with typed arguments, through a small registry; from this
single declaration `bobspec` derives the help text, argument validation, the
interactive shell, one-shot execution from the system shell and an IRAF-style
batch mode (`command @list`) in which each line of the list file is parsed
exactly like a command line. Failures
in batch mode are reported and skipped, and numerical results are collected in
plain-text tables.

Several numerical choices aim at correctness over a wide range of use cases.
Radial velocities are measured by cross-correlation following
@tonry1979, on a logarithmic wavelength grid, converting lags with the exact
inverse of the Doppler shift applied to the spectra, so that measurement and
correction are consistent; the template is extended beyond the observed window
so that no feature enters or leaves the correlation as the lag changes.
Uncertainties are given both by the @tonry1979 estimator and by a seeded Monte
Carlo simulation. Instrumental and rotational broadening [@gray2005] are applied
in velocity space, so that the resolving power and the projected rotational
velocity are exact at every wavelength. Signal-to-noise ratios use the DER_SNR
estimator [@stoehr2008]. Barycentric corrections are evaluated at mid-exposure,
and combined spectra store the flux-weighted mid-exposure time of their inputs.
In the co-addition of exposures, each spectrum is compared with the median of
the *other* spectra, avoiding the underestimate of the noise, and the resulting
over-rejection of positive noise fluctuations, that occurs when a spectrum is
compared with a median that includes itself.

The test suite (176 tests, run by continuous integration on Linux and macOS)
verifies the numerical results against synthetic spectra with known properties,
for instance the recovery of injected velocities to better than 0.01 km/s, the
conservation of equivalent widths under broadening, and the agreement between
Monte Carlo and empirical velocity scatter.

# Research impact statement

[AUTHOR: list the publications, theses or ongoing projects in which bob/bobspec
has been used (e.g. the analysis of FLAMES/GIRAFFE spectra of globular-cluster
stars), and any external users or groups. Evidence of realised impact is
required; near-term significance can be supported by the public test suite,
the documentation and the batch tools for large samples.]

# AI usage disclosure

Generative AI was used in the preparation of this version of the software.
Claude Opus 5.5 (Anthropic) was used in October 2026 to review the original
scripts for bugs, to refactor them into the present package, to implement the
`combine` and `read_eso_fits` commands from the author's specifications and
prototype code, to write the automatic tests and the documentation, and to
produce a first draft of this paper.

[AUTHOR: describe, truthfully and specifically, how you reviewed, modified and
validated the AI-generated code, tests, documentation and text (for example:
line-by-line review of the numerical modules, comparison with your previous
results on real spectra, checks of the test cases), and which architectural and
scientific decisions were made by you. JOSS requires this statement to be
accurate and complete.]

# Acknowledgements

[AUTHOR: funding, colleagues, data sources.]

# References
