"""bobspec: interactive and batch tools for 1-D stellar spectra.

The numerical core (I/O, resampling, broadening, continuum fitting, radial
velocities, SNR, barycentric corrections) is made of pure functions that work
on :class:`bobspec.io.Spec` objects and never touch matplotlib. The
interactive layer (:mod:`bobspec.interactive`) and the command line
(:mod:`bobspec.cli`) are thin wrappers around it.
"""

from bobspec.io import Spec, SpectrumReadError, read_spectrum, write_spectrum

__all__ = ["Spec", "SpectrumReadError", "read_spectrum", "write_spectrum"]
__version__ = "0.3.0"
