"""Original combine.py by the author (functions only), kept verbatim as a regression reference."""

# ruff: noqa
import numpy as np
from astropy.io import fits
from pathlib import Path


# ============================================================
# AUXILIARY FUNCTIONS
# ============================================================


def read_spectrum(filename):
    """Read a 1D spectrum from the PRIMARY HDU."""

    with fits.open(filename) as hdul:
        flux = np.asarray(hdul[0].data, dtype=float).squeeze()
        header = hdul[0].header.copy()

    if flux.ndim != 1:
        raise ValueError(f"{filename}: expected 1D spectrum, found shape {flux.shape}")

    return flux, header


def wavelength_from_header(header, npix):
    """
    Construct wavelength array from a linear FITS WCS.
    """

    crval = header["CRVAL1"]
    crpix = header.get("CRPIX1", 1.0)

    if "CDELT1" in header:
        cdelt = header["CDELT1"]
    elif "CD1_1" in header:
        cdelt = header["CD1_1"]
    else:
        raise KeyError("Neither CDELT1 nor CD1_1 found in header.")

    pixel = np.arange(npix, dtype=float) + 1.0

    wave = crval + (pixel - crpix) * cdelt

    return wave


def running_median(array, window):
    """Simple running median."""

    if window % 2 == 0:
        raise ValueError("window must be odd")

    n = len(array)
    half = window // 2

    result = np.empty(n)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)

        result[i] = np.nanmedian(array[lo:hi])

    return result


def local_robust_sigma(residual, window):
    """
    Estimate local noise using MAD.

    sigma = 1.4826 * MAD
    """

    center = running_median(residual, window)

    deviation = np.abs(residual - center)

    mad = running_median(deviation, window)

    sigma = 1.4826 * mad

    good = np.isfinite(sigma) & (sigma > 0)

    if not np.any(good):
        raise RuntimeError("Unable to estimate local noise.")

    global_sigma = np.nanmedian(sigma[good])

    sigma[~good] = global_sigma

    # Avoid unrealistically small local noise
    sigma = np.maximum(sigma, 0.25 * global_sigma)

    return sigma


def grow_boolean_mask(mask, npixels=1):
    """Expand a mask around detected cosmic rays."""

    output = mask.copy()

    for shift in range(1, npixels + 1):
        output[shift:] |= mask[:-shift]
        output[:-shift] |= mask[shift:]

    return output


# ============================================================
# MAIN ROUTINE
# ============================================================


def combine_spectra(
    files,
    output,
    cosmic_sigma=5.0,
    noise_window=51,
    grow_mask=True,
    grow_pixels=1,
    rescale_spectra=True,
    wavelength_tolerance_pixels=1,
    verbose=True,
):
    """
    Combine aligned 1D spectra with cosmic-ray rejection.

    Parameters
    ----------
    files : list
        List of input FITS spectra.

    output : str
        Name of output FITS file.

    cosmic_sigma : float
        Sigma threshold for positive cosmic-ray detection.

    noise_window : int
        Window used for local MAD noise estimate.

    grow_mask : bool
        Expand cosmic-ray mask to neighbouring pixels.

    grow_pixels : int
        Number of pixels used to expand the mask.

    rescale_spectra : bool
        Rescale spectra to a common flux level.

    wavelength_tolerance_pixels : float
        Maximum wavelength-grid difference allowed,
        expressed in pixels.

        Example:
            0.05 = 5% of one pixel.

    verbose : bool
        Print diagnostic information.

    Returns
    -------
    wave : ndarray
        Wavelength array.

    combined : ndarray
        Combined spectrum.

    error : ndarray
        Empirical standard error.

    n_used : ndarray
        Number of spectra contributing to each pixel.

    cosmic_masks : ndarray
        Boolean cosmic-ray masks.
    """
    # ========================================================
    # CHECK WHICH FILES EXIST
    # ========================================================

    existing_files = []
    missing_files = []

    for filename in files:
        if Path(filename).is_file():
            existing_files.append(filename)
        else:
            missing_files.append(filename)

    # Report missing files
    if missing_files:
        print("\nWARNING: the following files do not exist:")

        for filename in missing_files:
            print(f"   {filename}")

        print(f"\nContinuing with {len(existing_files)} available spectra.")

    # Replace original list with valid files only
    files = existing_files

    # ========================================================
    # CHECK NUMBER OF AVAILABLE SPECTRA
    # ========================================================

    if len(files) < 3:
        raise ValueError(
            "\nOnly "
            f"{len(files)} valid spectra are available.\n"
            "At least 3 spectra are required for reliable "
            "cosmic-ray rejection."
        )

    if noise_window % 2 == 0:
        raise ValueError("noise_window must be odd.")

    if verbose:
        print("\n")
        print("=" * 70)
        print("COMBINING SPECTRA")
        print("=" * 70)

        print(f"\nInput spectra requested : {len(existing_files) + len(missing_files)}")

        print(f"Available spectra       : {len(files)}")

        print(f"Missing spectra         : {len(missing_files)}")

    # --------------------------------------------------------
    # READ SPECTRA
    # --------------------------------------------------------

    spectra = []
    headers = []

    if verbose:
        print("\nReading spectra...\n")

    for filename in files:
        flux, header = read_spectrum(filename)

        spectra.append(flux)
        headers.append(header)

        if verbose:
            print(f"{Path(filename).name:50s} {len(flux):6d} pixels")

    # Check length

    lengths = [len(spectrum) for spectrum in spectra]

    if len(set(lengths)) != 1:
        raise ValueError("Input spectra have different numbers of pixels.")

    spectra = np.asarray(spectra, dtype=float)

    n_spectra, n_pixels = spectra.shape

    # --------------------------------------------------------
    # WAVELENGTH GRIDS
    # --------------------------------------------------------

    wave0 = wavelength_from_header(headers[0], n_pixels)

    pixel_size = np.nanmedian(np.abs(np.diff(wave0)))

    if verbose:
        print("\nWavelength-grid comparison")
        print("--------------------------")

        print(f"Reference spectrum: {wave0[0]:.8f} - {wave0[-1]:.8f} A")

        print(f"Pixel size: {pixel_size:.8f} A")

    max_grid_difference = 0.0

    for i in range(1, n_spectra):
        wave_i = wavelength_from_header(headers[i], n_pixels)

        delta = wave_i - wave0

        max_delta = np.nanmax(np.abs(delta))

        max_delta_pixels = max_delta / pixel_size

        max_grid_difference = max(max_grid_difference, max_delta_pixels)

        if verbose:
            print(f"\nSpectrum {i + 1}:")

            print(f"  start difference = {delta[0]:+.8e} A")

            print(f"  end difference   = {delta[-1]:+.8e} A")

            print(f"  maximum          = {max_delta:.8e} A")

            print(f"  maximum          = {max_delta_pixels:.5f} pixels")

        if max_delta_pixels > wavelength_tolerance_pixels:
            raise ValueError(
                "\nWavelength grids differ by "
                f"{max_delta_pixels:.4f} pixels.\n"
                "This exceeds the requested tolerance "
                f"of {wavelength_tolerance_pixels:.4f} "
                "pixels.\n"
                "Spectra should be aligned/rebinned "
                "before combination."
            )

    if verbose:
        print(f"\nMaximum grid difference: {max_grid_difference:.5f} pixels")

        print("Wavelength grids accepted.")

    # --------------------------------------------------------
    # INVALID PIXELS
    # --------------------------------------------------------

    invalid = ~np.isfinite(spectra)

    work = spectra.copy()

    work[invalid] = np.nan

    # --------------------------------------------------------
    # RELATIVE FLUX SCALING
    # --------------------------------------------------------

    scale_factors = np.ones(n_spectra)

    if rescale_spectra:
        preliminary_reference = np.nanmedian(work, axis=0)

        for i in range(n_spectra):
            good = np.isfinite(work[i]) & np.isfinite(preliminary_reference) & (preliminary_reference != 0)

            ratio = work[i, good] / preliminary_reference[good]

            scale_factors[i] = np.nanmedian(ratio)

            work[i] /= scale_factors[i]

    if verbose:
        print("\nRelative scale factors")
        print("----------------------")

        for filename, scale in zip(files, scale_factors):
            print(f"{Path(filename).name:50s} {scale:10.6f}")

    # --------------------------------------------------------
    # MEDIAN REFERENCE
    # --------------------------------------------------------

    reference = np.nanmedian(work, axis=0)

    # --------------------------------------------------------
    # RESIDUALS
    # --------------------------------------------------------

    residuals = work - reference[None, :]

    # --------------------------------------------------------
    # LOCAL NOISE
    # --------------------------------------------------------

    noise = np.empty_like(work)

    for i in range(n_spectra):
        noise[i] = local_robust_sigma(residuals[i], noise_window)

    # --------------------------------------------------------
    # COSMIC DETECTION
    # --------------------------------------------------------

    cosmic_masks = np.zeros(work.shape, dtype=bool)

    for i in range(n_spectra):
        significance = residuals[i] / noise[i]

        # IMPORTANT:
        # only POSITIVE outliers

        cosmic_masks[i] = significance > cosmic_sigma

        cosmic_masks[i] &= np.isfinite(work[i])

        if grow_mask:
            cosmic_masks[i] = grow_boolean_mask(cosmic_masks[i], grow_pixels)

    # --------------------------------------------------------
    # REPORT COSMICS
    # --------------------------------------------------------

    if verbose:
        print("\nCosmic-ray rejection")
        print("--------------------")

        for i, filename in enumerate(files):
            ncosmic = np.sum(cosmic_masks[i])

            print(f"{Path(filename).name:50s} {ncosmic:6d} pixels")

    # --------------------------------------------------------
    # CLEAN DATA
    # --------------------------------------------------------

    bad_mask = cosmic_masks | invalid

    clean = work.copy()

    clean[bad_mask] = np.nan

    # --------------------------------------------------------
    # FINAL COMBINATION
    # --------------------------------------------------------

    with np.errstate(invalid="ignore"):
        combined = np.nanmean(clean, axis=0)

    n_used = np.sum(np.isfinite(clean), axis=0)

    # --------------------------------------------------------
    # EMPIRICAL ERROR
    # --------------------------------------------------------

    with np.errstate(invalid="ignore", divide="ignore"):
        scatter = np.nanstd(clean, axis=0, ddof=1)

        combined_error = scatter / np.sqrt(n_used)

    combined_error[n_used < 2] = np.nan

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    if verbose:
        print("\nNumber of contributing spectra")

        print("------------------------------")

        for n in range(n_spectra + 1):
            number_pixels = np.sum(n_used == n)

            print(f"N = {n:2d}: {number_pixels:6d} pixels")

    # --------------------------------------------------------
    # COSMIC DIAGNOSTIC FILE
    # --------------------------------------------------------

    output_path = Path(output)

    diagnostic_file = output_path.parent / (output_path.stem + "_cosmics.txt")

    with open(diagnostic_file, "w") as f:
        f.write("# spectrum pixel wavelength_A flux reference significance\n")

        for i in range(n_spectra):
            indices = np.where(cosmic_masks[i])[0]

            for p in indices:
                sig = residuals[i, p] / noise[i, p]

                f.write(f"{i + 1:3d} {p:7d} {wave0[p]:13.6f} {work[i, p]:16.6f} {reference[p]:16.6f} {sig:12.4f}\n")
    # ========================================================
    # OUTPUT FITS -- ONLY FINAL COMBINED SPECTRUM
    # ========================================================

    header = headers[0].copy()

    header["NCOMBINE"] = (n_spectra, "Number of combined spectra")

    header["CRSIGMA"] = (cosmic_sigma, "Cosmic rejection threshold")

    header["CRWIN"] = (noise_window, "Local noise window")

    header["HISTORY"] = "Combined with robust cosmic-ray rejection."

    header["HISTORY"] = "Positive outliers detected relative to median spectrum."

    header["HISTORY"] = "Final spectrum = mean of accepted measurements."

    # Write ONLY the final combined 1D spectrum.
    # No additional HDUs/extensions are created.

    hdu = fits.PrimaryHDU(data=combined.astype(np.float32), header=header)

    hdu.writeto(output, overwrite=True)

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    if verbose:
        print("\nOutput:")
        print(f"  {output}")

        print("\nCosmic diagnostic:")
        print(f"  {diagnostic_file}")

        print("\nCombination completed.")

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return (wave0, combined, combined_error, n_used, cosmic_masks)

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    if verbose:
        print("\nOutput:")
        print(f"  {output}")

        print("\nCosmic diagnostic:")
        print(f"  {diagnostic_file}")

        print("\nCombination completed.")

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return (wave0, combined, combined_error, n_used, cosmic_masks)
