import math
import numpy as np
import astropy.units as u
from astropy.io import fits
import matplotlib.pylab as plt

# XMM-Newton PSF model (ELLBETA)
# Computes the 2D PSF image given position arrays and model parameters
def xmm_psf_model(x, y, params):
    # Extract parameters with default values
    pa = params.get('pa', 0.0)
    x0 = params.get('x0', 0.0)
    y0 = params.get('y0', 0.0)
    ellipticity = params.get('ellipticity', 0.0)
    alpha = params.get('alpha', 1.5)
    fwhm = params.get('fwhm', 1.0)
    r0 = params.get('r0', 1.0)
    norm = params.get('norm', 0.75)

    # Coordinate rotation to align with position angle
    pa_rad = math.radians(pa)
    xoff = (x - x0) * math.cos(pa_rad) + (y - y0) * math.sin(pa_rad)
    yoff = (y - y0) * math.cos(pa_rad) - (x - x0) * math.sin(pa_rad)

    # Compute King profile (beta model)
    r02 = r0 ** 2
    roff2 = xoff ** 2 + (yoff / (1.0 - ellipticity)) ** 2
    arg2 = roff2 / r02
    beta2d = 1.0 / (1.0 + arg2) ** alpha

    # Add Gaussian core if fwhm > 0
    if fwhm > 0:
        fwhm2 = fwhm ** 2
        groff2 = xoff ** 2 + (yoff / (1.0 - ellipticity)) ** 2
        arg2 = groff2 / fwhm2
        gauss2d = norm * np.exp(-4.0 * math.log(2) * arg2)
        summed = beta2d + gauss2d
        return summed / summed.max()  # Normalize to 1
    else:
        return beta2d

# Plot PSF image for given parameters
def plot_psf_image():
    # Detector dimensions in pixels (from XMM-Newton EPIC-pn Full Frame)
    xwidth = 376
    ywidth = 384

    # Pixel size: 150 micrometers = 0.15 mm
    pixel_size_m = 150e-6  # in meters
    # Plate scale: 4.1 arcsec per mm at focal length 7.5 m
    plate_scale_arcsec_per_mm = 206265 / 7500  # arcsec per mm
    plate_scale_arcsec_per_m = plate_scale_arcsec_per_mm * 1000  # arcsec per meter

    # Effective pixel size in arcsec
    pixel_size_arcsec = pixel_size_m * plate_scale_arcsec_per_m * u.arcsec

    # Image shape in pixels
    im_shape = (xwidth, ywidth)
    xcen, ycen = xwidth / 2.0, ywidth / 2.0

    # Meshgrid for pixel positions
    yg, xg = np.mgrid[:im_shape[0], :im_shape[1]]

    pars = {
        'x0': xcen, 'y0': ycen,
        'r0': 4.5 * u.arcsec / pixel_size_arcsec,
        'alpha': 1.5,
        'fwhm': 3.15 * u.arcsec / pixel_size_arcsec,
        'norm': 0.36,
        'pa': 0.0,
        'ellipticity': 0.0
    }

    psf_ima = xmm_psf_model(xg, yg, params=pars)

    # Plot extent in arcsec
    extent = [-(xwidth/2)*pixel_size_arcsec.value, (xwidth/2)*pixel_size_arcsec.value,
              -(ywidth/2)*pixel_size_arcsec.value, (ywidth/2)*pixel_size_arcsec.value]

    plt.figure(figsize=(10, 10))
    plt.imshow(psf_ima, cmap='viridis', origin='lower', extent=extent)
    plt.colorbar(label='Normalized PSF Intensity')
    plt.xlabel('Arcsec')
    plt.ylabel('Arcsec')
    plt.title('XMM-Newton EPIC-pn Full Frame PSF (ELLBETA Model)')
    plt.show()

# Generate PSF parameter table for SIXTE
def generate_psf_parameter_table(output_filename):
    energies = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 12.0]  # Photon energies in keV
    thetas = [0, 5, 10, 15]  # Off-axis angles in arcmin

    def compute_r0(energy, theta):
        a, b, c = 5.0, -0.3, 0.5
        return a * (energy / 1.0) ** b * (theta / 1.0) ** c  # r0 in arcsec

    def compute_alpha(energy, theta):
        a, b, c = 1.6, -0.05, 0.1
        return a * (energy / 1.0) ** b * (theta / 1.0) ** c  # alpha dimensionless

    fwhm_const = 3.0  # arcsec
    ellipticity_const = 0.0
    norm_const = 1.0

    rows = []
    for energy in energies:
        for theta in thetas:
            r0 = compute_r0(energy, theta)
            alpha = compute_alpha(energy, theta)
            rows.append((energy, theta, r0, alpha, fwhm_const, ellipticity_const, norm_const))

    coldefs = fits.ColDefs([
        fits.Column(name='ENERGY_KEV', format='E', array=[row[0] for row in rows]),
        fits.Column(name='THETA_ARCMIN', format='E', array=[row[1] for row in rows]),
        fits.Column(name='R0_ARCSEC', format='E', array=[row[2] for row in rows]),
        fits.Column(name='ALPHA', format='E', array=[row[3] for row in rows]),
        fits.Column(name='FWHM_ARCSEC', format='E', array=[row[4] for row in rows]),
        fits.Column(name='ELLIPTICITY', format='E', array=[row[5] for row in rows]),
        fits.Column(name='NORM', format='E', array=[row[6] for row in rows])
    ])

    hdu = fits.BinTableHDU.from_columns(coldefs, name='PSF')
    primary_hdu = fits.PrimaryHDU()

    hdul = fits.HDUList([primary_hdu, hdu])
    hdul.writeto(output_filename, overwrite=True)
    print(f"PSF parameter table saved as {output_filename}")

# Example usage
generate_psf_parameter_table('epn_psf_sixte_table.fits')
plot_psf_image()
