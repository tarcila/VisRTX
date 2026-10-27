"""Independent CPU interpretation of principledVolume.md for reference baking."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
XYZ_TO_RGB = np.array([[3.2404542, -1.5371385, -.4985314],
                       [-.969266, 1.8760108, .041556],
                       [.0556434, -.2040259, 1.0572252]])
RGB_Y = np.array([.2126729, .7151522, .0721750])


def observer():
    cie = np.loadtxt(ROOT/'CIE_xyz_1931_2deg.csv', delimiter=',')
    d65 = np.loadtxt(ROOT/'CIE_std_illum_D65.csv', delimiter=',')
    wavelengths = cie[:, 0]
    weights = np.ones(len(wavelengths))
    weights[[0, -1]] = .5
    daylight = np.interp(wavelengths, d65[:, 0], d65[:, 1])
    daylight /= np.sum(daylight*weights)
    normalization = 1 / np.sum(daylight*cie[:, 2]*weights)
    return wavelengths, cie[:, 1:]*weights[:, None]*normalization


def blackbody(temperatures):
    """Planck radiance per nm integrated over 360..830 nm; no artistic gain."""
    wavelength, cmf = observer()
    lam = wavelength * 1e-9
    temperature = np.asarray(temperatures, dtype=float)
    safe = np.maximum(temperature.reshape(-1, 1), 1.)
    exponent = (6.62607015e-34 * 299792458.) / (lam * 1.380649e-23 * safe)
    with np.errstate(over='ignore'):
        spectrum = (2*6.62607015e-34*299792458.**2 / lam**5) / np.expm1(exponent) * 1e-9
    xyz = spectrum @ cmf
    rgb = xyz @ XYZ_TO_RGB.T
    negative = (rgb < 0).any(axis=1)
    rgb = np.maximum(rgb, 0)
    y = rgb @ RGB_Y
    scale = np.ones(len(rgb))
    np.divide(xyz[:, 1], y, out=scale, where=negative & (y > 0))
    rgb *= scale[:, None]
    rgb[temperature.reshape(-1) <= 0] = 0
    return rgb.reshape(temperature.shape+(3,))


def coefficients_per_density(p):
    extinction = -np.log(p['transmissionColor']) / p['transmissionDistance']
    albedo = np.asarray(p['scatteringColor'])
    return extinction*albedo, extinction*(1-albedo)


def thermal_lookup(temperature, table):
    """1 K LUT interpolation, verified against direct integration in tests."""
    return np.stack([np.interp(temperature, np.arange(len(table)), table[:, c])
                     for c in range(3)], axis=-1)
