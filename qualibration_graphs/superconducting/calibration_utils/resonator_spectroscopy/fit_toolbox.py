# -*- coding: utf-8 -*-
"""
Resonator circle-fit toolbox.

Verbatim copy of CircleFit/fit_toolbox.py by Christian Schneider and David Zoepfl
(https://github.com/sebastianprobst/resonator_tools, arXiv:1410.3365).
Vendored here to avoid a DataModule dependency in the calibration node.
"""

import numpy as np
import scipy.optimize as spopt


def linear_fit_bg(freq, data, f_range=.1):
    """Fit a linear slope to the data.

    freq : list, np.array
        Frequencies
    data : list, np.array
        Complex data
    f_range : float
        Just f_range are used for the fit (get rid of the resonance)
    """
    fit_range = int(np.round(len(data)) * f_range)
    w = np.zeros(len(data))
    w[0:fit_range] = 1
    w[-fit_range:-1] = 1
    lin_fit = np.polyfit(freq, 20 * np.log10(np.abs(data)), 1, w=w)
    return lin_fit[0], lin_fit[1]


def fit_lorentzian(freq, data, Ql_init=None, fr_init=None, maxfev=10000):
    """Lorentzian Fit

    Parameters
    -----------
    freq : float
        Frequency array
    data : np.array
        Complex data
    Ql_init : float
        Initial guess for Ql
    fr_init : float
        Initial guess for resonance frequency
    """
    mag = np.abs(data)

    if Ql_init is None:
        Ql_init = freq.mean() / (freq[-1] - freq[0]) / 0.05

    if fr_init is None:
        fr_init = freq[np.argmin(mag)]

    A0 = mag[0]

    p, pcovs = spopt.curve_fit(lorentzian_abs, freq, mag,
                               p0=(A0, fr_init / Ql_init, Ql_init, fr_init),
                               maxfev=int(maxfev))
    return p, pcovs


def lorentzian_abs(x, A0, A1, Ql, x0):
    """Lorentzian function"""
    return np.abs((A0 - A1 / (np.pi * x0 / Ql + 1j * 2 * np.pi * (x - x0))))


def get_delay_arrays(freq_GHz, phase_deg, comb_slopes=True, f_range=0.1):
    """Estimate cable delay from linear phase slope.

    Standalone version of fit_toolbox.get_delay that takes raw arrays
    instead of a data_complex object.

    Parameters
    ----------
    freq_GHz : array  — frequencies in GHz
    phase_deg : array — unwrapped phase in degrees
    comb_slopes : bool — True = weighted fit over full range;
                         False = average of first/last segment slopes
    f_range : float   — fraction of data used at each edge

    Returns
    -------
    delay : float  — cable delay in ns
    offset : float — phase offset in degrees
    """
    fit_range = max(1, int(np.round(len(phase_deg) * f_range)))
    if comb_slopes:
        w = np.zeros(len(freq_GHz))
        w[0:fit_range] = 1
        w[-fit_range:] = 1
        lin_fit = np.polyfit(freq_GHz, phase_deg, 1, w=w)
        delay = -lin_fit[0] / 360   # deg / GHz → ns
        offset = lin_fit[1]
    else:
        lin_fit_first = np.polyfit(freq_GHz[:fit_range], phase_deg[:fit_range], 1)
        lin_fit_last  = np.polyfit(freq_GHz[-fit_range:], phase_deg[-fit_range:], 1)
        mean_slope = (lin_fit_first[0] + lin_fit_last[0]) / 2
        delay = -mean_slope / 360
        offset = (lin_fit_first[1] + lin_fit_last[1]) / 2
    return delay, offset


def tan_phase(x, theta0, Ql, fr, consRa):
    """Circuit Model phase behavior which is used for the fit """
    return theta0 + consRa * np.unwrap(2. * np.arctan(2. * Ql * (1. - x / fr)))


def phase_fit(f_data, z_data, theta0, Ql, fr, maxfev=10000):
    """Arctan fit of phase.

    Information is used to get offresonant point
    """
    p0 = (theta0, Ql, fr, 1)
    thetas = np.unwrap(np.angle(z_data))
    popt, pcov = spopt.curve_fit(tan_phase, f_data, thetas, p0=p0,
                                 maxfev=maxfev)
    return popt, pcov


def periodic_boundary(x, bound):
    return np.fmod(x, bound) - np.trunc(x / bound) * bound


def fit_circle_weights(freq, data, fr, Ql, weights):
    def res(params, data, weights):
        r, xc, yc = params
        r_calc = (data.real - xc) ** 2 + (data.imag - yc) ** 2
        return (r_calc - r ** 2) ** 2 * weights ** 2

    f_index = np.argwhere(freq >= fr)[0]
    r_guess = np.abs(data[0] - data[f_index]) / 2
    mid_vec = (data[0] + data[f_index]) / 2
    xc_guess = mid_vec.real
    yc_guess = mid_vec.imag
    f = spopt.leastsq(res, [r_guess, xc_guess, yc_guess], args=(data, weights),
                      full_output=True)
    r, xc, yc = f[0]
    return xc, yc, r


def notch_model(x, Ql, absQc, x0, phi0):
    return (1 - np.exp(1j * phi0) * Ql / absQc / (1 + 2j * Ql * (x - x0) / x0))


def notch_model_mag(x, Ql, absQc, x0):
    """Magnitude notch model — analytical solution for magnitude"""
    return np.abs(Ql) / np.abs(absQc) / np.sqrt(
        1 + 4 * (x - x0) ** 2 / x0 ** 2 * Ql ** 2)


def fit_model_notch(freq, data, Ql, absQc, fr, phi0, weights, max_nfev=1000):
    """Final fit of the notch model (complex) to get Ql, Qc, fr, phi0."""

    def res(params, f, data):
        Ql, Qc, x0, phi0 = params
        diff = notch_model(f, Ql, Qc, x0, phi0) - data
        z1d = np.zeros(data.size * 2, dtype=np.float64)
        z1d[0:z1d.size:2] = diff.real * weights
        z1d[1:z1d.size:2] = diff.imag * weights
        return z1d

    f = spopt.least_squares(res, [np.abs(Ql), np.abs(absQc), fr, phi0],
                            args=(freq, data),
                            verbose=0, max_nfev=max_nfev, xtol=2.3e-16,
                            ftol=2.3e-16,
                            bounds=([0, 0, 0, -np.pi],
                                    [10e12, 10e12, 20e9, np.pi]))
    cov = np.linalg.inv(np.dot(f.jac.T, f.jac))
    chi2dof = np.sum(f.fun**2) / (f.fun.size - f.x.size)
    cov *= chi2dof
    return f.x, cov


def fit_mag_notch(freq, data, Ql, absQc, fr, phi0, weights, ftol=1e-16):
    """Final fit of the notch model (magnitude only)."""

    def res(params, f, d, weights):
        Ql, Qc, x0 = params
        diff = (notch_model_mag(f, Ql, Qc, x0) - d) ** 2
        return diff * weights

    data1 = np.abs((1 - data) * np.exp(-1j * phi0))
    f = spopt.leastsq(res, [np.abs(Ql), np.abs(absQc), fr],
                      args=(freq, data1, weights),
                      full_output=True, xtol=ftol, ftol=ftol)
    return f[0], f[1]


def reflection_model(x, Ql, Qc, x0, phi0):
    return -1 * (1 - np.exp(1j * phi0) * (2 * Ql / Qc) / (
                1. + 2j * Ql * ((x - x0) / x0)))


def reflection_model_mag(x, Ql, Qc, x0, c):
    return c - (2 * Ql / Qc) ** 2 / (1. + 4 * Ql ** 2 * ((x - x0) / x0) ** 2)


def fit_model_refl(freq, data, Ql, Qc, fr, phi0, weights, ftol=2.3e-16):
    """Final fit of the reflection model (complex) to get Ql, Qc, fr, phi0."""

    def res(params, f, data):
        Ql, Qc, x0, phi0 = params
        diff = reflection_model(f, Ql, Qc, x0, phi0) - data
        z1d = np.zeros(data.size * 2, dtype=np.float64)
        z1d[0:z1d.size:2] = diff.real * weights
        z1d[1:z1d.size:2] = diff.imag * weights
        return z1d

    f = spopt.least_squares(
        res, [Ql, Qc, fr, phi0], args=(freq, data),
        bounds=([Ql - 0.5 * Ql, Qc - 0.5 * Qc, fr - 10 * fr / Ql, -np.pi],
                [Ql + 0.5 * Ql, Qc + 0.5 * Qc, fr + 10 * fr / Ql,  np.pi]),
        verbose=0, max_nfev=1000, xtol=2.3e-16, ftol=2.3e-16,
    )
    cov = np.linalg.inv(np.dot(f.jac.T, f.jac))
    chi2dof = np.sum(f.fun**2) / (f.fun.size - f.x.size)
    cov *= chi2dof
    return f.x, cov


def fit_mag_refl(freq, data, Ql, Qc, fr, phi0, weights, ftol=1e-16):
    """Final fit of the reflection model (magnitude only)."""

    def res(params, f, data):
        Ql, Qc, x0, c = params
        diff = (reflection_model_mag(f, Ql, Qc, x0, c) - np.abs(data)) ** 2
        return diff * weights

    f = spopt.leastsq(res, [Ql, Qc, fr, 1], args=(freq, data),
                      full_output=True, xtol=ftol, ftol=ftol)
    return f[0], f[1]


def subtract_linear_bg(freq, data, fit_range=0.1):
    """Subtracts background determined with a linear fit"""
    bg_slope, offset = linear_fit_bg(freq, data, fit_range)
    mag = 20 * np.log10(np.abs(data))
    f_rotate = freq[np.argmin(mag)]
    data_norm = (10 ** (0.05 * (mag - (freq - f_rotate) * bg_slope)) *
                 np.exp(1j * np.angle(data)))
    return data_norm, (f_rotate, bg_slope, offset)


def get_weights(freq, Ql, fr, weight_width):
    """Weighting function: uniform near resonance, 1/|f-fr| outside FWHM."""
    width = fr / Ql
    weights = np.ones(len(freq))
    outer_idx = np.abs(freq - fr) > width * weight_width
    weights[outer_idx] = (width * weight_width) / (np.abs(freq[outer_idx] - fr))
    return weights


def fit_theta0(freq, data, Ql, fr, zc):
    """Detect the off-resonance angle theta0 by arctan fit of the centred circle."""
    z_data_moved = data - zc
    theta0 = ((np.unwrap(np.angle(z_data_moved))[0] +
               np.unwrap(np.angle(z_data_moved))[-1]) / 2)
    fitparams, pcop = phase_fit(freq, z_data_moved, theta0, np.absolute(Ql), fr)
    theta0, Ql, fr, tmp = fitparams
    theta0 = periodic_boundary(theta0 + np.pi, 2 * np.pi)
    return theta0, (fitparams, pcop)
