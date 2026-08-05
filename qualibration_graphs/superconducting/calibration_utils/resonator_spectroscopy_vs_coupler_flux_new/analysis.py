"""Analysis utilities for resonator spectroscopy versus coupler flux calibration (``_new``).

Self-contained copy of ``resonator_spectroscopy_vs_flux/analysis.py`` with one
behavioural fix in :func:`process_raw_dataset`:

This node measures ONE qubit per qubit pair, and two pairs may share the same
measured control/target qubit (e.g. ``q4-5`` and ``q4-7`` both measure ``q4``).
The original code renamed the ``qubit_pair`` dimension to ``qubit`` using the
*measured-qubit* name, which produced **duplicate** ``qubit`` labels.  Every
downstream ``.sel(qubit=...)`` then returned 2 rows (2-D), and ``_fit_sinusoid``
crashed with ``IndexError: too many indices for array`` at ``flux = flux[finite]``.

Here the ``qubit`` dimension is keyed by the **unique qubit-pair name** and I/Q
is converted to volts positionally (replacing ``convert_IQ_to_V``'s
qubit-name-based alignment, which collapses on duplicate names).  With unique
labels the rest of the analysis is identical to the shared util.
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V  # noqa: F401


@dataclass
class FitParameters:
    """Stores the relevant node-specific fitted parameters used to update the state at the end of the node."""

    success: bool
    resonator_frequency: float
    frequency_shift: float
    min_offset: float
    idle_offset: float
    dv_phi0: float
    phi0_current: float
    m_pH: float


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit pair {q}: "
        s_idle_offset = f"\tidle offset: {fit_results[q]['idle_offset'] * 1e3:.0f} mV | "
        if np.isfinite(fit_results[q]["min_offset"]):
            s_min_offset = f"min offset: {fit_results[q]['min_offset'] * 1e3:.0f} mV | "
        else:
            s_min_offset = "min offset: not found | "
        s_freq = f"Resonator frequency: {1e-9 * fit_results[q]['resonator_frequency']:.3f} GHz | "
        s_shift = f"(shift of {1e-6 * fit_results[q]['frequency_shift']:.0f} MHz)\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_idle_offset + s_min_offset + s_freq + s_shift)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Process the raw dataset, keying the ``qubit`` dimension by the unique qubit-pair name.

    The measured-qubit names are not unique across pairs (two pairs can share a
    control/target qubit), so we key by the qubit-pair name instead and convert
    I/Q to volts positionally — fully robust to a shared measured qubit.
    """
    measured = node.namespace["qubits"]  # measured qubit objects (one per pair; may repeat)
    pair_names = list(node.namespace["qubit_pairs"].get_names())  # unique, in pair order

    # Re-key the qubit dimension to the unique qubit-pair name (positional; lengths match).
    ds = ds.assign_coords(qubit=("qubit", pair_names))

    # Convert the 'I' and 'Q' quadratures to volts (single-demod factor of 1),
    # keyed by the unique pair labels (avoids name-based alignment on duplicates).
    readout_lengths = xr.DataArray(
        [q.resonator.operations["readout"].length for q in measured],
        dims=["qubit"],
        coords={"qubit": pair_names},
    )
    ds = ds.assign({key: ds[key] * 2**12 / readout_lengths for key in ("I", "Q")})

    # Add the amplitude and phase to the raw dataset
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)

    # Add the absolute RF frequency per pair (the measured qubit's resonator), keyed positionally
    full_freq = np.array([ds.detuning + q.resonator.RF_frequency for q in measured])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}

    # Add the current axis of each qubit to the dataset coordinates for plotting
    current = ds.flux_bias / node.parameters.input_line_impedance_in_ohm
    ds = ds.assign_coords({"current": (["flux_bias"], current.data)})
    ds.current.attrs["long_name"] = "Current"
    ds.current.attrs["units"] = "A"
    # Add attenuated current to dataset
    attenuation_factor = 10 ** (-node.parameters.line_attenuation_in_db / 20)
    attenuated_current = ds.current * attenuation_factor
    ds = ds.assign_coords({"attenuated_current": (["flux_bias"], attenuated_current.values)})
    ds.attenuated_current.attrs["long_name"] = "Attenuated Current"
    ds.attenuated_current.attrs["units"] = "A"
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    For each qubit pair, locate the resonator vs coupler-flux trace (minimum IQ_abs per detuning),
    fit a sinusoid ``a*cos(2π*f*flux + phi) + offset`` to the peak positions, and derive the
    sweet-spot frequency, joint/min offsets, and mutual inductance metadata.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data (including ``IQ_abs``, ``detuning``, ``flux_bias``),
        with the ``qubit`` dimension keyed by the unique qubit-pair name.
    node : QualibrationNode
        Node context (qubits, parameters).

    Returns:
    --------
    tuple
        ``(fit_dataset, fit_results_dict)`` — xarray dataset with coordinates for plotting and
        per-pair :class:`FitParameters` values for state updates.
    """
    # Find the minimum of each frequency line to follow the resonance vs flux
    peak_freq = ds.IQ_abs.idxmin(dim="detuning")
    # Fit each pair's peak-position trace to a sinusoid independently so missing
    # points only affect that pair. Iterate positionally for full safety.
    fit_per_qubit = []
    for i in range(peak_freq.sizes["qubit"]):
        q = peak_freq.qubit.values[i]
        flux = peak_freq.flux_bias.values.astype(float)
        signal = peak_freq.isel(qubit=i).values.astype(float)
        params = _fit_sinusoid(flux, signal)
        q_fit = xr.DataArray(
            np.asarray(params)[None, :],
            dims=["qubit", "fit_vals"],
            coords={"qubit": [q], "fit_vals": ["a", "f", "phi", "offset"]},
        )
        fit_per_qubit.append(q_fit)
    fit_results_da = xr.concat(fit_per_qubit, dim="qubit")
    extrema, peak_freq_fit = _find_resonator_dip_extrema(peak_freq, fit_results_da)
    fit_dataset, fit_results = _extract_relevant_fit_parameters(fit_results_da, peak_freq, node, extrema, peak_freq_fit)
    return fit_dataset, fit_results


def _sinusoid_model(x: np.ndarray, a: float, f: float, phi: float, offset: float) -> np.ndarray:
    return a * np.cos(2 * np.pi * f * x + phi) + offset


def _fit_sinusoid(flux: np.ndarray, signal: np.ndarray) -> tuple[float, float, float, float]:
    """Fit ``y = a*cos(2π*f*x + phi) + offset`` to an irregular, possibly NaN-laden trace.

    The fit uses peak-to-peak amplitude and FFT-based frequency guesses computed on a
    uniform grid (linear interpolation to bridge missing samples). The output is
    canonicalised to ``a > 0`` and ``f > 0`` by folding the sign of ``a`` and ``f`` into
    the phase, with ``phi`` wrapped to ``[-π, π)``.

    Returns ``(np.nan,) * 4`` if the data are insufficient or the optimiser fails.
    """
    flux = np.asarray(flux, dtype=float).ravel()
    signal = np.asarray(signal, dtype=float).ravel()
    finite = np.isfinite(flux) & np.isfinite(signal)
    flux = flux[finite]
    signal = signal[finite]
    if flux.size < 5:
        return (np.nan, np.nan, np.nan, np.nan)

    order = np.argsort(flux)
    flux = flux[order]
    signal = signal[order]

    flux_min, flux_max = float(flux.min()), float(flux.max())
    span = flux_max - flux_min
    if span <= 0:
        return (np.nan, np.nan, np.nan, np.nan)

    offset0 = float(np.mean(signal))
    amp0 = float((np.max(signal) - np.min(signal)) / 2.0)
    if not amp0:
        return (np.nan, np.nan, np.nan, np.nan)

    # FFT-based frequency guess on a uniformly-resampled trace (avoids irregular sampling
    # caused by NaN-dropped points). The DC component is removed before the FFT.
    n_uniform = max(64, 2 * flux.size)
    flux_uniform = np.linspace(flux_min, flux_max, n_uniform)
    signal_uniform = np.interp(flux_uniform, flux, signal) - offset0
    spacing = (flux_max - flux_min) / (n_uniform - 1)
    fft_vals = np.fft.rfft(signal_uniform)
    fft_freqs = np.fft.rfftfreq(n_uniform, d=spacing)
    # Drop DC (first bin), pick the frequency with the largest magnitude
    if fft_vals.size > 1:
        idx = int(np.argmax(np.abs(fft_vals[1:]))) + 1
        f0_guess = float(fft_freqs[idx])
    else:
        f0_guess = 0.0
    if f0_guess <= 0:
        f0_guess = 1.0 / span

    # Phase guess: align cos(2π*f0*x + phi) with the data via the FFT phase
    if fft_vals.size > 1:
        phi0 = float(np.angle(fft_vals[idx]))
    else:
        phi0 = 0.0

    try:
        popt, _ = curve_fit(
            _sinusoid_model,
            flux,
            signal,
            p0=[amp0, f0_guess, phi0, offset0],
            maxfev=10000,
        )
    except (RuntimeError, ValueError):
        return (np.nan, np.nan, np.nan, np.nan)

    a, f, phi, offset = popt
    # Canonical form: a > 0, f > 0 (cos is even, so f → -f flips phi sign)
    if f < 0:
        f = -f
        phi = -phi
    if a < 0:
        a = -a
        phi = phi + np.pi
    phi = float(((phi + np.pi) % (2 * np.pi)) - np.pi)
    return (float(a), float(f), phi, float(offset))


def _max_position_closest_to_zero(a: float, f: float, phi: float) -> float:
    """Return the location of the maximum of ``a*cos(2π*f*t + phi)`` closest to ``t=0``."""
    if a >= 0:
        # cos(theta) = 1 at theta = 2πk → t = (2πk - phi)/(2πf)
        t_principal = -phi / (2 * np.pi * f)
    else:
        # a < 0 flips max and min: cos(theta) = -1 at theta = (2k+1)π
        t_principal = (np.pi - phi) / (2 * np.pi * f)
    period = 1.0 / np.abs(f)
    n = np.round(t_principal / period)
    return t_principal - n * period


def _shift_to_zero(t: float, period: float) -> float:
    """Shift t by integer multiples of period to bring it closest to zero."""
    n = np.round(t / period)
    return t - n * period


def _find_resonator_dip_extrema(
    peak_freq: xr.DataArray,
    fit_results_da: xr.DataArray,
    num_fit_points: int = 1001,
) -> tuple[dict[str, dict[str, float]], xr.DataArray]:
    """
    Compute the max and min of the fitted resonator-dip vs flux sinusoid for each qubit pair.

    The peak-position trace is fit to ``a*cos(2π*f*flux + phi) + offset``. Among the
    infinitely many maxima/minima of the sinusoid, the ones closest to ``flux=0`` are
    selected as the joint/idle and min offsets respectively. Each extremum is reported
    only when it lies inside the swept flux window — extrapolated extrema are returned
    as NaN.

    Parameters
    ----------
    peak_freq : xr.DataArray
        Per-pair resonator dip frequency (relative to RF) versus flux bias. Used to
        determine the swept flux window and the fine grid for the fitted curve.
    fit_results_da : xr.DataArray
        Sinusoidal fit parameters (``a``, ``f``, ``phi``, ``offset``) per pair.
    num_fit_points : int
        Number of points used to render the fitted sinusoid for plotting.
    """
    flux_min = float(peak_freq.flux_bias.min())
    flux_max = float(peak_freq.flux_bias.max())
    fit_flux = np.linspace(flux_min, flux_max, num_fit_points)
    extrema = {}
    fit_curves = []
    for qubit in peak_freq.qubit.values:
        params = fit_results_da.sel(qubit=qubit)
        a = float(params.sel(fit_vals="a"))
        f = float(params.sel(fit_vals="f"))
        phi = float(params.sel(fit_vals="phi"))
        offset = float(params.sel(fit_vals="offset"))

        if not np.all(np.isfinite([a, f, phi, offset])) or not np.abs(f) > 0:
            fit_curves.append(np.full_like(fit_flux, np.nan, dtype=float))
            extrema[str(qubit)] = {
                "max_offset": np.nan,
                "max_frequency_shift": np.nan,
                "min_offset": np.nan,
                "min_frequency_shift": np.nan,
            }
            continue

        period = 1.0 / np.abs(f)
        # Locate the max and min of the fitted sinusoid closest to zero flux
        t_max = _max_position_closest_to_zero(a, f, phi)
        t_min = _shift_to_zero(t_max + 0.5 * period, period)

        # Render fitted curve over swept range for plotting
        fit_curve = a * np.cos(2 * np.pi * f * fit_flux + phi) + offset
        fit_curves.append(fit_curve)

        # Sinusoid extrema values: |a| + offset (max) and -|a| + offset (min)
        max_value = abs(a) + offset
        min_value = -abs(a) + offset

        # Only report extrema that lie inside the swept flux window
        if flux_min <= t_max <= flux_max:
            max_offset = t_max
            max_frequency_shift = max_value
        else:
            max_offset = np.nan
            max_frequency_shift = np.nan

        if flux_min <= t_min <= flux_max:
            min_offset = t_min
            min_frequency_shift = min_value
        else:
            min_offset = np.nan
            min_frequency_shift = np.nan

        extrema[str(qubit)] = {
            "max_offset": max_offset,
            "max_frequency_shift": max_frequency_shift,
            "min_offset": min_offset,
            "min_frequency_shift": min_frequency_shift,
        }

    peak_freq_fit = xr.DataArray(
        np.asarray(fit_curves),
        dims=["qubit", "fit_flux_bias"],
        coords={"qubit": peak_freq.qubit.values, "fit_flux_bias": fit_flux},
        name="peak_freq_fit",
    )
    peak_freq_fit.attrs = {"long_name": "fitted resonator dip position", "units": "Hz"}
    peak_freq_fit.fit_flux_bias.attrs = {"long_name": "flux bias", "units": "V"}
    return extrema, peak_freq_fit


def _extract_relevant_fit_parameters(
    fit_results: xr.DataArray,
    peak_freq: xr.DataArray,
    node: QualibrationNode,
    extrema: dict[str, dict[str, float]],
    peak_freq_fit: xr.DataArray,
):
    """Add metadata to the fit dataset and fit result dictionary."""
    fit = xr.merge([fit_results.rename("fit_results"), peak_freq.rename("peak_freq"), peak_freq_fit])

    qubit_names = [str(q) for q in fit.qubit.values]
    flux_idle = xr.DataArray(
        [extrema[q]["max_offset"] for q in qubit_names],
        dims=["qubit"],
        coords={"qubit": fit.qubit.values},
    )
    fit = fit.assign_coords(idle_offset=("qubit", flux_idle.data))
    fit.idle_offset.attrs = {"long_name": "maximum resonator frequency flux bias", "units": "V"}

    flux_min = xr.DataArray(
        [extrema[q]["min_offset"] for q in qubit_names],
        dims=["qubit"],
        coords={"qubit": fit.qubit.values},
    )
    fit = fit.assign_coords(flux_min=("qubit", flux_min.data))
    fit.flux_min.attrs = {"long_name": "minimum frequency flux bias", "units": "V"}

    full_freq = np.array([q.resonator.RF_frequency for q in node.namespace["qubits"]])
    freq_shift = xr.DataArray(
        [extrema[q]["max_frequency_shift"] for q in qubit_names],
        dims=["qubit"],
        coords={"qubit": fit.qubit.values},
    )
    fit = fit.assign_coords(freq_shift=("qubit", freq_shift.data))
    fit.freq_shift.attrs = {"long_name": "frequency shift", "units": "Hz"}
    fit = fit.assign_coords(sweet_spot_frequency=("qubit", freq_shift.data + full_freq))
    fit.sweet_spot_frequency.attrs = {
        "long_name": "sweet spot frequency",
        "units": "Hz",
    }
    dv_phi0 = 2 * np.abs(flux_min - flux_idle)
    dv_phi0 = dv_phi0.where(np.isfinite(flux_min) & np.isfinite(flux_idle))

    attenuation_factor = 10 ** (-node.parameters.line_attenuation_in_db / 20)
    phi0_current = dv_phi0 / node.parameters.input_line_impedance_in_ohm * attenuation_factor
    m_pH = xr.where(
        np.isfinite(dv_phi0) & (dv_phi0 > 0),
        1e12 * 2.068e-15 / dv_phi0 / node.parameters.input_line_impedance_in_ohm * attenuation_factor,
        np.nan,
    )

    # Assess whether the fit was successful or not
    freq_success = np.abs(freq_shift.data) < node.parameters.frequency_span_in_mhz * 1e6
    nan_success = np.isnan(freq_shift.data) | np.isnan(flux_idle.data)
    success_criteria = freq_success & ~nan_success
    fit = fit.assign_coords(success=("qubit", success_criteria))

    fit_results_out = {
        q: FitParameters(
            success=bool(fit.sel(qubit=q).success.values),
            resonator_frequency=float(fit.sweet_spot_frequency.sel(qubit=q).values),
            frequency_shift=float(freq_shift.sel(qubit=q).values),
            min_offset=float(flux_min.sel(qubit=q).values),
            idle_offset=float(flux_idle.sel(qubit=q).values),
            dv_phi0=float(dv_phi0.sel(qubit=q).values),
            phi0_current=float(phi0_current.sel(qubit=q).values),
            m_pH=float(m_pH.sel(qubit=q).values),
        )
        for q in fit.qubit.values
    }

    return fit, fit_results_out
