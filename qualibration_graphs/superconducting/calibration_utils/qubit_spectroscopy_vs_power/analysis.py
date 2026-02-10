import numpy as np
import xarray as xr
from dataclasses import dataclass
from typing import Dict, Tuple

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V


# ----------------------------------------------------------------------
# RAW DATA PROCESSING
# ----------------------------------------------------------------------

def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """
    Process raw qubit spectroscopy vs power dataset:
      - Convert I/Q to Volts
      - Compute |IQ| and phase (NO normalization)
      - Add full RF frequency coordinate
    """

    ds = convert_IQ_to_V(ds, node.namespace["qubits"])

    ds = add_amplitude_and_phase(
        ds,
        dim="detuning",
        subtract_slope_flag=True,
    )

    full_freq = np.array(
        [
            ds.detuning + q.xy.RF_frequency
            for q in node.namespace["qubits"]
        ]
    )

    ds = ds.assign_coords(
        full_freq=(["qubit", "detuning"], full_freq)
    )

    ds.full_freq.attrs = {
        "long_name": "RF frequency",
        "units": "Hz",
    }

    return ds


# ----------------------------------------------------------------------
# PEAK + LINEWIDTH ANALYSIS
# ----------------------------------------------------------------------

def _peak_index(iq_abs, baseline, min_height):
    """
    Return index of peak if it exceeds minimum height above baseline.
    Otherwise return -1.
    """
    y = np.asarray(iq_abs)

    if np.all(np.isnan(y)):
        return -1

    idx = int(np.nanargmax(y))
    if y[idx] - baseline < min_height:
        return -1

    return idx


def _compute_fwhm_around_peak(detuning, signal, peak_idx) -> float:
    """
    Compute FWHM around a known peak index.
    Returns linewidth in Hz.
    """
    if peak_idx < 0:
        return np.nan

    x = np.asarray(detuning)
    y = np.asarray(signal)

    if np.all(np.isnan(y)):
        return np.nan

    y = y - np.nanmin(y)
    half_max = 0.5 * np.nanmax(y)

    above = y >= half_max
    if not np.any(above):
        return np.nan

    idx = np.where(above)[0]
    return x[idx[-1]] - x[idx[0]]


# ----------------------------------------------------------------------
# MAIN FIT FUNCTION
# ----------------------------------------------------------------------

def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, "FitParameters"]]:
    """
    Peak-based rough qubit spectroscopy analysis.
    """

    p = node.parameters
    machine = node.machine

    # --------------------------------------------------
    # Build per-qubit baseline and max IQ from QUAM
    # --------------------------------------------------
    qubit_names = ds.qubit.values

    baseline_iq_abs_v = xr.DataArray(
        [
            machine.resonator_amplitudes[q]["min_amplitude"]
            for q in qubit_names
        ],
        dims=["qubit"],
        coords={"qubit": qubit_names},
    )

    max_iq_abs_v = xr.DataArray(
        [
            machine.resonator_amplitudes[q]["max_amplitude"]
            for q in qubit_names
        ],
        dims=["qubit"],
        coords={"qubit": qubit_names},
    )

    # --------------------------------------------------
    # Minimum acceptable peak height (per qubit)
    # --------------------------------------------------
    min_peak_height = (
        p.min_peak_fraction
        * (max_iq_abs_v - baseline_iq_abs_v)
    )

    # --------------------------------------------------
    # Peak index vs power
    # --------------------------------------------------
    peak_index = xr.apply_ufunc(
        _peak_index,
        ds.IQ_abs,
        baseline_iq_abs_v,
        min_peak_height,
        input_core_dims=[["detuning"], [], []],
        vectorize=True,
        output_dtypes=[int],
    )

    ds["peak_index"] = peak_index
    ds.peak_index.attrs = {
        "long_name": "Detected peak index",
    }

    # --------------------------------------------------
    # Peak height
    # --------------------------------------------------
    ds["peak_height"] = xr.where(
        peak_index >= 0,
        ds.IQ_abs.isel(detuning=peak_index),
        np.nan,
    )

    ds.peak_height.attrs = {
        "long_name": "Peak height above baseline",
        "units": "V",
    }

    # --------------------------------------------------
    # Linewidth (FWHM)
    # --------------------------------------------------
    linewidth = xr.apply_ufunc(
        _compute_fwhm_around_peak,
        ds.detuning,
        ds.IQ_abs,
        peak_index,
        input_core_dims=[["detuning"], ["detuning"], []],
        vectorize=True,
        output_dtypes=[float],
    )

    ds["linewidth"] = linewidth
    ds.linewidth.attrs = {
        "long_name": "Spectroscopy linewidth (FWHM)",
        "units": "Hz",
    }

    # --------------------------------------------------
    # Valid power mask
    # --------------------------------------------------
    valid_power = (
        (ds.peak_index >= 0) &
        (ds.linewidth < p.linewidth_threshold_hz)
    )

    # --------------------------------------------------
    # Select highest safe power
    # --------------------------------------------------
    selected_power = (
        ds.power.where(valid_power)
        .max(dim="power")
        - p.power_buffer_db
    )

    ds["selected_power"] = selected_power
    ds.selected_power.attrs = {
        "long_name": "Selected spectroscopy power",
        "units": "dBm",
    }

    # --------------------------------------------------
    # Rough qubit frequency at selected power
    # --------------------------------------------------
    def _peak_frequency(full_freq, iq_abs, power, target_power):
        # If no valid power was selected, fail gracefully
        if np.isnan(target_power):
            return np.nan

        power = np.asarray(power)
        iq_abs = np.asarray(iq_abs)
        full_freq = np.asarray(full_freq)

        # Guard against all-NaN power arrays
        if np.all(np.isnan(power)):
            return np.nan

        # Find closest power index safely
        diff = np.abs(power - target_power)
        if np.all(np.isnan(diff)):
            return np.nan

        idx = int(np.nanargmin(diff))

        spectrum = iq_abs[idx]
        if np.all(np.isnan(spectrum)):
            return np.nan

        peak_idx = int(np.nanargmax(spectrum))
        return full_freq[peak_idx]


    rough_freq = xr.apply_ufunc(
        _peak_frequency,
        ds.full_freq,
        ds.IQ_abs,
        ds.power,
        ds.selected_power,
        input_core_dims=[
            ["detuning"],
            ["power", "detuning"],
            ["power"],
            [],
        ],
        vectorize=True,
        output_dtypes=[float],
    )

    ds["rough_qubit_frequency"] = rough_freq
    ds.rough_qubit_frequency.attrs = {
        "long_name": "Rough qubit frequency",
        "units": "Hz",
    }

    # --------------------------------------------------
    # Dummy fit results (pipeline compatibility)
    # --------------------------------------------------
    fit_results = {
        q.name: FitParameters(success=True)
        for q in node.namespace["qubits"]
    }

    return ds, fit_results


def log_fitted_results(*args, **kwargs):
    """No-op logger."""
    return


# ----------------------------------------------------------------------
# Fit interface
# ----------------------------------------------------------------------

@dataclass
class FitParameters:
    success: bool = False
