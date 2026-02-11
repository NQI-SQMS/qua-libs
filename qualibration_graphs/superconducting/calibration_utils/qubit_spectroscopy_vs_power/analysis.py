import numpy as np
import xarray as xr
from dataclasses import dataclass
from typing import Dict, Tuple

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V


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
        [ds.detuning + q.xy.RF_frequency for q in node.namespace["qubits"]]
    )

    ds = ds.assign_coords(
        full_freq=(["qubit", "detuning"], full_freq)
    )

    ds.full_freq.attrs = {
        "long_name": "RF frequency",
        "units": "Hz",
    }

    return ds


def _peak_index(iq_abs, baseline, min_height):
    y = np.asarray(iq_abs)

    if np.all(np.isnan(y)):
        return -1

    idx = int(np.nanargmax(y))
    if y[idx] - baseline < min_height:
        return -1

    return idx


def _compute_fwhm_around_peak(detuning, signal, peak_idx) -> float:
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


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, "FitParameters"]]:
    """
    Peak-based rough qubit spectroscopy vs power analysis.
    """

    p = node.parameters
    machine = node.machine

    qubit_names = ds.qubit.values

    baseline_iq_abs_v = xr.DataArray(
        [machine.resonator_amplitudes[q]["min_amplitude"] for q in qubit_names],
        dims=["qubit"],
        coords={"qubit": qubit_names},
    )

    max_iq_abs_v = xr.DataArray(
        [machine.resonator_amplitudes[q]["max_amplitude"] for q in qubit_names],
        dims=["qubit"],
        coords={"qubit": qubit_names},
    )

    min_peak_height = p.min_peak_fraction * (max_iq_abs_v - baseline_iq_abs_v)

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

    ds["peak_height"] = xr.where(
        peak_index >= 0,
        ds.IQ_abs.isel(detuning=peak_index),
        np.nan,
    )

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

    valid_power = (
        (ds.peak_index >= 0)
        & (ds.linewidth < p.linewidth_threshold_hz)
    )

    selected_power = (
        ds.power.where(valid_power)
        .max(dim="power")
        - p.power_buffer_db
    )

    ds["selected_power"] = selected_power

    def _peak_frequency(full_freq, iq_abs, power, target_power):
        if np.isnan(target_power):
            return np.nan

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

    fit_results = {
        q: FitParameters(
            selected_power=ds.sel(qubit=q).selected_power.values.__float__(),
            rough_qubit_frequency=ds.sel(qubit=q).rough_qubit_frequency.values.__float__(),
            linewidth=ds.sel(qubit=q).linewidth.min(dim="power").values.__float__(),
            success=bool(
                np.isfinite(ds.sel(qubit=q).selected_power)
                and np.isfinite(ds.sel(qubit=q).rough_qubit_frequency)
            ),
        )
        for q in ds.qubit.values
    }

    return ds, fit_results


def log_fitted_results(*args, **kwargs):
    #TODO
    return


# ----------------------------------------------------------------------
# Fit interface
# ----------------------------------------------------------------------

@dataclass
class FitParameters:
    """Spectroscopy vs power fit results for a single qubit"""

    selected_power: float
    rough_qubit_frequency: float
    linewidth: float
    success: bool
