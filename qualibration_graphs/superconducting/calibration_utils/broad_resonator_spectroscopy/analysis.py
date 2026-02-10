import logging
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import numpy as np
import xarray as xr
from scipy.signal import find_peaks

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class FitParameters:
    """Stores the relevant resonator spectroscopy experiment fit parameters for a single qubit"""
    frequencies: List[float]
    success: bool


# -----------------------------------------------------------------------------
# Logging utilities
# -----------------------------------------------------------------------------

def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_freq = "Detected resonator frequencies: "
        for f in fit_results[q]["frequencies"]:
            s_freq += f"{1e-9 * f:.3f} GHz, "
        s_freq += "\t"

        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"

        log_callable(s_qubit + s_freq)


# -----------------------------------------------------------------------------
# Dataset preprocessing
# -----------------------------------------------------------------------------

def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)

    full_freq = np.array(
        [
            ds.detuning + q.resonator.RF_frequency
            for q in node.namespace["qubits"]
        ]
    )

    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


# -----------------------------------------------------------------------------
# Main fitting entry point
# -----------------------------------------------------------------------------

def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Detect resonator dips and select the one closest to the LO frequency.
    """

    fit_results = peaks_dips_all(
        ds.IQ_abs,
        dim="detuning",
        prominence_factor=node.parameters.peak_prominence,
        height=node.parameters.peak_height,
        threshold=node.parameters.peak_threshold,
    )

    fit_data, fit_results = _extract_relevant_fit_parameters(fit_results, node)
    return fit_data, fit_results


# -----------------------------------------------------------------------------
# Dip selection logic
# -----------------------------------------------------------------------------

def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """
    Select the dip closest to the initial resonator frequency for each qubit.
    """

    fit.attrs = {"long_name": "frequency", "units": "Hz"}

    qubits = node.namespace["qubits"]
    qubit_names = fit.positions.coords["qubit"].values

    # Initial guess: LO frequency per qubit
    f_guess = np.array(
        [q.resonator.RF_frequency for q in qubits]
    )

    f_guess_da = xr.DataArray(
        f_guess, coords={"qubit": qubit_names}, dims=("qubit",)
    )

    # Absolute frequencies of all detected dips
    res_freq_da = fit.positions + f_guess_da
    res_freq_da.attrs = {"long_name": "resonator frequency", "units": "Hz"}

    # Distance from initial guess
    dist_da = abs(res_freq_da - f_guess_da)
    dist_da = dist_da.where(~np.isnan(res_freq_da))

    # Does each qubit have at least one dip?
    has_any_dip = dist_da.notnull().any(dim="dip")

    # SAFE argmin (critical fix)
    safe_dist = dist_da.fillna(np.inf)
    closest_idx = safe_dist.argmin(dim="dip")

    # Selected resonator frequency
    selected_freq = res_freq_da.isel(dip=closest_idx)
    selected_freq = selected_freq.where(has_any_dip)

    # Success flag
    success = has_any_dip


    fit = fit.assign_coords(
        res_freq_all=res_freq_da,
        res_freq=selected_freq,
        success=("qubit", success.values),
    )

    fit.res_freq.attrs = {
        "long_name": "selected resonator frequency",
        "units": "Hz",
    }

    # Build results dict
    fit_results: Dict[str, FitParameters] = {}
    for i, qname in enumerate(qubit_names):
        if success.values[i]:
            fit_results[qname] = FitParameters(
                frequencies=[float(selected_freq.sel(qubit=qname).values)],
                success=True,
            )
        else:
            fit_results[qname] = FitParameters(
                frequencies=[],
                success=False,
            )

    return fit, fit_results


# -----------------------------------------------------------------------------
# Dip detection (ALL dips)
# -----------------------------------------------------------------------------

def peaks_dips_all(
    da: xr.DataArray,
    dim: str,
    prominence_factor: float = 5.0,
    remove_baseline: bool = True,
    height: Optional[float] = None,
    threshold: Optional[float] = None,
    width: Optional[Tuple[float, float]] = None,
) -> xr.Dataset:
    """
    Find all dips (local minima) in `da` along dimension `dim`.

    Returns a Dataset with one DataArray:
        positions(qubit, dip)
    """

    if dim not in da.coords:
        raise KeyError(f"Coordinate '{dim}' not found in DataArray")

    x_vals = da.coords[dim].values

    if "qubit" in da.dims:
        qubits = list(da.coords["qubit"].values)
        series_list = [da.sel(qubit=q).values.squeeze() for q in qubits]
    else:
        qubits = ["_single"]
        series_list = [da.values.squeeze()]

    positions_list = []

    for y in series_list:
        y = np.asarray(y).squeeze()
        if y.ndim != 1 or y.size == 0:
            positions_list.append(np.array([], dtype=float))
            continue

        # Remove slow baseline if requested
        y_proc = y
        if remove_baseline and len(y) >= 3:
            try:
                window = min(10, max(1, len(y) // 10))
                kernel = np.ones(window) / window
                baseline = np.convolve(y, kernel, mode="same")
                y_proc = y - baseline
            except Exception:
                y_proc = y

        # Noise estimate → prominence
        try:
            if len(y) >= 10:
                roll = np.convolve(y, np.ones(10) / 10, mode="same")
                sigma = float(np.std(y - roll))
            else:
                sigma = float(np.std(y_proc))
        except Exception:
            sigma = float(np.std(y_proc))

        prominence = (
            max(0.0, prominence_factor * sigma)
            if prominence_factor
            else None
        )

        # Find minima via inverted signal
        inv = -y_proc
        try:
            peaks_idx, _ = find_peaks(
                inv,
                prominence=prominence,
                height=height,
                threshold=threshold,
                width=width,
            )
        except Exception:
            peaks_idx = np.array([], dtype=int)

        if peaks_idx.size > 0:
            idx_clipped = np.clip(peaks_idx, 0, len(x_vals) - 1)
            pos = x_vals[idx_clipped]
        else:
            pos = np.array([], dtype=float)

        positions_list.append(pos)

    # Rectangular array padded with NaN
    max_len = max((len(p) for p in positions_list), default=0)
    data = np.full((len(positions_list), max_len), np.nan)

    for i, p in enumerate(positions_list):
        data[i, : len(p)] = p

    coords = {
        "qubit": qubits,
        "dip": np.arange(max_len),
    }

    positions_da = xr.DataArray(
        data,
        coords=coords,
        dims=("qubit", "dip"),
        attrs={
            "long_name": "dip positions",
            "units": da.coords[dim].attrs.get("units", ""),
        },
    )

    return xr.Dataset({"positions": positions_da})
