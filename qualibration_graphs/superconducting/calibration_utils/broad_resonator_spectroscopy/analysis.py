import logging
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import numpy as np
import xarray as xr
from scipy.signal import find_peaks

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from qualibration_libs.analysis import peaks_dips


@dataclass
class FitParameters:
    """Stores the relevant resonator spectroscopy experiment fit parameters for a single qubit"""

    frequencies: List[float]
    success: bool


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
        s_qubit = f"Results for qubit {q}: "
        s_freq = "Detected resonator frequencies: "
        for f in fit_results[q]['frequencies']:
            s_freq += f"{1e-9 * f:.3f} GHz, "
        s_freq += "\t"    
        # s_freq = f"\tResonator frequency: {1e-9 * fit_results[q]['frequencies']:.3f} GHz | "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_freq)

def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    full_freq = np.array([ds.detuning + q.resonator.frequency_converter_up.LO_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the T1 relaxation time for each qubit according to ``a * np.exp(t * decay) + offset``.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node : QualibrationNode
        The QUAlibrate node.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """
    # Fit the resonator line
    # fit_results = peaks_dips(ds.IQ_abs, "detuning")

    # Find dips in magnitude (IQ_abs)
    fit_results = peaks_dips_all(ds.IQ_abs,
                                 "detuning",
                                 prominence_factor=node.parameters.peak_prominence,
                                 height=node.parameters.peak_height,
                                 threshold=node.parameters.peak_threshold,
                                 min_distance_from_zero=node.parameters.min_frequency_distance_from_zero_mhz * 1e6)

    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(fit_results, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Select the dip closest to the initial resonator frequency for each qubit."""

    fit.attrs = {"long_name": "frequency", "units": "Hz"}

    qubits = node.namespace["qubits"]
    qubit_names = fit.positions.coords["qubit"].values

    # Initial guess frequencies (one per qubit)
    f_guess = np.array(
        [q.resonator.frequency_converter_up.LO_frequency for q in qubits]
    )

    # Broadcast for xarray math
    f_guess_da = xr.DataArray(
        f_guess, coords={"qubit": qubit_names}, dims=("qubit",)
    )

    # Absolute frequencies of all detected dips
    res_freq_da = fit.positions + f_guess_da
    res_freq_da.attrs = {"long_name": "resonator frequency", "units": "Hz"}

    # Distance from initial guess
    dist_da = np.abs(res_freq_da - f_guess_da)

    # Mask invalid entries
    dist_da = dist_da.where(~np.isnan(res_freq_da))

    # Index of closest dip per qubit
    closest_idx = dist_da.argmin(dim="dip")

    # Selected resonator frequency per qubit
    selected_freq = res_freq_da.isel(dip=closest_idx)

    # Success: at least one dip detected
    success = ~np.isnan(selected_freq)

    # Attach to dataset
    fit = fit.assign_coords(
        res_freq_all=res_freq_da,
        res_freq=selected_freq,
        success=("qubit", success.values),
    )

    fit.res_freq.attrs = {"long_name": "selected resonator frequency", "units": "Hz"}

    # Build fit_results dict
    fit_results = {}
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


def peaks_dips_all(
    da: xr.DataArray,
    dim: str,
    prominence_factor: float = 5.0,
    min_distance: int = 1,
    remove_baseline: bool = True,
    height: Optional[float] = None,
    threshold: Optional[float] = None,
    min_distance_from_zero: Optional[float] = None,
) -> xr.Dataset:
    """
    Find all dips (local minima) in the data array along dimension `dim`.

    This is a lightweight alternative to `peaks_dips` which returns only the
    most prominent peak. Here we return all detected dip positions for each
    entry along the leading dimension (usually `qubit`). The returned dataset
    contains a single DataArray `positions` with dims (`qubit`, `dip`) where
    missing entries are filled with `nan`.

    Parameters
    ----------
    da : xr.DataArray
        Input data array containing the signal to search for dips.
    dim : str
        The coordinate name along which to search for dips (e.g., 'detuning').
    prominence_factor : float
        Multiplier for the estimated noise (std) to set the prominence threshold.
    min_distance : int
        Minimum number of samples between detected dips.
    remove_baseline : bool
        If True, subtract a rolling baseline before detection (helps with slow drifts).
    height : float or None
        Passed to `scipy.signal.find_peaks` as the `height` argument (see
        scipy docs). Use to require a minimum peak height.
    threshold : float or None
        Passed to `scipy.signal.find_peaks` as the `threshold` argument.
    min_distance_from_zero : float or None
        If provided, filter out detected dips within this distance (in same units as `dim`)
        from zero on the dimension axis.

    Returns
    -------
    xr.Dataset
        Dataset with a single DataArray `positions` of shape (n_qubits, n_dips_max)
        giving the coordinate values (along `dim`) of each detected dip.
    """
    if dim not in da.coords:
        raise KeyError(f"Coordinate '{dim}' not found in DataArray")

    # Prepare coordinate axis values
    x_vals = da.coords[dim].values

    # If we have a 'qubit' dimension, iterate over it; otherwise handle single series
    if "qubit" in da.dims:
        qubits = list(da.coords["qubit"].values)
        series_list = [da.sel(qubit=q).values.squeeze() for q in qubits]
    else:
        qubits = ["_single"]
        series_list = [da.values.squeeze()]

    positions_list = []
    for y in series_list:
        # ensure 1D
        y = np.asarray(y).squeeze()
        if y.ndim != 1 or y.size == 0:
            positions_list.append(np.array([], dtype=float))
            continue

        # optionally remove slow baseline using a rolling mean
        y_proc = y.copy()
        if remove_baseline and len(y) >= 3:
            try:
                # estimate baseline with simple moving average of window 10
                window = min(10, max(1, len(y) // 10))
                kernel = np.ones(window) / window
                baseline = np.convolve(y, kernel, mode="same")
                y_proc = y - baseline
            except Exception:
                y_proc = y

        # estimate noise and set prominence threshold
        try:
            # rolling mean estimate of noise similar to peaks_dips
            if len(y) >= 10:
                roll = np.convolve(y, np.ones(10) / 10, mode="same")
                sigma = float(np.std(y - roll))
            else:
                sigma = float(np.std(y_proc)) if y_proc.size > 0 else 0.0
        except Exception:
            sigma = float(np.std(y_proc)) if y_proc.size > 0 else 0.0

        if prominence_factor:
            prominence = max(0.0, prominence_factor * sigma)
        else:
            prominence = None

        # find minima by inverting the signal and using scipy.find_peaks
        inv = -1.0 * y_proc
        try:
            peaks_idx, props = find_peaks(
                inv, prominence=prominence, distance=min_distance, height=height, threshold=threshold
            )
        except Exception:
            peaks_idx = np.array([], dtype=int)

        if peaks_idx.size > 0:
            # map indices to coordinate values
            idx_clipped = np.clip(peaks_idx, 0, len(x_vals) - 1)
            pos = x_vals[idx_clipped]

            # filter out dips too close to zero if threshold provided
            if min_distance_from_zero is not None:
                pos = pos[np.abs(pos) >= min_distance_from_zero]
        else:
            pos = np.array([], dtype=float)

        positions_list.append(pos)

    # Build rectangular array padded with nan
    max_len = max((len(p) for p in positions_list), default=0)
    data = np.full((len(positions_list), max_len), np.nan, dtype=float)
    for i, p in enumerate(positions_list):
        if p.size > 0:
            data[i, : p.size] = p

    # Create coords
    dip_idx = np.arange(max_len)
    coords = {"dip": dip_idx}
    if "qubit" in da.dims:
        coords["qubit"] = da.coords["qubit"].values
        dims = ("qubit", "dip")
    else:
        coords["qubit"] = qubits
        dims = ("qubit", "dip")

    positions_da = xr.DataArray(data, coords=coords, dims=dims)
    positions_da.attrs = {"long_name": "dip positions", "units": da.coords[dim].attrs.get("units", "")}

    return xr.Dataset({"positions": positions_da})