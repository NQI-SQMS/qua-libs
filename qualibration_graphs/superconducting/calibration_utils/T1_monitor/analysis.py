"""Analysis utilities for the T1 monitor node (node 29).

Reuses the T1 fitting logic from calibration_utils/T1/analysis.py.
The monitor node calls fit_single_dataset once per iteration and accumulates
the results externally.
"""
import logging
import numpy as np
import xarray as xr
from dataclasses import dataclass
from typing import Dict, List

from qualibration_libs.data import convert_IQ_to_V
from qualibration_libs.analysis import fit_decay_exp


@dataclass
class T1MonitorResult:
    """Accumulated T1 vs time result for a single qubit."""

    qubit: str
    t_min: List[float]
    """Elapsed time at each iteration [minutes]."""
    T1_us: List[float]
    """Fitted T1 at each iteration [µs]."""
    T1_error_us: List[float]
    """Fit uncertainty on T1 at each iteration [µs]."""


def process_raw_dataset(ds: xr.Dataset, use_state_discrimination: bool, qubits) -> xr.Dataset:
    """Convert IQ to voltage if state discrimination is not used."""
    if not use_state_discrimination:
        ds = convert_IQ_to_V(ds, qubits)
    return ds


def fit_single_dataset(ds: xr.Dataset, use_state_discrimination: bool) -> Dict[str, dict]:
    """Fit a single T1 sweep and return {qubit: {"T1_us", "T1_error_us", "success"}}.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset with dimension ``idle_time`` and either ``state`` or ``I`` variable.
    use_state_discrimination : bool
        Whether the dataset contains a ``state`` variable.

    Returns
    -------
    dict
        Per-qubit fit results with keys ``T1_us``, ``T1_error_us``, ``success``.
    """
    signal = ds.state if use_state_discrimination else ds.I
    fit_data = fit_decay_exp(signal, "idle_time")

    results = {}
    for q in ds.qubit.values:
        fd = fit_data.sel(qubit=q)
        decay = float(fd.sel(fit_vals="decay").values)
        decay_var = float(fd.sel(fit_vals="decay_decay").values)

        if decay >= 0 or not np.isfinite(decay):
            results[q] = {"T1_us": float("nan"), "T1_error_us": float("nan"), "success": False}
            continue

        tau_ns = -1.0 / decay
        tau_error_ns = tau_ns * (np.sqrt(decay_var) / abs(decay))
        success = (tau_ns > 16) and (tau_error_ns / tau_ns < 1)

        results[q] = {
            "T1_us": tau_ns * 1e-3,
            "T1_error_us": tau_error_ns * 1e-3,
            "success": success,
        }
    return results


def build_dataset(monitor_results: Dict[str, "T1MonitorResult"]) -> xr.Dataset:
    """Convert accumulated T1MonitorResult objects into an xarray Dataset.

    The returned dataset has dimensions ``(qubit, iteration)`` and variables:
    - ``T1_us``: fitted T1 [µs]
    - ``T1_error_us``: fit uncertainty on T1 [µs]
    - ``elapsed_min``: wall-clock time since experiment start [minutes]

    The dataset is suitable for storage via ``node.results["ds"]`` so that
    ``node.save()`` persists it as an H5 file for later retrieval.
    """
    qubit_names = list(monitor_results.keys())
    n_iter = max(len(r.t_min) for r in monitor_results.values())

    def _pad(lst, length):
        arr = np.full(length, float("nan"))
        arr[: len(lst)] = lst
        return arr

    T1 = np.stack([_pad(monitor_results[q].T1_us, n_iter) for q in qubit_names])
    T1_err = np.stack([_pad(monitor_results[q].T1_error_us, n_iter) for q in qubit_names])
    t_min = np.stack([_pad(monitor_results[q].t_min, n_iter) for q in qubit_names])

    return xr.Dataset(
        {
            "T1_us": xr.DataArray(
                T1,
                dims=["qubit", "iteration"],
                attrs={"long_name": "T1 relaxation time", "units": "µs"},
            ),
            "T1_error_us": xr.DataArray(
                T1_err,
                dims=["qubit", "iteration"],
                attrs={"long_name": "T1 fit uncertainty", "units": "µs"},
            ),
            "elapsed_min": xr.DataArray(
                t_min,
                dims=["qubit", "iteration"],
                attrs={"long_name": "elapsed time", "units": "min"},
            ),
        },
        coords={
            "qubit": qubit_names,
            "iteration": np.arange(n_iter),
        },
    )


def log_monitor_summary(monitor_results: Dict[str, T1MonitorResult], log_callable=None):
    """Log a summary of the T1 monitor run."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for qubit, res in monitor_results.items():
        valid = [t for t, s in zip(res.T1_us, [True] * len(res.T1_us)) if np.isfinite(t)]
        if valid:
            log_callable(
                f"T1 monitor for {qubit}: {len(valid)} successful fits, "
                f"mean T1 = {np.mean(valid):.2f} µs, "
                f"std = {np.std(valid):.2f} µs"
            )
        else:
            log_callable(f"T1 monitor for {qubit}: no successful fits.")
