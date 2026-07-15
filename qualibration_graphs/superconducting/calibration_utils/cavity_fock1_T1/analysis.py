"""
Analysis utilities for the cavity Fock |1⟩ T1 experiment.

The sequence prepares the cavity Fock |1⟩ state via the D-SNAP-D protocol
(two displacements bracketing a SNAP₀(2π) gate), waits a variable time tau,
then reads out via PNRS: a selective qubit π-pulse at the n=1 dressed qubit
frequency maps P(n=1) → P(|e⟩).

The signal P_e(t) follows a simple exponential decay from which T1 of the
cavity Fock |1⟩ state is extracted.
"""
import logging
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
try:
    from qualibration_libs.data import apply_confusion_correction_to_dataset
except ImportError:
    def apply_confusion_correction_to_dataset(ds, node):
        raise NotImplementedError(
            "apply_confusion_correction_to_dataset not available in this qualibration_libs version"
        )
from qualibration_libs.analysis import fit_decay_exp


@dataclass
class Fock1T1Fit:
    """Fit results for a single qubit's Fock |1⟩ T1 experiment."""

    T1_ns: float
    """Cavity Fock |1⟩ lifetime T1 [ns]."""
    T1_error_ns: float
    """1-σ uncertainty on T1 [ns]."""
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, result in fit_results.items():
        status = "SUCCESS" if result["success"] else "FAIL"
        T1_us = result["T1_ns"] * 1e-3
        T1_err_us = result["T1_error_ns"] * 1e-3
        log_callable(
            f"Results for qubit {q}: {status}\n"
            f"\tFock |1⟩ T1 = {T1_us:.1f} ± {T1_err_us:.1f} µs"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert I/Q to Volts if not using state discrimination."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    else:
        ds = apply_confusion_correction_to_dataset(ds, node)
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, Fock1T1Fit]]:
    """Fit simple exponential decay in state (or I) vs idle_time for each qubit.

    Model: P_e(t) = a * exp(decay * t) + offset   where decay < 0.
    T1 = -1 / decay.
    """
    if node.parameters.use_state_discrimination:
        fit_vals = fit_decay_exp(ds.state, "idle_time")
    else:
        fit_vals = fit_decay_exp(ds.I, "idle_time")

    ds_fit = xr.merge([ds, fit_vals.rename("fit")])

    fit_results = {}
    for q in ds_fit.qubit.values:
        fit_q = ds_fit.fit.sel(qubit=q)
        decay = float(fit_q.sel(fit_vals="decay").item())
        decay_var = float(fit_q.sel(fit_vals="decay_decay").item())

        if np.isfinite(decay) and decay < 0:
            T1_ns = -1.0 / decay  # decay is in 1/ns (idle_time axis is in ns)
            T1_error_ns = T1_ns * (np.sqrt(max(decay_var, 0.0)) / abs(decay))
            success = bool(T1_ns > 16.0 and (T1_error_ns / T1_ns) < 1.0)
        else:
            T1_ns = float("nan")
            T1_error_ns = float("nan")
            success = False

        fit_results[q] = Fock1T1Fit(
            T1_ns=T1_ns,
            T1_error_ns=T1_error_ns,
            success=success,
        )

    return ds_fit, fit_results
