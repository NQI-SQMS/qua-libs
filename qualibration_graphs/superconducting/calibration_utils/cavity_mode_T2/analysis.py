"""
Analysis utilities for the cavity mode T2 Ramsey experiment.

The sequence creates a Fock-state superposition (|0>+|1>)/sqrt(2) in the cavity,
waits a variable time tau with an artificial detuning imprinted via frame rotation,
then retrieves and measures.  The qubit state oscillates as a decaying sinusoid
from which T2ramsey of the cavity mode is extracted.
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
        raise NotImplementedError("apply_confusion_correction_to_dataset not available in this qualibration_libs version")
from qualibration_libs.analysis import fit_oscillation_decay_exp


@dataclass
class FitParameters:
    """Fit results for a single qubit's cavity mode T2 Ramsey experiment."""
    T2ramsey_ns: float
    """Cavity mode T2 Ramsey time [ns]."""
    freq_offset_hz: float
    """Fitted oscillation frequency offset from the applied detuning [Hz]."""
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, result in fit_results.items():
        status = "SUCCESS" if result["success"] else "FAIL"
        T2_us = result["T2ramsey_ns"] * 1e-3
        log_callable(
            f"Results for qubit {q}: {status}\n"
            f"\tCavity T2ramsey: {T2_us:.1f} us | "
            f"freq offset: {result['freq_offset_hz'] * 1e-3:.3f} kHz"
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
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit decaying sinusoid in state (or I) vs idle_time for each qubit."""
    if node.parameters.use_state_discrimination:
        fit_vals = fit_oscillation_decay_exp(ds.state, "idle_time")
    else:
        fit_vals = fit_oscillation_decay_exp(ds.I, "idle_time")

    ds_fit = xr.merge([ds, fit_vals.rename("fit")])

    fit_results = {}
    for q in ds_fit.qubit.values:
        fit_q = ds_fit.fit.sel(qubit=q)
        decay = float(fit_q.sel(fit_vals="decay").item())
        f = float(fit_q.sel(fit_vals="f").item())
        a = float(fit_q.sel(fit_vals="a").item())

        if np.isfinite(decay) and decay > 0 and np.isfinite(a) and a > 0:
            T2ramsey_ns = float(decay) * 1e9  # decay is in seconds from fit_oscillation_decay_exp
            # freq_offset = difference between fitted freq and applied detuning
            freq_offset_hz = float(f) * 1e9 - node.parameters.ramsey_detuning_hz
            success = True
        else:
            T2ramsey_ns = float("nan")
            freq_offset_hz = float("nan")
            success = False

        fit_results[q] = FitParameters(
            T2ramsey_ns=T2ramsey_ns,
            freq_offset_hz=freq_offset_hz,
            success=success,
        )

    return ds_fit, fit_results
