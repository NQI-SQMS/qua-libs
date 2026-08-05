"""Analysis utilities for the cavity active reset test.

The sequence prepares Fock |1⟩, drives the sideband reset for a variable
flat-top duration, then reads out the remaining cavity population via the
inverse sideband.  P(0) = 1 − P(|e⟩) is reported as a function of reset
duration.  No formal fit is performed; we report the duration at which
P(0) first exceeds 0.95.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V

from calibration_utils.shared import apply_confusion_matrix_correction


def apply_confusion_correction_to_dataset(ds, node):
    return apply_confusion_matrix_correction(ds, node.namespace["qubits"])


@dataclass
class CavityResetFit:
    """Summary metrics for a single qubit's cavity reset test."""

    t95_ns: float
    """Reset flat-top duration [ns] at which P(0) first exceeds 0.95.
    Set to NaN if P0_max never reaches 0.95."""

    P0_max: float
    """Maximum P(0) = 1 − P(|e⟩) observed over the sweep."""

    success: bool
    """True if P0_max > 0.9."""


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, result in fit_results.items():
        status = "SUCCESS" if result["success"] else "FAIL"
        t95 = result["t95_ns"]
        t95_str = f"{t95 * 1e-3:.1f} µs" if np.isfinite(t95) else "never reached"
        log_callable(
            f"Results for qubit {q}: {status}\n"
            f"\tP0_max = {result['P0_max']:.3f}\n"
            f"\tt(P0>0.95) = {t95_str}"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert I/Q to Volts and compute P0 = 1 − state (or 1 − I_rot)."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
        ds["P0"] = 1.0 - ds["I"]
    else:
        if node.parameters.use_confusion_matrix_correction:
            ds = apply_confusion_correction_to_dataset(ds, node)
        ds["P0"] = 1.0 - ds["state"]
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, CavityResetFit]]:
    """Extract t95 and P0_max for each qubit from the P0 vs reset_duration curve."""
    fit_results = {}
    for q in ds.qubit.values:
        P0 = ds["P0"].sel(qubit=q).values
        t_ns = ds["reset_duration"].values

        P0_max = float(np.nanmax(P0))

        above = np.where(P0 >= 0.95)[0]
        t95 = float(t_ns[above[0]]) if len(above) > 0 else float("nan")

        success = P0_max > 0.9
        fit_results[q] = CavityResetFit(t95_ns=t95, P0_max=P0_max, success=success)

    return ds, fit_results
