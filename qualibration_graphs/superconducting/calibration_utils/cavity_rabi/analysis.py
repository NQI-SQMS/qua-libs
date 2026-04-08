"""
Analysis utilities for the cavity f0g1 Rabi experiment.

The sequence prepares the qubit in |f⟩, drives the f0g1 sideband with varying
amplitude, swaps back and measures.  The qubit state oscillates as a function of
the drive amplitude (Rabi-like fringe).  The π-amplitude is extracted from the
first minimum of the state (deepest point in the first Rabi lobe).
"""
import logging
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from qualibration_libs.analysis import fit_oscillation_decay_exp


@dataclass
class FitParameters:
    """Fit results for a single qubit's cavity Rabi experiment."""
    pi_amp: float
    """Calibrated π-pulse amplitude [V] for the f0g1 channel."""
    pi_over_2_amp: float
    """Calibrated π/2-pulse amplitude [V] for the f0g1 channel."""
    period: float
    """Rabi oscillation period in amplitude units (relative)."""
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        log_callable(
            f"Results for qubit {q}: {status}\n"
            f"\tf0g1 π-amp: {1e3 * res['pi_amp']:.2f} mV | "
            f"π/2-amp: {1e3 * res['pi_over_2_amp']:.2f} mV | "
            f"period: {res['period']:.4f} (amp units)"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Convert raw counts to Volts if not using state discrimination."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit the Rabi oscillation vs amplitude for each qubit.

    We fit a decaying cosine to the state (or I) vs amplitude sweep.  The π-pulse
    amplitude is the first minimum of the fitted curve (half-period from 0).

    Returns
    -------
    ds_fit : xr.Dataset
    fit_results : dict[str, FitParameters]
    """
    sideband_drive = _get_sideband_drive(node)
    base_amp = sideband_drive.operations[node.parameters.operation].amplitude

    if node.parameters.use_state_discrimination:
        signal = ds.state
    else:
        signal = ds.I

    # Fit oscillation (decaying cosine) vs amplitude axis
    fit = fit_oscillation_decay_exp(signal, "amp_factor")
    ds_fit = xr.merge([ds, fit.rename("fit")])

    fit_results = {}
    for q in ds.qubit.values:
        a     = float(fit.sel(qubit=q, fit_vals="a").values)
        f     = float(fit.sel(qubit=q, fit_vals="f").values)
        phi   = float(fit.sel(qubit=q, fit_vals="phi").values)
        decay = float(fit.sel(qubit=q, fit_vals="decay").values)

        if f <= 0 or not np.isfinite(f):
            fit_results[q] = FitParameters(
                pi_amp=float("nan"),
                pi_over_2_amp=float("nan"),
                period=float("nan"),
                success=False,
            )
            continue

        # Period in amplitude-factor units
        period_af = 1.0 / f
        # π-amp: half a period from 0
        pi_amp_factor = 0.5 * period_af
        pi_over_2_amp_factor = 0.25 * period_af

        pi_amp = pi_amp_factor * base_amp
        pi_over_2_amp = pi_over_2_amp_factor * base_amp

        # Success: pi_amp within the swept range
        max_amp = node.parameters.max_amp_factor * base_amp
        success = (
            np.isfinite(pi_amp)
            and 0 < pi_amp <= max_amp
        )

        fit_results[q] = FitParameters(
            pi_amp=pi_amp,
            pi_over_2_amp=pi_over_2_amp,
            period=period_af * base_amp,
            success=success,
        )

    return ds_fit, fit_results


def _get_sideband_drive(node: QualibrationNode):
    """Return the sideband_drive channel for the cavity_transmon_pair whose
    cavity_mode_name matches node.parameters.mode_name."""
    mode_name = node.parameters.mode_name
    for pair in node.machine.cavity_transmon_pairs.values():
        if pair.cavity_mode_name == mode_name:
            return pair.sideband_drive
    raise KeyError(f"No cavity_transmon_pair with cavity_mode_name='{mode_name}'")
