"""
Analysis utilities for the cavity f0g1 spectroscopy experiment.

The f0g1 sideband shows a DIP in the qubit state population when the drive is
on resonance (the qubit transitions from |f,0⟩ to |g,1⟩ and is left in |g⟩
after the back-swap π_ef pulse).  We therefore negate the signal before calling
peaks_dips so that the Lorentzian detection algorithm finds the correct feature.
"""
import logging
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V

try:
    from qualibration_libs.data import apply_confusion_correction_to_dataset
except ImportError:
    def apply_confusion_correction_to_dataset(ds, node):
        raise NotImplementedError("apply_confusion_correction_to_dataset not available in this qualibration_libs version")
from qualibration_libs.analysis import peaks_dips


@dataclass
class FitParameters:
    """Fit results for a single qubit's cavity f0g1 spectroscopy experiment."""
    frequency: float
    """Fitted f0g1 sideband frequency [Hz]."""
    fwhm: float
    """Fitted FWHM of the sideband dip [Hz]."""
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        log_callable(
            f"Results for qubit {q}: {status}\n"
            f"\tf0g1 frequency: {1e-9 * res['frequency']:.4f} GHz | "
            f"FWHM: {1e-3 * res['fwhm']:.1f} kHz"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Convert raw I/Q counts to Volts (if not using state discrimination)."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
        ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    else:
        ds = apply_confusion_correction_to_dataset(ds, node)
    # Add the full RF frequency as a coordinate
    sideband_drive = _get_sideband_drive(node)
    full_freq = ds.detuning + sideband_drive.RF_frequency
    ds = ds.assign_coords(full_freq=("detuning", full_freq.values))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit the cavity f0g1 sideband dip for each qubit.

    The sideband appears as a DIP in the state population (or in I_rot for raw IQ
    measurements).  We negate the signal so that peaks_dips finds it as a peak.
    """
    if node.parameters.use_state_discrimination:
        signal = ds.state
    else:
        shifts = np.abs(ds.IQ_abs - ds.IQ_abs.mean(dim="detuning")).idxmax(dim="detuning")
        angle = np.arctan2(
            ds.sel(detuning=shifts).Q - ds.Q.mean(dim="detuning"),
            ds.sel(detuning=shifts).I - ds.I.mean(dim="detuning"),
        )
        I_rot = ds.I * np.cos(angle) + ds.Q * np.sin(angle)
        ds = ds.assign({"I_rot": I_rot})
        signal = I_rot

    # Negate to turn dip → peak for the standard peaks_dips detector
    fit_vals = peaks_dips(-signal, dim="detuning", prominence_factor=3)
    ds_fit = xr.merge([ds, fit_vals])
    ds_fit, fit_results = _extract_fit_parameters(ds_fit, node)
    return ds_fit, fit_results


def _get_sideband_drive(node: QualibrationNode):
    """Return the sideband_drive channel for the cavity_transmon_pair whose
    cavity_mode_name matches node.parameters.mode_name."""
    mode_name = node.parameters.mode_name
    for pair in node.machine.cavity_transmon_pairs.values():
        if pair.cavity_mode_name == mode_name:
            return pair.sideband_drive
    raise KeyError(f"No cavity_transmon_pair with cavity_mode_name='{mode_name}'")


def _extract_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Extract the sideband frequency and FWHM from the fitted dataset."""
    sideband_drive = _get_sideband_drive(node)
    f0g1_rf_freq = sideband_drive.RF_frequency

    span_hz = node.parameters.frequency_span_in_mhz * 1e6

    fit_results = {}
    for q in fit.qubit.values:
        pos = float(fit.position.sel(qubit=q).values)
        width = float(fit.width.sel(qubit=q).values)

        freq = f0g1_rf_freq + pos
        fwhm = abs(width)

        position_found = bool(np.isfinite(pos))
        freq_in_range = bool(abs(pos) < span_hz)
        success = position_found and freq_in_range

        fit_results[q] = FitParameters(
            frequency=freq,
            fwhm=fwhm,
            success=success,
        )

    fit = fit.assign(
        success=xr.DataArray(
            [fit_results[q].success for q in fit.qubit.values],
            dims=["qubit"],
            coords={"qubit": fit.qubit},
        )
    )
    return fit, fit_results


def update_state(node: QualibrationNode, fit_results: Dict[str, FitParameters]):
    """Update the sideband_drive RF_frequency in the QUAM state."""
    sideband_drive = _get_sideband_drive(node)
    for q_name, res in fit_results.items():
        if res.success:
            sideband_drive.RF_frequency = res.frequency
            break  # single-qubit system — one update per sideband drive
