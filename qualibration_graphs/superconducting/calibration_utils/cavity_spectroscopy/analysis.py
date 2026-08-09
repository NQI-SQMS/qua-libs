"""
Analysis utilities for the cavity spectroscopy experiment.

The sequence sweeps the cavity drive frequency and probes photon occupation via
a narrow-bandwidth qubit pulse (selective_x180). When the cavity drive is on
resonance, photons are deposited; the dispersive coupling shifts the qubit
frequency by χ per photon, so the selective probe pulse goes off-resonant and
the qubit excitation probability drops.

The result is a DIP in qubit state vs. cavity drive detuning centred on the
bare cavity resonance. We negate the signal before calling peaks_dips so that
the Lorentzian detection algorithm finds the correct feature.
"""
import logging
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V

from calibration_utils.shared import apply_confusion_matrix_correction


def apply_confusion_correction_to_dataset(ds, node):
    return apply_confusion_matrix_correction(ds, node.namespace["qubits"])
from qualibration_libs.analysis import peaks_dips


@dataclass
class FitParameters:
    """Fit results for a single qubit's cavity spectroscopy experiment."""
    frequency: float
    """Fitted cavity resonance frequency [Hz]."""
    fwhm: float
    """Fitted FWHM of the resonance dip [Hz]."""
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        log_callable(
            f"Results for qubit {q}: {status}\n"
            f"\tCavity resonance: {1e-9 * res['frequency']:.6f} GHz | "
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
    cavity_mode = _get_cavity_mode(node)
    full_freq = ds.detuning + cavity_mode.cavity_mode_drive.RF_frequency
    ds = ds.assign_coords(full_freq=("detuning", full_freq.values))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit the cavity resonance dip for each qubit.

    The cavity resonance appears as a DIP in the state population (or in I_rot
    for raw IQ measurements).  We negate the signal so that peaks_dips finds
    it as a peak.

    Returns
    -------
    ds_fit : xr.Dataset
        Dataset augmented with fit variables (position, width, …).
    fit_results : dict
        Per-qubit :class:`FitParameters`.
    """
    if node.parameters.use_state_discrimination:
        signal = ds.state
    else:
        # Rotate IQ to align signal along I axis
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


def _get_cavity_mode(node: QualibrationNode):
    """Return the CavityMode object for the cavity_name parameter."""
    cavity_name = node.parameters.cavity_name
    for cav in node.machine.cavities.values():
        mode = getattr(cav, cavity_name, None)
        if mode is not None:
            return mode
    raise KeyError(f"Cavity mode '{cavity_name}' not found in machine.cavities")


def _extract_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Extract the cavity resonance frequency and FWHM from the fitted dataset."""
    cavity_mode = _get_cavity_mode(node)
    rf_freq = cavity_mode.cavity_mode_drive.RF_frequency

    span_hz = node.parameters.frequency_span_in_mhz * 1e6

    fit_results = {}
    for q in fit.qubit.values:
        pos = float(fit.position.sel(qubit=q).values)
        width = float(fit.width.sel(qubit=q).values)

        freq = rf_freq + pos
        fwhm = abs(width)

        position_found = np.isfinite(pos)
        freq_in_range = abs(pos) < span_hz
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
    """Update the cavity drive RF_frequency in the QUAM state."""
    cavity_name = node.parameters.cavity_name
    for cav in node.machine.cavities.values():
        mode = getattr(cav, cavity_name, None)
        if mode is None:
            continue
        for q_name, res in fit_results.items():
            if res.success:
                mode.cavity_mode_drive.RF_frequency = res.frequency
                break  # single-qubit system — one update per cavity mode
