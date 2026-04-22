import logging
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from calibration_utils.error_codes import (
    ResonatorPunchOutErrorCode,
    ResonatorPunchOutCorrectiveAction,
)
# from qualibration_libs.analysis import peaks_dips  # Not used - now finding min values directly


# =========================
# Fit parameter container
# =========================

@dataclass
class FitParameters:
    success: bool
    resonator_frequency: float  # legacy: freq_shift + RF_frequency (kept for logging)
    frequency_shift: float
    optimal_power: float
    freq_low_abs: float = 0.0   # absolute resonator frequency at LOW power (use this to update state)
    error_code: int = ResonatorPunchOutErrorCode.SUCCESS  # Error diagnostic code
    corrective_action: int = ResonatorPunchOutCorrectiveAction.NONE  # Corrective action code
    action_magnitude: float = 0.0  # Magnitude of the corrective action


# =========================
# Logging
# =========================

def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for q in fit_results.keys():
        error_code = ResonatorPunchOutErrorCode(fit_results[q].get("error_code", 0))
        s_qubit = f"Results for qubit {q}: "
        s_power = f"Optimal readout power: {fit_results[q]['optimal_power']:.2f} dBm | "
        s_freq = f"Resonator frequency: {1e-9 * fit_results[q]['resonator_frequency']:.3f} GHz | "
        s_shift = f"(shift of {1e-6 * fit_results[q]['frequency_shift']:.3f} MHz)\n"
        s_error = f"Error code: {error_code.name} ({error_code.value})\n"

        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"

        log_callable(s_qubit + s_error + s_power + s_freq + s_shift)


# =========================
# Raw data processing
# =========================

def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):

    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)

    full_freq = np.array(
        [ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]]
    )

    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}

    ds = ds.assign(
        IQ_abs_norm=ds.IQ_abs / ds.IQ_abs.mean(dim="detuning")
    )

    return ds


# =========================
# Shift-based fit logic
# =========================

def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:

    powers = ds.power.values
    if len(powers) != 2:
        raise ValueError(
            "Shift-based resonator analysis requires exactly 2 power points."
        )

    # Use Python floats to avoid numpy scalar issues with xr.where below
    P_low, P_high = float(powers[0]), float(powers[-1])

    freq_low = []
    freq_high = []

    for q in node.namespace["qubits"]:
        name = q.name

        # Use isel (integer position) instead of sel (label matching) to guarantee
        # the power dimension is dropped regardless of float precision issues.
        # squeeze() removes any accidental size-1 dimensions before idxmin.
        low_power_data = ds.sel(qubit=name).isel(power=0).IQ_abs.squeeze()
        high_power_data = ds.sel(qubit=name).isel(power=-1).IQ_abs.squeeze()

        f0 = low_power_data.idxmin(dim="detuning").item()
        f1 = high_power_data.idxmin(dim="detuning").item()

        freq_low.append(f0)
        freq_high.append(f1)

    freq_low = xr.DataArray(freq_low, dims="qubit", coords={"qubit": ds.qubit})
    freq_high = xr.DataArray(freq_high, dims="qubit", coords={"qubit": ds.qubit})

    # Convention: freq_shift = freq_high - freq_low must be positive for punch-out
    # (For this device, dispersive shift moves the resonator to HIGHER frequencies at high power)
    freq_shift = freq_high - freq_low

    # Decision rule (direction-agnostic: punch-out can shift resonance either way)
    shift_threshold = node.parameters.frequency_shift_threshold_in_hz
    abs_shift = abs(freq_shift)

    # Punch-out detected when |shift| is above threshold regardless of sign
    large_shift = abs_shift > shift_threshold

    # If punch-out detected → use LOW power for readout (before Kerr nonlinearity dominates)
    optimal_power = xr.where(
        large_shift,
        P_low,
        P_high,
    )

    ds_fit = ds.assign_coords(
        freq_shift=("qubit", freq_shift.data),
        freq_low=("qubit", freq_low.data),   # low-power resonance detuning (Hz from RF_frequency)
        optimal_power=("qubit", optimal_power.data),
    )

    return _extract_relevant_fit_parameters(ds_fit, node)


# =========================
# Result extraction
# =========================

def _extract_relevant_fit_parameters(
    fit: xr.Dataset, node: QualibrationNode
):
    """Extract fit parameters and determine success based on punch-out detection."""

    # Calculate absolute resonator frequencies from detunings and base RF frequency
    base_freq = np.array(
        [q.resonator.RF_frequency for q in node.namespace["qubits"]]
    )
    freq_low_abs = fit.freq_low + base_freq        # actual low-power resonance frequency

    fit = fit.assign_coords(
        freq_low_abs=("qubit", freq_low_abs.data),
    )
    fit.freq_low_abs.attrs = {"long_name": "low-power resonator frequency", "units": "Hz"}

    # Data validity checks — direction-agnostic: accept shift in either direction
    no_nans = ~(np.isnan(fit.freq_shift.data) | np.isnan(fit.optimal_power.data))
    # |shift| must be within the swept span
    freq_in_range = np.abs(fit.freq_shift.data) < node.parameters.frequency_span_in_mhz * 1e6

    # Punch-out detection: |shift| above threshold
    shift_threshold = node.parameters.frequency_shift_threshold_in_hz
    punchout_detected = np.abs(fit.freq_shift.data) > shift_threshold

    # Success requires: valid data, shift within span, AND punch-out above threshold
    success = no_nans & freq_in_range & punchout_detected

    fit = fit.assign_coords(success=("qubit", success.data))

    # Build results dictionary with error codes
    fit_results = {}
    qubit_list = fit.qubit.values.tolist()
    for q in fit.qubit.values:
        q_idx = qubit_list.index(q)
        q_success = bool(fit.sel(qubit=q).success)
        q_shift = float(fit.freq_shift.sel(qubit=q))
        q_abs_shift = abs(q_shift)
        q_no_nans = bool(no_nans[q_idx])
        q_freq_in_range = bool(freq_in_range[q_idx])
        q_punchout_detected = bool(punchout_detected[q_idx])

        # Determine error code (checked in priority order)
        if q_success:
            error_code = ResonatorPunchOutErrorCode.SUCCESS
        elif not q_no_nans:
            error_code = ResonatorPunchOutErrorCode.INVALID_DATA
        elif not q_freq_in_range:
            error_code = ResonatorPunchOutErrorCode.INVALID_DATA
        elif q_abs_shift < 1e3:  # Essentially zero shift
            error_code = ResonatorPunchOutErrorCode.NO_SHIFT_DETECTED
        else:
            error_code = ResonatorPunchOutErrorCode.SHIFT_BELOW_THRESHOLD

        fit_results[q] = FitParameters(
            success=q_success,
            resonator_frequency=float(fit.freq_low_abs.sel(qubit=q)),
            frequency_shift=q_shift,
            optimal_power=float(fit.optimal_power.sel(qubit=q)),
            freq_low_abs=float(fit.freq_low_abs.sel(qubit=q)),
            error_code=int(error_code),
        )

    return fit, fit_results
