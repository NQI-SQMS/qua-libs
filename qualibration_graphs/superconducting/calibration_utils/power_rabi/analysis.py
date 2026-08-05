import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation
from qualibration_libs.data import convert_IQ_to_V
from calibration_utils.shared import apply_confusion_matrix_correction


def apply_confusion_correction_to_dataset(ds, node):
    return apply_confusion_matrix_correction(ds, node.namespace["qubits"])
from calibration_utils.error_codes import PowerRabiErrorCode, PowerRabiCorrectiveAction

# Thresholds for period-count classification (in adaptive mode)
_TOO_MANY_PERIODS_THRESHOLD = 2.0   # more than 2 full oscillations → base amp too high
_TOO_FEW_PERIODS_THRESHOLD = 0.5    # less than 0.5 oscillations  → base amp too low


@dataclass
class FitParameters:
    """Stores the relevant power Rabi experiment fit parameters for a single qubit."""

    opt_amp_prefactor: float
    opt_amp: float
    operation: str
    success: bool
    num_periods: float = float("nan")
    """Number of Rabi oscillation periods detected across the amplitude sweep."""
    chi2: float = float("nan")
    """Residual chi-squared: SS_res / ((N-4)·a²); chi2 > 2 → NO_OSCILLATION."""
    error_code: int = int(PowerRabiErrorCode.SUCCESS)
    """PowerRabiErrorCode: classification of the calibration result."""
    corrective_action: int = int(PowerRabiCorrectiveAction.NONE)
    """PowerRabiCorrectiveAction: action applied (or to be applied) to fix the issue."""
    action_magnitude: float = 0.0
    """Magnitude of the corrective action (interpretation depends on corrective_action)."""
    pulse_length_ns: float = float("nan")
    """Pulse length used during this calibration run [ns]. Saved to QUAM on success."""


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
        s_amp = (
            f"The calibrated {fit_results[q]['operation']} amplitude: "
            f"{1e3 * fit_results[q]['opt_amp']:.2f} mV "
            f"(x{fit_results[q]['opt_amp_prefactor']:.2f})\n "
        )
        num_periods = fit_results[q].get("num_periods", float("nan"))
        if np.isfinite(num_periods):
            s_amp += f"Rabi periods in sweep: {num_periods:.2f}\n "
        chi2_val = fit_results[q].get("chi2", float("nan"))
        if np.isfinite(chi2_val):
            s_amp += f"Residual chi2: {chi2_val:.3f}\n "
        error_code = fit_results[q].get("error_code", 0)
        if error_code != 0:
            s_amp += f"Error code: {PowerRabiErrorCode(error_code).name}\n "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_amp)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    else:
        ds = apply_confusion_correction_to_dataset(ds, node)

    _is_ef = not hasattr(node.parameters, "operation")
    if _is_ef:
        full_amp = np.array([ds.amp_prefactor * q.xy.operations["EF_x180"].amplitude for q in node.namespace["qubits"]])
    else:
        full_amp = np.array(
            [ds.amp_prefactor * q.xy.operations[node.parameters.operation].amplitude for q in node.namespace["qubits"]]
        )
    ds = ds.assign_coords(full_amp=(["qubit", "amp_prefactor"], full_amp))
    ds.full_amp.attrs = {"long_name": "pulse amplitude", "units": "V"}

    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the qubit frequency and FWHM for each qubit in the dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node_parameters : Parameters
        Parameters related to the node, including whether state discrimination is used.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """
    max_pulses = getattr(node.parameters, "max_number_pulses_per_sweep", 1)
    _is_ef = not hasattr(node.parameters, "max_number_pulses_per_sweep")
    operation = getattr(node.parameters, "operation", "EF_x180")
    if max_pulses == 1:
        ds_fit = ds if "nb_of_pulses" not in ds.dims else ds.sel(nb_of_pulses=1)
        # Fit the power Rabi oscillations
        if node.parameters.use_state_discrimination:
            fit_vals = fit_oscillation(ds_fit.state, "amp_prefactor")
        else:
            fit_vals = fit_oscillation(ds_fit.I, "amp_prefactor")

        ds_fit = xr.merge([ds, fit_vals.rename("fit")])
    else:
        ds_fit = ds
        # Get the average along the number of pulses axis to identify the best pulse amplitude
        if node.parameters.use_state_discrimination:
            ds_fit["data_mean"] = ds.state.mean(dim="nb_of_pulses")
        else:
            ds_fit["data_mean"] = ds.I.mean(dim="nb_of_pulses")
        if (ds.nb_of_pulses.data[0] % 2 == 0 and operation == "x180") or (
            ds.nb_of_pulses.data[0] % 2 != 0 and operation != "x180"
        ):
            ds_fit["opt_amp_prefactor"] = ds_fit["data_mean"].idxmin(dim="amp_prefactor")
        else:
            ds_fit["opt_amp_prefactor"] = ds_fit["data_mean"].idxmax(dim="amp_prefactor")

    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _compute_chi2(fit: xr.Dataset) -> xr.DataArray:
    """
    Compute the residual chi-squared for each qubit's oscillation fit.

        chi2 = SS_res / ((N - 4) * a_fit²)

    where SS_res is the sum of squared residuals between I and the
    reconstructed sinusoid, N is the number of sweep points, and a_fit is
    the fitted oscillation amplitude.

    Interpretation:
        chi2 ≤ 2  →  oscillation detected (amplitude ≥ residual RMS / √2)
        chi2 > 2  →  residuals dominate; no Rabi oscillation (NO_OSCILLATION)
    """
    from qualibration_libs.analysis import oscillation as _oscillation_fn

    x = fit.amp_prefactor.values  # (N,)
    N = len(x)
    P = 4  # free parameters: a, f, phi, offset

    chi2_values = []
    for q in fit.qubit.values:
        fit_q = fit.sel(qubit=q)

        # Use state discrimination data when available, else fall back to I
        if "I" in fit_q:
            i_data = fit_q.I
            if "nb_of_pulses" in i_data.dims:
                i_data = i_data.sel(nb_of_pulses=1)
        else:
            i_data = fit_q.state
            if "nb_of_pulses" in i_data.dims:
                i_data = i_data.sel(nb_of_pulses=1)
        data = i_data.values  # (N,)

        a = float(fit_q.fit.sel(fit_vals="a").item())
        f = float(fit_q.fit.sel(fit_vals="f").item())
        phi = float(fit_q.fit.sel(fit_vals="phi").item())
        offset = float(fit_q.fit.sel(fit_vals="offset").item())

        if not np.isfinite(a) or a <= 0.0 or N <= P:
            chi2_values.append(float("inf"))
            continue

        fitted = _oscillation_fn(x, a, f, phi, offset)
        SS_res = float(np.sum((data - fitted) ** 2))
        chi2_values.append(SS_res / ((N - P) * a ** 2))

    return xr.DataArray(chi2_values, dims=["qubit"], coords={"qubit": fit.qubit})


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    max_pulses = getattr(node.parameters, "max_number_pulses_per_sweep", 1)
    _is_ef = not hasattr(node.parameters, "operation")
    operation = getattr(node.parameters, "operation", "EF_x180" if _is_ef else "x180")

    if max_pulses == 1:
        # Process the fit parameters to get the right amplitude
        # Generate the fitted sinusoid and find its first maximum directly
        from qualibration_libs.analysis import oscillation

        # Reconstruct fitted curve for each qubit and find maximum
        factors = []
        for q in fit.qubit.values:
            fit_q = fit.sel(qubit=q)
            amp_q = fit_q.fit.sel(fit_vals="a").item()
            freq_q = fit_q.fit.sel(fit_vals="f").item()
            phase_q = fit_q.fit.sel(fit_vals="phi").item()
            offset_q = fit_q.fit.sel(fit_vals="offset").item()

            # Generate fitted curve for this qubit
            fitted_curve_q = oscillation(
                fit.amp_prefactor.values,
                amp_q,
                freq_q,
                phase_q,
                offset_q
            )

            # Find the FIRST pi-pulse extremum — the smallest amplitude that
            # produces a pi rotation, not a 3pi/5pi harmonic.
            # np.argmax/argmin would return the global extremum, which due to
            # floating-point sampling may be the LAST peak when num_periods > 1.
            # scipy find_peaks locates local extrema and we take the first one.
            from scipy.signal import find_peaks as _find_peaks

            starts_high = fitted_curve_q[0] > offset_q
            if starts_high:
                # Descending start: pi pulse at the first local minimum
                troughs, _ = _find_peaks(-fitted_curve_q)
                extremum_idx = int(troughs[0]) if len(troughs) > 0 else int(np.argmin(fitted_curve_q))
            else:
                # Ascending start: pi pulse at the first local maximum
                peaks, _ = _find_peaks(fitted_curve_q)
                extremum_idx = int(peaks[0]) if len(peaks) > 0 else int(np.argmax(fitted_curve_q))
            factor_q = fit.amp_prefactor.values[extremum_idx]
            factors.append(factor_q)

        factor = xr.DataArray(factors, dims="qubit", coords={"qubit": fit.qubit})
        fit = fit.assign({"opt_amp_prefactor": factor})
        fit.opt_amp_prefactor.attrs = {
            "long_name": "factor to get a pi pulse",
            "units": "Hz",
        }
        if _is_ef:
            current_amps = xr.DataArray(
                [q.xy.operations["EF_x180"].amplitude for q in node.namespace["qubits"]],
                coords=dict(qubit=fit.qubit.data),
            )
        else:
            current_amps = xr.DataArray(
                [q.xy.operations[node.parameters.operation].amplitude for q in node.namespace["qubits"]],
                coords=dict(qubit=fit.qubit.data),
            )
        opt_amp = factor * current_amps
        fit = fit.assign({"opt_amp": opt_amp})
        fit.opt_amp.attrs = {"long_name": "x180 pulse amplitude", "units": "V"}

    else:
        current_amps = xr.DataArray(
            [q.xy.operations[operation].amplitude for q in node.namespace["qubits"]],
            coords=dict(qubit=fit.qubit.data),
        )
        fit = fit.assign({"opt_amp": fit.opt_amp_prefactor * current_amps})
        fit.opt_amp.attrs = {
            "long_name": f"{operation} pulse amplitude",
            "units": "V",
        }

    # Assess whether the fit was successful or not
    # Check 1: No NaN values in fitted parameters
    nan_success = np.isnan(fit.opt_amp_prefactor) | np.isnan(fit.opt_amp)

    # Check 2: Rabi oscillation detected via residual chi-squared.
    # chi2 = SS_res / ((N - 4) * a²); chi2 > 2 → NO_OSCILLATION
    if max_pulses == 1:
        chi2 = _compute_chi2(fit)
        rabi_amp_sufficient = chi2 <= 2.0
    else:
        # Multi-pulse sweep: chi2 not applicable, skip check
        chi2 = xr.DataArray(
            [float("nan")] * len(fit.qubit.values),
            dims=["qubit"], coords={"qubit": fit.qubit},
        )
        rabi_amp_sufficient = True

    success_criteria = ~nan_success & rabi_amp_sufficient
    fit = fit.assign({"success": success_criteria})

    # Compute number of Rabi periods per qubit (only possible for single-pulse sweep)
    sweep_range = float(
        fit.amp_prefactor.values[-1] - fit.amp_prefactor.values[0]
    ) if max_pulses == 1 else float("nan")

    # Populate the FitParameters class with fitted values and error codes
    fit_results = {}
    for q in fit.qubit.values:
        q_success = fit.sel(qubit=q).success.values.__bool__()
        q_opt_amp_prefactor = fit.sel(qubit=q).opt_amp_prefactor.values.__float__()
        q_opt_amp = fit.sel(qubit=q).opt_amp.values.__float__()

        # Determine number of periods and error code
        if max_pulses == 1:
            freq_q = float(fit.sel(qubit=q).fit.sel(fit_vals="f").item())
            num_periods = abs(freq_q) * sweep_range
            q_chi2 = float(chi2.sel(qubit=q).item())

            if not q_success:
                error_code = PowerRabiErrorCode.NO_OSCILLATION
            elif num_periods > _TOO_MANY_PERIODS_THRESHOLD:
                error_code = PowerRabiErrorCode.TOO_MANY_PERIODS
            elif num_periods < _TOO_FEW_PERIODS_THRESHOLD:
                error_code = PowerRabiErrorCode.TOO_FEW_PERIODS
            else:
                error_code = PowerRabiErrorCode.SUCCESS
        else:
            # With error amplification we cannot count periods from a single-pulse fit
            num_periods = float("nan")
            q_chi2 = float("nan")
            error_code = PowerRabiErrorCode.SUCCESS if q_success else PowerRabiErrorCode.NO_OSCILLATION

        # Record the pulse length used in this run
        q_obj = next((qb for qb in node.namespace["qubits"] if qb.name == q), None)
        try:
            pulse_length_ns = float(q_obj.xy.operations[operation].length) if q_obj else float("nan")
        except Exception:
            pulse_length_ns = float("nan")

        fit_results[q] = FitParameters(
            opt_amp_prefactor=q_opt_amp_prefactor,
            opt_amp=q_opt_amp,
            operation=operation,
            success=q_success,
            num_periods=num_periods,
            chi2=q_chi2,
            error_code=int(error_code),
            corrective_action=int(PowerRabiCorrectiveAction.NONE),
            action_magnitude=0.0,
            pulse_length_ns=pulse_length_ns,
        )
    return fit, fit_results