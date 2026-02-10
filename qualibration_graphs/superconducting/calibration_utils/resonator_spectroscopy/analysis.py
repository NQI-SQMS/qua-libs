import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from qualibration_libs.analysis import peaks_dips


@dataclass
class FitParameters:
    """Stores the relevant resonator spectroscopy experiment fit parameters for a single qubit"""

    frequency: float
    fwhm: float
    min_amplitude: float
    max_amplitude: float
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
        s_freq = f"\tResonator frequency: {1e-9 * fit_results[q]['frequency']:.3f} GHz | "
        s_fwhm = f"FWHM: {1e-3 * fit_results[q]['fwhm']:.1f} kHz | "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_freq + s_fwhm)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    full_freq = np.array([ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]])
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
    fit_results = peaks_dips(ds.IQ_abs, "detuning")
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(fit_results, ds, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(
    fit: xr.Dataset,
    ds: xr.Dataset,
    node: QualibrationNode,
):
    """Add metadata to the dataset and extract fit results."""

    # ------------------
    # Frequency metadata
    # ------------------
    fit.attrs = {"long_name": "frequency", "units": "Hz"}

    full_freq = np.array([q.resonator.RF_frequency for q in node.namespace["qubits"]])
    res_freq = fit.position + full_freq

    fit = fit.assign_coords(res_freq=("qubit", res_freq.data))
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}

    # -----
    # FWHM
    # -----
    fwhm = np.abs(fit.width)
    fit = fit.assign_coords(fwhm=("qubit", fwhm.data))
    fit.fwhm.attrs = {"long_name": "resonator fwhm", "units": "Hz"}

    # -------------------------------
    # Amplitude extraction (ABS(IQ))
    # -------------------------------
    # IMPORTANT: use RAW dataset, not fit dataset
    iq_abs = ds.IQ_abs  # already in Volts after convert_IQ_to_V

    min_amp = iq_abs.min(dim="detuning")
    max_amp = iq_abs.max(dim="detuning")

    fit = fit.assign_coords(
        min_amplitude=("qubit", min_amp.data),
        max_amplitude=("qubit", max_amp.data),
    )

    fit.min_amplitude.attrs = {"units": "V", "long_name": "min |IQ|"}
    fit.max_amplitude.attrs = {"units": "V", "long_name": "max |IQ|"}

    # ------------------
    # Success criteria
    # ------------------
    freq_success = (
        np.abs(res_freq.data)
        < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    )
    fwhm_success = (
        np.abs(fwhm.data)
        < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    )

    success_criteria = freq_success & fwhm_success
    fit = fit.assign_coords(success=("qubit", success_criteria))

    # ------------------
    # Results dictionary
    # ------------------
    fit_results = {
        q: {
            "frequency": fit.sel(qubit=q).res_freq.values.item(),
            "fwhm": fit.sel(qubit=q).fwhm.values.item(),
            "min_amplitude": fit.sel(qubit=q).min_amplitude.values.item(),
            "max_amplitude": fit.sel(qubit=q).max_amplitude.values.item(),
            "success": bool(fit.sel(qubit=q).success.values),
        }
        for q in fit.qubit.values
    }

    return fit, fit_results
