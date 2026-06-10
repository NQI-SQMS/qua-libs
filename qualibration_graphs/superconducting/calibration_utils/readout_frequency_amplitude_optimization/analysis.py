import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr
from sklearn.mixture import GaussianMixture

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from calibration_utils.iq_blobs import fit_raw_data as fit_iq_blobs
from calibration_utils.iq_blobs.analysis import FitParameters as FitParametersIQblobs


@dataclass
class FitParameters(FitParametersIQblobs):
    """Stores the relevant 2D readout frequency/amplitude optimization fit parameters for a single qubit"""

    optimal_frequency: float = 0
    optimal_detuning: float = 0
    optimal_amplitude: float = 0
    optimal_amp_prefactor: float = 0


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
        s_freq = (
            f"\tOptimal readout frequency: {1e-9 * fit_results[q]['optimal_frequency']:.4f} GHz "
            f"(detuning {1e-6 * fit_results[q]['optimal_detuning']:.3f} MHz) | "
        )
        s_amp = f"optimal readout amplitude: {1e3 * fit_results[q]['optimal_amplitude']:.3f} mV | "
        s_fid = f"readout fidelity: {fit_results[q]['readout_fidelity']:.1f} %\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_freq + s_amp + s_fid)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    # Skip if the data has already been processed
    if ~np.all([var in ds.data_vars for var in ["Ig", "Qg", "Ie", "Qe"]]):
        return ds
    ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=["Ig", "Qg", "Ie", "Qe"])

    # Add the absolute readout amplitude to the dataset
    readout_amplitudes = np.array(
        [ds.amp_prefactor * q.resonator.operations["readout"].amplitude for q in node.namespace["qubits"]]
    )
    ds = ds.assign_coords(readout_amplitude=(["qubit", "amp_prefactor"], readout_amplitudes))
    ds.readout_amplitude.attrs = {"long_name": "readout amplitude", "units": "V"}

    # Add the absolute readout RF frequency to the dataset
    full_freq = np.array([ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "Readout RF frequency", "units": "Hz"}

    # Rearrange the data to combine I_g and I_e into I, and Q_g and Q_e into Q
    ds_rearranged = xr.Dataset()
    # Combine Ig and Ie into I
    ds_rearranged["I"] = xr.concat([ds.Ig, ds.Ie], dim="state")
    ds_rearranged["I"] = ds_rearranged["I"].assign_coords(state=[0, 1])
    # Combine Qg and Qe into Q
    ds_rearranged["Q"] = xr.concat([ds.Qg, ds.Qe], dim="state")
    ds_rearranged["Q"] = ds_rearranged["Q"].assign_coords(state=[0, 1])
    # Copy other coordinates and data variables
    for var in ds.coords:
        if var not in ds_rearranged.coords:
            ds_rearranged[var] = ds[var]

    for var in ds.data_vars:
        if var not in ["Ig", "Ie", "Qg", "Qe"]:
            ds_rearranged[var] = ds[var]

    # Replace the original dataset with the rearranged one
    ds = ds_rearranged
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the readout fidelity for each (frequency, amplitude) point and find the combination that maximizes it.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node : QualibrationNode
        The node containing the parameters used for the analysis.

    Returns:
    --------
    Tuple[xr.Dataset, xr.Dataset, dict[str, FitParameters]]
        The dataset containing the 2D fit results, the dataset containing the IQ blobs at the
        optimal point and the fit results for each qubit.
    """
    ds_fit = ds

    def apply_fit_gmm(I, Q):
        I_mean = np.mean(I, axis=1)
        Q_mean = np.mean(Q, axis=1)
        means_init = [[I_mean[0], Q_mean[0]], [I_mean[1], Q_mean[1]]]
        precisions_init = [1 / ((np.mean(np.var(I, axis=1)) + np.mean(np.var(Q, axis=1))) / 2)] * 2
        clf = GaussianMixture(
            n_components=2,
            covariance_type="spherical",
            means_init=means_init,
            precisions_init=precisions_init,
            tol=1e-5,
            reg_covar=1e-12,
        )
        X = np.array([np.array(I).flatten(), np.array(Q).flatten()]).T
        clf.fit(X)
        meas_fidelity = (
            np.sum(clf.predict(np.array([I[0], Q[0]]).T) == 0) / len(I[0])
            + np.sum(clf.predict(np.array([I[1], Q[1]]).T) == 1) / len(I[1])
        ) / 2
        loglikelihood = clf.score_samples(X)
        max_ll = np.max(loglikelihood)
        outliers = np.sum(loglikelihood > np.log(0.01) + max_ll) / len(X)
        return np.array([meas_fidelity, outliers])

    # Fit a 2-component Gaussian Mixture Model at every (qubit, detuning, amp_prefactor) point
    fit_data = xr.apply_ufunc(
        apply_fit_gmm,
        ds_fit.I,
        ds_fit.Q,
        input_core_dims=[["state", "n_runs"], ["state", "n_runs"]],
        output_core_dims=[["fit_vals"]],
        vectorize=True,
    )
    fit_data = fit_data.assign_coords(fit_vals=["meas_fidelity", "outliers"])
    ds_fit = xr.merge([ds, fit_data.rename("fit_data")])

    # Extract the relevant fitted parameters
    ds_fit, fit_results, ds_iq_blobs = _extract_relevant_fit_parameters(ds_fit, node)

    return ds_fit, ds_iq_blobs, fit_results


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    fidelity = ds_fit.fit_data.sel(fit_vals="meas_fidelity")
    outliers = ds_fit.fit_data.sel(fit_vals="outliers")

    # Discard points for which the GMM fit found too many outliers
    valid_fidelity = fidelity.where(outliers >= node.parameters.outliers_threshold)
    ds_fit["valid_fidelity"] = valid_fidelity

    # Find the (detuning, amp_prefactor) combination that maximizes the fidelity, for each qubit
    optimal_index = valid_fidelity.argmax(dim=["detuning", "amp_prefactor"], skipna=True)
    ds_fit["optimal_detuning"] = ds_fit.detuning.isel(detuning=optimal_index["detuning"])
    ds_fit["optimal_frequency"] = ds_fit.full_freq.isel(detuning=optimal_index["detuning"])
    ds_fit["optimal_amp_prefactor"] = ds_fit.amp_prefactor.isel(amp_prefactor=optimal_index["amp_prefactor"])
    ds_fit["optimal_amplitude"] = ds_fit.readout_amplitude.isel(amp_prefactor=optimal_index["amp_prefactor"])
    ds_fit["best_fidelity"] = valid_fidelity.isel(
        detuning=optimal_index["detuning"], amp_prefactor=optimal_index["amp_prefactor"]
    )

    # Select the I/Q single-shot data at the optimal point to run the iq_blobs analysis
    best_point = ds_fit.isel(detuning=optimal_index["detuning"], amp_prefactor=optimal_index["amp_prefactor"])
    ds_temp = xr.Dataset(
        {
            "Ig": best_point.I.sel(state=0).drop_vars("state"),
            "Ie": best_point.I.sel(state=1).drop_vars("state"),
            "Qg": best_point.Q.sel(state=0).drop_vars("state"),
            "Qe": best_point.Q.sel(state=1).drop_vars("state"),
        }
    )
    ds_iq_blobs, iq_blobs_fit_results = fit_iq_blobs(ds_temp, node)

    fit_results = {}
    for q in ds_fit.qubit.values:
        # Create a dictionary of the existing iq_blobs attributes (iw_angle, thresholds, confusion_matrix, ...)
        params_dict = iq_blobs_fit_results[q].__dict__
        # Add the 2D-optimization-specific fields
        params_dict["optimal_frequency"] = float(ds_fit["optimal_frequency"].sel(qubit=q))
        params_dict["optimal_detuning"] = float(ds_fit["optimal_detuning"].sel(qubit=q))
        params_dict["optimal_amplitude"] = float(ds_fit["optimal_amplitude"].sel(qubit=q))
        params_dict["optimal_amp_prefactor"] = float(ds_fit["optimal_amp_prefactor"].sel(qubit=q))
        fit_results[q] = FitParameters(**params_dict)
    return ds_fit, fit_results, ds_iq_blobs
