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
    """Stores the relevant readout frequency/amplitude optimization fit parameters for a single qubit."""

    optimal_amplitude: float = 0
    optimal_frequency: float = 0


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results.

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    log_callable : callable, optional
        Logger for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_freq = f"\tOptimal readout frequency: {1e-9 * fit_results[q]['optimal_frequency']:.5f} GHz\n"
        s_amp = f"\tOptimal readout amplitude: {1e3 * fit_results[q]['optimal_amplitude']:.3f} mV\n"
        s_fid = f"\tReadout fidelity: {fit_results[q]['readout_fidelity']:.1f} %\n"
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
    # Add the absolute readout frequency to the dataset
    full_freq = np.array([ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    # Rearrange the data to combine I_g and I_e into I, and Q_g and Q_e into Q
    ds_rearranged = xr.Dataset()
    ds_rearranged["I"] = xr.concat([ds.Ig, ds.Ie], dim="state")
    ds_rearranged["I"] = ds_rearranged["I"].assign_coords(state=[0, 1])
    ds_rearranged["Q"] = xr.concat([ds.Qg, ds.Qe], dim="state")
    ds_rearranged["Q"] = ds_rearranged["Q"].assign_coords(state=[0, 1])
    for var in ds.coords:
        if var not in ds_rearranged.coords:
            ds_rearranged[var] = ds[var]
    for var in ds.data_vars:
        if var not in ["Ig", "Ie", "Qg", "Qe"]:
            ds_rearranged[var] = ds[var]
    ds = ds_rearranged
    return ds


def _apply_fit_gmm(I, Q):
    """Fast single-shot GMM-based fidelity proxy, used to locate the optimal grid point."""
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


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, xr.Dataset, dict[str, FitParameters]]:
    """
    Jointly optimize the readout frequency and amplitude for each qubit in the dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node : QualibrationNode
        The node, used to access qubits and parameters.

    Returns:
    --------
    Tuple[xr.Dataset, xr.Dataset, dict[str, FitParameters]]
        The dataset with the fit metric added, the IQ blob dataset evaluated at the optimum, and the per-qubit
        fit results.
    """
    ds_fit = ds

    # Fast fidelity proxy over the full (detuning, amp_prefactor) grid, used only to locate the optimum.
    fit_data = xr.apply_ufunc(
        _apply_fit_gmm,
        ds_fit.I,
        ds_fit.Q,
        input_core_dims=[["state", "n_runs"], ["state", "n_runs"]],
        output_core_dims=[["fit_vals"]],
        vectorize=True,
    )
    fit_data = fit_data.assign_coords(fit_vals=["meas_fidelity", "outliers"])
    ds_fit = xr.merge([ds, fit_data.rename("fit_data")])

    fit_data, fit_results, ds_iq_blobs = _extract_relevant_fit_parameters(ds_fit, node)

    return fit_data, ds_iq_blobs, fit_results


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset, node: QualibrationNode):
    """Locate the optimal (detuning, amp_prefactor) point per qubit and refine the readout statistics there."""

    meas_fidelity = ds_fit.fit_data.sel(fit_vals="meas_fidelity")
    outliers = ds_fit.fit_data.sel(fit_vals="outliers")
    valid_fidelity = meas_fidelity.where(outliers >= node.parameters.outliers_threshold)

    # Stack the 2D (detuning, amp_prefactor) grid into a single dimension to find the joint argmax per qubit.
    stacked = valid_fidelity.stack(point=("detuning", "amp_prefactor"))
    best_idx = stacked.fillna(-np.inf).argmax(dim="point")
    best_detuning = stacked["detuning"][best_idx]
    best_amp_prefactor = stacked["amp_prefactor"][best_idx]

    ds_fit = ds_fit.assign_coords(
        optimal_detuning=("qubit", best_detuning.values),
        optimal_amp_prefactor=("qubit", best_amp_prefactor.values),
    )

    # Select, for each qubit, the data at its own optimal (detuning, amp_prefactor) point.
    best_data = ds_fit.sel(
        detuning=ds_fit["optimal_detuning"],
        amp_prefactor=ds_fit["optimal_amp_prefactor"],
        method="nearest",
    )

    Ig = best_data.I.sel(state=0).drop_vars("state")
    Qg = best_data.Q.sel(state=0).drop_vars("state")
    Ie = best_data.I.sel(state=1).drop_vars("state")
    Qe = best_data.Q.sel(state=1).drop_vars("state")
    ds_temp = xr.Dataset({"Ig": Ig, "Ie": Ie, "Qg": Qg, "Qe": Qe})
    ds_iq_blobs, _fit_results = fit_iq_blobs(ds_temp, node)

    fit_results = {}
    for q in ds_fit.qubit.values:
        params_dict = _fit_results[q].__dict__.copy()
        params_dict["optimal_amplitude"] = float(best_data["readout_amplitude"].sel(qubit=q))
        params_dict["optimal_frequency"] = float(best_data["full_freq"].sel(qubit=q))
        fit_results[q] = FitParameters(**params_dict)
    return ds_fit, fit_results, ds_iq_blobs
