import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from calibration_utils.shared import apply_confusion_matrix_correction


def apply_confusion_correction_to_dataset(ds, node):
    return apply_confusion_matrix_correction(ds, node.namespace["qubits"])


@dataclass
class FitParameters:
    """Stores the relevant DRAG calibration fit parameters for a single qubit"""

    alpha: float
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    log_callable : callable, optional
        Callable for logging. If None, uses the module logger.

    Returns:
    --------
    None
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_alpha = f"\tDRAG coefficient alpha: {fit_results[q]['alpha']:.4f}\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_alpha)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    else:
        ds = apply_confusion_correction_to_dataset(ds, node)
    # Add the DRAG coefficient alpha to the dataset
    alpha = np.array(
        [q.xy.operations[node.parameters.operation].alpha * ds.alpha_prefactor for q in node.namespace["qubits"]]
    )
    ds = ds.assign_coords(alpha=(["qubit", "alpha_prefactor"], alpha))
    ds.alpha.attrs = {"long_name": "DRAG coefficient alpha"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Find the optimal DRAG coefficient for each qubit using Gaussian-smoothed
    sum optimisation.

    The method sums I (or state) over all pulse-number iterations, applies a
    1-D Gaussian filter along the alpha axis for noise robustness, then picks
    the alpha that minimises the smoothed curve — matching the analysis used
    in the manual DRAG calibration notebook.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data (output of process_raw_dataset).
    node : QualibrationNode
        Node carrying parameters including smooth_sigma.

    Returns:
    --------
    ds_fit : xr.Dataset
        Dataset with added variables: summed_data, smoothed_data, optimal_alpha, success.
    fit_results : dict[str, FitParameters]
        Per-qubit optimal alpha and success flag.
    """
    ds_fit = ds
    sigma = node.parameters.smooth_sigma

    if node.parameters.use_state_discrimination:
        raw_sum = ds.state.sum(dim="nb_of_pulses")
    else:
        raw_sum = ds.I.sum(dim="nb_of_pulses")

    ds_fit["summed_data"] = raw_sum
    ds_fit["summed_data"].attrs = {"long_name": "Sum over pulse iterations"}

    # Gaussian smoothing per qubit along the alpha_prefactor axis
    smoothed_vals = np.stack([
        gaussian_filter1d(raw_sum.sel(qubit=q).values, sigma=sigma)
        for q in raw_sum.qubit.values
    ])
    ds_fit["smoothed_data"] = xr.DataArray(
        smoothed_vals,
        dims=["qubit", "alpha_prefactor"],
        coords={"qubit": raw_sum.qubit, "alpha_prefactor": raw_sum.alpha_prefactor},
        attrs={"long_name": f"Gaussian-smoothed sum (sigma={sigma})"},
    )
    # Propagate the alpha coordinate to smoothed_data
    if "alpha" in ds_fit.coords:
        ds_fit["smoothed_data"] = ds_fit["smoothed_data"].assign_coords(
            alpha=ds_fit["alpha"]
        )

    # Optimal alpha: minimise the smoothed curve
    best_idx = ds_fit["smoothed_data"].argmin("alpha_prefactor")
    ds_fit["optimal_alpha"] = ds_fit["smoothed_data"].alpha[:, best_idx]

    ds_fit, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return ds_fit, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add success metadata to the dataset and build the fit_results dict."""
    nan_success = np.isnan(fit.optimal_alpha)
    snr_success = (
        np.abs(
            (fit["summed_data"].min("alpha_prefactor") - fit["summed_data"].mean("alpha_prefactor"))
            / fit["summed_data"].std("alpha_prefactor")
        )
        > 2
    )
    success_criteria = ~nan_success & snr_success
    fit = fit.assign({"success": success_criteria})
    fit_results = {
        q: FitParameters(
            alpha=float(fit.sel(qubit=q)["optimal_alpha"]),
            success=bool(fit.sel(qubit=q).success),
        )
        for q in fit.qubit.values
    }
    node.outcomes = {q: "successful" if fit_results[q].success else "fail" for q in fit.qubit.values}
    return fit, fit_results
