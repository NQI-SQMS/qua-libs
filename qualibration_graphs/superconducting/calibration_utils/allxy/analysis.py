import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V

# All-XY sequences. Each pulse name must match an operation defined on qubit.xy, with "I"
# denoting the identity (played as a wait of duration x180_length).
# See Reed's Thesis: https://rsl.yale.edu/sites/default/files/files/RSL_Theses/reed.pdf
SEQUENCE = [
    ("I", "I"),
    ("x180", "x180"),
    ("y180", "y180"),
    ("x180", "y180"),
    ("y180", "x180"),
    ("x90", "I"),
    ("y90", "I"),
    ("x90", "y90"),
    ("y90", "x90"),
    ("x90", "y180"),
    ("y90", "x180"),
    ("x180", "y90"),
    ("y180", "x90"),
    ("x90", "x180"),
    ("x180", "x90"),
    ("y90", "y180"),
    ("y180", "y90"),
    ("x180", "I"),
    ("y180", "I"),
    ("x90", "x90"),
    ("y90", "y90"),
]
SEQUENCE_LABELS = [f"{g1}-{g2}" for g1, g2 in SEQUENCE]
# Ideal excited-state population for each of the 21 sequences above: the first 5 pairs should
# leave the qubit in the ground state, the next 12 in an equal superposition, and the last 4 in
# the excited state.
IDEAL_POPULATION = np.array([0.0] * 5 + [0.5] * 12 + [1.0] * 4)


@dataclass
class FitParameters:
    """Stores the relevant AllXY fit parameters for a single qubit"""

    allxy_error: float
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
        s_error = f"\tAllXY mean absolute error: {fit_results[q]['allxy_error']:.4f}\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_error)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = ds.assign_coords(sequence_label=("sequence", SEQUENCE_LABELS))
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Compare the measured outcome of each of the 21 AllXY sequences against the ideal population
    and extract a per-qubit AllXY error metric.

    The raw outcome (state population, or 'I' quadrature when state discrimination is not used)
    is normalized using the first 5 (ideally ground-state) and last 4 (ideally excited-state)
    sequences as references, then compared point-by-point to the ideal population.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data (output of process_raw_dataset).
    node : QualibrationNode
        Node carrying the fit parameters, including the error_threshold.

    Returns:
    --------
    ds_fit : xr.Dataset
        Dataset with the added variables: normalized_population, allxy_error.
    fit_results : dict[str, FitParameters]
        Per-qubit AllXY error and success flag.
    """
    population = ds.state if node.parameters.use_state_discrimination else ds.I

    ground_ref = population.isel(sequence=slice(0, 5)).mean(dim="sequence")
    excited_ref = population.isel(sequence=slice(17, 21)).mean(dim="sequence")
    normalized_population = (population - ground_ref) / (excited_ref - ground_ref)

    ideal_population = xr.DataArray(IDEAL_POPULATION, dims="sequence", coords={"sequence": ds.sequence})
    allxy_error = np.abs(normalized_population - ideal_population).mean(dim="sequence")

    ds_fit = ds.assign(normalized_population=normalized_population, allxy_error=allxy_error)
    ds_fit["normalized_population"].attrs = {"long_name": "Normalized population"}
    ds_fit["allxy_error"].attrs = {"long_name": "AllXY mean absolute error"}

    fit_results = {
        q: FitParameters(
            allxy_error=float(ds_fit.sel(qubit=q)["allxy_error"]),
            success=bool(ds_fit.sel(qubit=q)["allxy_error"] < node.parameters.error_threshold),
        )
        for q in ds_fit.qubit.values
    }
    node.outcomes = {q: "successful" if fit_results[q].success else "failed" for q in ds_fit.qubit.values}
    return ds_fit, fit_results
