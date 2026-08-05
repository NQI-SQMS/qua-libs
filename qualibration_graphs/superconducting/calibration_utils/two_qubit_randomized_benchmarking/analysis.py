# SPDX-License-Identifier: EUPL-1.2
# Copyright (C) 2026 Q.M Technologies Ltd. / Soon Teh
# Copyright (C) 2026 Q.M Technologies Ltd. / Hiroyuki Inoue
# Copyright (C) 2026 RIKEN / András Gunyhó
# Licensed under the EUPL v1.2.
# See: https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12

import logging
import pickle as pkl
from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from uncertainties import ufloat
from uncertainties import unumpy as unp

from calibration_utils.cr_utils import get_cr_duration
from calibration_utils.data_process_utils import reshape_control_target_val2dim
from calibration_utils.common_utils.fitting_tools import fit_power_law
from calibration_utils.common_utils.physics_tools import coherence_limit


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    p: float
    p_sem: float
    error_per_clifford: float
    error_per_gate: float
    epg_eval_method: Literal["interleaved", "2Q_decomposition", "1Q_2Q_decomposition", "N/A"]
    coherence_limit: float
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
        s_qubit = f"Results for qubit pair {q}: "
        s_epg = ""
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
            s_epg += f"Error per Clifford:       {fit_results[q]['error_per_clifford']:.3%}\n"
            s_epg += f"Error per two-qubit gate: {fit_results[q]['error_per_gate']:.3%}\n"
            s_epg += f"Eval method:              {fit_results[q]['epg_eval_method']}\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_epg)
    pass


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    ds = reshape_control_target_val2dim(ds, state_discrimination=True)

    # Map integer 0..3 to basis labels
    basis_labels = ["00", "01", "10", "11"]

    # Compute joint outcomes = 2*c + t
    joint_ds = 2 * ds.sel(control_target="c").state + ds.sel(control_target="t").state
    # joint_ds dims: (qubit_pair, nb_of_shots, correction_phases_2pi, initial_state)

    # One-hot encode outcomes [0,1,2,3] along a new "measured_state" dim
    joint_ds = (
        xr.apply_ufunc(
            lambda arr: np.eye(4, dtype=int)[arr],
            joint_ds,
            input_core_dims=[["nb_of_shots"]],
            output_core_dims=[["nb_of_shots", "measured_state"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[int],
        ).sum(dim="nb_of_shots")
        / ds.nb_of_shots.size
    )
    joint_ds = joint_ds.assign_coords(measured_state=("measured_state", basis_labels))

    return xr.Dataset({"state": joint_ds})  # DataArray can't be saved by qualibrate, so convert back to dataset


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
    # Calculate coherence limit
    qps = ds.qubit_pair.values
    # Index the live qubit-pair objects by name and iterate over the *dataset's* pair axis so
    # coh_vals stays aligned with `qps`. The dataset may hold a different set/order of pairs than
    # node.namespace["qubit_pairs"] -- e.g. 60d loads runs whose qubit_pairs were unset (None) and
    # fell back to all machine pairs. Enumerating the namespace list directly then overruns
    # coh_vals (IndexError) or silently misaligns it against the data.
    pair_by_name = {qp.name: qp for qp in node.namespace["qubit_pairs"]}
    coh_vals = np.full(len(qps), np.nan, dtype=float)
    for i, qp_name in enumerate(qps):
        qp = pair_by_name.get(str(qp_name))
        if qp is None:
            coh_vals[i] = 0.0
            continue
        qc = qp.qubit_control
        qt = qp.qubit_target
        T1_list = [qc.T1, qt.T1]
        T2_list = [qc.T2echo, qt.T2echo]
        # CZ nodes carry an `operation` parameter and take the gate length from the calibrated
        # CZ flux pulse; CR nodes keep the cross-resonance pulse-train duration (get_cr_duration).
        op = getattr(node.parameters, "operation", None)
        if op is not None:
            try:
                flux_pulse = qp.macros[op].flux_pulse_qubit
                gate_length = float(getattr(flux_pulse, "length", None) or flux_pulse.flat_length) * 1e-9
            except Exception:
                gate_length = None
        else:
            gate_length = get_cr_duration(node, qp=qp, with_x180=True) * 1e-9

        if None in T1_list or None in T2_list or gate_length is None:
            coh_vals[i] = 0.0
        else:
            coh_vals[i] = coherence_limit(nQ=2, T1_list=T1_list, T2_list=T2_list, gatelen=gate_length)

    ds_coherence_limits = xr.DataArray(
        coh_vals,
        coords={"qubit_pair": qps},
        dims=("qubit_pair",),
        name="coherence_limit",
    )

    # Averaging over sequences
    ds_fit = ds.sel(measured_state="00").state
    ds_fit_mean = ds_fit.mean(dim="nb_of_sequences").rename("data_mean")
    ds_fit_sem = ds_fit.std(dim="nb_of_sequences", ddof=1).rename("data_sem") / np.sqrt(ds.nb_of_sequences.size)

    # Fit to the model
    param_names = ["p", "A", "B"]
    fit_data = xr.DataArray(
        np.full((len(qps), len(param_names)), np.nan, float),
        coords={"qubit_pair": qps, "param": param_names},
        dims=("qubit_pair", "param"),
        name="fit_data",
    )
    fit_data_sem = xr.DataArray(
        np.full((len(qps), len(param_names)), np.nan, float),
        coords={"qubit_pair": qps, "param": param_names},
        dims=("qubit_pair", "param"),
        name="fit_data_sem",
    )
    try:
        for qp in qps:
            x = ds_fit_mean.sel(qubit_pair=qp).depths.values.squeeze()
            y = ds_fit_mean.sel(qubit_pair=qp).values.squeeze()
            y_err = ds_fit_sem.sel(qubit_pair=qp).values.squeeze()

            popt, perr = fit_power_law(x, y, y_err)

            # Store the fit results in the dataset
            fit_data.loc[dict(qubit_pair=qp)] = popt
            fit_data_sem.loc[dict(qubit_pair=qp)] = perr
    except Exception as e:
        print(f"Fit failed with error: {e}")

    # Combine mean, std, and fit results into a single dataset
    ds_fit = xr.merge(
        [
            ds_fit_mean,
            ds_fit_sem,
            fit_data,
            fit_data_sem,
            ds_coherence_limits,
        ]
    )

    # Extract the relevant fitted parameters
    ds_fit, fit_results = _extract_relevant_fit_parameters(ds_fit, node)

    return ds_fit, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    # Extract the decay rate
    p = xr.DataArray(
        [
            ufloat(
                fit.fit_data.sel(qubit_pair=qp, param="p").values, fit.fit_data_sem.sel(qubit_pair=qp, param="p").values
            )
            for qp in fit.qubit_pair.values
        ],
        coords={"qubit_pair": fit.qubit_pair.values},
        dims=["qubit_pair"],
    )
    # EPC from here: https://qiskit.org/textbook/ch-quantum-hardware/randomized-benchmarking.html#Step-5:-Fit-the-results
    n_qubits = 2
    d = 2**n_qubits
    epc = (1 - p) * (1 - 1 / d)
    epc_nom = xr.apply_ufunc(unp.nominal_values, epc)
    epc_sem = xr.apply_ufunc(unp.std_devs, epc)

    # Assess whether the fit was successful or not
    nan_success = xr.apply_ufunc(np.isnan, epc_nom)
    rb_success = (0 < epc_nom) & (epc_nom < 1)
    success_criteria = ~nan_success & rb_success

    # avg_num_gate = decomposition_stats["avg_num_1q_gate"] + decomposition_stats["avg_num_CNOT_gate"]
    # Decomposition numbers (example)
    # N1=8.25, N2=1.5 https://journals.aps.org/prx/abstract/10.1103/PhysRevX.9.021011
    # N1=12.2167, N2=1.5 https://arxiv.org/pdf/1712.06550v2
    try:
        decomposition_stats = pkl.load(
            open("./calibration_utils/two_qubit_randomized_benchmarking/2q_Clifford_gen_CNOT_stats.pkl", "rb")
        )
    except Exception:
        decomposition_stats = pkl.load(
            open("../calibration_utils/two_qubit_randomized_benchmarking/2q_Clifford_gen_CNOT_stats.pkl", "rb")
        )
    avg_num_1q_gate = decomposition_stats["avg_num_1q_gate"]
    avg_num_CNOT_gate = decomposition_stats["avg_num_CNOT_gate"]

    # Calculation of single two-qubit gate error from Clifford error
    # https://arxiv.org/src/1712.06550v2/anc/threeq_supp.pdf
    # https://qiskit-community.github.io/qiskit-experiments/manuals/verification/randomized_benchmarking.html#id10
    # Calculation depends on whether previous data (e.g. interleaving, 1QRB, ...) is available
    epg_dict = {}
    if node.parameters.interleaved_CNOT:
        print("\nInterleaving:")
        for qp in fit.qubit_pair.values:
            qp_extras = node.machine.qubit_pairs[qp].extras
            if "2QRB_p" in qp_extras:
                p_ref = ufloat(qp_extras["2QRB_p"], qp_extras.get("2QRB_p_sem", 0))
                p_int = p.sel(qubit_pair=qp).values.item()
                epg = (1 - p_int / p_ref) * (1 - 1 / d)
                epg_dict[qp] = (epg, "interleaved")
            else:
                epg_dict[qp] = (ufloat(0, 0), "N/A")
                print(f"-> Reference p not found for {qp}. Cannot compute interleaved CNOT error.")
    else:
        print("\nNot interleaving:")
        for qp in fit.qubit_pair.values:
            qp_obj = node.machine.qubit_pairs[qp]
            qc = qp_obj.qubit_control
            qt = qp_obj.qubit_target
            p_2Q = p.sel(qubit_pair=qp).values.item()
            if "1QRB_p" in qc.extras and "1QRB_p" in qt.extras:
                print(f"-> Using 1QRB data to improve estimate for {qp}.")
                p_1Qc = ufloat(qc.extras["1QRB_p"], qc.extras.get("1QRB_p_sem", 0))
                p_1Qt = ufloat(qt.extras["1QRB_p"], qt.extras.get("1QRB_p_sem", 0))
                denominator = (
                    p_1Qc ** (avg_num_1q_gate / 2)
                    + p_1Qt ** (avg_num_1q_gate / 2)
                    + 3 * (p_1Qc * p_1Qt) ** (avg_num_1q_gate / 2)
                )
                p_2Qgate = (5 * p_2Q / denominator) ** (1 / avg_num_CNOT_gate)
                eval_method = "1Q_2Q_decomposition"
            else:
                print(f"-> Assuming 2Q error dominates for {qp} since no 1QRB data is available.")
                p_2Qgate = p_2Q ** (1 / avg_num_CNOT_gate)
                eval_method = "2Q_decomposition"
            epg = (1 - p_2Qgate) * (1 - 1 / d)
            epg_dict[qp] = (epg, eval_method)

    # Merge epg_dict into fit
    epg_xr = xr.DataArray(
        [epg_dict[qp][0].nominal_value for qp in fit.qubit_pair.values],
        coords={"qubit_pair": fit.qubit_pair.values},
        dims=["qubit_pair"],
    )
    epg_sem_xr = xr.DataArray(
        [epg_dict[qp][0].std_dev for qp in fit.qubit_pair.values],
        coords={"qubit_pair": fit.qubit_pair.values},
        dims=["qubit_pair"],
    )
    epg_method_xr = xr.DataArray(
        [epg_dict[qp][1] for qp in fit.qubit_pair.values],
        coords={"qubit_pair": fit.qubit_pair.values},
        dims=["qubit_pair"],
    )

    # Gather all relevant fit parameters into the dataset
    fit = fit.assign(
        {
            "error_per_clifford": epc_nom,
            "error_per_clifford_sem": epc_sem,
            "error_per_gate": epg_xr,
            "error_per_gate_sem": epg_sem_xr,
            "epg_eval_method": epg_method_xr,
            "success": success_criteria,
        }
    )

    # Save fitting results
    fit_results = {
        qp: FitParameters(
            p=p.sel(qubit_pair=qp).values.item().nominal_value,
            p_sem=p.sel(qubit_pair=qp).values.item().std_dev,
            error_per_clifford=epc.sel(qubit_pair=qp).values.item().nominal_value,
            error_per_gate=epg_dict[qp][0].nominal_value,
            epg_eval_method=epg_dict[qp][1],
            coherence_limit=fit.coherence_limit.sel(qubit_pair=qp).values.item(),
            success=fit.sel(qubit_pair=qp).success.values.item(),
        )
        for qp in fit.qubit_pair.values
    }
    node.outcomes = {qp: "successful" if fit_results[qp].success else "fail" for qp in fit.qubit_pair.values}

    return fit, fit_results
