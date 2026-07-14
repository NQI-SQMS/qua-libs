import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from calibration_utils.iq_blobs import fit_raw_data as fit_iq_blobs
from calibration_utils.iq_blobs.analysis import FitParameters as FitParametersIQblobs
from scipy.optimize import minimize_scalar


@dataclass
class FitParameters(FitParametersIQblobs):
    """Stores the relevant readout frequency/amplitude optimization fit parameters for a single qubit."""

    optimal_amplitude: float = 0
    optimal_frequency: float = 0


def log_fitted_results(fit_results: Dict, log_callable=None):
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
    if ~np.all([var in ds.data_vars for var in ["Ig", "Qg", "Ie", "Qe"]]):
        return ds
    ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=["Ig", "Qg", "Ie", "Qe"])
    readout_amplitudes = np.array(
        [ds.amp_prefactor * q.resonator.operations["readout"].amplitude for q in node.namespace["qubits"]]
    )
    ds = ds.assign_coords(readout_amplitude=(["qubit", "amp_prefactor"], readout_amplitudes))
    ds.readout_amplitude.attrs = {"long_name": "readout amplitude", "units": "V"}
    full_freq = np.array([ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
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


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, xr.Dataset, dict[str, FitParameters]]:
    ds_fit = ds

    # True 2-state fidelity at every (detuning, amp_prefactor) grid point
    def _true_fidelity(I, Q):
        """I/Q shape: (2, n_runs). Returns scalar fidelity."""
        return _ge_fidelity(I[0], Q[0], I[1], Q[1])

    fidelity = xr.apply_ufunc(
        _true_fidelity,
        ds_fit.I,
        ds_fit.Q,
        input_core_dims=[["state", "n_runs"], ["state", "n_runs"]],
        output_core_dims=[[]],
        vectorize=True,
    )
    fidelity.attrs = {"long_name": "g/e discrimination fidelity"}
    ds_fit = xr.merge([ds, fidelity.rename("fidelity_ge")])

    ds_fit, ds_iq_blobs, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return ds_fit, ds_iq_blobs, fit_results


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset, node: QualibrationNode):
    stacked = ds_fit.fidelity_ge.stack(point=("detuning", "amp_prefactor"))
    best_idx = stacked.fillna(-np.inf).argmax(dim="point")
    best_detuning = stacked["detuning"][best_idx]
    best_amp_prefactor = stacked["amp_prefactor"][best_idx]

    ds_fit = ds_fit.assign_coords(
        optimal_detuning=("qubit", best_detuning.values),
        optimal_amp_prefactor=("qubit", best_amp_prefactor.values),
    )

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
    return ds_fit, ds_iq_blobs, fit_results


# ── True fidelity helpers (two-cut algorithm) ────────────────────────────────

def _rotate_iq(I, Q, angle):
    C, S = np.cos(angle), np.sin(angle)
    return I * C - Q * S, I * S + Q * C


def _ge_rotation_angle(Ig, Qg, Ie, Qe):
    angle = np.arctan2(np.mean(Qe) - np.mean(Qg), np.mean(Ig) - np.mean(Ie))
    C, S = np.cos(angle), np.sin(angle)
    if np.mean((Ig - Ie) * C - (Qg - Qe) * S) > 0:
        angle += np.pi
    return float(angle)


def _ge_fidelity(Ig, Qg, Ie, Qe) -> float:
    """True 2-state discrimination fidelity via optimal threshold search."""
    angle = _ge_rotation_angle(Ig, Qg, Ie, Qe)
    Ig_rot, _ = _rotate_iq(Ig, Qg, angle)
    Ie_rot, _ = _rotate_iq(Ie, Qe, angle)
    result = minimize_scalar(
        lambda t: np.sum(Ig_rot > t) + np.sum(Ie_rot < t),
        bounds=(min(Ig_rot.min(), Ie_rot.min()), max(Ig_rot.max(), Ie_rot.max())),
        method="bounded",
    )
    t = result.x
    gg = np.sum(Ig_rot < t) / len(Ig_rot)
    ee = np.sum(Ie_rot >= t) / len(Ie_rot)
    return float((gg + ee) / 2)
