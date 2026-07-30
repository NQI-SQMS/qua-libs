import logging
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import xarray as xr
from scipy.optimize import minimize_scalar

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


@dataclass
class FitParameters:
    optimal_detuning: float
    max_fidelity: float
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        status = "SUCCESS" if fit_results[q]["success"] else "FAIL"
        log_callable(
            f"Results for qubit {q}: {status}\n"
            f"\tOptimal GEF frequency shift: {1e-6 * fit_results[q]['optimal_detuning']:.3f} MHz\n"
            f"\tMax GEF fidelity: {fit_results[q]['max_fidelity'] * 100:.2f} %\n"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=["Ig", "Qg", "Ie", "Qe", "If", "Qf"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """Compute true 3-state fidelity at every frequency point and find the optimum."""
    qubits = node.namespace["qubits"]
    qubit_names = [q.name for q in qubits]
    n_freq = len(ds.coords["frequency"])

    fidelity_map = np.zeros((len(qubit_names), n_freq))

    for qi, q_name in enumerate(qubit_names):
        for fi in range(n_freq):
            Ig = ds["Ig"].sel(qubit=q_name).isel(frequency=fi).values
            Qg = ds["Qg"].sel(qubit=q_name).isel(frequency=fi).values
            Ie = ds["Ie"].sel(qubit=q_name).isel(frequency=fi).values
            Qe = ds["Qe"].sel(qubit=q_name).isel(frequency=fi).values
            If = ds["If"].sel(qubit=q_name).isel(frequency=fi).values
            Qf = ds["Qf"].sel(qubit=q_name).isel(frequency=fi).values
            fidelity_map[qi, fi] = _gef_fidelity(Ig, Qg, Ie, Qe, If, Qf)

    fidelity_da = xr.DataArray(
        fidelity_map,
        dims=["qubit", "frequency"],
        coords={"qubit": qubit_names, "frequency": ds.coords["frequency"]},
        attrs={"long_name": "g/e/f discrimination fidelity"},
    )
    ds_fit = ds.assign(fidelity_gef=fidelity_da)

    fit_results = {}
    opt_detunings = []
    for q_name in qubit_names:
        fid_q = fidelity_da.sel(qubit=q_name)
        # Light smoothing before argmax to reduce shot noise
        fid_smooth = fid_q.rolling(frequency=3, center=True, min_periods=1).mean()
        best_fi = int(fid_smooth.argmax(dim="frequency"))
        opt_detuning = float(ds.coords["frequency"].values[best_fi])
        max_fid = float(fid_q.isel(frequency=best_fi).values)
        success = np.isfinite(opt_detuning) and np.isfinite(max_fid) and max_fid > 1.0 / 3.0

        fit_results[q_name] = FitParameters(
            optimal_detuning=opt_detuning,
            max_fidelity=max_fid,
            success=success,
        )
        opt_detunings.append(opt_detuning)

    ds_fit = ds_fit.assign_coords(optimal_detuning=("qubit", opt_detunings))
    return ds_fit, fit_results


# ── True 3-state fidelity helpers ─────────────────────────────────────────────

def _rotate_iq(I, Q, angle):
    C, S = np.cos(angle), np.sin(angle)
    return I * C - Q * S, I * S + Q * C


def _ge_rotation_angle(Ig, Qg, Ie, Qe) -> float:
    angle = np.arctan2(np.mean(Qe) - np.mean(Qg), np.mean(Ig) - np.mean(Ie))
    C, S = np.cos(angle), np.sin(angle)
    if np.mean((Ig - Ie) * C - (Qg - Qe) * S) > 0:
        angle += np.pi
    return float(angle)


def _gef_fidelity(Ig, Qg, Ie, Qe, If, Qf) -> float:
    angle = _ge_rotation_angle(Ig, Qg, Ie, Qe)
    Ig_rot, Qg_rot = _rotate_iq(Ig, Qg, angle)
    Ie_rot, Qe_rot = _rotate_iq(Ie, Qe, angle)
    If_rot, Qf_rot = _rotate_iq(If, Qf, angle)

    ef_i = np.concatenate([Ie_rot, If_rot])
    res_i = minimize_scalar(
        lambda t: np.sum(Ig_rot > t) + np.sum(ef_i < t),
        bounds=(min(Ig_rot.min(), ef_i.min()), max(Ig_rot.max(), ef_i.max())),
        method="bounded",
    )
    t_i = res_i.x

    res_q = minimize_scalar(
        lambda t: np.sum(Qf_rot > t) + np.sum(Qe_rot < t),
        bounds=(min(Qe_rot.min(), Qf_rot.min()), max(Qe_rot.max(), Qf_rot.max())),
        method="bounded",
    )
    t_q = res_q.x

    def classify(I_r, Q_r):
        not_g = I_r > t_i
        n = len(I_r)
        return float(np.sum(~not_g) / n), float(np.sum(not_g & (Q_r >= t_q)) / n), float(np.sum(not_g & (Q_r < t_q)) / n)

    gg, ge, gf = classify(Ig_rot, Qg_rot)
    eg, ee, ef = classify(Ie_rot, Qe_rot)
    fg, fe, ff = classify(If_rot, Qf_rot)

    return float(1.0 - (ge + eg + fg + gf + ef + fe) / 2.0)
