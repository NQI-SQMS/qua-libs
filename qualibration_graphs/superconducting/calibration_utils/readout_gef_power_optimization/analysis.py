import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import minimize_scalar

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


@dataclass
class FitParameters:
    """Fit results for a single qubit's GEF readout power optimisation."""

    optimal_amp_factor: float
    optimal_amp: float
    max_fidelity: float
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        log_callable(
            f"GEF readout power optimisation for qubit {q}: {status}\n"
            f"\tOptimal amp factor = {res['optimal_amp_factor']:.3f}  "
            f"(amplitude = {1e3 * res['optimal_amp']:.2f} mV)\n"
            f"\tMax GEF fidelity = {res['max_fidelity'] * 100:.2f} %"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=["Ig", "Qg", "Ie", "Qe", "If", "Qf"])
    return ds


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Find the amplitude that maximises the true 3-state GEF discrimination fidelity."""
    qubits = node.namespace["qubits"]
    qubit_names = [q.name for q in qubits]
    n_amp = len(ds.coords["amp_prefactor"])

    fidelity_map = np.zeros((len(qubit_names), n_amp))

    for qi, q_name in enumerate(qubit_names):
        for ai in range(n_amp):
            Ig = ds["Ig"].sel(qubit=q_name).isel(amp_prefactor=ai).values
            Qg = ds["Qg"].sel(qubit=q_name).isel(amp_prefactor=ai).values
            Ie = ds["Ie"].sel(qubit=q_name).isel(amp_prefactor=ai).values
            Qe = ds["Qe"].sel(qubit=q_name).isel(amp_prefactor=ai).values
            If = ds["If"].sel(qubit=q_name).isel(amp_prefactor=ai).values
            Qf = ds["Qf"].sel(qubit=q_name).isel(amp_prefactor=ai).values
            fidelity_map[qi, ai] = _gef_fidelity(Ig, Qg, Ie, Qe, If, Qf)

    fidelity_da = xr.DataArray(
        fidelity_map,
        dims=["qubit", "amp_prefactor"],
        coords={"qubit": qubit_names, "amp_prefactor": ds.coords["amp_prefactor"]},
        attrs={"long_name": "g/e/f discrimination fidelity"},
    )
    ds_fit_list = []
    fit_results = {}

    for qubit in qubits:
        q = qubit.name
        fid_q = fidelity_da.sel(qubit=q)

        # Light smoothing before argmax to reduce shot noise
        fid_smooth = fid_q.rolling(amp_prefactor=3, center=True, min_periods=1).mean()
        best_ai = int(fid_smooth.argmax(dim="amp_prefactor"))
        opt_factor = float(ds.coords["amp_prefactor"].isel(amp_prefactor=best_ai).values)
        current_amp = float(qubit.resonator.operations[node.parameters.operation].amplitude)
        opt_amp = opt_factor * current_amp
        max_fid = float(fid_q.isel(amp_prefactor=best_ai).values)

        success = np.isfinite(opt_factor) and opt_factor > 0 and max_fid > 1.0 / 3.0

        fit_results[q] = FitParameters(
            optimal_amp_factor=opt_factor,
            optimal_amp=opt_amp,
            max_fidelity=max_fid,
            success=success,
        )

        ds_fit_list.append(
            xr.Dataset({
                "fidelity_gef": fid_q,
                "optimal_amp_factor": xr.DataArray(opt_factor),
                "optimal_amp": xr.DataArray(opt_amp),
                "max_fidelity": xr.DataArray(max_fid),
                "success": xr.DataArray(success),
            }).expand_dims(qubit=[q])
        )

    ds_fit = xr.concat(ds_fit_list, dim="qubit")
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
