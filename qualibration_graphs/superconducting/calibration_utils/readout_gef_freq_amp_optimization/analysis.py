"""
Analysis for the GEF readout frequency × amplitude 2D optimisation.

Computes the true 3-state discrimination fidelity (from a 3×3 confusion matrix) at every
(frequency, amplitude) grid point and identifies the joint optimum.  This is distinct from the
1D nodes (14, 14b) which use the worst-case pairwise IQ distance as a cheaper proxy.

Algorithm (per grid point):
  1. Rotate IQ data by the angle that maximises I-axis g/e separation.
  2. Find the I-axis threshold separating g from {e, f} via scalar minimisation.
  3. Find the Q-axis threshold separating e from f (among non-g points) via scalar minimisation.
  4. Build the 3×3 confusion matrix and return fidelity = 1 - (sum of off-diagonal errors) / 2.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import minimize, minimize_scalar

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


# ─────────────────────────────────────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FitParameters:
    """Optimal GEF readout parameters extracted from the 2D frequency × amplitude sweep."""

    optimal_frequency: float
    """Absolute resonator readout frequency at the optimal point (Hz)."""
    optimal_gef_detuning: float
    """IF detuning relative to the current GEF operating frequency at the optimal point (Hz).
    This value is added to qubit.resonator.GEF_frequency_shift during state update."""
    optimal_amp_scale: float
    """Amplitude scale factor at the optimal point (relative to current readout amplitude)."""
    optimal_amplitude_v: float
    """Absolute readout amplitude at the optimal point (V)."""
    max_fidelity: float
    """Maximum 3-state discrimination fidelity achieved (fraction, 0–1)."""
    iw_angle: float
    """IQ rotation angle that maximises I-axis g/e separation (rad).
    Subtracted from integration_weights_angle during state update."""
    ge_threshold: float
    """Optimal g vs {e,f} I-axis discrimination threshold (V)."""
    rus_threshold: float
    """Repeat-Until-Success threshold derived from the ground-state histogram peak (V)."""
    gef_threshold_i: float
    """I-axis threshold separating g from {e, f} (V).  Alias of ge_threshold for clarity."""
    gef_threshold_q: float
    """Q-axis threshold separating e from f among non-g points (V)."""
    confusion_matrix: list
    """Row-normalised 3×3 confusion matrix as a nested list.
    Row/column ordering: [g, e, f].  confusion_matrix[i][j] = P(detect j | true i)."""
    success: bool
    """True when all fit values are finite and max_fidelity > 1/3 (better than random)."""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert raw IQ streams from OPX fixed-point units to Volts and attach derived coordinates.

    Adds:
      - readout_amplitude_v (qubit, amplitude): absolute readout amplitude in V.
      - abs_frequency (qubit, frequency): absolute readout frequency in Hz,
          computed as q.resonator.RF_frequency + current_GEF_shift + df.
    """
    ds = convert_IQ_to_V(
        ds, node.namespace["qubits"],
        IQ_list=["Ig", "Qg", "Ie", "Qe", "If", "Qf"],
    )

    amp_scales = ds.coords["amplitude"].values       # shape (n_amp,)
    if_detunings = ds.coords["frequency"].values     # shape (n_freq,), Hz offsets from GEF freq

    readout_amp_v = np.array(
        [amp_scales * q.resonator.operations[node.parameters.operation].amplitude
         for q in node.namespace["qubits"]]
    )  # (n_qubits, n_amp)

    gef_shift = np.array(
        [q.resonator.GEF_frequency_shift or 0.0 for q in node.namespace["qubits"]]
    )  # (n_qubits,)
    abs_freq = np.array(
        [q.resonator.RF_frequency + gef_shift[qi] + if_detunings
         for qi, q in enumerate(node.namespace["qubits"])]
    )  # (n_qubits, n_freq)

    ds = ds.assign_coords(
        readout_amplitude_v=(["qubit", "amplitude"], readout_amp_v),
        abs_frequency=(["qubit", "frequency"], abs_freq),
    )
    ds.coords["readout_amplitude_v"].attrs = {"long_name": "readout amplitude", "units": "V"}
    ds.coords["abs_frequency"].attrs = {"long_name": "absolute readout frequency", "units": "Hz"}
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Compute the GEF fidelity map and extract the optimal (frequency, amplitude) per qubit.

    Returns
    -------
    ds_fit : xr.Dataset
        Raw dataset augmented with the ``fidelity_gef`` DataArray and optimal-point coordinates.
    fit_results : dict[str, FitParameters]
        Per-qubit fit parameters.
    """
    qubits = node.namespace["qubits"]

    fidelity_gef = _compute_gef_fidelity_map(ds, qubits)
    ds_fit = ds.assign(fidelity_gef=fidelity_gef)

    fit_results = {}
    for q in qubits:
        q_name = q.name
        fid_q = fidelity_gef.sel(qubit=q_name)  # (frequency, amplitude)

        flat_idx = int(fid_q.values.argmax())
        fi, ai = np.unravel_index(flat_idx, fid_q.shape)

        opt_detuning = float(fid_q.coords["frequency"].values[fi])
        opt_freq = float(ds_fit.coords["abs_frequency"].sel(qubit=q_name).values[fi])
        opt_amp_scale = float(fid_q.coords["amplitude"].values[ai])
        opt_amp_v = float(ds_fit.coords["readout_amplitude_v"].sel(qubit=q_name).values[ai])
        max_fid = float(fid_q.values[fi, ai])

        Ig = ds_fit["Ig"].sel(qubit=q_name).isel(frequency=fi, amplitude=ai).values
        Qg = ds_fit["Qg"].sel(qubit=q_name).isel(frequency=fi, amplitude=ai).values
        Ie = ds_fit["Ie"].sel(qubit=q_name).isel(frequency=fi, amplitude=ai).values
        Qe = ds_fit["Qe"].sel(qubit=q_name).isel(frequency=fi, amplitude=ai).values
        If = ds_fit["If"].sel(qubit=q_name).isel(frequency=fi, amplitude=ai).values
        Qf = ds_fit["Qf"].sel(qubit=q_name).isel(frequency=fi, amplitude=ai).values

        iw_angle, ge_threshold, rus_threshold = _compute_ge_discriminator_params(Ig, Qg, Ie, Qe)
        t_gef_i, t_gef_q, confusion_matrix = _compute_gef_confusion_matrix(
            Ig, Qg, Ie, Qe, If, Qf, iw_angle
        )

        success = (
            np.isfinite(iw_angle)
            and np.isfinite(ge_threshold)
            and np.isfinite(rus_threshold)
            and max_fid > 1.0 / 3.0
        )

        fit_results[q_name] = FitParameters(
            optimal_frequency=opt_freq,
            optimal_gef_detuning=opt_detuning,
            optimal_amp_scale=opt_amp_scale,
            optimal_amplitude_v=opt_amp_v,
            max_fidelity=max_fid,
            iw_angle=iw_angle,
            ge_threshold=ge_threshold,
            rus_threshold=rus_threshold,
            gef_threshold_i=t_gef_i,
            gef_threshold_q=t_gef_q,
            confusion_matrix=confusion_matrix,
            success=success,
        )

    # Attach optimal-point coordinates to ds_fit for plotting
    opt_detunings = [fit_results[q.name].optimal_gef_detuning for q in qubits]
    opt_amp_scales = [fit_results[q.name].optimal_amp_scale for q in qubits]
    ds_fit = ds_fit.assign_coords(
        optimal_gef_detuning=("qubit", opt_detunings),
        optimal_amp_scale=("qubit", opt_amp_scales),
    )

    return ds_fit, fit_results


def log_fitted_results(fit_results: Dict, log_callable=None) -> None:
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q_name, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        log_callable(
            f"GEF readout 2D optimisation for qubit {q_name}: {status}\n"
            f"  Optimal frequency   : {res['optimal_frequency'] / 1e9:.6f} GHz\n"
            f"  GEF detuning added  : {res['optimal_gef_detuning'] / 1e6:.3f} MHz\n"
            f"  Optimal amp scale   : {res['optimal_amp_scale']:.4f}\n"
            f"  Optimal amplitude   : {res['optimal_amplitude_v'] * 1e3:.3f} mV\n"
            f"  Max GEF fidelity    : {res['max_fidelity'] * 100:.2f} %\n"
            f"  IW angle            : {res['iw_angle'] * 180 / np.pi:.2f} deg\n"
            f"  GEF I-threshold     : {res['gef_threshold_i'] * 1e3:.3f} mV\n"
            f"  GEF Q-threshold     : {res['gef_threshold_q'] * 1e3:.3f} mV\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_gef_fidelity_map(ds: xr.Dataset, qubits: List) -> xr.DataArray:
    """Compute true 3-state fidelity for every (qubit, frequency, amplitude) grid point."""
    qubit_names = [q.name for q in qubits]
    n_freq = len(ds.coords["frequency"])
    n_amp = len(ds.coords["amplitude"])
    fidelity = np.zeros((len(qubit_names), n_freq, n_amp))

    for qi, q_name in enumerate(qubit_names):
        for fi in range(n_freq):
            for ai in range(n_amp):
                Ig = ds["Ig"].sel(qubit=q_name).isel(frequency=fi, amplitude=ai).values
                Qg = ds["Qg"].sel(qubit=q_name).isel(frequency=fi, amplitude=ai).values
                Ie = ds["Ie"].sel(qubit=q_name).isel(frequency=fi, amplitude=ai).values
                Qe = ds["Qe"].sel(qubit=q_name).isel(frequency=fi, amplitude=ai).values
                If = ds["If"].sel(qubit=q_name).isel(frequency=fi, amplitude=ai).values
                Qf = ds["Qf"].sel(qubit=q_name).isel(frequency=fi, amplitude=ai).values
                fidelity[qi, fi, ai] = _gef_fidelity(Ig, Qg, Ie, Qe, If, Qf)

    return xr.DataArray(
        fidelity,
        dims=["qubit", "frequency", "amplitude"],
        coords={
            "qubit": qubit_names,
            "frequency": ds.coords["frequency"],
            "amplitude": ds.coords["amplitude"],
        },
        attrs={"long_name": "g/e/f discrimination fidelity"},
    )


def _rotate_iq(I: np.ndarray, Q: np.ndarray, angle: float):
    C, S = np.cos(angle), np.sin(angle)
    return I * C - Q * S, I * S + Q * C


def _ge_rotation_angle(Ig, Qg, Ie, Qe) -> float:
    """Angle that rotates IQ so that the e blob sits to the right of the g blob on the I axis."""
    angle = np.arctan2(np.mean(Qe) - np.mean(Qg), np.mean(Ig) - np.mean(Ie))
    C, S = np.cos(angle), np.sin(angle)
    if np.mean((Ig - Ie) * C - (Qg - Qe) * S) > 0:
        angle += np.pi
    return float(angle)


def _gef_fidelity(Ig, Qg, Ie, Qe, If, Qf) -> float:
    """True 3-state discrimination fidelity at a single (frequency, amplitude) point."""
    angle = _ge_rotation_angle(Ig, Qg, Ie, Qe)
    Ig_rot, Qg_rot = _rotate_iq(Ig, Qg, angle)
    Ie_rot, Qe_rot = _rotate_iq(Ie, Qe, angle)
    If_rot, Qf_rot = _rotate_iq(If, Qf, angle)

    # I-axis threshold: g vs {e, f}
    ef_i = np.concatenate([Ie_rot, If_rot])
    res_i = minimize_scalar(
        lambda t: np.sum(Ig_rot > t) + np.sum(ef_i < t),
        bounds=(min(Ig_rot.min(), ef_i.min()), max(Ig_rot.max(), ef_i.max())),
        method="bounded",
    )
    t_i = res_i.x

    # Q-axis threshold: e vs f (among non-g)
    res_q = minimize_scalar(
        lambda t: np.sum(Qf_rot > t) + np.sum(Qe_rot < t),
        bounds=(min(Qe_rot.min(), Qf_rot.min()), max(Qe_rot.max(), Qf_rot.max())),
        method="bounded",
    )
    t_q = res_q.x

    gg, ge, gf = _classify_three(Ig_rot, Qg_rot, t_i, t_q)
    eg, ee, ef = _classify_three(Ie_rot, Qe_rot, t_i, t_q)
    fg, fe, ff = _classify_three(If_rot, Qf_rot, t_i, t_q)

    return float(1.0 - (ge + eg + fg + gf + ef + fe) / 2.0)


def _classify_three(I_rot, Q_rot, t_i, t_q):
    """Classify single-shot IQ points into g / e / f using two thresholds."""
    n = len(I_rot)
    not_g = I_rot > t_i
    is_e = not_g & (Q_rot >= t_q)
    is_f = not_g & (Q_rot < t_q)
    is_g = ~not_g
    return float(np.sum(is_g) / n), float(np.sum(is_e) / n), float(np.sum(is_f) / n)


def _compute_ge_discriminator_params(Ig, Qg, Ie, Qe):
    """Rotation angle, Nelder-Mead g/e threshold, and RUS threshold from g/e shots."""
    angle = _ge_rotation_angle(Ig, Qg, Ie, Qe)
    Ig_rot, _ = _rotate_iq(Ig, Qg, angle)
    Ie_rot, _ = _rotate_iq(Ie, Qe, angle)

    hist_counts, hist_edges = np.histogram(Ig_rot, bins=100)
    rus_threshold = float(hist_edges[1:][np.argmax(hist_counts)])

    x0 = 0.5 * (float(np.mean(Ig_rot)) + float(np.mean(Ie_rot)))
    fit = minimize(
        lambda t: np.sum(Ig_rot > t[0]) + np.sum(Ie_rot < t[0]),
        x0=[x0],
        method="Nelder-Mead",
    )
    ge_threshold = float(fit.x[0])

    return float(angle), ge_threshold, rus_threshold


def _compute_gef_confusion_matrix(Ig, Qg, Ie, Qe, If, Qf, iw_angle):
    """Full 3×3 confusion matrix and GEF thresholds at a given operating point."""
    Ig_rot, Qg_rot = _rotate_iq(Ig, Qg, iw_angle)
    Ie_rot, Qe_rot = _rotate_iq(Ie, Qe, iw_angle)
    If_rot, Qf_rot = _rotate_iq(If, Qf, iw_angle)

    ef_i = np.concatenate([Ie_rot, If_rot])
    res_i = minimize_scalar(
        lambda t: np.sum(Ig_rot > t) + np.sum(ef_i < t),
        bounds=(min(Ig_rot.min(), ef_i.min()), max(Ig_rot.max(), ef_i.max())),
        method="bounded",
    )
    t_gef_i = res_i.x

    res_q = minimize_scalar(
        lambda t: np.sum(Qf_rot > t) + np.sum(Qe_rot < t),
        bounds=(min(Qe_rot.min(), Qf_rot.min()), max(Qe_rot.max(), Qf_rot.max())),
        method="bounded",
    )
    t_gef_q = res_q.x

    gg, ge, gf = _classify_three(Ig_rot, Qg_rot, t_gef_i, t_gef_q)
    eg, ee, ef = _classify_three(Ie_rot, Qe_rot, t_gef_i, t_gef_q)
    fg, fe, ff = _classify_three(If_rot, Qf_rot, t_gef_i, t_gef_q)

    confusion = [[gg, ge, gf], [eg, ee, ef], [fg, fe, ff]]
    return float(t_gef_i), float(t_gef_q), confusion
