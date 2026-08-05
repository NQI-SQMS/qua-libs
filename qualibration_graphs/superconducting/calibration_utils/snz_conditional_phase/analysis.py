"""Analysis functions for the SNZ conditional phase measurement.

Fits oscillations along the frame (phase tomography) dimension for each
(amplitude, t_phi_eff, control_axis) slice, then computes the conditional
phase difference between control-on and control-off.  Finds the optimal
operating point using a leakage-first threshold approach.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation, oscillation

from calibration_utils.snz_b_over_a import decompose_t_phi_eff


@dataclass
class FitResults:
    """Stores the extracted results for a single qubit pair."""

    optimal_amplitude: float
    optimal_t_phi_eff: float
    optimal_t_phi: int
    optimal_b_over_a: float
    optimal_phase: float
    optimal_leakage: float
    success: bool


def fix_oscillation_phi_2pi(fit_data):
    """Extract phase from oscillation fit and normalize to [0, 1] (units of 2pi)."""
    phase = fit_data.sel(fit_vals="phi")
    phase = (phase / (2 * np.pi)) % 1
    return phase


def _circ_dist_to_half(z):
    """Circular distance of values in [0,1) to 0.5, returned in [0, 0.5]."""
    return np.abs(((z - 0.5 + 0.5) % 1.0) - 0.5)


def _find_optimal_point(qp_ds, leak_percentile=20.0):
    """Find the optimal (amplitude, t_phi_eff) using leakage-first threshold.

    1. Get f-state leakage (control_axis=1, averaged over frame).
    2. Keep only points where leakage < percentile(leakage, leak_percentile).
    3. Among those, find the point with phase_diff closest to 0.5 (pi).

    Returns (opt_amp, opt_tpe, opt_phase, opt_leakage, success).
    """
    _fail = (1.0, 0.0, 0, 1.0, float("nan"), float("nan"), False)

    if "f_state_control" in qp_ds.data_vars:
        leakage = qp_ds.f_state_control.sel(control_axis=1).mean(dim="frame")
    elif "I_control" in qp_ds.data_vars:
        leakage = qp_ds.I_control
        if "control_axis" in leakage.dims:
            leakage = leakage.sel(control_axis=1)
        if "frame" in leakage.dims:
            leakage = leakage.mean(dim="frame")
    else:
        return _fail

    if "phase_diff" not in qp_ds.data_vars:
        return _fail

    phase = qp_ds.phase_diff

    # Align by coordinate labels, then force the same dimension order before
    # switching to NumPy. phase_diff is commonly (t_phi_eff, amplitude) while
    # leakage is (amplitude, t_phi_eff).
    phase_aligned, leakage_aligned = xr.align(phase, leakage, join="inner")
    if not {"amplitude", "t_phi_eff"}.issubset(phase_aligned.dims):
        return _fail
    if not set(leakage_aligned.dims).issubset(set(phase_aligned.dims)):
        return _fail
    phase_aligned = phase_aligned.transpose("t_phi_eff", "amplitude")
    leakage_aligned = leakage_aligned.transpose(*phase_aligned.dims)

    phase_vals = phase_aligned.values
    leak_vals = leakage_aligned.values

    finite = np.isfinite(phase_vals) & np.isfinite(leak_vals)
    if not phase_vals.size or not leak_vals.size or not np.any(finite):
        return _fail

    threshold = np.nanpercentile(leak_vals[finite], leak_percentile)
    mask = (leak_vals <= threshold) & finite

    if not np.any(mask):
        mask = finite

    circ_dist = _circ_dist_to_half(phase_vals)
    circ_dist[~mask] = np.inf

    if not np.any(np.isfinite(circ_dist)):
        return _fail

    flat_idx = np.nanargmin(circ_dist)
    idx = np.unravel_index(flat_idx, circ_dist.shape)
    idx_by_dim = dict(zip(phase_aligned.dims, idx))

    amp_coord = phase_aligned.amplitude.values
    tpe_coord = phase_aligned.t_phi_eff.values

    opt_amp = float(amp_coord[idx_by_dim["amplitude"]])
    opt_tpe = float(tpe_coord[idx_by_dim["t_phi_eff"]])
    opt_t_phi, opt_ba = decompose_t_phi_eff(opt_tpe)
    opt_phase = float(phase_vals[idx])
    opt_leakage = float(leak_vals[idx])

    return opt_amp, opt_tpe, opt_t_phi, opt_ba, opt_phase, opt_leakage, True


def log_fitted_results(fit_results: Dict[str, FitResults], log_callable=None):
    """Log the fitted results for every qubit pair."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qp_name, fr in fit_results.items():
        status = "SUCCESS" if fr.success else "FAIL"
        msg = (
            f"Results for qubit pair {qp_name}: {status}!\n"
            f"\tOptimal amplitude  : {fr.optimal_amplitude:.6f} (relative)\n"
            f"\tOptimal t_phi_eff  : {fr.optimal_t_phi_eff:.4f} ns\n"
            f"\t  -> t_phi         : {fr.optimal_t_phi} ns\n"
            f"\t  -> B/A           : {fr.optimal_b_over_a:.4f}\n"
            f"\tConditional phase  : {fr.optimal_phase:.4f} (2\u03c0 units, target=0.5)\n"
            f"\tLeakage at optimum : {fr.optimal_leakage:.4f}"
        )
        log_callable(msg)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Add absolute-amplitude coordinate to the raw dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset straight from the OPX.
    node : QualibrationNode
        Node carrying qubit-pair objects and parameters.

    Returns
    -------
    xr.Dataset
        Dataset enriched with an ``amp_full`` coordinate (volts).
    """
    qubit_pairs = node.namespace["qubit_pairs"]
    operation = node.parameters.operation

    def abs_amp(qp, amp_rel):
        return amp_rel * qp.macros[operation].flux_pulse_qubit.amplitude

    ds = ds.assign_coords(
        {
            "amp_full": (
                ["qubit_pair", "amplitude"],
                np.array([abs_amp(qp, ds.amplitude.values) for qp in qubit_pairs]),
            )
        }
    )
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitResults]]:
    """Fit oscillations along the frame dimension, compute phase difference,
    and find the optimal operating point.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset with dimensions
        ``(qubit_pair, amplitude, t_phi_eff, frame, control_axis)``.
    node : QualibrationNode
        Node with qubit-pair objects and parameters.

    Returns
    -------
    tuple of (xr.Dataset, dict)
        The enriched dataset (with ``fitted`` and ``phase_diff`` data
        variables) and a dictionary of :class:`FitResults`.
    """
    ds_fit = ds.groupby("qubit_pair").apply(_fit_routine)

    qubit_pairs = node.namespace["qubit_pairs"]
    leak_percentile = node.parameters.leak_percentile
    fit_results: Dict[str, FitResults] = {}

    for qp in qubit_pairs:
        qp_ds = ds_fit.sel(qubit_pair=qp.name)
        opt_amp, opt_tpe, opt_t_phi, opt_ba, opt_phase, opt_leak, success = _find_optimal_point(qp_ds, leak_percentile)
        fit_results[qp.name] = FitResults(
            optimal_amplitude=opt_amp,
            optimal_t_phi_eff=opt_tpe,
            optimal_t_phi=opt_t_phi,
            optimal_b_over_a=opt_ba,
            optimal_phase=opt_phase,
            optimal_leakage=opt_leak,
            success=success,
        )

    return ds_fit, fit_results


def _fit_routine(da):
    """Fit oscillations for a single qubit pair.

    Mirrors the workflow used by 33_cz_conditional_phase_error_amp:
    iterate over the outer sweep, fit the target signal vs frame for each
    (amplitude, control_axis) combination, then compute the same
    (control=0 - control=1) modulo-1 phase convention.
    Computes phase_diff = (phase[control=0] - phase[control=1]) % 1.
    """
    data_var = "state_target" if "state_target" in da else "I_target"
    tpe_vals = da.t_phi_eff.values

    fitted_list = []
    phase_diff_list = []

    for tpe in tpe_vals:
        da_sel = da.sel(t_phi_eff=tpe)
        fit_data = fit_oscillation(da_sel[data_var], "frame")

        fitted_curve = (
            oscillation(
                da_sel.frame,
                fit_data.sel(fit_vals="a"),
                fit_data.sel(fit_vals="f"),
                fit_data.sel(fit_vals="phi"),
                fit_data.sel(fit_vals="offset"),
            )
            .rename("fitted")
            .expand_dims(t_phi_eff=[tpe])
        )
        fitted_list.append(fitted_curve)

        phase = fix_oscillation_phi_2pi(fit_data)
        phase_diff = (
            ((phase.sel(control_axis=0) - phase.sel(control_axis=1)) % 1)
            .rename("phase_diff")
            .expand_dims(t_phi_eff=[tpe])
        )
        phase_diff_list.append(phase_diff)

    to_assign = {}
    if fitted_list:
        to_assign["fitted"] = xr.concat(fitted_list, dim="t_phi_eff")
    if phase_diff_list:
        to_assign["phase_diff"] = xr.concat(phase_diff_list, dim="t_phi_eff")

    if to_assign:
        da = da.assign(to_assign)

    return da
