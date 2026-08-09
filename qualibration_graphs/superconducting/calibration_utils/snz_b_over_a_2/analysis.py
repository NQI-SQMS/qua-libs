"""Analysis functions for the SNZ t_phi_eff scan.

This module provides processing, fitting, and logging utilities for
the 2-D SNZ experiment (amplitude x t_phi_eff).
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode


@dataclass
class FitResults:
    """Stores the extracted results for a single qubit pair."""

    optimal_t_phi_eff: float
    optimal_amplitude: float
    min_leakage: float
    success: bool


def _field(fr, name):
    """Read a fit-result field whether ``fr`` is a :class:`FitResults` dataclass
    instance or a plain dict (e.g. produced by ``dataclasses.asdict``).

    The node passes ``node.results["fit_results"]`` -- which is
    ``{name: asdict(FitResults)}`` -- to this function, so ``fr`` is a dict at
    runtime. Supporting both forms avoids the ``'dict' object has no attribute
    'success'`` error while remaining usable with raw dataclass instances.
    """
    return fr[name] if isinstance(fr, dict) else getattr(fr, name)


def log_fitted_results(fit_results: Dict[str, FitResults], log_callable=None):
    """Log the fitted results for every qubit pair.

    Parameters
    ----------
    fit_results : dict
        Mapping of qubit-pair name to :class:`FitResults` *or* to its
        ``asdict`` dictionary (the node passes the dict form).
    log_callable : callable, optional
        Logger function.  Falls back to the module logger when *None*.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qp_name, fr in fit_results.items():
        status = "SUCCESS" if _field(fr, "success") else "FAIL"
        msg = (
            f"Results for qubit pair {qp_name}: {status}!\n"
            f"\tOptimal t_phi_eff : {_field(fr, 'optimal_t_phi_eff'):.4f} ns\n"
            f"\tOptimal amplitude : {_field(fr, 'optimal_amplitude'):.6f} (relative)\n"
            f"\tMin leakage       : {_field(fr, 'min_leakage'):.4f}"
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
    """Find the optimal (amplitude, t_phi_eff) point from the 2-D data.

    The optimal point is chosen by minimizing leakage to the |f> state
    of the control qubit (when using state discrimination) or by
    minimizing the control-qubit signal otherwise.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset with ``amplitude`` and ``t_phi_eff`` dimensions.
    node : QualibrationNode
        Node with qubit-pair objects.

    Returns
    -------
    tuple of (xr.Dataset, dict)
        The enriched dataset and a dictionary of :class:`FitResults`
        keyed by qubit-pair name.
    """
    qubit_pairs = node.namespace["qubit_pairs"]
    fit_results: Dict[str, FitResults] = {}

    opt_tpe_list = []
    opt_amp_list = []

    for qp in qubit_pairs:
        qp_name = qp.name
        qp_ds = ds.sel(qubit_pair=qp_name)

        try:
            if "f_state_control" in qp_ds.data_vars:
                leakage = qp_ds.f_state_control
            elif "I_control" in qp_ds.data_vars:
                leakage = qp_ds.I_control
            else:
                raise KeyError("No suitable leakage variable found")

            min_idx = leakage.argmin(dim=["amplitude", "t_phi_eff"])
            opt_amp_idx = int(min_idx["amplitude"])
            opt_tpe_idx = int(min_idx["t_phi_eff"])
            opt_amp = float(qp_ds.amplitude.values[opt_amp_idx])
            opt_tpe = float(qp_ds.t_phi_eff.values[opt_tpe_idx])
            min_leak = float(leakage.values[opt_amp_idx, opt_tpe_idx])
            success = True
        except Exception:
            opt_amp = 1.0
            opt_tpe = 0.0
            min_leak = float("nan")
            success = False

        opt_tpe_list.append(opt_tpe)
        opt_amp_list.append(opt_amp)
        fit_results[qp_name] = FitResults(
            optimal_t_phi_eff=opt_tpe,
            optimal_amplitude=opt_amp,
            min_leakage=min_leak,
            success=success,
        )

    ds = ds.assign_coords(
        {
            "optimal_t_phi_eff": ("qubit_pair", opt_tpe_list),
            "optimal_amplitude": ("qubit_pair", opt_amp_list),
        }
    )
    return ds, fit_results
