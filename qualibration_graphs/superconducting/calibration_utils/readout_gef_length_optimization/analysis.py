import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode


@dataclass
class FitParameters:
    """Fit results for a single qubit's GEF readout length optimisation."""

    optimal_readout_length_ns: int
    """Optimal readout pulse length [ns]."""
    optimal_distance: float
    """min(D_ge, D_ef, D_gf) at the optimal length [V]."""
    num_divisions: int
    distances: List[float]
    """min-distance curve vs readout length."""
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, r in fit_results.items():
        status = "SUCCESS!" if r["success"] else "FAIL!"
        log_callable(
            f"GEF readout length for qubit {q}: optimal = {r['optimal_readout_length_ns']} ns, "
            f"min-distance = {1e3 * r['optimal_distance']:.2f} mV --> {status}"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Compute pairwise centroid distances at each accumulated readout length."""
    qubits = node.namespace["qubits"]
    division_length_ns = node.parameters.division_length_in_ns

    # Collect distance curves per qubit into a structured Dataset
    all_Dge, all_Def, all_Dgf, all_Dist = [], [], [], []
    q_names = []

    for q in qubits:
        qname = q.name
        # Each variable shape: (n_divisions,) — already averaged over shots
        Ig = ds[f"Ig_{qname}"].values.astype(float)
        Qg = ds[f"Qg_{qname}"].values.astype(float)
        Ie = ds[f"Ie_{qname}"].values.astype(float)
        Qe = ds[f"Qe_{qname}"].values.astype(float)
        If_ = ds[f"If_{qname}"].values.astype(float)
        Qf = ds[f"Qf_{qname}"].values.astype(float)

        Dge = np.sqrt((Ig - Ie) ** 2 + (Qg - Qe) ** 2)
        Def = np.sqrt((Ie - If_) ** 2 + (Qe - Qf) ** 2)
        Dgf = np.sqrt((Ig - If_) ** 2 + (Qg - Qf) ** 2)
        Dist = np.minimum(np.minimum(Dge, Def), Dgf)

        all_Dge.append(Dge)
        all_Def.append(Def)
        all_Dgf.append(Dgf)
        all_Dist.append(Dist)
        q_names.append(qname)

    n_div = len(all_Dge[0])
    lengths_ns = (np.arange(n_div) + 1) * division_length_ns

    coords = {
        "qubit": q_names,
        "readout_length_ns": lengths_ns,
    }

    ds_proc = xr.Dataset(
        {
            "Dge":      xr.DataArray(np.array(all_Dge), dims=["qubit", "readout_length_ns"], coords=coords),
            "Def":      xr.DataArray(np.array(all_Def), dims=["qubit", "readout_length_ns"], coords=coords),
            "Dgf":      xr.DataArray(np.array(all_Dgf), dims=["qubit", "readout_length_ns"], coords=coords),
            "Distance": xr.DataArray(np.array(all_Dist), dims=["qubit", "readout_length_ns"], coords=coords),
        }
    )
    return ds_proc


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Find the readout length that maximises min(D_ge, D_ef, D_gf)."""
    qubits = node.namespace["qubits"]
    division_length_ns = node.parameters.division_length_in_ns
    fit_results = {}
    opt_lengths = []

    for q in qubits:
        qname = q.name
        dist = ds.sel(qubit=qname).Distance.values.astype(float)

        # Light smoothing to reduce noise
        kernel = np.ones(3) / 3
        dist_smooth = np.convolve(dist, kernel, mode="same")

        valid = np.isfinite(dist_smooth)
        if valid.any():
            best_div = int(np.nanargmax(dist_smooth))
            opt_len_ns = int((best_div + 1) * division_length_ns)
            opt_dist = float(dist[best_div])
            success = True
        else:
            opt_len_ns = int(len(dist) * division_length_ns)
            opt_dist = float("nan")
            success = False

        opt_lengths.append(opt_len_ns)
        fit_results[qname] = FitParameters(
            optimal_readout_length_ns=opt_len_ns,
            optimal_distance=opt_dist,
            num_divisions=len(dist),
            distances=dist.tolist(),
            success=success,
        )

    ds_fit = ds.assign(
        {
            "optimal_readout_length_ns": xr.DataArray(
                opt_lengths,
                dims=["qubit"],
                coords={"qubit": [q.name for q in qubits]},
            )
        }
    )
    return ds_fit, fit_results
