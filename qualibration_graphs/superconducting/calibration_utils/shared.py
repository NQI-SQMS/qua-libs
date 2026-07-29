"""Shared helper functions for superconducting calibration nodes."""

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def apply_confusion_matrix_correction(ds: xr.Dataset, qubits) -> xr.Dataset:
    """Correct averaged state probabilities for ge readout errors using stored confusion matrices."""
    if "state" not in ds.data_vars:
        return ds
    cm_inv_map = {}
    for qubit in qubits:
        cm = getattr(qubit.resonator, "confusion_matrix", None)
        if cm is not None:
            cm_inv_map[qubit.name] = np.linalg.inv(np.array(cm))
    if not cm_inv_map:
        return ds
    corrected = ds.state.copy(deep=True)
    for q_name, cm_inv in cm_inv_map.items():
        s = ds.state.sel(qubit=q_name).values.astype(float)
        p_meas = np.stack([1.0 - s, s], axis=-1)
        p_true = (cm_inv @ p_meas.T).T
        corrected.loc[dict(qubit=q_name)] = p_true[..., 1]
    return ds.assign(state=corrected)


# ---------------------------------------------------------------------------
# Node-level lookup helpers (take Qualibrate node — cannot live in quam-builder)
# ---------------------------------------------------------------------------

def _get_cavity_mode(node):
    """Return the cavity mode object matching node.parameters.mode_name."""
    mode_name = node.parameters.mode_name
    for cav in node.machine.cavities.values():
        mode = getattr(cav, mode_name, None)
        if mode is not None:
            return mode
    raise KeyError(f"Cavity mode '{mode_name}' not found in machine.cavities")


def _get_pair(node):
    """Return the CavityTransmonPair for the current mode, or None if not found."""
    mode_name = node.parameters.mode_name
    for pair in node.machine.cavity_transmon_pairs.values():
        if pair.cavity_mode_name == mode_name:
            return pair
    return None


def _get_pair_components(node):
    """Return (pair, qubit, sideband_drive, cav_mode) for the requested mode."""
    mode_name = node.parameters.mode_name
    for pair in node.machine.cavity_transmon_pairs.values():
        if pair.cavity_mode_name == mode_name:
            qubit = node.machine.qubits[pair.qubit_name]
            cav_mode = next(
                (getattr(cav, mode_name, None)
                 for cav in node.machine.cavities.values()
                 if getattr(cav, mode_name, None) is not None),
                None,
            )
            return pair, qubit, pair.sideband_drive, cav_mode
    raise KeyError(f"No cavity_transmon_pair with cavity_mode_name='{mode_name}'")
