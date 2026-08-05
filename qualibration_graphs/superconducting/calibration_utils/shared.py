"""Shared helper functions for superconducting calibration nodes."""

import numpy as np
import xarray as xr

from qm.qua import align, strict_timing_


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
        p_true = np.einsum("jk,...k->...j", cm_inv, p_meas)
        corrected.loc[dict(qubit=q_name)] = p_true[..., 1]
    return ds.assign(state=corrected)


# ---------------------------------------------------------------------------
# Hardware / QUA helpers
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


def _get_transition_rf(pair, sideband_drive, k):
    """Return the calibrated RF frequency for the f{k}g{k+1} sideband transition.

    Reads from pair.transitions, falling back to pair.extras, then to
    sideband_drive.RF_frequency.
    """
    tr = pair.transitions.get(f"f{k}g{k+1}")
    if tr is not None and tr.RF_frequency is not None:
        return float(tr.RF_frequency)
    extras_key = f"f{k}g{k+1}_RF_frequency"
    if extras_key in (pair.extras or {}):
        return float(pair.extras[extras_key])
    return float(sideband_drive.RF_frequency)


def _ge_if_at_fock(pair, qubit, k):
    """Return the qubit ge IF when cavity is in Fock |k⟩.

    Computes k × chi (linear dispersive shift) and adds delta_f_focka from
    pair.transitions["f{k-1}g{k}"] when calibrated.  delta_f_focka is the
    nonlinear correction beyond the linear approximation; it is zero for k=1
    by convention since chi is defined from that measurement.
    """
    chi = pair.chi if pair.chi is not None else 0.0
    base = int(qubit.xy.intermediate_frequency) + int(k * chi)
    if k > 0:
        tr = pair.transitions.get(f"f{k-1}g{k}")
        if tr is not None and tr.delta_f_focka is not None:
            return base + int(tr.delta_f_focka)
    return base


def _ef_if_at_fock(pair, qubit, k):
    """Return the qubit ef IF when cavity is in Fock |k⟩.

    Uses ef_delta_f_focka from pair.transitions["f{k-1}g{k}"] when calibrated
    (deviation of the ef-ge gap from the bare anharmonicity due to photon-number-
    dependent Kerr shift), falling back to bare anharmonicity + k × chi.
    """
    ge_if_k = _ge_if_at_fock(pair, qubit, k)
    if k > 0:
        tr = pair.transitions.get(f"f{k-1}g{k}")
        if tr is not None and tr.ef_delta_f_focka is not None:
            return ge_if_k + int(qubit.anharmonicity) + int(tr.ef_delta_f_focka)
    chi = pair.chi if pair.chi is not None else 0.0
    return ge_if_k + int(qubit.anharmonicity) + int(k * chi)


def _resolve_sb_op(sideband_drive, k):
    """Return the operation name for f{k}g{k+1}, falling back to sideband_flat_top."""
    per_level = f"f{k}g{k+1}_pi"
    return per_level if per_level in sideband_drive.operations else "sideband_flat_top"


def _fock_prep_qua(fock_level, pair, qubit, sideband_drive):
    """Generate QUA statements to prepare Fock |fock_level⟩ in the cavity.

    Iterates j = 0 … fock_level-1; each iteration:
      - prepares qubit in |f⟩ using Fock-j-corrected ge/ef frequencies
      - sets sideband IF to the f{j}g{j+1} transition
      - plays the sideband π-pulse  →  cavity advances from |j⟩ to |j+1⟩,
        qubit returns to |g⟩ naturally.

    The ge and ef π-pulses at step j are driven at the dispersive-shifted
    frequencies for j photons in the cavity (delta_f_focka and ef_delta_f_focka
    from pair.transitions["f{j-1}g{j}"], with linear-dispersive fallbacks).
    """
    for j in range(fock_level):
        sb_rf_j = _get_transition_rf(pair, sideband_drive, j)
        if_offset_j = int(sb_rf_j - sideband_drive.RF_frequency)
        target_if_j = int(sideband_drive.intermediate_frequency) + if_offset_j

        tr_j = pair.transitions.get(f"f{j}g{j+1}")
        flat_top_clk_j = (tr_j.pi_flat_top_length_ns // 4) if (tr_j and tr_j.pi_flat_top_length_ns) else None

        ge_if_j = _ge_if_at_fock(pair, qubit, j)
        ef_if_j = _ef_if_at_fock(pair, qubit, j)

        qubit.xy.update_frequency(ge_if_j)
        qubit.xy.play("x180")
        qubit.xy.update_frequency(ef_if_j)
        qubit.xy.play("EF_x180")

        align(qubit.xy.name, sideband_drive.name)
        sideband_drive.update_frequency(target_if_j)
        with strict_timing_():
            sideband_drive.play("sideband_ramp_up")
            if flat_top_clk_j is not None:
                sideband_drive.play("sideband_square", duration=flat_top_clk_j)
            else:
                sideband_drive.play("sideband_square")
            sideband_drive.play("sideband_ramp_down")

        align(sideband_drive.name, qubit.xy.name)
