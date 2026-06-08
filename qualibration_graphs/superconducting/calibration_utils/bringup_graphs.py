"""
Shared subgraph builders and utility helpers for fixed-frequency transmon bringup.

# ── Migration note (2026-05-27) ────────────────────────────────────────────────
# The following have been REMOVED as part of the calib_framework migration:
#   - All FSM condition functions for resonator/qubit bringup:
#       should_retry_resonator_discovery, should_repeat_punch_out,
#       should_repeat_spec_vs_power, should_repeat_rabi_amplitude,
#       should_restart_qubit_calibration
#   - BO cost helpers: spec_vs_power_bo_cost, power_rabi_bo_cost
#   - Graph builders: build_resonator_bringup, build_qubit_calibration,
#       build_x180_fine_calibration
#   - import of calibration_utils.error_codes (error_codes.py deleted)
#
# These are fully replaced by:
#   calib_framework.bo.node_controller.BONodeController (BIC + GP-BO retry)
#   calib_framework.core.bic.BICDiagnoser            (fit diagnosis)
#   calib_framework.core.orchestrator.CausalOrchestrator (DAG routing)
#
# See qualibrate_graphs/bringup_causal.py for the new bringup graph.
# The frozen FSM baseline is at:
#   calibrations/1Q_calibrations/92_BASELINE_bringup_fsm_frozen.py
# ──────────────────────────────────────────────────────────────────────────────

Used by:
  - calibrations/1Q_calibrations/05_x180_fine_calibration_graph.py  (helpers only)
  - calibrations/1Q_calibrations/92_BASELINE_bringup_fsm_frozen.py  (FROZEN — do not use)
  - EF and cavity bringup graphs
"""

import logging
from enum import IntEnum
from typing import List, Optional

import numpy as np
from qualibrate import GraphParameters, QualibrationGraph, QualibrationLibrary, QualibrationNode

logger = logging.getLogger(__name__)


# ── Minimal PowerRabiErrorCode (inlined — error_codes.py removed 2026-05-27) ──
# Used only by the EF bringup section below.  The bringup FSM functions that
# previously used this enum have been deleted; this stub preserves the values
# needed by build_ef_bringup without re-introducing error_codes.py.

class _PowerRabiErrorCode(IntEnum):
    SUCCESS = 0
    NO_OSCILLATION = 1
    TOO_MANY_PERIODS = 2
    TOO_FEW_PERIODS = 3


# ── QUA utility helpers ────────────────────────────────────────────────────────
# These are shared across multiple graphs and are NOT FSM logic; they are
# retained as pure utility functions.


def _get_machine(node):
    """Return the machine from a node or subgraph.

    When looping over a subgraph (QualibrationGraph), the condition function
    receives the subgraph itself rather than a QualibrationNode.  Subgraphs
    store child elements in ``_elements`` (not ``nodes``), and the machine
    lives on the individual child nodes after they have run.
    """
    if hasattr(node, "machine") and node.machine is not None:
        return node.machine
    for child in node._elements.values():
        if hasattr(child, "machine") and child.machine is not None:
            return child.machine
        grandchild_machine = _get_machine(child) if hasattr(child, "_elements") else None
        if grandchild_machine is not None:
            return grandchild_machine
    return None


def _ensure_temp_calibration(machine, qubit_name: str):
    """Return TemporaryCalibrationData for *qubit_name*, creating it if absent."""
    from quam_config.my_quam import TemporaryCalibrationData

    if machine.temp_calibration is None:
        machine.temp_calibration = {}
    if qubit_name not in machine.temp_calibration:
        machine.temp_calibration[qubit_name] = TemporaryCalibrationData()
    temp = machine.temp_calibration[qubit_name]
    for field in (
        "initial_x180_amplitude", "initial_qubit_f01", "initial_rf_frequency",
        "qubit_calibration_succeeded",   # True=OK, False=exhausted, None=unknown
    ):
        if not hasattr(temp, field):
            object.__setattr__(temp, field, None)
    return temp


def _restore_initial_state(machine, target: str, loop_state: dict) -> None:
    """Restore x180/x90 amplitude and f_01/RF_frequency to their pre-loop values.

    After restoring in-memory values, ``machine.save()`` is called to persist
    the restored state to the JSON file.  This is necessary because the Ramsey
    node's ``update_state`` already called ``machine.save()`` with the
    Ramsey-modified frequencies; without an explicit save here those would
    remain in the persistent state even though the in-memory values are correct.
    """
    q = machine.qubits[target]
    init_amp = loop_state["initial_x180_amplitude"].get(target)
    init_x90_amp = loop_state["initial_x90_amplitude"].get(target)
    init_f01 = loop_state["initial_f01"].get(target)
    init_rf = loop_state["initial_rf_frequency"].get(target)
    if init_amp is not None:
        q.xy.operations["x180"].amplitude = init_amp
        if "x90" in q.xy.operations:
            q.xy.operations["x90"].amplitude = init_x90_amp or init_amp / 2.0
        logger.info(f"[X180 fine] {target}: Restored x180 amplitude to {1e3 * init_amp:.2f} mV.")
    if init_f01 is not None:
        q.f_01 = init_f01
        q.xy.RF_frequency = init_rf
        logger.info(
            f"[X180 fine] {target}: Restored f_01 to {init_f01 / 1e9:.6f} GHz."
        )
    try:
        machine.save()
    except Exception as exc:
        logger.warning(f"[X180 fine] {target}: machine.save() after restore failed: {exc}")


# ── EF-transition condition function ──────────────────────────────────────────

def should_repeat_ef_spec(node: QualibrationNode, target: str) -> bool:
    """Retry EF spectroscopy when no transition peak was found.

    The node's own update_state() handles adaptive span adjustments before
    the next iteration.  No temp_calibration state is needed for EF (the
    EF frequency is derived from the known anharmonicity, not a blind search).
    """
    if node.outcomes.get(target) == "failed":
        logger.info(f"{target}: EF spectroscopy failed; retrying.")
        return True
    logger.info(f"{target}: EF spectroscopy succeeded.")
    return False


# ── Inner subgraph parameter stubs (EF and cavity) ────────────────────────────

class _EFCalibrationSubgraphParameters(GraphParameters):
    qubits: List[str] = ["q0"]


class _EFDiscoverySubgraphParameters(GraphParameters):
    qubits: List[str] = ["q0"]


class _EFRabiRamseySubgraphParameters(GraphParameters):
    qubits: List[str] = ["q0"]


class _CavityCalibrationSubgraphParameters(GraphParameters):
    qubits: List[str] = ["q0"]


# ── EF bringup subgraph builder ────────────────────────────────────────────────

def build_ef_bringup(
    graph: QualibrationGraph, library: QualibrationLibrary
) -> QualibrationGraph:
    """Build and return the ``ef_bringup`` subgraph.

    Sequence::

        ef_discovery [loop: should_repeat_ef_discovery, max_ef_discovery_iterations]:
          ef_spectroscopy  [loop: should_repeat_ef_spec, max_ef_spec_iterations]
          → ef_tentative_rabi   (amplitude-only convergence, no Octave gain changes)
          If NO_OSCILLATION: blacklist EF freq, restart ef_discovery.
        → ef_rabi_ramsey [loop: should_repeat_ef_calibration until EF detuning converges]:
              ef_power_rabi [inner loop: amplitude only, max_ef_rabi_iterations]
              → ef_ramsey
        → ef_T1
        → gef_readout_frequency_optimization
        → gef_iq_blobs

    The EF power Rabi (13_power_rabi_ef) never modifies the Octave gain —
    it only adjusts EF_x180.amplitude.
    """
    p = graph.parameters

    # Per-call isolated loop states
    _ef_loop_state: dict = {"initialized": {}, "detuning_history": {}}

    # ── should_repeat_ef_discovery ────────────────────────────────────────────
    def should_repeat_ef_discovery(node: QualibrationNode, target: str) -> bool:
        """Restart EF spectroscopy when the tentative Rabi shows no oscillation.

        On NO_OSCILLATION the current EF frequency estimate is recorded in
        temp_calibration as a hint for subsequent spectroscopy attempts.
        Any other outcome (SUCCESS, TOO_MANY, TOO_FEW) means the EF transition
        was found and we proceed to the fine-calibration loop.
        """
        tentative_node = getattr(node, "_elements", {}).get("ef_tentative_rabi")
        if tentative_node is None:
            return False

        error_code = (
            tentative_node.results.get("fit_results", {})
            .get(target, {})
            .get("error_code", int(_PowerRabiErrorCode.SUCCESS))
        )

        if error_code == int(_PowerRabiErrorCode.NO_OSCILLATION):
            logger.warning(
                f"[EF discovery] {target}: Tentative EF Rabi found no oscillation. "
                "Blacklisting EF frequency estimate and restarting spectroscopy."
            )
            machine = _get_machine(node)
            if machine is not None:
                try:
                    temp = _ensure_temp_calibration(machine, target)
                    q = machine.qubits[target]
                    # EF drive freq ≈ qubit f_01 + anharmonicity
                    ef_freq = float(q.f_01) + float(q.anharmonicity)
                    if not hasattr(temp, "blacklisted_ef_frequencies"):
                        object.__setattr__(temp, "blacklisted_ef_frequencies", [])
                    if ef_freq not in temp.blacklisted_ef_frequencies:
                        temp.blacklisted_ef_frequencies.append(ef_freq)
                        logger.info(
                            f"[EF discovery] {target}: Blacklisted EF freq "
                            f"{ef_freq / 1e9:.6f} GHz."
                        )
                except Exception as exc:
                    logger.warning(f"[EF discovery] {target}: Could not store EF blacklist: {exc}")
            return True  # restart ef_discovery (spectroscopy + tentative rabi)

        # SUCCESS / TOO_MANY / TOO_FEW → EF transition found, proceed.
        return False

    # ── _should_repeat_ef_rabi_amplitude ─────────────────────────────────────
    def _should_repeat_ef_rabi_amplitude(node: QualibrationNode, target: str) -> bool:
        """Retry EF power_rabi when amplitude mismatch is detected.

        Replaces the removed should_repeat_rabi_amplitude for EF-specific use.
        TOO_MANY/TOO_FEW: node's update_state has already rescaled amplitude.
        NO_OSCILLATION: stop — the discovery loop will restart.
        """
        error_code = (
            node.results.get("fit_results", {})
            .get(target, {})
            .get("error_code", int(_PowerRabiErrorCode.SUCCESS))
        )
        if error_code in (
            int(_PowerRabiErrorCode.TOO_MANY_PERIODS),
            int(_PowerRabiErrorCode.TOO_FEW_PERIODS),
        ):
            logger.info(
                f"{target}: EF Rabi amplitude mismatch "
                f"({_PowerRabiErrorCode(error_code).name}); retrying."
            )
            return True
        if error_code != int(_PowerRabiErrorCode.SUCCESS):
            logger.info(
                f"{target}: EF Rabi failed ({_PowerRabiErrorCode(error_code).name}); stopping."
            )
        return False

    # ── should_repeat_ef_calibration ─────────────────────────────────────────
    def should_repeat_ef_calibration(node: QualibrationNode, target: str) -> bool:
        """Loop ef_power_rabi → ef_ramsey until |EF detuning| < ef_freq_threshold_hz."""
        if not _ef_loop_state["initialized"].get(target, False):
            _ef_loop_state["detuning_history"][target] = []
            _ef_loop_state["initialized"][target] = True

        if node.outcomes.get(target) == "failed":
            logger.warning(f"[EF fine] {target}: Fit failed — stopping EF calibration loop.")
            _ef_loop_state["initialized"][target] = False
            return False

        _ramsey_node = getattr(node, "_elements", {}).get("ef_ramsey")
        _ramsey_results = _ramsey_node.results if _ramsey_node is not None else {}
        freq_offset = (
            _ramsey_results.get("fit_results", {})
            .get(target, {})
            .get("freq_offset", None)
        )

        if freq_offset is None:
            _ef_loop_state["initialized"][target] = False
            return False

        abs_offset = abs(freq_offset)
        _ef_loop_state["detuning_history"][target].append(abs_offset)
        logger.info(
            f"[EF fine] {target}: |EF detuning| = {abs_offset / 1e3:.2f} kHz, "
            f"threshold = {p.ef_freq_threshold_hz / 1e3:.0f} kHz."
        )

        if abs_offset < p.ef_freq_threshold_hz:
            logger.info(
                f"[EF fine] {target}: Converged after "
                f"{len(_ef_loop_state['detuning_history'][target])} iteration(s)."
            )
            _ef_loop_state["initialized"][target] = False
            return False

        return True

    # ── Graph construction ────────────────────────────────────────────────────
    with QualibrationGraph.build(
        "ef_bringup",
        parameters=_EFCalibrationSubgraphParameters(),
    ) as ef_bringup:

        # ── ef_discovery: spectroscopy + tentative Rabi ───────────────────────
        with QualibrationGraph.build(
            "ef_discovery",
            parameters=_EFDiscoverySubgraphParameters(),
        ) as ef_discovery:

            ef_spec = library.nodes["12_qubit_spectroscopy_EF"].copy(
                name="ef_spectroscopy",
                frequency_span_in_mhz=p.ef_spec_frequency_span_mhz,
                frequency_step_in_mhz=p.ef_spec_frequency_step_mhz,
                operation=p.ef_spec_operation,
                operation_len_in_ns=p.ef_spec_operation_len_in_ns,
                operation_amplitude_factor=p.ef_spec_amplitude_factor,
                num_shots=p.ef_spec_num_shots,
                target_peak_width=p.ef_spec_target_peak_width,
                update_pulses_amplitude=p.ef_spec_update_pulses_amplitude,
                find_dip=p.ef_spec_find_dip,
                update_integration_weights_angle=False,
            )
            ef_discovery.add_node(ef_spec)
            ef_discovery.loop(
                ef_spec,
                on=should_repeat_ef_spec,
                max_iterations=p.max_ef_spec_iterations,
            )

            # Tentative Rabi: amplitude-only convergence, no Octave gain changes.
            # 13_power_rabi_ef.update_state only sets EF_x180.amplitude — safe.
            ef_tentative_rabi = library.nodes["13_power_rabi_ef"].copy(
                name="ef_tentative_rabi",
                min_amp_factor=p.ef_rabi_min_amp_factor,
                max_amp_factor=p.ef_rabi_max_amp_factor,
                amp_factor_step=p.ef_rabi_amp_factor_step,
                num_shots=p.ef_rabi_num_shots,
            )
            ef_discovery.add_node(ef_tentative_rabi)
            ef_discovery.connect(ef_spec, ef_tentative_rabi)

        ef_bringup.add_node(ef_discovery)
        ef_bringup.loop(
            ef_discovery,
            on=should_repeat_ef_discovery,
            max_iterations=p.max_ef_discovery_iterations,
        )

        # ── EF fine calibration: power_rabi → ramsey [convergence loop] ───────
        with QualibrationGraph.build(
            "ef_rabi_ramsey",
            parameters=_EFRabiRamseySubgraphParameters(),
        ) as ef_rabi_ramsey:

            # Amplitude-only convergence inner loop (no Octave gain changes).
            ef_power_rabi = library.nodes["13_power_rabi_ef"].copy(
                name="ef_power_rabi",
                min_amp_factor=p.ef_rabi_min_amp_factor,
                max_amp_factor=p.ef_rabi_max_amp_factor,
                amp_factor_step=p.ef_rabi_amp_factor_step,
                num_shots=p.ef_rabi_num_shots,
            )
            ef_rabi_ramsey.add_node(ef_power_rabi)
            ef_rabi_ramsey.loop(
                ef_power_rabi,
                on=_should_repeat_ef_rabi_amplitude,
                max_iterations=p.ef_rabi_max_amplitude_iterations,
            )

            ef_ramsey = library.nodes["06b_ramsey_ef"].copy(
                name="ef_ramsey",
                num_shots=p.ef_ramsey_num_shots,
                frequency_detuning_in_mhz=p.ef_ramsey_frequency_detuning_in_mhz,
                min_wait_time_in_ns=p.ef_ramsey_min_wait_time_in_ns,
                max_wait_time_in_ns=p.ef_ramsey_max_wait_time_in_ns,
                wait_time_num_points=p.ef_ramsey_wait_time_num_points,
                log_or_linear_sweep=p.ef_ramsey_log_or_linear_sweep,
            )
            ef_rabi_ramsey.add_node(ef_ramsey)
            ef_rabi_ramsey.connect(ef_power_rabi, ef_ramsey)

        ef_bringup.add_node(ef_rabi_ramsey)
        ef_bringup.loop(
            ef_rabi_ramsey,
            on=should_repeat_ef_calibration,
            max_iterations=p.ef_max_iterations,
        )

        # ── EF T1 ─────────────────────────────────────────────────────────────
        ef_t1 = library.nodes["05b_T1_ef"].copy(
            name="ef_T1",
            num_shots=p.ef_t1_num_shots,
            min_wait_time_in_ns=p.ef_t1_min_wait_time_ns,
            max_wait_time_in_ns=p.ef_t1_max_wait_time_ns,
            wait_time_num_points=p.ef_t1_wait_time_num_points,
            log_or_linear_sweep=p.ef_t1_log_or_linear_sweep,
        )
        ef_bringup.add_node(ef_t1)

        # ── GEF readout frequency optimization ────────────────────────────────
        gef_freq_opt = library.nodes["14_gef_frequency_optimization"].copy(
            name="gef_readout_frequency_optimization",
            num_shots=p.gef_freq_opt_num_shots,
            frequency_span_in_mhz=p.gef_freq_opt_frequency_span_mhz,
            frequency_step_in_mhz=p.gef_freq_opt_frequency_step_mhz,
        )
        ef_bringup.add_node(gef_freq_opt)

        # ── GEF IQ blobs ──────────────────────────────────────────────────────
        gef_iq_blobs = library.nodes["15_iq_blobs_gef"].copy(
            name="gef_iq_blobs",
            num_shots=p.gef_iq_blobs_num_shots,
        )
        ef_bringup.add_node(gef_iq_blobs)

        # ── Connections ───────────────────────────────────────────────────────
        ef_bringup.connect(ef_discovery, ef_rabi_ramsey)
        ef_bringup.connect(ef_rabi_ramsey, ef_t1)
        ef_bringup.connect(ef_t1, gef_freq_opt)
        ef_bringup.connect(gef_freq_opt, gef_iq_blobs)

    return ef_bringup


# ── Cavity mode bringup subgraph builder ───────────────────────────────────────

def build_cavity_bringup(
    graph: QualibrationGraph, library: QualibrationLibrary
) -> QualibrationGraph:
    """Build and return the ``cavity_bringup`` subgraph.

    Sequence (all sequential, no retry loops)::

        cavity_mode_spectroscopy
        → displacement_calibration
        → cavity_T1
        → parity_time_measurement

    The cavity mode is selected via ``graph.parameters.cavity_mode_name``.
    Should be appended after the EF bringup (or readout_power_opt if EF is
    skipped).

    The caller is responsible for adding the returned subgraph to the outer
    graph and connecting it.

    Reads the following attributes from ``graph.parameters``::

        cavity_mode_name,
        cavity_spec_frequency_span_mhz, cavity_spec_frequency_step_mhz,
        cavity_spec_amplitude_factor, cavity_spec_num_shots,
        cavity_disp_amp_min, cavity_disp_amp_max, cavity_disp_amp_points,
        cavity_disp_num_shots,
        cavity_t1_min_wait_ns, cavity_t1_max_wait_ns, cavity_t1_num_points,
        cavity_t1_num_shots,
        parity_min_delay_ns, parity_max_delay_ns, parity_delay_step_ns,
        parity_num_shots
    """
    p = graph.parameters
    mode = p.cavity_mode_name

    with QualibrationGraph.build(
        "cavity_bringup",
        parameters=_CavityCalibrationSubgraphParameters(),
    ) as cavity_bringup:

        cav_spec = library.nodes["21_cavity_mode_spectroscopy"].copy(
            name="cavity_mode_spectroscopy",
            mode_name=mode,
            frequency_span_in_mhz=p.cavity_spec_frequency_span_mhz,
            frequency_step_in_mhz=p.cavity_spec_frequency_step_mhz,
            operation=p.cavity_spec_operation,
            operation_len_in_ns=p.cavity_spec_operation_len_in_ns,
            operation_amplitude_factor=p.cavity_spec_amplitude_factor,
            num_shots=p.cavity_spec_num_shots,
            qubit_probe_operation=p.cavity_spec_qubit_probe_operation,
            use_state_discrimination=p.cavity_spec_use_state_discrimination,
            min_dip_fraction=p.cavity_spec_min_dip_fraction,
        )
        cavity_bringup.add_node(cav_spec)

        displ = library.nodes["22_displacement_calibration_vacuum"].copy(
            name="displacement_calibration",
            mode_name=mode,
            amp_min=p.cavity_disp_amp_min,
            amp_max=p.cavity_disp_amp_max,
            amp_points=p.cavity_disp_amp_points,
            num_shots=p.cavity_disp_num_shots,
            qubit_pulse=p.cavity_disp_qubit_pulse,
            cavity_reset_type=p.cavity_disp_cavity_reset_type,
            active_reset=p.cavity_disp_active_reset,
            use_state_discrimination=p.cavity_disp_use_state_discrimination,
        )
        cavity_bringup.add_node(displ)

        cav_t1 = library.nodes["23_cavity_coherent_T1"].copy(
            name="cavity_T1",
            mode_name=mode,
            min_wait_time_in_ns=p.cavity_t1_min_wait_ns,
            max_wait_time_in_ns=p.cavity_t1_max_wait_ns,
            wait_time_num_points=p.cavity_t1_num_points,
            num_shots=p.cavity_t1_num_shots,
            log_or_linear_sweep=p.cavity_t1_log_or_linear_sweep,
            displacement_scale=p.cavity_t1_displacement_scale,
            use_state_discrimination=p.cavity_t1_use_state_discrimination,
            cavity_reset_type=p.cavity_t1_cavity_reset_type,
        )
        cavity_bringup.add_node(cav_t1)

        parity = library.nodes["30_parity_time_measurement"].copy(
            name="parity_time_measurement",
            mode_name=mode,
            min_delay_ns=p.parity_min_delay_ns,
            max_delay_ns=p.parity_max_delay_ns,
            delay_step_ns=p.parity_delay_step_ns,
            num_shots=p.parity_num_shots,
            displacement_scale=p.parity_displacement_scale,
            use_state_discrimination=p.parity_use_state_discrimination,
            cavity_reset_type=p.parity_cavity_reset_type,
        )
        cavity_bringup.add_node(parity)

        cavity_bringup.connect(cav_spec, displ)
        cavity_bringup.connect(displ, cav_t1)
        cavity_bringup.connect(cav_t1, parity)

    return cavity_bringup
