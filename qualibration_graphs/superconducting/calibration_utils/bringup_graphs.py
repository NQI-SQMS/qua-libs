"""
Shared subgraph builders and condition functions for fixed-frequency transmon bring-up.

Used by:
  - calibrations/1Q_calibrations/02f_resonator_bringup_graph.py
  - calibrations/1Q_calibrations/03d_qubit_bringup_graph.py
  - calibrations/1Q_calibrations/92_calibration_graph_bringup_fixed_frequency_transmon_adaptive.py
  - calibrations/1Q_calibrations/93_ef_bringup_graph.py
  - calibrations/1Q_calibrations/94_cavity_bringup_graph.py
  - calibrations/1Q_calibrations/05_x180_fine_calibration_graph.py  (helpers only)
"""

import logging
from typing import List

from qualibrate import GraphParameters, QualibrationGraph, QualibrationLibrary, QualibrationNode

from calibration_utils.error_codes import PowerRabiErrorCode, DisplacementVacuumErrorCode

logger = logging.getLogger(__name__)


# ── Inner subgraph parameter stubs ────────────────────────────────────────────
# Minimal classes for the nested subgraphs; actual calibration parameters are
# read from the outer graph via graph.parameters.

class _ResonatorDiscoverySubgraphParameters(GraphParameters):
    qubits: List[str] = ["q0"]
    multiplexed: bool = False


class _ResonatorBringUpSubgraphParameters(GraphParameters):
    qubits: List[str] = ["q0"]
    multiplexed: bool = False


class _QubitCalibrationSubgraphParameters(GraphParameters):
    qubits: List[str] = ["q0"]


# ── Condition functions ────────────────────────────────────────────────────────

def should_retry_resonator_discovery(node, target: str) -> bool:
    """Retry the resonator_discovery subgraph when high-power spectroscopy finds no dip."""
    if node.outcomes.get(target) == "failed":
        logger.info(
            f"{target}: Resonator dip not confirmed at high power. "
            f"Retrying broad spectroscopy with blacklisted frequencies excluded."
        )
        return True
    # Success: clear initial resonator fields so stale values don't persist
    machine = _get_machine(node)
    if machine is not None and machine.temp_calibration is not None:
        temp = machine.temp_calibration.get(target)
        if temp is not None:
            temp.initial_resonator_f01 = None
            temp.initial_resonator_RF_frequency = None
    logger.info(f"{target}: Resonator discovery succeeded.")
    return False


def should_repeat_punch_out(node: QualibrationNode, target: str) -> bool:
    """Retry the punch-out node if it failed (adaptive power span adjusts automatically)."""
    if node.outcomes.get(target) == "failed":
        logger.info(f"{target}: Punch-out failed; retrying.")
        return True
    logger.info(f"{target}: Punch-out succeeded.")
    return False


def should_repeat_spec_vs_power(node: QualibrationNode, target: str) -> bool:
    """Retry spec_vs_power if no peak was found.

    The adaptive span/power logic in the node's update_state automatically
    expands the search range before the next iteration.
    """
    if node.outcomes.get(target) == "failed":
        logger.info(f"{target}: Spectroscopy vs power failed; retrying with expanded span.")
        return True
    # Exiting loop (succeeded): clear adaptive fields so stale values don't persist
    machine = getattr(node, "machine", None)
    if machine is not None and machine.temp_calibration is not None:
        temp = machine.temp_calibration.get(target)
        if temp is not None:
            temp.adaptive_frequency_span_mhz = None
            temp.adaptive_power_shift_dbm = None
            temp.adaptive_num_shots = None
    logger.info(f"{target}: Spectroscopy vs power succeeded.")
    return False


def should_repeat_rabi_amplitude(node: QualibrationNode, target: str) -> bool:
    """Retry power_rabi only when the failure is an amplitude mismatch.

    TOO_MANY_PERIODS or TOO_FEW_PERIODS: the adaptive update_state has already
    rescaled the base amplitude (new_amp = old_amp / num_periods) so the next
    iteration will converge toward ~1 oscillation period.

    NO_OSCILLATION: do NOT retry here – escalate to the outer loop so the full
    frequency-search sequence can restart with the current frequency blacklisted.

    Note: the node may report outcome "succeeded" even for TOO_MANY/TOO_FEW,
    so we check the error code directly rather than gating on the outcome string.
    """
    error_code = (
        node.results.get("fit_results", {})
        .get(target, {})
        .get("error_code", int(PowerRabiErrorCode.SUCCESS))
    )
    if error_code in (
        int(PowerRabiErrorCode.TOO_MANY_PERIODS),
        int(PowerRabiErrorCode.TOO_FEW_PERIODS),
    ):
        logger.info(
            f"{target}: Rabi amplitude mismatch "
            f"({PowerRabiErrorCode(error_code).name}); "
            "retrying with rescaled base amplitude."
        )
        return True
    if error_code != int(PowerRabiErrorCode.SUCCESS):
        logger.info(
            f"{target}: Rabi failed ({PowerRabiErrorCode(error_code).name}); "
            "escalating to full frequency search."
        )
    return False


def should_repeat_displacement_vacuum(node: QualibrationNode, target: str) -> bool:
    """Retry the displacement vacuum-population calibration while the adaptive
    amplitude/duration escalation loop is still trying to reach target_n_sigma
    coverage of the Gaussian fit.

    Checks the error_code directly (not node.outcomes, since the node's own
    update_state forces outcome to "failed" while escalating even though the
    underlying fit converged fine) so escalation iterations are distinguished
    from a genuine fit failure.
    """
    error_code = (
        node.results.get("fit_results", {})
        .get(target, {})
        .get("error_code", int(DisplacementVacuumErrorCode.SUCCESS))
    )
    if error_code == int(DisplacementVacuumErrorCode.INSUFFICIENT_RANGE_COVERAGE):
        logger.info(
            f"{target}: Displacement vacuum calibration coverage insufficient "
            f"({DisplacementVacuumErrorCode(error_code).name}); "
            "retrying with escalated amplitude/duration."
        )
        return True
    if error_code == int(DisplacementVacuumErrorCode.FIT_FAILED):
        logger.warning(f"{target}: Displacement vacuum calibration fit failed; not retrying automatically.")
        return False
    logger.info(f"{target}: Displacement vacuum calibration succeeded with full coverage.")
    return False


def should_restart_qubit_calibration(node, target: str) -> bool:
    """Restart the full calibration subgraph if it failed.

    This handles the NO_OSCILLATION case: the qubit frequency is now blacklisted
    in temp_calibration, so the next spec_vs_power run will avoid it.

    On success, the post-bringup x180 amplitude, qubit f_01, and RF_frequency are
    saved to temp_calibration so that the x180 fine calibration can restore them if
    its first Ramsey or time_rabi iteration fails.
    """
    if node.outcomes.get(target) == "failed":
        logger.info(
            f"{target}: Qubit calibration failed; restarting frequency search."
        )
        # Mark as not yet successfully calibrated so x180 fine calibration can
        # detect exhaustion if the outer loop never exits via the success path.
        machine = _get_machine(node)
        if machine is not None:
            temp = _ensure_temp_calibration(machine, target)
            temp.qubit_calibration_succeeded = False
        return True
    # Succeeded: snapshot the bringup result so fine calibration can roll back to it.
    # IMPORTANT: _get_machine() returns the first non-None machine found in
    # _elements (insertion order: spec_vs_power → qubit_spec → time_rabi).
    # spec_vs_power's machine was loaded before time_rabi ran, so its x180
    # amplitude is stale (pre-calibration).  We must read the amplitude from
    # the time_rabi node's machine, which was updated by update_state.
    _pr_elem = getattr(node, "_elements", {}).get("time_rabi")
    machine = (
        _get_machine(_pr_elem)
        if _pr_elem is not None
        else _get_machine(node)
    ) or _get_machine(node)
    if machine is not None:
        q = machine.qubits[target]
        temp = _ensure_temp_calibration(machine, target)
        temp.initial_x180_amplitude = float(q.xy.operations["x180"].amplitude)
        temp.initial_qubit_f01 = float(q.f_01)
        temp.initial_rf_frequency = float(q.xy.RF_frequency)
        temp.qubit_calibration_succeeded = True
        logger.info(
            f"[Qubit bringup] {target}: Saved fine-calibration backup – "
            f"x180={1e3 * temp.initial_x180_amplitude:.2f} mV, "
            f"f_01={temp.initial_qubit_f01 / 1e9:.6f} GHz."
        )
        # Persist the backup to disk immediately.  The condition function runs
        # after the last node's node.save(), so without this explicit call the
        # backup would only live in memory and would be lost when the next
        # graph.run() loads the machine fresh from state.json.
        try:
            machine.save()
        except Exception as exc:
            logger.warning(
                f"[Qubit bringup] {target}: machine.save() after backup failed: {exc}"
            )
    logger.info(f"{target}: Qubit calibration succeeded.")
    return False


# ── Subgraph builders ─────────────────────────────────────────────────────────

def build_resonator_bringup(
    graph: QualibrationGraph, library: QualibrationLibrary
) -> QualibrationGraph:
    """Build and return the ``resonator_bringup`` subgraph.

    Sequence::

        broad_resonator_spectroscopy
        → resonator_spectroscopy_high_power  [loop: retry_resonator_discovery]
        → resonator_punch_out                [loop: repeat_punch_out]
        → resonator_spectroscopy_low_power

    Reads the following attributes from ``graph.parameters``::

        multiplexed
        broad_frequency_span_mhz, broad_frequency_step_mhz, broad_num_shots,
        broad_peak_prominence, broad_peak_width, blacklist_exclusion_radius_mhz,
        broad_readout_power_dbm, broad_max_amp
        high_power_frequency_span_mhz, high_power_frequency_step_mhz,
        high_power_num_shots, high_power_readout_power_dbm, high_power_max_amp,
        punch_out_frequency_span_mhz, punch_out_frequency_step_mhz,
        punch_out_min_power_dbm, punch_out_max_power_dbm, punch_out_num_power_points,
        punch_out_max_amp, punch_out_num_shots, punch_out_frequency_shift_threshold_hz,
        use_adaptive_span
        low_power_frequency_span_mhz, low_power_frequency_step_mhz,
        low_power_num_shots, low_power_readout_power_dbm, low_power_max_amp
        max_resonator_discovery_iterations, max_punch_out_iterations
    """
    p = graph.parameters
    with QualibrationGraph.build(
        "resonator_bringup",
        parameters=_ResonatorBringUpSubgraphParameters(),
    ) as resonator_bringup:

        # Inner: broad scan → high-power confirmation
        with QualibrationGraph.build(
            "resonator_discovery",
            parameters=_ResonatorDiscoverySubgraphParameters(),
        ) as resonator_discovery:

            broad_res_spec = library.nodes["02d_broad_resonator_spectroscopy"].copy(
                name="broad_resonator_spectroscopy",
                multiplexed=p.multiplexed,
                frequency_span_in_mhz=100.0,
                frequency_step_in_mhz=0.1,
                num_shots=50,
                peak_prominence=10.0,
                peak_width=[1.0, 5.0],
                peak_height=None,
                peak_threshold=None,
                blacklist_exclusion_radius_mhz=10.0,
                readout_power_dbm=0.0,
                max_amp=0.4,
            )
            resonator_discovery.add_node(broad_res_spec)

            high_power_res_spec = library.nodes["02a_resonator_spectroscopy"].copy(
                name="resonator_spectroscopy_high_power",
                multiplexed=p.multiplexed,
                frequency_span_in_mhz=5.0,
                frequency_step_in_mhz=0.01,
                num_shots=50,
                readout_power_dbm=0.0,
                max_amp=0.4,
                save_readout_amplitude=True,
            )
            resonator_discovery.add_node(high_power_res_spec)
            resonator_discovery.connect(broad_res_spec, high_power_res_spec)

        resonator_bringup.add_node(resonator_discovery)
        resonator_bringup.loop(
            resonator_discovery,
            on=should_retry_resonator_discovery,
            max_iterations=p.max_resonator_discovery_iterations,
        )

        # Punch-out: find optimal readout power via Kerr shift
        resonator_punch_out = library.nodes["02e_resonator_punch_out"].copy(
            name="resonator_punch_out",
            multiplexed=p.multiplexed,
            frequency_span_in_mhz=2.0,
            frequency_step_in_mhz=0.01,
            min_power_dbm=-20,
            max_power_dbm=0,
            num_power_points=10,
            max_amp=0.4,
            num_shots=200,
            frequency_shift_threshold_in_hz=0.1e6,
            use_adaptive_span=p.use_adaptive_span,
            sweep_left_offset_mhz=1.0,
        )
        resonator_bringup.add_node(resonator_punch_out)
        resonator_bringup.loop(
            resonator_punch_out,
            on=should_repeat_punch_out,
            max_iterations=p.max_punch_out_iterations,
        )

        # Low-power fine spectroscopy: precise frequency at optimal power
        low_power_res_spec = library.nodes["02a_resonator_spectroscopy"].copy(
            name="resonator_spectroscopy_low_power",
            multiplexed=p.multiplexed,
            frequency_span_in_mhz=5.0,
            frequency_step_in_mhz=0.01,
            num_shots=100,
            readout_power_dbm=None,
            max_amp=0.2,
            save_readout_amplitude=True,
            run_circle_fit=True,
        )
        resonator_bringup.add_node(low_power_res_spec)

        resonator_bringup.connect(resonator_discovery, resonator_punch_out)
        resonator_bringup.connect(resonator_punch_out, low_power_res_spec)

    return resonator_bringup


def build_qubit_calibration(
    graph: QualibrationGraph, library: QualibrationLibrary
) -> QualibrationGraph:
    """Build and return the ``qubit_calibration`` subgraph (without the outer loop).

    Sequence::

        qubit_spectroscopy_vs_power  [inner loop: repeat_spec_vs_power]
        → time_rabi

    The 1D qubit spectroscopy step is omitted: the power-broadening fit inside
    spec_vs_power already provides a well-calibrated frequency and amplitude.

    The caller is responsible for adding the returned subgraph to the outer
    graph and registering the outer loop with ``should_restart_qubit_calibration``.

    Reads the following attributes from ``graph.parameters``::

        multiplexed
        spec_vs_power_frequency_span_mhz, spec_vs_power_frequency_step_mhz,
        spec_vs_power_num_power_points, spec_vs_power_num_shots,
        spec_vs_power_min_power_dbm, spec_vs_power_max_power_dbm,
        spec_vs_power_operation, spec_vs_power_operation_len_ns,
        spec_vs_power_max_amplitude_opx, spec_vs_power_rabi_target_periods,
        spec_vs_power_rabi_sweep_max_duration_ns
        time_rabi_min_duration_ns, time_rabi_max_duration_ns, time_rabi_duration_step_ns,
        time_rabi_num_shots, time_rabi_operation_amplitude_factor, time_rabi_drive_power_dbm
        max_spec_vs_power_iterations
    """
    p = graph.parameters
    with QualibrationGraph.build(
        "qubit_calibration",
        parameters=_QubitCalibrationSubgraphParameters(),
    ) as qubit_calibration:

        # 1. Spec vs power: find qubit frequency, fit broadening, set saturation/x180 amplitude
        spec_vs_power = library.nodes["03c_qubit_spectroscopy_vs_power"].copy(
            name="qubit_spectroscopy_vs_power",
            use_adaptive_span=p.spec_vs_power_use_adaptive_span,
            multiplexed=p.multiplexed,
            frequency_span_in_mhz=300.0,
            frequency_step_in_mhz=1.0,
            num_power_points=10,
            num_shots=100,
            min_power_dbm=-60,
            max_power_dbm=10,
            operation="saturation",
            operation_len_in_ns=20_000,
            linewidth_threshold_hz=1e6,
            max_amplitude_opx=0.24,
            min_amplitude_opx=0.01,
            power_buffer_db=-0.0,
            signal_source="I_rot",
            peak_persistence_lookahead=0,
            peak_persistence_freq_tolerance_hz=5e6,
            rabi_target_periods=1,
            rabi_sweep_max_duration_ns=300.0,
        )
        qubit_calibration.add_node(spec_vs_power)
        qubit_calibration.loop(
            spec_vs_power,
            on=should_repeat_spec_vs_power,
            max_iterations=p.max_spec_vs_power_iterations,
        )

        # 2. Time Rabi: measure π-pulse duration using the saturation pulse at the
        #    amplitude set by spec_vs_power's broadening fit.
        time_rabi = library.nodes["04c_time_rabi"].copy(
            name="time_rabi",
            multiplexed=p.multiplexed,
            min_duration_ns=16,
            max_duration_ns=300,
            duration_step_ns=4,
            num_shots=200,
            operation="saturation",
            operation_amplitude_factor=1.0,
            drive_power_dbm=None,
            max_amplitude_opx=0.1,
        )
        qubit_calibration.add_node(time_rabi)

        qubit_calibration.connect(spec_vs_power, time_rabi)

    return qubit_calibration


# ── X180 fine-calibration helpers ─────────────────────────────────────────────
# These are also used directly by 05_x180_fine_calibration_graph.py.

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
    if target not in loop_state["initial_x180_amplitude"]:
        return
    q = machine.qubits[target]
    q.xy.operations["x180"].amplitude = loop_state["initial_x180_amplitude"][target]
    q.xy.operations["x90"].amplitude = loop_state["initial_x90_amplitude"][target]
    q.f_01 = loop_state["initial_f01"][target]
    q.xy.RF_frequency = loop_state["initial_rf_frequency"][target]
    logger.info(
        f"[X180 fine] {target}: Restored x180={1e3 * loop_state['initial_x180_amplitude'][target]:.2f} mV, "
        f"f_01={loop_state['initial_f01'][target] / 1e9:.6f} GHz."
    )
    temp = (machine.temp_calibration or {}).get(target)
    if temp is not None:
        if hasattr(temp, "initial_x180_amplitude"):
            temp.initial_x180_amplitude = None
        if hasattr(temp, "initial_qubit_f01"):
            temp.initial_qubit_f01 = None
        if hasattr(temp, "initial_rf_frequency"):
            temp.initial_rf_frequency = None
    # Persist the restored state so the JSON file reflects the rollback.
    # The Ramsey node already called machine.save() with modified frequencies;
    # without this call those would survive in state.json.
    try:
        machine.save()
    except Exception as exc:
        logger.warning(f"[X180 fine] {target}: machine.save() after restore failed: {exc}")


class _X180FineCalibrationSubgraphParameters(GraphParameters):
    qubits: List[str] = ["q0"]
    multiplexed: bool = False


class _RabiRamseySubgraphParameters(GraphParameters):
    qubits: List[str] = ["q0"]


def build_x180_fine_calibration(
    graph: QualibrationGraph, library: QualibrationLibrary
) -> QualibrationGraph:
    """Build and return the ``x180_fine_calibration`` subgraph.

    Iteratively refines the x180 pulse amplitude and qubit frequency::

        ramsey → power_rabi  [loop until |detuning| < x180_freq_threshold_hz]

    On fit failure the pre-loop state is restored.  The loop state is isolated
    per call so concurrent or sequential graphs do not share state.

    Reads the following attributes from ``graph.parameters``::

        multiplexed
        x180_rabi_min_amp_factor, x180_rabi_max_amp_factor,
        x180_rabi_amp_factor_step, x180_rabi_num_shots,
        x180_rabi_max_number_pulses_per_sweep
        x180_ramsey_num_shots, x180_ramsey_frequency_detuning_in_mhz,
        x180_ramsey_max_wait_time_in_ns, x180_ramsey_wait_time_num_points,
        x180_ramsey_log_or_linear_sweep
        x180_freq_threshold_hz, x180_max_iterations
    """
    p = graph.parameters

    # Isolated per-call loop state — no shared module-level globals
    _loop_state: dict = {
        "initialized": {},
        "any_failed": False,
        "initial_x180_amplitude": {},
        "initial_x90_amplitude": {},
        "initial_f01": {},
        "initial_rf_frequency": {},
        "detuning_history": {},
    }

    def should_repeat_x180_calibration(node: QualibrationNode, target: str) -> bool:
        machine = _get_machine(node)

        if not any(_loop_state["initialized"].values()):
            _loop_state["any_failed"] = False

        if _loop_state["any_failed"]:
            _restore_initial_state(machine, target, _loop_state)
            _loop_state["initialized"][target] = False
            return False

        q = machine.qubits[target]

        if not _loop_state["initialized"].get(target, False):
            temp = _ensure_temp_calibration(machine, target)

            # Guard: qubit_calibration exhausted all iterations without success.
            # qubit_calibration_succeeded is set to False on every failed attempt
            # and True only when qubit_calibration exits via the success path.
            if getattr(temp, "qubit_calibration_succeeded", None) is False:
                logger.warning(
                    f"[X180 fine] {target}: Qubit calibration exhausted all iterations "
                    "without success — skipping x180 fine calibration entirely."
                )
                return False

            if temp.initial_x180_amplitude is not None:
                _loop_state["initial_x180_amplitude"][target] = temp.initial_x180_amplitude
                _loop_state["initial_x90_amplitude"][target] = temp.initial_x180_amplitude / 2
                _loop_state["initial_f01"][target] = temp.initial_qubit_f01 or float(q.f_01)
                # Use the RF_frequency snapshotted at bringup; fall back to current if absent.
                if hasattr(temp, "initial_rf_frequency") and temp.initial_rf_frequency is not None:
                    _loop_state["initial_rf_frequency"][target] = temp.initial_rf_frequency
                else:
                    _loop_state["initial_rf_frequency"][target] = float(q.xy.RF_frequency)
            else:
                _loop_state["initial_x180_amplitude"][target] = float(q.xy.operations["x180"].amplitude)
                _loop_state["initial_x90_amplitude"][target] = float(q.xy.operations["x90"].amplitude)
                _loop_state["initial_f01"][target] = float(q.f_01)
                _loop_state["initial_rf_frequency"][target] = float(q.xy.RF_frequency)
                temp.initial_x180_amplitude = _loop_state["initial_x180_amplitude"][target]
                temp.initial_qubit_f01 = _loop_state["initial_f01"][target]
            _loop_state["detuning_history"][target] = []
            _loop_state["initialized"][target] = True
            logger.info(
                f"[X180 fine] {target}: Initial state captured – "
                f"x180={1e3 * _loop_state['initial_x180_amplitude'][target]:.2f} mV, "
                f"f_01={_loop_state['initial_f01'][target] / 1e9:.6f} GHz."
            )

        if node.outcomes.get(target) == "failed":
            logger.warning(
                f"[X180 fine] {target}: Fit failed. "
                "Restoring initial state and stopping loop."
            )
            _restore_initial_state(machine, target, _loop_state)
            _loop_state["initialized"][target] = False
            _loop_state["any_failed"] = True
            return False

        _ramsey_node = node._elements.get("ramsey")
        _ramsey_results = _ramsey_node.results if _ramsey_node is not None else {}
        freq_offset = (
            _ramsey_results.get("fit_results", {})
            .get(target, {})
            .get("freq_offset", None)
        )

        if freq_offset is None:
            history = _loop_state["detuning_history"][target]
            last_f01 = (
                _loop_state["initial_f01"][target] - sum(history)
                if history
                else _loop_state["initial_f01"][target]
            )
            freq_offset = last_f01 - float(q.f_01)

        abs_offset = abs(freq_offset)
        _loop_state["detuning_history"][target].append(abs_offset)

        f01 = float(q.f_01) or 1.0
        pct = abs_offset / f01 * 100.0
        logger.info(
            f"[X180 fine] {target}: |detuning| = {abs_offset / 1e3:.2f} kHz "
            f"({pct:.4f}%),  threshold = {p.x180_freq_threshold_hz / 1e3:.0f} kHz."
        )

        if abs_offset < p.x180_freq_threshold_hz:
            logger.info(
                f"[X180 fine] {target}: Converged after "
                f"{len(_loop_state['detuning_history'][target])} iteration(s)."
            )
            temp = (machine.temp_calibration or {}).get(target)
            if temp is not None:
                if hasattr(temp, "initial_x180_amplitude"):
                    temp.initial_x180_amplitude = None
                if hasattr(temp, "initial_qubit_f01"):
                    temp.initial_qubit_f01 = None
                if hasattr(temp, "initial_rf_frequency"):
                    temp.initial_rf_frequency = None
            _loop_state["initialized"][target] = False
            return False

        return True

    with QualibrationGraph.build(
        "x180_fine_calibration",
        parameters=_X180FineCalibrationSubgraphParameters(),
    ) as x180_fine_calibration:

        with QualibrationGraph.build(
            "ramsey_rabi",
            parameters=_RabiRamseySubgraphParameters(),
        ) as ramsey_rabi:

            power_rabi = library.nodes["04b_power_rabi"].copy(
                name="power_rabi",
                multiplexed=p.multiplexed,
                min_amp_factor=0.001,
                max_amp_factor=1.99,
                amp_factor_step=0.005,
                num_shots=100,
                operation="x180",
                operation_length_in_ns=None,
                max_number_pulses_per_sweep=1,
                update_x90=True,
                use_adaptive=p.x180_rabi_use_adaptive,
            )
            ramsey_rabi.add_node(power_rabi)
            # Inner loop: retry power_rabi until the period count is correct
            # (TOO_MANY or TOO_FEW), before proceeding to Ramsey.
            ramsey_rabi.loop(
                power_rabi,
                on=should_repeat_rabi_amplitude,
                max_iterations=p.x180_rabi_max_amplitude_iterations,
            )

            ramsey = library.nodes["06a_ramsey"].copy(
                name="ramsey",
                multiplexed=p.multiplexed,
                num_shots=100,
                frequency_detuning_in_mhz=0.1,
                min_wait_time_in_ns=16,
                max_wait_time_in_ns=100_000,
                wait_time_num_points=200,
                log_or_linear_sweep="linear",
                x180_operation="x180",
            )
            ramsey_rabi.add_node(ramsey)
            ramsey_rabi.connect(power_rabi, ramsey)

        x180_fine_calibration.add_node(ramsey_rabi)
        x180_fine_calibration.loop(
            ramsey_rabi,
            on=should_repeat_x180_calibration,
            max_iterations=p.x180_max_iterations,
        )

    return x180_fine_calibration


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
              ef_ramsey
              → ef_power_rabi [inner loop: amplitude only, max_ef_rabi_iterations]
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
            .get("error_code", int(PowerRabiErrorCode.SUCCESS))
        )

        if error_code == int(PowerRabiErrorCode.NO_OSCILLATION):
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

    # ── should_repeat_ef_calibration ─────────────────────────────────────────
    def should_repeat_ef_calibration(node: QualibrationNode, target: str) -> bool:
        """Loop ef_ramsey → ef_power_rabi until |EF detuning| < ef_freq_threshold_hz."""
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

            # Ramsey runs first, refining the EF frequency using the amplitude
            # already set by ef_tentative_rabi, before the power Rabi
            # amplitude-only convergence inner loop (no Octave gain changes).
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
                on=should_repeat_rabi_amplitude,
                max_iterations=p.ef_rabi_max_amplitude_iterations,
            )
            ef_rabi_ramsey.connect(ef_ramsey, ef_power_rabi)

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
        cavity_spec_subtract_baseline,
        cavity_disp_amp_min, cavity_disp_amp_max, cavity_disp_amp_points,
        cavity_disp_num_shots, cavity_disp_subtract_baseline,
        cavity_disp_use_adaptive, cavity_disp_target_n_sigma,
        max_displacement_vacuum_iterations,
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
            subtract_baseline=p.cavity_spec_subtract_baseline,
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
            subtract_baseline=p.cavity_disp_subtract_baseline,
            use_adaptive=p.cavity_disp_use_adaptive,
            target_n_sigma=p.cavity_disp_target_n_sigma,
        )
        cavity_bringup.add_node(displ)
        cavity_bringup.loop(
            displ,
            on=should_repeat_displacement_vacuum,
            max_iterations=p.max_displacement_vacuum_iterations,
        )

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
