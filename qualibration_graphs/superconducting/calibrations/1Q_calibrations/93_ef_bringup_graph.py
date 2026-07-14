# %%
"""
EF-Transition Bring-Up Graph

Calibrates the EF (|e>→|f>) transition of a fixed-frequency transmon:

  ef_discovery [loop: retry if no EF oscillation]:
    ef_spectroscopy  [loop: retry on no peak]
    → ef_tentative_rabi       (checks oscillation; blacklists EF freq on failure)
  → ef_rabi_ramsey [loop: until |EF detuning| converges]:
      ef_ramsey
      → ef_power_rabi [inner loop: amplitude convergence]
  → ef_T1
  → gef_readout_optimization (subgraph):
      gef_readout_length_optimization
      → gef_readout_frequency_optimization
      → gef_readout_power_optimization
      → gef_iq_blobs
"""

import logging
from typing import List

from qualibrate import GraphParameters, QualibrationGraph, QualibrationLibrary, QualibrationNode
from calibration_utils.bringup_graphs import (
    _EFDiscoverySubgraphParameters,
    _EFRabiRamseySubgraphParameters,
    build_gef_readout_optimization,
    should_repeat_ef_spec,
    should_repeat_rabi_amplitude,
    _get_machine,
    _ensure_temp_calibration,
)
from calibration_utils.error_codes import PowerRabiErrorCode

logger = logging.getLogger(__name__)

library = QualibrationLibrary.get_active_library()


# ─── Top-level parameters ─────────────────────────────────────────────────────

class EFBringUpParameters(GraphParameters):
    """Graph-flow parameters for the EF-transition bring-up graph."""

    qubits: List[str] = ["q1"]
    # ── Iteration limits ──────────────────────────────────────────────────────
    max_ef_discovery_iterations: int = 3
    max_ef_spec_iterations: int = 3
    ef_max_iterations: int = 5
    ef_rabi_max_amplitude_iterations: int = 5
    # ── Convergence threshold ─────────────────────────────────────────────────
    ef_freq_threshold_hz: float = 50_000.0
    """Stop the ef_power_rabi → ef_ramsey loop when |EF detuning| < this value [Hz]."""


# ─── Graph construction ───────────────────────────────────────────────────────

with QualibrationGraph.build(
    "ef_bringup_graph",
    parameters=EFBringUpParameters(),
) as graph:
    p = graph.parameters

    # ── Local condition functions (closures over p) ───────────────────────────

    def should_repeat_ef_discovery(node: QualibrationNode, target: str) -> bool:
        """Restart EF spectroscopy when the tentative Rabi shows no oscillation."""
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
            return True
        return False

    _ef_loop_state: dict = {"initialized": {}, "detuning_history": {}}

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

    # ── ef_discovery: spectroscopy + tentative Rabi ───────────────────────────
    with QualibrationGraph.build(
        "ef_discovery",
        parameters=_EFDiscoverySubgraphParameters(),
    ) as ef_discovery:

        ef_spec = library.nodes["12_qubit_spectroscopy_EF"].copy(
            name="ef_spectroscopy",
            frequency_span_in_mhz=300.0,
            frequency_step_in_mhz=1.0,
            operation="saturation",
            operation_len_in_ns=20_000,
            operation_amplitude_factor=1.0,
            num_shots=100,
            target_peak_width=3e6,
            update_pulses_amplitude=False,
            find_dip=False,
            update_integration_weights_angle=False,
        )
        ef_discovery.add_node(ef_spec)
        ef_discovery.loop(
            ef_spec,
            on=should_repeat_ef_spec,
            max_iterations=p.max_ef_spec_iterations,
        )

        ef_tentative_rabi = library.nodes["13_power_rabi_ef"].copy(
            name="ef_tentative_rabi",
            min_amp_factor=0.001,
            max_amp_factor=1.9,
            amp_factor_step=0.01,
            num_shots=200,
        )
        ef_discovery.add_node(ef_tentative_rabi)
        ef_discovery.connect(ef_spec, ef_tentative_rabi)

    graph.add_node(ef_discovery)
    graph.loop(
        ef_discovery,
        on=should_repeat_ef_discovery,
        max_iterations=p.max_ef_discovery_iterations,
    )

    # ── EF fine calibration: ef_ramsey → ef_power_rabi [convergence loop] ─────
    with QualibrationGraph.build(
        "ef_rabi_ramsey",
        parameters=_EFRabiRamseySubgraphParameters(),
    ) as ef_rabi_ramsey:

        ef_ramsey = library.nodes["06b_ramsey_ef"].copy(
            name="ef_ramsey",
            num_shots=200,
            frequency_detuning_in_mhz=0.1,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=100_000,
            wait_time_num_points=100,
            log_or_linear_sweep="linear",
        )
        ef_rabi_ramsey.add_node(ef_ramsey)

        ef_power_rabi = library.nodes["13_power_rabi_ef"].copy(
            name="ef_power_rabi",
            min_amp_factor=0.001,
            max_amp_factor=1.9,
            amp_factor_step=0.01,
            num_shots=200,
        )
        ef_rabi_ramsey.add_node(ef_power_rabi)
        ef_rabi_ramsey.loop(
            ef_power_rabi,
            on=should_repeat_rabi_amplitude,
            max_iterations=p.ef_rabi_max_amplitude_iterations,
        )
        ef_rabi_ramsey.connect(ef_ramsey, ef_power_rabi)

    graph.add_node(ef_rabi_ramsey)
    graph.loop(
        ef_rabi_ramsey,
        on=should_repeat_ef_calibration,
        max_iterations=p.ef_max_iterations,
    )

    # ── EF T1 ─────────────────────────────────────────────────────────────────
    ef_t1 = library.nodes["05b_T1_ef"].copy(
        name="ef_T1",
        num_shots=500,
        min_wait_time_in_ns=16,
        max_wait_time_in_ns=300_000,
        wait_time_num_points=100,
        log_or_linear_sweep="linear",
    )
    graph.add_node(ef_t1)

    # ── GEF readout optimization (length → frequency → power → IQ blobs) ──────
    gef_readout_opt = build_gef_readout_optimization(graph, library)
    graph.add_node(gef_readout_opt)

    # ── Execution order ────────────────────────────────────────────────────────
    graph.connect(ef_discovery, ef_rabi_ramsey)
    graph.connect(ef_rabi_ramsey, ef_t1)
    graph.connect(ef_t1, gef_readout_opt)


graph.run()
