# %%
"""
EF Discovery Graph (Adaptive)

Standalone graph that finds the EF transition frequency via a self-correcting loop.
This is the EF analogue of the ge_discovery graph (03d_qubit_bringup_graph.py).

  ┌─ outer loop (restart on NO_OSCILLATION, max_ef_discovery_iterations) ─────┐
  │  ef_spectroscopy  [inner loop: retry on no peak, max_ef_spec_iterations]  │
  │  → ef_tentative_rabi   (checks oscillation; blacklists EF freq on failure) │
  └────────────────────────────────────────────────────────────────────────────┘

Run this before the full ef_bringup_graph (93) when you need to locate the EF
transition from scratch without running the entire EF bringup sequence.
"""

import logging
from typing import List

from qualibrate import GraphParameters, QualibrationGraph, QualibrationLibrary, QualibrationNode
from calibration_utils.bringup_graphs import (
    _EFDiscoverySubgraphParameters,
    should_repeat_ef_spec,
    _get_machine,
    _ensure_temp_calibration,
)
from calibration_utils.error_codes import PowerRabiErrorCode

logger = logging.getLogger(__name__)

library = QualibrationLibrary.get_active_library()

test_qubits = ["q1"]


# ─── Top-level parameters ─────────────────────────────────────────────────────

class EFDiscoveryParameters(GraphParameters):
    """Graph-flow parameters for the standalone EF discovery graph."""

    qubits: List[str] = test_qubits
    max_ef_discovery_iterations: int = 3
    max_ef_spec_iterations: int = 3


# ─── Graph construction ───────────────────────────────────────────────────────

with QualibrationGraph.build(
    "ef_discovery",
    parameters=EFDiscoveryParameters(),
) as graph:
    p = graph.parameters

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

    # ── ef_discovery subgraph: spectroscopy + tentative Rabi ─────────────────
    with QualibrationGraph.build(
        "ef_discovery_inner",
        parameters=_EFDiscoverySubgraphParameters(),
    ) as ef_discovery_inner:

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
        ef_discovery_inner.add_node(ef_spec)
        ef_discovery_inner.loop(
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
        ef_discovery_inner.add_node(ef_tentative_rabi)
        ef_discovery_inner.connect(ef_spec, ef_tentative_rabi)

    graph.add_node(ef_discovery_inner)
    graph.loop(
        ef_discovery_inner,
        on=should_repeat_ef_discovery,
        max_iterations=p.max_ef_discovery_iterations,
    )


graph.run()
