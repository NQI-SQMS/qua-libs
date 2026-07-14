# %%
"""
Qubit Optimization Graph (Adaptive)

This graph finds the ideal qubit parameters through a self-correcting nested loop.
The three nodes form an inner calibration subgraph that is repeated until the Rabi
calibration succeeds:

  ┌─ outer loop (restart on NO_OSCILLATION) ──────────────────────────────────┐
  │  spec_vs_power ──► qubit_spec ──► power_rabi                              │
  │  [ inner loop ]                   [ inner loop ]                          │
  │  (span expansion)                 (amplitude rescaling)                   │
  └────────────────────────────────────────────────────────────────────────────┘

Retry logic:
  - power_rabi TOO_MANY / TOO_FEW periods → only power_rabi is retried after
    the adaptive amplitude rescaling (new_amp = old_amp / num_periods).
  - power_rabi NO_OSCILLATION → current qubit frequency is blacklisted in
    temp_calibration and the outer loop restarts from spec_vs_power to find
    a new candidate frequency.
"""

from typing import List

from qualibrate import (
    GraphParameters,
    QualibrationGraph,
    QualibrationLibrary,
)
from calibration_utils.bringup_graphs import (
    build_qubit_calibration,
    should_restart_qubit_calibration,
)

library = QualibrationLibrary.get_active_library()

test_qubits = ["q1"]


class QubitOptimizationParameters(GraphParameters):
    """Graph-flow parameters for the adaptive qubit optimization graph."""

    qubits: List[str] = test_qubits
    multiplexed: bool = False
    spec_vs_power_use_adaptive_span: bool = True
    max_spec_vs_power_iterations: int = 5
    max_qubit_calibration_iterations: int = 3


with QualibrationGraph.build(
    "qubit_optimization",
    parameters=QubitOptimizationParameters(),
) as graph:
    qubit_calibration = build_qubit_calibration(graph, library)
    graph.add_node(qubit_calibration)
    graph.loop(
        qubit_calibration,
        on=should_restart_qubit_calibration,
        max_iterations=graph.parameters.max_qubit_calibration_iterations,
    )

graph.run()
