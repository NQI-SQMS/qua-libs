# %%
"""
GE Bring-Up Graph (Adaptive FSM)

Full automated GE (ground-excited) bring-up sequence for a fixed-frequency
transmon qubit. EF-transition and cavity-mode bring-up live in their own
dedicated graphs (93_ef_bringup_graph.py, 94_cavity_bringup_graph.py) and
should be run after this graph.

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  1.  resonator_bringup (subgraph):                                      │
  │        resonator_discovery [loop: retry on no dip]:                     │
  │          broad_resonator_spectroscopy                                   │
  │          ──► resonator_spectroscopy_high_power                          │
  │        ──► resonator_punch_out        [loop: retry on failure]          │
  │        ──► resonator_spectroscopy_low_power                             │
  │  2.  qubit_calibration (subgraph, nested loops):                        │
  │        qubit_spectroscopy_vs_power    [inner loop: span expansion]      │
  │          (power-broadening fit → sets saturation & x180 amplitude)      │
  │        ──► time_rabi (saturation pulse)                                 │
  │        [outer loop: restart on NO_OSCILLATION → new freq search]        │
  │  3.  x180_fine_calibration (subgraph):                                  │
  │        rabi_ramsey [loop: repeat until freq converges]                  │
  │          power_rabi ──► ramsey                                          │
  │  4.  T1                                                                 │
  │  5.  ge_readout_optimization (subgraph):                                │
  │        readout_length_optimization                                      │
  │        ──► readout_frequency_optimization                               │
  │        ──► readout_power_optimization                                   │
  │        ──► iq_blobs                                                     │
  └─────────────────────────────────────────────────────────────────────────┘

Graph-level parameters control graph FLOW (iteration limits, convergence
thresholds, opt-in flags).  Node-specific measurement parameters are baked
into the node copies in bringup_graphs.py and can be edited there or through
the Qualibrate GUI at the individual node level.
"""

from typing import List

from qualibrate import (
    GraphParameters,
    QualibrationGraph,
    QualibrationLibrary,
)
from calibration_utils.bringup_graphs import (
    build_resonator_bringup,
    build_qubit_calibration,
    build_x180_fine_calibration,
    build_ge_readout_optimization,
    should_restart_qubit_calibration,
    _resolve_x180_fine_params,
)


library = QualibrationLibrary.get_active_library()

test_qubits = ["q1"]


# ─── Top-level parameters ─────────────────────────────────────────────────────

class TransmonBringUpParameters(GraphParameters):
    """Graph-flow parameters for the GE bring-up graph.

    Only parameters that control graph STRUCTURE or LOOP BEHAVIOUR live here.
    Node-specific measurement parameters (frequency spans, shot counts, etc.)
    are set directly in the node copies inside bringup_graphs.py.
    """

    qubits: List[str] = test_qubits
    multiplexed: bool = False

    # ── Iteration limits ──────────────────────────────────────────────────────
    max_resonator_discovery_iterations: int = 5
    max_punch_out_iterations: int = 5
    max_spec_vs_power_iterations: int = 5
    max_qubit_calibration_iterations: int = 3
    x180_max_iterations: int = 10
    x180_rabi_max_amplitude_iterations: int = 5

    # ── Adaptive behaviour ─────────────────────────────────────────────────────
    use_adaptive_span: bool = True
    spec_vs_power_use_adaptive_span: bool = True
    x180_rabi_use_adaptive: bool = True

    # ── Convergence thresholds ─────────────────────────────────────────────────
    x180_freq_threshold_hz: float = 50_000.0


# ─── Graph construction ───────────────────────────────────────────────────────

with QualibrationGraph.build(
    "ge_bringup_graph",
    parameters=TransmonBringUpParameters(),
) as graph:

    # ── 1. Resonator bring-up ─────────────────────────────────────────────────
    resonator_bringup = build_resonator_bringup(graph, library)
    graph.add_node(resonator_bringup)

    # ── 2. Qubit calibration (FSM: spec-vs-power → time Rabi) ─────────────────
    qubit_calibration = build_qubit_calibration(graph, library)
    graph.add_node(qubit_calibration)
    graph.loop(
        qubit_calibration,
        on=should_restart_qubit_calibration,
        max_iterations=graph.parameters.max_qubit_calibration_iterations,
    )

    # ── 3. X180 fine calibration (power_rabi → ramsey loop) ───────────────────
    x180_fine_calibration = build_x180_fine_calibration(graph, library)
    graph.add_node(x180_fine_calibration)

    # ── 4. T1 ─────────────────────────────────────────────────────────────────
    t1 = library.nodes["05_T1"].copy(
        name="T1",
        num_shots=200,
        min_wait_time_in_ns=16,
        max_wait_time_in_ns=500_000,
        wait_time_num_points=100,
        log_or_linear_sweep="linear",
    )
    graph.add_node(t1)

    # ── 5. GE readout optimization (length → frequency → power → IQ blobs) ────
    ge_readout_opt = build_ge_readout_optimization(graph, library)
    graph.add_node(ge_readout_opt)

    # ── Execution order ────────────────────────────────────────────────────────
    graph.connect(resonator_bringup, qubit_calibration)
    graph.connect(qubit_calibration, x180_fine_calibration,
                  resolve_params=_resolve_x180_fine_params)
    graph.connect(x180_fine_calibration, t1)
    graph.connect(t1, ge_readout_opt)


graph.run()
