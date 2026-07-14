# %%
"""
GE Readout Optimization Graph

Optimizes the readout for two-level (g/e) state discrimination in sequence:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  1.  readout_length_optimization   (08d)                                │
  │  2.  readout_frequency_optimization (08a)                               │
  │  3.  readout_power_optimization    (08b)                                │
  │  4.  iq_blobs                      (07)                                 │
  └─────────────────────────────────────────────────────────────────────────┘

This graph is also used as a subgraph inside ge_bringup_graph (92).
Node-specific measurement parameters are baked into the node copies in
bringup_graphs.py and can be adjusted there or through the Qualibrate GUI.
"""

from typing import List

from qualibrate import GraphParameters, QualibrationGraph, QualibrationLibrary

library = QualibrationLibrary.get_active_library()


# ─── Top-level parameters ─────────────────────────────────────────────────────

class GEReadoutOptParameters(GraphParameters):
    qubits: List[str] = ["q1"]
    multiplexed: bool = False


# ─── Graph construction ───────────────────────────────────────────────────────

with QualibrationGraph.build(
    "ge_readout_optimization_graph",
    parameters=GEReadoutOptParameters(),
) as graph:

    # ── 1. Readout length optimization ────────────────────────────────────────
    readout_length_opt = library.nodes["08d_readout_length_optimization"].copy(
        name="readout_length_optimization",
        max_readout_length_in_ns=12000,
        division_length_in_ns=160,
        num_shots=2000,
        readout_operation="readout",
        cos_weight_name="iw1",
        sin_weight_name="iw2",
        minus_sin_weight_name="iw3",
    )
    graph.add_node(readout_length_opt)

    # ── 2. Readout frequency optimization ─────────────────────────────────────
    readout_freq_opt = library.nodes["08a_readout_frequency_optimization"].copy(
        name="readout_frequency_optimization",
        multiplexed=graph.parameters.multiplexed,
        num_shots=100,
        frequency_span_in_mhz=20.0,
        frequency_step_in_mhz=0.1,
    )
    graph.add_node(readout_freq_opt)

    # ── 3. Readout power optimization ─────────────────────────────────────────
    readout_power_opt = library.nodes["08b_readout_power_optimization"].copy(
        name="readout_power_optimization",
        num_shots=2000,
        start_amp=0.5,
        end_amp=1.5,
        num_amps=10,
        outliers_threshold=0.98,
        plot_raw=False,
    )
    graph.add_node(readout_power_opt)

    # ── 4. IQ blobs ───────────────────────────────────────────────────────────
    iq_blobs = library.nodes["07_iq_blobs"].copy(
        name="iq_blobs",
        num_shots=2000,
    )
    graph.add_node(iq_blobs)

    # ── Execution order ────────────────────────────────────────────────────────
    graph.connect(readout_length_opt, readout_freq_opt)
    graph.connect(readout_freq_opt, readout_power_opt)
    graph.connect(readout_power_opt, iq_blobs)


graph.run()
