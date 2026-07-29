# %%
"""
GEF Readout Optimization Graph

Optimizes the readout for three-level (g/e/f) state discrimination in sequence:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  0.  gef_dispersive_shift               (20b)                           │
  │  1.  gef_readout_length_optimization   (14c)                            │
  │  2.  gef_readout_frequency_optimization (14)                            │
  │  3.  gef_readout_power_optimization    (14b)                            │
  │  4.  gef_iq_blobs                      (15)                             │
  └─────────────────────────────────────────────────────────────────────────┘

This graph is also used as a subgraph inside ef_bringup_graph (93).
Node-specific measurement parameters are baked into the node copies in
bringup_graphs.py and can be adjusted there or through the Qualibrate GUI.

Prerequisites:
  - GE bringup completed (ge_bringup_graph, 92).
  - EF transition calibrated (ef_bringup_graph without gef_readout step, 93).
  - EF_x180 pulse calibrated (node 13).
"""

from typing import List

from qualibrate import GraphParameters, QualibrationGraph, QualibrationLibrary

library = QualibrationLibrary.get_active_library()


# ─── Top-level parameters ─────────────────────────────────────────────────────

class GEFReadoutOptParameters(GraphParameters):
    qubits: List[str] = ["q1"]


# ─── Graph construction ───────────────────────────────────────────────────────

with QualibrationGraph.build(
    "gef_readout_optimization_graph",
    parameters=GEFReadoutOptParameters(),
) as graph:

    # ── 0. GEF dispersive shift ───────────────────────────────────────────────
    gef_dispersive_shift = library.nodes["20b_dispersive_shift_gef"].copy(
        name="gef_dispersive_shift",
        num_shots=200,
        frequency_span_in_mhz=5.0,
        frequency_step_in_mhz=0.05,
    )
    graph.add_node(gef_dispersive_shift)

    # ── 1. GEF readout length optimization ───────────────────────────────────
    gef_length_opt = library.nodes["14c_readout_gef_length_optimization"].copy(
        name="gef_readout_length_optimization",
        num_shots=2000,
        max_readout_length_in_ns=4000,
        division_length_in_ns=16,
        readout_operation="readout",
        cos_weight_name="iw1",
        sin_weight_name="iw2",
        minus_sin_weight_name="iw3",
    )
    graph.add_node(gef_length_opt)

    # ── 2. GEF readout frequency optimization ─────────────────────────────────
    gef_freq_opt = library.nodes["14_gef_frequency_optimization"].copy(
        name="gef_readout_frequency_optimization",
        num_shots=100,
        frequency_span_in_mhz=2.0,
        frequency_step_in_mhz=0.01,
    )
    graph.add_node(gef_freq_opt)

    # ── 3. GEF readout power optimization ─────────────────────────────────────
    gef_power_opt = library.nodes["14b_readout_gef_power_optimization"].copy(
        name="gef_readout_power_optimization",
        num_shots=200,
        min_amp_factor=0.1,
        max_amp_factor=1.9,
        num_amps=30,
    )
    graph.add_node(gef_power_opt)

    # ── 4. GEF IQ blobs ───────────────────────────────────────────────────────
    gef_iq_blobs = library.nodes["15_iq_blobs_gef"].copy(
        name="gef_iq_blobs",
        num_shots=2000,
    )
    graph.add_node(gef_iq_blobs)

    # ── Execution order ────────────────────────────────────────────────────────
    graph.connect(gef_dispersive_shift, gef_length_opt)
    graph.connect(gef_length_opt, gef_freq_opt)
    graph.connect(gef_freq_opt, gef_power_opt)
    graph.connect(gef_power_opt, gef_iq_blobs)


graph.run()
