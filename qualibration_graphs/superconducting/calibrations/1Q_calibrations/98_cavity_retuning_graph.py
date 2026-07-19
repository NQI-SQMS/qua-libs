# %%
"""
Cavity Mode Retuning Graph

Recalibrates the displacement amplitude scale (A_1ph) and the dispersive
coupling rate (χ_eff / parity mapping time) for a single cavity mode.

  displacement_calibration   (vacuum state; finds 1-photon amplitude)  [loop]
  → parity_time_measurement  (optimal parity mapping time and χ_eff)

Use this graph after the full cavity bringup (94) has been run at least once.
It skips spectroscopy and coherence measurements, targeting only the two
parameters most likely to drift between cooldowns or after hardware changes:
the displacement amplitude scale and the qubit–cavity dispersive shift.
"""

from typing import List

from qualibrate import GraphParameters, QualibrationGraph, QualibrationLibrary
from calibration_utils.bringup_graphs import should_repeat_displacement_vacuum

library = QualibrationLibrary.get_active_library()


class CavityRetuningParameters(GraphParameters):
    """Graph-flow parameters for the cavity mode retuning graph."""

    qubits: List[str] = ["q1"]
    cavity_mode_name: str = "alice"
    """Which cavity mode to recalibrate: 'alice' or 'bob'."""
    max_displacement_vacuum_iterations: int = 5


with QualibrationGraph.build(
    "cavity_retuning_graph",
    parameters=CavityRetuningParameters(),
) as graph:

    mode = graph.parameters.cavity_mode_name

    displ = library.nodes["22_displacement_calibration_vacuum"].copy(
        name="displacement_calibration",
        mode_name=mode,
        amp_min=-4.0,
        amp_max=4.0,
        amp_points=61,
        num_shots=200,
        qubit_pulse="selective_x180",
        cavity_reset_type="thermal",
        use_state_discrimination=True,
        subtract_baseline=True,
        use_adaptive=True,
        target_n_sigma=5.0,
    )
    graph.add_node(displ)
    graph.loop(
        displ,
        on=should_repeat_displacement_vacuum,
        max_iterations=graph.parameters.max_displacement_vacuum_iterations,
    )

    parity = library.nodes["28_parity_time_measurement"].copy(
        name="parity_time_measurement",
        mode_name=mode,
        min_delay_ns=16,
        max_delay_ns=4000,
        delay_step_ns=16,
        num_shots=1000,
        displacement_scale=0.5,
        use_state_discrimination=True,
        cavity_reset_type="thermal",
    )
    graph.add_node(parity)

    graph.connect(displ, parity)


graph.run()
