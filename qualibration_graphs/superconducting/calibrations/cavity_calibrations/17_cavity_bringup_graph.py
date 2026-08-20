# %%
"""
Cavity Mode Bring-Up Graph

Calibrates a single cavity mode (alice or bob):

  cavity_mode_spectroscopy
  → displacement_calibration     (vacuum state; finds 1-photon amplitude)  [loop]
  → cavity_T1                    (coherent T1)
  → cavity_T2                    (coherent T2 Ramsey)
  → parity_time_measurement      (optimal parity mapping time)
  → fock1_T1                     (Fock |1> photon lifetime)
  → fock1_T2                     (Fock |1> T2 Ramsey)
"""

from typing import List

from qualibrate import GraphParameters, QualibrationGraph, QualibrationLibrary
from calibration_utils.bringup_graphs import should_repeat_displacement_vacuum

library = QualibrationLibrary.get_active_library()


class CavityBringUpParameters(GraphParameters):
    """Graph-flow parameters for the cavity mode bring-up graph."""

    qubits: List[str] = ["q1"]
    cavity_mode_name: str = "alice"
    """Which cavity mode to calibrate: 'alice' or 'bob'."""
    max_displacement_vacuum_iterations: int = 5


with QualibrationGraph.build(
    "cavity_bringup_graph",
    parameters=CavityBringUpParameters(),
) as graph:

    mode = graph.parameters.cavity_mode_name

    cav_spec = library.nodes["02_cavity_mode_spectroscopy"].copy(
        name="cavity_mode_spectroscopy",
        mode_name=mode,
        frequency_span_in_mhz=25.0,
        frequency_step_in_mhz=0.1,
        operation="saturation",
        operation_len_in_ns=1_000,
        operation_amplitude_factor=0.01,
        num_shots=100,
        qubit_probe_operation="selective_x180",
        use_state_discrimination=True,
        min_dip_fraction=0.05,
        subtract_baseline=True,
    )
    graph.add_node(cav_spec)

    displ = library.nodes["03_displacement_calibration_vacuum"].copy(
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

    cav_t1 = library.nodes["04_cavity_coherent_T1"].copy(
        name="cavity_T1",
        mode_name=mode,
        min_wait_time_in_ns=40,
        max_wait_time_in_ns=5_000_000,
        wait_time_num_points=21,
        num_shots=500,
        log_or_linear_sweep="linear",
        displacement_alpha=1.0,
        use_state_discrimination=True,
        cavity_reset_type="thermal",
    )
    graph.add_node(cav_t1)

    cav_t2 = library.nodes["08_cavity_coherent_T2"].copy(
        name="cavity_T2",
        mode_name=mode,
        min_wait_time_in_ns=40,
        max_wait_time_in_ns=2_000_000,
        wait_time_num_points=21,
        num_shots=500,
        displacement_alpha=1.0,
        ramsey_detuning_hz=1000.0,
        use_state_discrimination=True,
        cavity_reset_type="thermal",
    )
    graph.add_node(cav_t2)

    parity = library.nodes["09_parity_time_measurement"].copy(
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

    fock1_t1 = library.nodes["11_cavity_fock1_T1"].copy(
        name="fock1_T1",
        mode_name=mode,
        min_wait_time_in_ns=40,
        max_wait_time_in_ns=5_000_000,
        wait_time_num_points=21,
        num_shots=500,
        fock1_prep_method="sideband",
        fock1_alpha1=1.0,
        fock1_alpha2=-0.59,
        use_state_discrimination=True,
        cavity_reset_type="thermal",
    )
    graph.add_node(fock1_t1)

    fock1_t2 = library.nodes["12_cavity_fock1_T2"].copy(
        name="fock1_T2",
        mode_name=mode,
        min_wait_time_in_ns=40,
        max_wait_time_in_ns=2_000_000,
        wait_time_num_points=21,
        num_shots=500,
        fock1_prep_method="sideband",
        fock1_alpha1=1.0,
        fock1_alpha2=-0.59,
        ramsey_detuning_hz=400.0,
        use_state_discrimination=True,
        cavity_reset_type="thermal",
    )
    graph.add_node(fock1_t2)

    graph.connect(cav_spec, displ)
    graph.connect(displ, cav_t1)
    graph.connect(cav_t1, cav_t2)
    graph.connect(cav_t2, parity)
    graph.connect(parity, fock1_t1)
    graph.connect(fock1_t1, fock1_t2)


graph.run()
