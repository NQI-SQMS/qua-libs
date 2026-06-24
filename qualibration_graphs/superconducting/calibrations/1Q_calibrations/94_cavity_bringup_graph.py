# %%
"""
Cavity Mode Bring-Up Graph

Calibrates a single cavity mode (alice or bob):

  cavity_mode_spectroscopy
  → displacement_calibration     (vacuum state; finds 1-photon amplitude)
  → cavity_T1                    (coherent T1)
  → parity_time_measurement      (optimal parity mapping time)
"""

from typing import List

from qualibrate import GraphParameters, QualibrationGraph, QualibrationLibrary
from calibration_utils.bringup_graphs import build_cavity_bringup

library = QualibrationLibrary.get_active_library()


class CavityBringUpParameters(GraphParameters):
    """Graph-flow and measurement parameters for the cavity mode bring-up graph."""

    qubits: List[str] = ["q1"]
    cavity_mode_name: str = "alice"
    """Which cavity mode to calibrate: 'alice' or 'bob'."""

    # ── Cavity mode spectroscopy ───────────────────────────────────────────────
    cavity_spec_frequency_span_mhz: float = 25.0
    cavity_spec_frequency_step_mhz: float = 0.1
    cavity_spec_operation: str = "saturation"
    cavity_spec_operation_len_in_ns: int = 1_000
    cavity_spec_amplitude_factor: float = 0.01
    cavity_spec_num_shots: int = 100
    cavity_spec_qubit_probe_operation: str = "selective_x180"
    cavity_spec_use_state_discrimination: bool = True
    cavity_spec_min_dip_fraction: float = 0.05
    cavity_spec_subtract_baseline: bool = True

    # ── Displacement calibration ──────────────────────────────────────────────
    cavity_disp_amp_min: float = -4.0
    cavity_disp_amp_max: float = 4.0
    cavity_disp_amp_points: int = 61
    cavity_disp_num_shots: int = 200
    cavity_disp_qubit_pulse: str = "selective_x180"
    """'selective_x180' or 'x180'."""
    cavity_disp_cavity_reset_type: str = "thermal"
    """'thermal' or 'active_sideband'."""
    cavity_disp_active_reset: bool = True
    cavity_disp_use_state_discrimination: bool = True
    cavity_disp_subtract_baseline: bool = True
    cavity_disp_use_adaptive: bool = True
    cavity_disp_target_n_sigma: float = 5.0
    max_displacement_vacuum_iterations: int = 5

    # ── Cavity T1 ──────────────────────────────────────────────────────────────
    cavity_t1_min_wait_ns: int = 40
    cavity_t1_max_wait_ns: int = 5_000_000
    cavity_t1_num_points: int = 21
    cavity_t1_num_shots: int = 500
    cavity_t1_log_or_linear_sweep: str = "linear"
    """'log' or 'linear'."""
    cavity_t1_displacement_scale: float = 1.0
    """scale=1 -> 1 photon (calibrated by displacement_calibration)."""
    cavity_t1_use_state_discrimination: bool = True
    cavity_t1_cavity_reset_type: str = "thermal"

    # ── Parity time measurement ───────────────────────────────────────────────
    parity_min_delay_ns: int = 16
    parity_max_delay_ns: int = 4000
    parity_delay_step_ns: int = 16
    parity_num_shots: int = 1000
    parity_displacement_scale: float = 0.5
    parity_use_state_discrimination: bool = True
    parity_cavity_reset_type: str = "thermal"


with QualibrationGraph.build(
    "cavity_bringup_graph",
    parameters=CavityBringUpParameters(),
) as graph:
    cavity_bringup = build_cavity_bringup(graph, library)
    graph.add_node(cavity_bringup)


graph.run()
