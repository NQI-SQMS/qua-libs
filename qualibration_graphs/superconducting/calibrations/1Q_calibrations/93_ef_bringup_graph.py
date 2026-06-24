# %%
"""
EF-Transition Bring-Up Graph

Calibrates the EF (|e>→|f>) transition of a fixed-frequency transmon:

  ef_discovery [loop: retry if no EF oscillation]:
    ef_spectroscopy  [loop: retry on no peak]
    → ef_tentative_rabi       (checks oscillation; blacklists EF freq on failure)
  → ef_rabi_ramsey [loop: until |EF detuning| converges]:
      ef_power_rabi [inner loop: amplitude convergence]
      → ef_ramsey
  → ef_T1
  → gef_readout_frequency_optimization
  → gef_iq_blobs
"""

from typing import List

from qualibrate import GraphParameters, QualibrationGraph, QualibrationLibrary
from calibration_utils.bringup_graphs import build_ef_bringup

library = QualibrationLibrary.get_active_library()


class EFBringUpParameters(GraphParameters):
    """Graph-flow and measurement parameters for the EF-transition bring-up graph."""

    qubits: List[str] = ["q1"]
    # Iteration limits
    max_ef_discovery_iterations: int = 3
    max_ef_spec_iterations: int = 3
    ef_max_iterations: int = 5
    ef_rabi_max_amplitude_iterations: int = 5
    # Convergence threshold
    ef_freq_threshold_hz: float = 50_000.0
    """Stop the ef_power_rabi → ef_ramsey loop when |EF detuning| < this value [Hz]."""

    # ── EF spectroscopy ────────────────────────────────────────────────────────
    ef_spec_frequency_span_mhz: float = 300.0
    ef_spec_frequency_step_mhz: float = 1.0
    ef_spec_operation: str = "saturation"
    ef_spec_operation_len_in_ns: int = 20_000
    ef_spec_amplitude_factor: float = 1.0
    ef_spec_num_shots: int = 100
    ef_spec_find_dip: bool = False
    """True for reflection / SRF setups."""
    ef_spec_target_peak_width: float = 3e6
    ef_spec_update_pulses_amplitude: bool = False

    # ── EF power Rabi (amplitude-only convergence) ────────────────────────────
    ef_rabi_min_amp_factor: float = 0.001
    ef_rabi_max_amp_factor: float = 1.9
    ef_rabi_amp_factor_step: float = 0.01
    ef_rabi_num_shots: int = 200

    # ── EF Ramsey ──────────────────────────────────────────────────────────────
    ef_ramsey_num_shots: int = 200
    ef_ramsey_frequency_detuning_in_mhz: float = 0.1
    ef_ramsey_min_wait_time_in_ns: int = 16
    ef_ramsey_max_wait_time_in_ns: int = 100_000
    ef_ramsey_wait_time_num_points: int = 100
    ef_ramsey_log_or_linear_sweep: str = "linear"

    # ── EF T1 ──────────────────────────────────────────────────────────────────
    ef_t1_num_shots: int = 500
    ef_t1_min_wait_time_ns: int = 16
    ef_t1_max_wait_time_ns: int = 300_000
    ef_t1_wait_time_num_points: int = 100
    ef_t1_log_or_linear_sweep: str = "linear"

    # ── GEF readout frequency optimization ────────────────────────────────────
    gef_freq_opt_num_shots: int = 100
    gef_freq_opt_frequency_span_mhz: float = 2.0
    gef_freq_opt_frequency_step_mhz: float = 0.01

    # ── GEF IQ blobs ───────────────────────────────────────────────────────────
    gef_iq_blobs_num_shots: int = 2000


with QualibrationGraph.build(
    "ef_bringup_graph",
    parameters=EFBringUpParameters(),
) as graph:
    ef_bringup = build_ef_bringup(graph, library)
    graph.add_node(ef_bringup)


graph.run()
