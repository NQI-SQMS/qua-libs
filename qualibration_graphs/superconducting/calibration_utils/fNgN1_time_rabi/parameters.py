from typing import Literal, Optional
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    fock_level: int = 0
    """Which |n⟩→|n+1⟩ sideband transition to calibrate (0=f0g1, 1=f1g2, ...)."""
    num_shots: int = 200
    """Number of averages per sweep point."""
    mode_name: str = "alice"
    """Which cavity mode: 'alice' or 'bob'."""
    min_duration_ns: int = 16
    """Minimum sideband pulse duration [ns] (multiple of 4)."""
    max_duration_ns: int = 20000
    """Maximum sideband pulse duration [ns] (exclusive, multiple of 4)."""
    duration_step_ns: int = 4
    """Duration sweep step [ns] (multiple of 4)."""
    use_state_discrimination: bool = True
    """True → measure qubit state (recommended). False → measure raw I/Q."""
    use_displaced_threshold: bool = False
    """When True and use_state_discrimination is True, use pair.ge_iq_threshold_displaced
    instead of the vacuum readout threshold (calibrated by node 26j)."""
    use_confusion_matrix_correction: bool = False
    """Apply ge confusion matrix correction to averaged state probabilities."""
    cavity_reset_type: Literal["thermal", "active_sideband"] = "thermal"
    """How to reset the cavity at the end of each shot.
    'thermal'         - wait thermalization_time_factor × T1 (passive decay).
    'active_sideband' - cascade sideband π-pulses to actively remove photons;
                        requires calibrated sideband operations (nodes 26 / 26b)."""
    cavity_active_cooling_fock_n: int = 1
    """Starting Fock level for active sideband cooling (only used when
    cavity_reset_type='active_sideband').  Set to 1 for thermal state cooling."""
    sideband_pulse_duration_ns: Optional[int] = None
    """Override the sideband pulse flat-top duration [ns] during active cooling.
    When None, the calibrated pi_flat_top_length_ns from pair.transitions is used.
    Must be a multiple of 4 ns."""
    cavity_thermalization_time_ns: Optional[int] = None
    """Override the cavity thermalization wait [ns] when cavity_reset_type='thermal'.
    When None, the cavity's T1 × thermalization_time_factor is used.
    Useful for long-T1 cavities (e.g. SRF) where the default would be impractically long."""
    ramp_length_ns: Optional[int] = None
    """Override the ramp length [ns] for each edge (rise or fall) of the FlatTopGaussianPulse.
    Total pulse length = flat_top + 2 * ramp_length_ns. Must be a multiple of 4.
    When None, the ramp length is preserved from the current pulse definition."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
