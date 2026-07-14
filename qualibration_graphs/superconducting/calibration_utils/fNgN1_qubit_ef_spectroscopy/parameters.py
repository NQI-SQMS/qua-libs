from typing import Literal, Optional
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    fock_level: int = 0
    """Which sideband transition to calibrate (0=f0g1, 1=f1g2, ...). Internally prepares |fock_level+1⟩ photons."""
    num_shots: int = 200
    """Number of averages."""
    mode_name: str = "alice"
    """Which cavity mode: 'alice' or 'bob'."""
    frequency_span_in_mhz: float = 2.0
    """Frequency span to sweep around the ef_chi_focka-shifted ef frequency [MHz]."""
    frequency_step_in_mhz: float = 0.05
    """Step size for the frequency sweep [MHz]."""
    operation: str = "saturation"
    """Pulse name on the qubit xy channel to play."""
    operation_amplitude_factor: float = 1.0
    """Pre-factor applied to the operation amplitude."""
    operation_len_in_ns: Optional[int] = None
    """Override pulse length [ns]. None → use the pulse's own length."""
    use_state_discrimination: bool = True
    """True → measure qubit state. False → measure raw I/Q."""
    use_confusion_matrix_correction: bool = False
    """Apply ge confusion matrix correction to averaged state probabilities."""
    cavity_reset_type: Literal["thermal", "active_sideband"] = "thermal"
    """How to reset the cavity at the end of each shot.
    'thermal'         – wait thermalization_time_factor × T1 (passive decay).
    'active_sideband' – cascade sideband π-pulses to actively remove photons."""
    cavity_active_cooling_fock_n: int = 1
    """Starting Fock level for active sideband cooling (only used when
    cavity_reset_type='active_sideband').  Set to 1 for thermal state cooling."""
    sideband_pulse_duration_ns: Optional[int] = None
    """Override the sideband pulse flat-top duration [ns] during active cooling.
    When None, the calibrated pi_flat_top_length_ns from pair.transitions is used."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
