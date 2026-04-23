from typing import Optional
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform."""
    frequency_span_in_mhz: float = 100
    """Frequency span to sweep around the current f0g1 RF_frequency [MHz]."""
    frequency_step_in_mhz: float = 0.25
    """Step size for the frequency sweep [MHz]."""
    operation: str = "saturation"
    """Pulse name on the f0g1 channel to apply during the sweep (long square saturation pulse)."""
    operation_amplitude_factor: float = 1.0
    """Pre-factor applied to the operation amplitude."""
    operation_len_in_ns: Optional[int] = None
    """Override pulse length [ns]. None → use the pulse's own length."""
    mode_name: str = "alice"
    """Which cavity mode to calibrate: attribute name on the Cavity object (e.g. 'alice' or 'bob')."""
    min_dip_fraction: float = 0.1
    """Minimum dip depth as a fraction of the state population range. Dips smaller than this are rejected."""
    use_state_discrimination: bool = True
    """True → measure qubit state (recommended). False → measure raw I/Q."""
    cavity_thermalization_time_ns: Optional[int] = None
    """Override the cavity thermalization wait at the start of each shot [ns].
    When set, this value is used instead of cavity_mode.T1 × thermalization_time_factor.
    Useful for long-T1 cavities (e.g. SRF) where the default would be impractically long."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
