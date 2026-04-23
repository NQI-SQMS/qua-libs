from typing import Optional
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform."""
    frequency_span_in_mhz: float = 100
    """Frequency span to sweep around the current cavity drive RF_frequency [MHz]."""
    frequency_step_in_mhz: float = 0.25
    """Step size for the frequency sweep [MHz]."""
    operation: str = "saturation"
    """Pulse name on the cavity drive channel to apply during the sweep."""
    operation_amplitude_factor: float = 1.0
    """Pre-factor applied to the operation amplitude."""
    operation_len_in_ns: Optional[int] = None
    """Override pulse length [ns]. None → use the pulse's own length."""
    cavity_name: str = "alice"
    """Which cavity mode to calibrate: 'alice' or 'bob'."""
    min_dip_fraction: float = 0.1
    """Minimum dip depth as a fraction of the state population range. Dips smaller than this are rejected."""
    use_state_discrimination: bool = True
    """True → measure qubit state (recommended). False → measure raw I/Q."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
