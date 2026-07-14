from typing import Optional
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages."""
    min_amp_factor: float = 0.01
    """Minimum amplitude factor relative to the f0g1 saturation pulse amplitude."""
    max_amp_factor: float = 1.99
    """Maximum amplitude factor relative to the f0g1 saturation pulse amplitude."""
    amp_factor_step: float = 0.01
    """Step size for the amplitude factor sweep."""
    mode_name: str = "alice"
    """Which cavity mode to calibrate: attribute name on the Cavity object (e.g. 'alice' or 'bob')."""
    operation: str = "sideband_flat_top"
    """Name of the sideband flat-top operation whose amplitude will be calibrated."""
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
