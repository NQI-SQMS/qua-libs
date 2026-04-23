from typing import Optional
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 200
    """Number of averages per sweep point. Default is 200."""
    mode_name: str = "alice"
    """Which cavity mode to probe: attribute name on the Cavity object (e.g. 'alice' or 'bob')."""

    min_duration_ns: int = 16
    """Minimum f0g1 pulse duration in ns. Must be a multiple of 4 (one clock cycle)."""
    max_duration_ns: int = 2000
    """Maximum f0g1 pulse duration in ns (exclusive). Must be a multiple of 4."""
    duration_step_ns: int = 4
    """Step size for the duration sweep in ns. Must be a multiple of 4."""

    operation: str = "f0g1_pi"
    """Operation to play on the f0g1 channel for the time Rabi sweep."""

    use_state_discrimination: bool = True
    """True -> measure qubit state (recommended). False -> measure raw I/Q."""
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
