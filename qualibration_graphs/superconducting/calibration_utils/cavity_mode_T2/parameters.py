from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    mode_name: str = "alice"
    """Which cavity mode to probe: attribute name on the Cavity object (e.g. 'alice' or 'bob')."""

    ramsey_detuning_hz: float = 1000.0
    """Artificial detuning applied via frame rotation [Hz]. Creates Ramsey oscillation fringes
    needed to extract T2ramsey. Should be >> 1/T2 to resolve several oscillation periods."""

    use_state_discrimination: bool = True
    """True -> measure qubit state (recommended). False -> measure raw I/Q."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    IdleTimeNodeParameters,
    QubitsExperimentNodeParameters,
):
    pass
