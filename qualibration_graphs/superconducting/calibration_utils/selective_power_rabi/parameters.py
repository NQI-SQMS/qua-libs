from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    operation: str = "selective_x180"
    """Which operation (pulse name) to calibrate amplitude for."""

    selective_length_ns: int = 10000
    """Length of the selective pulse in ns (controls bandwidth ≈ 1/length)."""

    num_shots: int = 100
    """Number of averages."""

    min_amp_ratio: float = -1.8
    """Minimum amplitude pre-factor for the sweep."""

    max_amp_ratio: float = 1.8
    """Maximum amplitude pre-factor for the sweep."""

    amp_points: int = 101
    """Number of amplitude sweep points."""

    use_state_discrimination: bool = True
    """True → measure qubit state; False → measure raw I/Q."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
