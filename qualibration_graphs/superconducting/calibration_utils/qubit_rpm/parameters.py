from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 500
    """Number of averages per sweep point."""
    min_amp_factor: float = 0.0
    """Minimum EF amplitude scale factor (0 = no rotation)."""
    max_amp_factor: float = 4.0
    """Maximum EF amplitude scale factor (4 = two full Rabi oscillations)."""
    amp_factor_step: float = 0.02
    """Step size for the amplitude scale sweep."""
    ef_operation: str = "EF_x180"
    """Name of the EF π-pulse operation on qubit.xy."""
    use_state_discrimination: bool = True
    """True → measure qubit state (recommended). False → measure raw I/Q."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
