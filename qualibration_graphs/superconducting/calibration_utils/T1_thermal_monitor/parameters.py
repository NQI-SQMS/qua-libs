from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 200
    """Number of averages per sweep (shared by T1 and RPM sweeps)."""
    n_iter: int = 60
    """Number of (T1 + RPM) iterations to repeat."""
    min_amp_factor: float = 0.0
    """Minimum EF amplitude scale factor for the RPM sweep."""
    max_amp_factor: float = 2.0
    """Maximum EF amplitude scale factor for the RPM sweep."""
    amp_factor_step: float = 0.05
    """Step size for the RPM amplitude scale sweep."""
    ef_operation: str = "EF_x180"
    """Name of the EF π-pulse operation on qubit.xy."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
