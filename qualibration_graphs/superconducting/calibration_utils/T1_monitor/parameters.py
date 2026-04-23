from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters, IdleTimeNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 200
    """Number of averages per T1 sweep iteration."""
    n_iter: int = 100
    """Number of T1 measurements to repeat."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
