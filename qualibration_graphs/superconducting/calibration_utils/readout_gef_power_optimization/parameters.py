from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 200
    """Number of averages per amplitude point. Default is 200."""
    min_amp_factor: float = 0.1
    """Minimum readout amplitude scale factor (relative to current amplitude). Default is 0.1."""
    max_amp_factor: float = 1.9
    """Maximum readout amplitude scale factor (relative to current amplitude). Default is 1.9."""
    num_amps: int = 30
    """Number of amplitude steps. Default is 30."""
    operation: str = "readout"
    """Readout operation name. Default is 'readout'."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
