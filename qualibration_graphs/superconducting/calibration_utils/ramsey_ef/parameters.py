from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters

from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters, IdleTimeNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_detuning_in_mhz: float = 1.0
    """Frequency detuning in MHz applied symmetrically (±) around the EF drive frequency. Default is 1.0 MHz."""
    ef_x180_operation: str = "EF_x180"
    """Name of the EF pi-pulse operation. The pi/2 pulses are played with amplitude_scale=0.5."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
