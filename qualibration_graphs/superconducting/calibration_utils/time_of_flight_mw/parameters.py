from typing import Optional
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    time_of_flight_in_ns: Optional[int] = None
    """Time of flight in nanoseconds. None means use the value already stored in the machine state."""
    readout_amplitude_in_dBm: Optional[float] = -12
    """Readout amplitude in dBm. Default is -12 dBm."""
    readout_length_in_ns: Optional[int] = 1000
    """Readout length in nanoseconds. Default is 1000 ns."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
