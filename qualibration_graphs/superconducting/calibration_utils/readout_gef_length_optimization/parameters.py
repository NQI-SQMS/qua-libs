from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 2000
    """Number of averages per division point. Default is 2000."""
    max_readout_length_in_ns: int = 4000
    """Readout pulse length to use during the sweep [ns]. Must be a multiple of 4. Default is 4000 ns."""
    division_length_in_ns: int = 16
    """Accumulated demodulation chunk size [ns]. Must be a multiple of 4. Default is 16 ns."""
    readout_operation: str = "readout"
    """Name of the readout operation. Default is 'readout'."""
    cos_weight_name: str = "iw1"
    """Name of the cosine integration weight used for accumulated demodulation."""
    sin_weight_name: str = "iw2"
    """Name of the sine integration weight used for accumulated demodulation."""
    minus_sin_weight_name: str = "iw3"
    """Name of the minus-sine integration weight used for accumulated demodulation."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
