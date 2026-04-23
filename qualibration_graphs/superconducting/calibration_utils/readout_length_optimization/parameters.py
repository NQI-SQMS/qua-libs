from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 10000
    """Number of single-shot measurements for each state. Default is 10000."""
    max_readout_length_in_ns: int = 8000
    """Readout pulse length to use during the sweep [ns]. Must be a multiple of 4."""
    division_length_in_ns: int = 16
    """Accumulated demodulation chunk size in nanoseconds (must be a multiple of 4). Default is 16 ns."""
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
