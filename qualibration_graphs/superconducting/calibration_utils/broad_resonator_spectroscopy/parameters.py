from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 30.0
    """Span of frequencies to sweep in MHz. Default is 30 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""
    peak_height: float = 2.0
    """Height of the peaks to detect"""
    peak_threshold: float = None
    """Required vertical distance between peak and its neighboring samples"""
    peak_prominence: float = 0.5
    """Required prominence of the peaks to detect"""
    min_frequency_distance_from_zero_mhz: float = 1.0
    """Minimum distance from zero detuning (in MHz) to accept detected dips. Default is 1.0 MHz."""
    peak_merge_distance_hz: float = 100e3
    """Merge distance for magnitude and phase detections (Hz). Phase peaks closer than this to magnitude peaks are discarded. Default is 100 kHz."""
    

class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
