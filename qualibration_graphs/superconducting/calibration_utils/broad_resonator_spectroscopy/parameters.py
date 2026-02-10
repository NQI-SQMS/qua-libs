from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters

from typing import Optional

class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 100.0
    """Span of frequencies to sweep in MHz. Default is 100 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""
    peak_height: Optional[float] = None
    """Height of the peaks to detect"""
    peak_threshold: Optional[float] = None
    """Required vertical distance between peak and its neighboring samples"""
    peak_prominence: Optional[float] = 2
    """Required prominence of the peaks to detect"""
    peak_width : Optional[tuple[float, float]] = (1, 10.0)
    """Required width of the peaks to detect in samples (min, max)"""
    

class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
