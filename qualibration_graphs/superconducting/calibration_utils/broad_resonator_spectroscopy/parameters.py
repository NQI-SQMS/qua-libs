from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
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
    blacklist_exclusion_radius_mhz: float = 10.0
    """Exclusion radius around blacklisted resonator frequencies in MHz.
    Detected dips within this distance of a blacklisted frequency are ignored.
    Default is 10.0 MHz."""
    readout_power_dbm: Optional[float] = None
    """Readout power in dBm for the broad spectroscopy sweep.
    If None, the current QUAM state power is used unchanged.
    The QUAM state is reverted to its original value after the node finishes."""
    max_amp: float = 0.1
    """Maximum readout pulse amplitude (OPX units, 0–0.5).
    Only used when readout_power_dbm is set. Default is 0.1."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
