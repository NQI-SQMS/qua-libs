from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 200
    """Number of averages per resonator frequency point."""
    frequency_span_in_mhz: float = 5.0
    """Frequency span around the current resonator RF_frequency [MHz]."""
    frequency_step_in_mhz: float = 0.05
    """Step size for the frequency sweep [MHz]."""
    min_dip_contrast: float = 0.05
    """Minimum relative contrast for a dip to be accepted (fraction of baseline amplitude)."""
    lo_leakage_exclusion_mhz: float = 10.0
    """Exclude peaks within ±lo_leakage_exclusion_mhz of the LO frequency [MHz]."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
