from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1000
    """Number of runs to perform. Default is 1000."""
    frequency_span_in_mhz: float = 60
    """Span of frequencies to sweep around the resonator frequency, in MHz. Default is 60 MHz."""
    frequency_num_points: int = 30
    """Number of frequency points to sweep. Default is 30."""
    min_amp_factor: float = 0.25
    """Minimum readout amplitude, as a prefactor of the nominal readout amplitude. Default is 0.25."""
    max_amp_factor: float = 4.0
    """Maximum readout amplitude, as a prefactor of the nominal readout amplitude. Default is 4.0."""
    num_amps: int = 5
    """Number of amplitudes to sweep (log-spaced). Default is 5."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
