from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of individual shots per (frequency, amplitude) point. Default is 100."""
    frequency_span_in_mhz: float = 4
    """Span of frequencies to sweep around the resonator IF in MHz. Default is 4 MHz."""
    frequency_step_in_mhz: float = 0.2
    """Step size for the frequency sweep in MHz. Default is 0.2 MHz."""
    start_amp: float = 0.5
    """Start readout amplitude pre-factor. Default is 0.5."""
    end_amp: float = 1.5
    """End readout amplitude pre-factor. Default is 1.5."""
    num_amps: int = 11
    """Number of amplitudes to sweep. Default is 11."""
    outliers_threshold: float = 0.98
    """Outliers threshold used to discard unreliable (frequency, amplitude) points when looking for the
    optimum. Default is 0.98."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
