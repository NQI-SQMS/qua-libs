from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""

    frequency_span_in_mhz: float = 2
    """Span of frequencies to sweep in MHz. Default is 2 MHz."""

    frequency_step_in_mhz: float = 0.05
    """Step size for frequency sweep in MHz. Default is 0.05 MHz."""

    min_power_dbm: int = -50
    """Lower readout power (used as reference). Default is -50 dBm."""

    max_power_dbm: int = -25
    """Higher readout power (used to test Kerr shift). Default is -25 dBm."""

    num_power_points: int = 2
    """Number of power points. Must be exactly 2 for shift-based analysis."""

    max_amp: float = 0.1
    """Maximum readout amplitude for the experiment. Default is 0.1."""

    frequency_shift_threshold_in_hz: float = 2e5
    """
    Minimum absolute frequency shift (in Hz) between low and high power
    required to declare a Kerr-induced shift.
    
    Typical values: 1e5 - 5e5 Hz, depending on resonator linewidth.
    """


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
