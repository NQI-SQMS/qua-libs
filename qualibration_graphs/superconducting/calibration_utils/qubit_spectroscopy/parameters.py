from typing import Optional
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 100
    """Span of frequencies to sweep in MHz. Default is 100 MHz."""
    frequency_step_in_mhz: float = 0.25
    """Step size for frequency sweep in MHz. Default is 0.25 MHz."""
    operation: str = "saturation"
    """Type of operation to perform. Default is "saturation"."""
    operation_amplitude_factor: float = 1.0
    """Amplitude pre-factor for the operation. Default is 1.0."""
    operation_len_in_ns: Optional[int] = None
    """Length of the operation in nanoseconds. Default is the predefined pulse length."""
    target_peak_width: float = 3e6
    """Target peak width in Hz. Default is 3e6 Hz."""
    update_pulses_amplitude: bool = False
    """Whether to update the saturation pulse and x180/x90 pulse amplitudes based on the peak width. Default is False"""
    find_dip: bool = False
    """Set True for reflection readout where the qubit appears as a dip in I_rot (e.g. SRF setups)."""
    signal_source: str = "I_rot"
    """Signal used for analysis and plotting: 'I_rot' (PCA-rotated quadrature) or 'IQ_abs' (magnitude).
    When 'IQ_abs' is chosen the integration weight angle is NOT updated."""
    update_iw_angle: bool = True
    """When False, skip updating integration_weights_angle in update_state even when signal_source='I_rot'.
    Set False in bringup graphs where the angle was already calibrated by qubit_spectroscopy_vs_power."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
