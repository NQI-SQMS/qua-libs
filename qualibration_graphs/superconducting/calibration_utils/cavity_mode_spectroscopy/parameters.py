from typing import Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform."""
    mode_name: str = "alice"
    """Which cavity mode to probe: attribute name on the Cavity object (e.g. 'alice' or 'bob')."""
    frequency_span_in_mhz: float = 400.0
    """Frequency span to sweep around the current cavity_mode_drive RF_frequency [MHz]."""
    frequency_step_in_mhz: float = 1.0
    """Frequency step for the sweep [MHz]."""
    operation: str = "saturation"
    """Pulse name on the cavity_mode_drive channel to play during the sweep."""
    operation_amplitude_factor: float = 1.0
    """Pre-factor applied to the operation amplitude."""
    operation_len_in_ns: Optional[int] = None
    """Override pulse length [ns]. None → use the pulse's own length."""
    qubit_probe_operation: str = "selective_x180"
    """Qubit operation to probe photon number via dispersive coupling.
    A narrow-bandwidth pulse (e.g. 'selective_x180') is most sensitive:
    when photons are present the qubit frequency shifts and the probe pulse
    becomes off-resonant, producing a dip in the excitation probability."""
    use_state_discrimination: bool = True
    """True → measure qubit state (recommended). False → measure raw I/Q."""
    use_confusion_matrix_correction: bool = False
    """Apply ge confusion matrix correction to averaged state probabilities."""
    min_dip_fraction: float = 0.05
    """Minimum dip depth as a fraction of the signal range. Shallower dips are rejected."""
    subtract_baseline: bool = True
    """If True, run a second sub-sequence per frequency point WITHOUT the qubit
    probe pulse and subtract its averaged IQ from the signal IQ before any
    state-discrimination threshold is applied -- removes the amplitude/frequency-
    dependent IQ offset caused by the cavity probe drive leaking into the
    readout, the same artifact handled in displacement_calibration_vacuum.
    If False, only the original single-sequence protocol runs."""
    cavity_thermalization_time_ns: Optional[int] = None
    """Override the cavity thermalization wait time [ns]. When set, this value is used
    instead of cavity_mode.T1 × cavity_mode.thermalization_time_factor.
    Useful when T1 is very long (e.g. SRF cavities) and you want a shorter practical wait."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for 27_cavity_mode_spectroscopy."""
