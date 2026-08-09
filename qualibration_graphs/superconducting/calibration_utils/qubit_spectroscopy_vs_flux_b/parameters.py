"""Parameters for qubit spectroscopy versus flux calibration."""

from typing import Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for qubit spectroscopy versus flux."""

    num_shots: int = 50
    """Number of averages to perform. Default is 50."""
    operation: str = "saturation"
    """Operation to perform. Default is "saturation"."""
    operation_amplitude_factor: float = 0.1
    """Amplitude factor for the operation. Default is 0.1."""
    operation_len_in_ns: Optional[int] = None
    """Length of the operation in ns. Default is the predefined pulse length."""
    detuning_in_mhz: float = 0.0
    """Center the frequency sweep this many MHz BELOW the qubit's bare 0->1 (GE)
    frequency, instead of symmetrically around it.

    Node 09 always centers the XY-drive sweep on the GE frequency (detuning 0),
    but the transmon frequency only bends DOWNWARD as |flux| grows, so the upper
    half of a GE-centered window is structurally empty noise. Set this positive
    to slide the whole window down onto the flux arch, so a much narrower
    ``frequency_span_in_mhz`` still captures it. 0.0 reproduces the node-09
    GE-centered behavior.

    If the resulting intermediate frequency would exceed the +-400 MHz limit, the
    node transparently shifts the upconverter (LO) for the duration of the scan
    and reverts it before analysis. Default is 0 MHz."""
    frequency_span_in_mhz: float = 100.0
    """Frequency span in MHz. Default is 100 MHz."""
    frequency_step_in_mhz: float = 0.5
    """Frequency step in MHz. Default is 0.5 MHz."""
    flux_offset_span_in_v: float = 0.05
    """Minimum flux bias offset in volts. Default is -0.02 V."""
    num_flux_points: int = 11
    """Number of flux points. Default is 11."""
    input_line_impedance_in_ohm: Optional[int] = 50
    """Input line impedance in ohms. Default is 50 Ohm."""
    line_attenuation_in_db: Optional[int] = 0
    """Line attenuation in dB. Default is 0 dB."""
    settle_time_in_ns: int = 20000
    """Settle time for flux to wait in ns. Default is 5000 ns."""
    buffer_time_in_ns: int = 100
    """Buffer time for flux before readout in ns. Default is 100 ns."""
    target_flux: Optional[float] = None
    """(parabola mode only) Flux center [V] around which to isolate the primary
    arch for parabola fitting. If None, the default per-qubit value is taken
    from ``qubit.z.joint_offset`` or ``qubit.z.independent_offset`` depending on
    the qubit's ``flux_point``; if neither is set, 0.0 V is used."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameters for qubit spectroscopy versus flux calibration."""

    pass
