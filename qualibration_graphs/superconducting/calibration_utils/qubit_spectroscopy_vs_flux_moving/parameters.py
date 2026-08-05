"""Parameter definitions for moving-qubit spectroscopy vs flux calibration."""

from typing import ClassVar, Literal, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for moving-qubit spectroscopy vs flux calibration."""

    num_shots: int = 50
    """Number of averages to perform. Default is 50."""
    operation: str = "saturation"
    """Operation to perform on the moving qubit. Default is "saturation"."""
    operation_amplitude_factor: float = 0.1
    """Amplitude factor for the operation. Default is 0.1."""
    operation_len_in_ns: Optional[int] = None
    """Length of the operation in ns. Default is the predefined pulse length."""
    frequency_span_in_mhz: float = 100.0
    """Frequency span in MHz, centered on the stationary qubit's frequency (not the moving
    qubit's own bare frequency), and played through the moving qubit's XY line. Default is
    100 MHz."""
    frequency_step_in_mhz: float = 0.5
    """Frequency step in MHz. Default is 0.5 MHz."""
    flux_offset_span_in_v: float = 0.05
    """Flux bias offset span in volts, applied around the moving qubit's own independent idle
    offset (qubit.z.independent_offset). Default is 0.05 V."""
    num_flux_points: int = 11
    """Number of flux points. Default is 11."""
    input_line_impedance_in_ohm: Optional[int] = 50
    """Input line impedance in ohms. Default is 50 Ohm."""
    line_attenuation_in_db: Optional[int] = 0
    """Line attenuation in dB. Default is 0 dB."""
    quantity: Literal["IQ_abs", "I", "Q"] = "IQ_abs"
    """Which quadrature to plot: 'IQ_abs' (always positive), 'I', or 'Q'. Default is 'IQ_abs'."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for moving-qubit spectroscopy vs flux calibration."""

    targets_name: ClassVar[str] = "qubit_pairs"
