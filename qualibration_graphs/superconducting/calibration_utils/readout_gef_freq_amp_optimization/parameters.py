from typing import Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 500
    """Number of single-shot samples per (frequency, amplitude) grid point. Default is 500."""
    frequency_span_in_mhz: float = 2.0
    """Span of the frequency sweep around the current GEF operating frequency, in MHz. Default is 2 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Frequency step size, in MHz. Default is 0.1 MHz."""
    min_amp_factor: float = 0.5
    """Minimum readout amplitude as a scale factor of the current amplitude. Default is 0.5."""
    max_amp_factor: float = 1.5
    """Maximum readout amplitude as a scale factor of the current amplitude. Default is 1.5."""
    num_amps: int = 11
    """Number of amplitude steps (linearly spaced). Default is 11."""
    max_readout_amplitude_v: Optional[float] = None
    """Hard cap on the absolute readout amplitude (V). Amplitude steps exceeding this are skipped. Default is None."""
    operation: str = "readout"
    """Readout operation name. Default is 'readout'."""
    ef_pi_pulse: str = "EF_x180"
    """Name of the EF π pulse used to prepare the |f⟩ state. Default is 'EF_x180'."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
