from typing import Literal, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 20
    """Number of averages to perform. Default is 20."""
    operation: Literal["x180", "x90"] = "x180"
    """Type of operation to play. Default is "x180"."""
    frequency_span_in_mhz: float = 10
    """Frequency span of the detuning sweep in MHz. Default is 10."""
    frequency_step_in_mhz: float = 0.1
    """Frequency step of the detuning sweep in MHz. Default is 0.1."""
    max_number_pulses_per_sweep: int = 20
    """Maximum number of drive pulses. Default is 20."""
    alpha_setpoint: Optional[float] = 0.5
    """Optional setpoint for the DRAG coefficient alpha. Default is 0.5."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
