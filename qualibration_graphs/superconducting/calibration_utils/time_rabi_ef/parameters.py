from typing import Optional
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 50
    """Number of averages to perform. Default is 50."""
    min_duration_ns: int = 16
    """Minimum pulse duration in ns (multiple of 4). Default is 16."""
    max_duration_ns: int = 2000
    """Maximum pulse duration in ns (exclusive). Default is 2000."""
    duration_step_ns: int = 4
    """Step size in ns (must be a multiple of 4). Default is 4."""
    ef_x180_operation: str = "EF_x180"
    """Name of the EF pi pulse operation whose duration is swept. Default is 'EF_x180'."""
    operation_amplitude_factor: float = 1.0
    """Amplitude prefactor applied to the EF drive pulse. Default is 1.0."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
