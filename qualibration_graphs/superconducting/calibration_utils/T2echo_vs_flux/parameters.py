"""Parameters for T2-echo-versus-flux coherence-time characterization."""

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for the T2-echo-versus-flux sweep."""

    num_shots: int = 100
    """Number of averages to perform per (flux, idle-time) point. Default is 100."""
    flux_span: float = 0.02
    """Full span of the flux-bias sweep in volts, centered on the qubit flux
    point. The sweep runs over ``[-flux_span/2, +flux_span/2]``. Default is 0.02 V."""
    flux_num: int = 11
    """Number of flux-bias points to sample. Default is 11."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameters for the T2-echo-versus-flux characterization node."""

    pass
