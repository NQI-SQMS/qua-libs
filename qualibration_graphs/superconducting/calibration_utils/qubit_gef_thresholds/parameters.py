from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    mode_name: str = "alice"
    """CavityTransmonPair mode name — used to locate the pair whose extras dict
    will be updated with the calibrated thresholds."""

    num_shots: int = 10000
    """Single-shot measurements collected per qubit state (g, e, f).
    Larger counts give more accurate threshold estimates; 10 000 is typical."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
