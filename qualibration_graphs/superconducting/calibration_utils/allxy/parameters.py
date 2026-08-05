from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """
    Parameters for configuring an AllXY calibration experiment.

    Attributes:
        num_shots (int): Number of averages performed for each of the 21 AllXY sequences. Default is 200.
        error_threshold (float): Maximum allowed mean absolute deviation, in normalized population units,
            between the measured and ideal AllXY outcomes for a qubit to be considered successful. Default is 0.1.
    """

    num_shots: int = 200
    error_threshold: float = 0.1


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
