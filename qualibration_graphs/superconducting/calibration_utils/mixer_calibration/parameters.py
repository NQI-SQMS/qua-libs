from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    calibrate_resonator: bool = True
    """Whether to calibrate the resonator. Default is True."""
    calibrate_drive: bool = True
    """Whether to calibrate the xy drive. Default is True."""
    calibrate_cavity_drive: bool = True
    """Whether to calibrate the cavity mode drive channels. Default is True."""
    calibrate_sideband_drive: bool = True
    """Whether to calibrate the f0g1 sideband drive channels in cavity_transmon_pairs.
    Skipped automatically if sideband_drive.RF_frequency or the LO frequency is not yet set
    (i.e., before running node 21 f0g1 spectroscopy)."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
