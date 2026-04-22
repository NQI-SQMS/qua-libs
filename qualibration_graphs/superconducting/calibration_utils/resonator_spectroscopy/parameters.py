from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters

from typing import Optional


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 30.0
    """Span of frequencies to sweep in MHz. Default is 30 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""
    readout_power_dbm: Optional[float] = None
    """Readout power in dBm for the spectroscopy sweep.
    If None, the current QUAM state power is used unchanged.
    The QUAM state is reverted to its original value after the node finishes."""
    max_amp: float = 0.1
    """Maximum readout pulse amplitude (OPX units, 0–0.5).
    Only used when readout_power_dbm is set. Default is 0.1."""
    save_readout_amplitude: bool = True
    """When True (default) and readout_power_dbm is set, permanently save the calibrated
    readout power/amplitude to the QUAM state after a successful run.
    Set to False to keep the QUAM state readout power unchanged (e.g. when using this node
    only for frequency calibration and the power is set just to improve the SNR)."""



class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
