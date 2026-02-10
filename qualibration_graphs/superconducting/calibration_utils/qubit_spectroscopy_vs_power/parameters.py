from typing import Optional
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    # ------------------------------------------------------------------
    # User-facing power sweep (dBm)
    # ------------------------------------------------------------------
    min_power_dbm: float = -60
    max_power_dbm: float = -10
    num_power_points: int = 10

    # ------------------------------------------------------------------
    # OPX amplitude constraints
    # ------------------------------------------------------------------
    max_amplitude_opx: float = 0.1
    min_amplitude_opx: float = 0.01

    # ------------------------------------------------------------------
    # Spectroscopy parameters
    # ------------------------------------------------------------------
    frequency_span_in_mhz: float = 50
    frequency_step_in_mhz: float = 0.25

    # ------------------------------------------------------------------
    # Averaging
    # ------------------------------------------------------------------
    num_shots: int = 100

    # ------------------------------------------------------------------
    # XY operation
    # ------------------------------------------------------------------
    operation: str = "saturation"
    operation_len_in_ns: Optional[int] = None

    # ------------------------------------------------------------------
    # Qubit peak analysis parameters
    # ------------------------------------------------------------------
    linewidth_threshold_hz: float = 2e6
    """
    Linewidth (FWHM) threshold above which the spectroscopy
    is considered power-broadened.
    """

    power_buffer_db: float = 3.0
    """
    Safety margin below the critical power (in dB).
    """

    min_peak_fraction: float = 0.1
    """
    Minimum acceptable peak height as a fraction of the difference.
    """



class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
