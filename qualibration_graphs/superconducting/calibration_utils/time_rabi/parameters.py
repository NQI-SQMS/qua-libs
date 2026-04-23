from typing import Optional, Literal
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 50
    """Number of averages to perform."""

    min_duration_ns: int = 16
    """Minimum qubit pulse duration in ns. Must be a multiple of 4 (one clock cycle)."""
    max_duration_ns: int = 2000
    """Maximum qubit pulse duration in ns (exclusive). Must be a multiple of 4."""
    duration_step_ns: int = 4
    """Step size for the duration sweep in ns. Must be a multiple of 4."""

    operation: str = "x180"
    """Operation to play for the time Rabi sweep. Default is 'x180'."""

    operation_amplitude_factor: float = 1.0
    """Amplitude scale factor applied to the pulse during the sweep.
    1.0 uses the amplitude stored in the QUAM state."""

    drive_power_dbm: Optional[float] = None
    """XY drive output power in dBm.  If set, the qubit XY channel is temporarily
    configured to this power before the sweep and reverted in update_state.
    If None (default), the power from the QUAM state is used unchanged."""

    max_amplitude_opx: float = 0.1
    """Maximum OPX output amplitude (0–0.5 V) used when drive_power_dbm is set."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
