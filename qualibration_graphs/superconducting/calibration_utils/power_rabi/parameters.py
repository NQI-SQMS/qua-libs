from typing import Literal, Optional, Protocol, runtime_checkable

import numpy as np
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class BasePowerRabiParameters(RunnableParameters):
    """Parameters shared by both 04b (GE power Rabi) and 12b (EF power Rabi) nodes."""

    num_shots: int = 50
    """Number of averages to perform. Default is 50."""
    min_amp_factor: float = 0.001
    """Minimum amplitude factor for the operation. Default is 0.001."""
    max_amp_factor: float = 1.99
    """Maximum amplitude factor for the operation. Default is 1.99."""
    amp_factor_step: float = 0.005
    """Step size for the amplitude factor. Default is 0.005."""


class NodeSpecificParameters(BasePowerRabiParameters):
    """04b-specific parameters (GE power Rabi with optional error amplification)."""

    operation: str = "x180"
    """Name of the QUAM operation (pulse) to calibrate. Default is 'x180'.
    Can be any operation defined in qubit.xy.operations, e.g. 'x90', 'selective_x180'."""
    operation_length_in_ns: Optional[int] = None
    """Pulse length in ns to use for this run. If None (default), uses the length
    currently stored in the QUAM state for the selected operation. If set, overrides
    the QUAM length before the QUA program is generated; the override is saved to
    the QUAM state on a successful fit."""
    max_number_pulses_per_sweep: int = 1
    """Maximum number of Rabi pulses per sweep (error amplification). Default is 1."""
    update_x90: bool = True
    """Flag to update the x90 pulse amplitude after calibrating x180. Default is True."""
    use_adaptive: bool = False
    """Enable adaptive calibration. When True:
    - If no oscillation is found, the current qubit frequency is added to the blacklist in temp_calibration.
    - If too many periods (>2) are found, the base pulse amplitude is scaled down so the next run
      shows ~1 period (new_amp = current_amp / num_periods).
    - If too few periods (<0.8) are found, the base pulse amplitude is scaled up similarly.
    Default is False."""


class EfNodeSpecificParameters(BasePowerRabiParameters):
    """12b EF-specific parameters."""

    operation: str = "EF_x180"
    """Name of the QUAM operation (pulse) to calibrate on the EF transition. Default is 'EF_x180'.
    Can be any EF operation defined in qubit.xy.operations, e.g. 'selective_EF_x180'.
    When set to 'EF_x180', the drive frequency is automatically shifted to the EF transition
    before the pulse and back to GE afterwards. For other operations (e.g. selective_EF_x180)
    that are already defined at the correct frequency, no frequency update is performed."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for 04b_power_rabi."""


class EfParameters(
    NodeParameters,
    CommonNodeParameters,
    EfNodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for 12b_power_rabi_ef (EF transition)."""


@runtime_checkable
class HasErrorAmplification(Protocol):
    """Structural typing for objects supporting error amplification controls."""

    max_number_pulses_per_sweep: int
    operation: str


def get_number_of_pulses(node_parameter: BasePowerRabiParameters):
    """Return array of number of pulses for error amplification.

    For EF node (12b) the default behaviour is a single pulse sweep (equivalent to max_number_pulses_per_sweep = 1).
    """
    # If the parameter object lacks error amplification attributes, default to single pulse.
    if not isinstance(node_parameter, HasErrorAmplification):
        return np.array([1], dtype=int)

    if node_parameter.max_number_pulses_per_sweep > 1:
        _op = node_parameter.operation
        if _op in ["x90", "-x90", "y90", "-y90"]:
            # x90-type: use multiples of 4 for error amplification
            N_pulses = np.arange(2, node_parameter.max_number_pulses_per_sweep, 4).astype(int)
        else:
            # x180-type (x180, selective_x180, EF_x180, …): use odd numbers
            N_pulses = np.arange(1, node_parameter.max_number_pulses_per_sweep, 2).astype(int)
    else:
        N_pulses = np.linspace(
            1,
            node_parameter.max_number_pulses_per_sweep,
            node_parameter.max_number_pulses_per_sweep,
        ).astype(int)[::2]
    return N_pulses
