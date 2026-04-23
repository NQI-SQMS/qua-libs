from typing import Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 400
    """Number of averages per amplitude point."""
    mode_name: str = "alice"
    """Which cavity mode to probe: 'alice' or 'bob'."""
    amp_min: float = -2.0
    """Minimum displacement amplitude_scale (can be negative)."""
    amp_max: float = 2.0
    """Maximum displacement amplitude_scale."""
    amp_points: int = 101
    """Number of amplitude points (linearly spaced from amp_min to amp_max)."""
    parity_time_ns: Optional[int] = None
    """Fixed Ramsey wait time for parity measurement [ns].
    If None, computed from chi_hz: t = 1 / (4 * chi_hz) = 1 / (2 * peak_spacing).
    Must be a multiple of 4 ns."""
    chi_hz: Optional[float] = None
    """Dispersive coupling χ/(2π) [Hz] (Hamiltonian constant, H/ħ = χ a†a σz),
    used to compute parity_time_ns when parity_time_ns is None.
    If also None, read from cavity_transmon_pairs in the machine state."""
    use_state_discrimination: bool = True
    """True → measure qubit state (0/1). False → measure raw I/Q."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
