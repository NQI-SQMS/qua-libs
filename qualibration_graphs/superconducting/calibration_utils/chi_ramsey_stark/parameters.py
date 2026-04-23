from typing import List, Literal, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    mode_name: str = "alice"
    """Which cavity mode to probe ('alice' or 'bob')."""
    num_shots: int = 1000
    """Number of averages per (amplitude, delay) point."""
    min_delay_ns: int = 16
    """Minimum Ramsey delay [ns]. Must be a multiple of 4."""
    max_delay_ns: int = 3000
    """Maximum Ramsey delay [ns]."""
    delay_step_ns: int = 16
    """Step between Ramsey delays [ns]. Must be a multiple of 4."""
    ring_up_ns: int = 2000
    """Time to wait for the cavity to reach photon steady-state after turning on
    the CW drive [ns].  Approximately 3/κ for the storage cavity.  Increase for
    high-Q (long-lifetime) cavities."""
    cavity_amplitudes: List[float] = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
    """List of cavity CW drive amplitude scales to sweep (dimensionless, [0, 2)).
    The first value should always be 0.0 (reference Ramsey with no cavity
    photons).  Steady-state photon number n̄ ∝ A²."""
    artificial_detuning_hz: int = 200_000
    """Artificial detuning applied to the qubit XY drive during Ramsey [Hz].
    Shifts the oscillation frequency so that the Ramsey fringes are visible even
    at A=0.  Recommended: 100–500 kHz, well within the Ramsey decay envelope."""
    cavity_reset_type: Literal["thermal", "active_sideband"] = "thermal"
    """How to reset the cavity between shots.
    'thermal'         — wait thermalization_time_factor × T1 for passive decay.
    'active_sideband' — drive f0g1 π-pulses to actively remove photons."""
    cavity_active_cooling_fock_n: int = 1
    """Starting Fock level for active sideband cooling (only used when
    cavity_reset_type='active_sideband')."""
    f0g1_pulse_duration_ns: Optional[int] = None
    """Override the f0g1 sideband pulse duration [ns] during active cooling.
    When None, the calibrated f0g1_pi pulse length is used."""
    use_state_discrimination: bool = True
    """True → measure qubit state using IQ discrimination threshold.
    False → measure raw I/Q signal."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
