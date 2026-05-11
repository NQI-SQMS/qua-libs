from typing import Literal, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1000
    """Number of averages per amplitude point."""
    mode_name: str = "alice"
    """Which cavity mode to calibrate: 'alice' or 'bob'."""
    amp_min: float = 0.0
    """Minimum displacement amplitude_scale. Use 0.0 to sweep only positive amplitudes."""
    amp_max: float = 2.0
    """Maximum displacement amplitude_scale."""
    amp_points: int = 51
    """Number of amplitude points (linearly spaced from amp_min to amp_max)."""
    qubit_pulse: str = "selective_x180"
    """Qubit π-pulse operation to use for the vacuum-state probe.
    'selective_x180' — spectrally narrow, flips qubit only when cavity is in |0⟩.
    'x180'           — broadband π-pulse (use when selective_x180 is not yet calibrated)."""
    cavity_reset_type: Literal["thermal", "active_sideband"] = "thermal"
    """How to reset the cavity before each displacement.
    'thermal'         — passive decay (wait thermalization_time_factor × T1).
    'active_sideband' — drive f0g1 π-pulses to actively remove photons; requires a
                        calibrated f0g1_pi on the sideband_drive of the CavityTransmonPair."""
    cavity_active_cooling_fock_n: int = 1
    """Starting Fock level for active sideband cooling (used only when
    cavity_reset_type='active_sideband'). Set to 1 for thermal state cooling."""
    f0g1_pulse_duration_ns: Optional[int] = None
    """Override the f0g1 sideband pulse duration [ns] during active cooling.
    When None (default), the calibrated f0g1_pi pulse length is used.
    Set to a longer value (e.g. several ms) to ensure the cavity photon
    decoheres fully during each cooling step, at the cost of longer reset time.
    Must be a multiple of 4 ns."""
    active_reset: bool = True
    """If True, apply D(-a) immediately after measurement to return the cavity
    toward vacuum, replacing passive thermalization for the next shot."""
    subtract_baseline: bool = True
    """If True, run a second sub-sequence WITHOUT the qubit π-pulse at every amplitude
    point and subtract its averaged IQ from the signal IQ before any state-discrimination
    threshold is applied.  This removes the amplitude-dependent IQ offset caused by
    cross-Kerr coupling between the cavity mode and the readout resonator.
    If False, only the standard single-sequence protocol (with π-pulse) is executed,
    matching the original behaviour with no cross-Kerr correction."""
    use_state_discrimination: bool = True
    """True → measure qubit state (0/1). False → measure raw I quadrature."""
    normalize_plot: bool = False
    """When True and use_state_discrimination=False, normalize the plotted I signal
    to [0, 1] so the peak at a=0 is 1 and the baseline is 0.
    Has no effect when use_state_discrimination=True."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
