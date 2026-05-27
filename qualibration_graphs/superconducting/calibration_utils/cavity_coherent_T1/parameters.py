from typing import Literal, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
    IdleTimeNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1000
    """Number of averages per time point."""
    mode_name: str = "alice"
    """Which cavity mode to probe: 'alice' or 'bob'."""
    cavity_reset_type: Literal["thermal", "active_sideband"] = "thermal"
    """How to reset the cavity before each displacement.
    'thermal'         — wait thermalization_time_factor × T1 (passive decay).
    'active_sideband' — drive f0g1 π-pulses to actively remove photons; requires a
                        calibrated f0g1_pi operation on the sideband_drive of the
                        corresponding CavityTransmonPair."""
    cavity_active_cooling_fock_n: int = 1
    """Starting Fock level for active sideband cooling (only used when
    cavity_reset_type='active_sideband').  Set to 1 for thermal state cooling."""
    f0g1_pulse_duration_ns: Optional[int] = None
    """Override the f0g1 sideband pulse duration [ns] during active cooling.
    When None (default), the calibrated f0g1_pi pulse length is used.
    Set to a longer value (e.g. several ms) to ensure the cavity photon
    decoheres fully during each cooling step, at the cost of longer reset time.
    Must be a multiple of 4 ns."""
    displacement_alpha: float = 1.0
    """Desired coherent-state amplitude α for the displacement pulse.
    The actual QUA amplitude_scale is computed at runtime as
        amplitude_scale = displacement_alpha / displacement_alpha_max
    where displacement_alpha_max is read from the CavityTransmonPair in the
    QuAM state (set by node 30/32 calibration).
    displacement_alpha = 1 → |α|² = 1 mean photon after calibration.
    The runtime check ensures amplitude_scale stays within the QUA hardware
    limit ±(2 − 2^−16)."""
    delay_repeats: int = 1
    """Number of times to repeat the wait per point.  Extends the effective
    sweep range: total time = delay_repeats × t_per_rep × 4 ns.
    min/max_wait_time_in_ns define the *per-repeat* range, so the full sweep
    spans [min, delay_repeats × max] ns.  The x-axis always shows total time."""
    subtract_baseline: bool = False
    """If True, run a second sub-sequence WITHOUT the selective qubit π-pulse at every
    time point and subtract its averaged IQ from the signal IQ before any
    state-discrimination is applied.  This removes the time-dependent IQ offset caused
    by cross-Kerr coupling between the cavity mode and the readout resonator (the
    cross-Kerr shift changes as the cavity photon number decays during the T1 sweep).
    If False, only the standard single-sequence protocol is executed."""
    use_state_discrimination: bool = True
    """True → measure qubit state. False → measure raw I quadrature."""
    normalize_plot: bool = False
    """When True and use_state_discrimination=False, normalize the plotted I signal
    to [0, 1] so the decay starts at 1 (full vacuum-state signal) and the baseline
    is 0.  Has no effect when use_state_discrimination=True."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    # Override IdleTimeNodeParameters defaults to match cavity T1 timescales
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 5_000_000
    wait_time_num_points: int = 51
    log_or_linear_sweep: Literal["log", "linear"] = "log"
