from pydantic import field_validator
from typing import Literal, Optional

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
    IdleTimeNodeParameters,
)

_AMP_MAX = 2.0 - 2**-16  # QUA hardware limit


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1000
    """Number of averages per time point."""
    mode_name: str = "alice"
    """Which cavity mode to probe: 'alice' or 'bob'."""
    cavity_reset_type: Literal["thermal", "active_sideband"] = "thermal"
    """How to reset the cavity before each displacement.
    'thermal'        — wait thermalization_time_factor × T1 (passive decay).
    'active_sideband'— drive f0g1 π-pulses to actively remove photons; requires a
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
    displacement_scale: float = 1.0
    """Amplitude scale for the displacement pulse.
    After node 30/32 calibration: scale=1 → 1 photon.  Use scale>1 for a
    brighter initial coherent state (stronger signal at early times).
    Must be in (-2, +2) — the QUA hardware limit is ±(2 - 2^-16)."""
    delay_repeats: int = 1
    """Number of times to repeat the wait per point.  Extends the effective
    sweep range: total time = delay_repeats × t_per_rep × 4 ns.
    min/max_wait_time_in_ns define the *per-repeat* range, so the full sweep
    spans [min, delay_repeats × max] ns.  The x-axis always shows total time."""
    use_state_discrimination: bool = True
    """True → measure qubit state. False → measure raw I quadrature."""
    normalize_plot: bool = False
    """When True and use_state_discrimination=False, normalize the plotted I signal
    to [0, 1] so the decay starts at 1 (full vacuum-state signal) and the baseline
    is 0.  Has no effect when use_state_discrimination=True."""

    @field_validator("displacement_scale")
    @classmethod
    def _check_amp_scale(cls, v: float) -> float:
        if abs(v) > _AMP_MAX:
            raise ValueError(
                f"displacement_scale={v} exceeds the QUA hardware limit "
                f"±{_AMP_MAX:.6f} (2 - 2^-16). Use a value in (-2, +2)."
            )
        return v


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
