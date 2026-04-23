from typing import Literal, Optional

from pydantic import field_validator

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters

_AMP_MAX = 2.0 - 2**-16  # QUA hardware limit


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages."""
    mode_name: str = "alice"
    """Which cavity mode to probe: 'alice' or 'bob'."""
    displacement_scale: float = 1.0
    """Amplitude scale for the cavity displacement pulse.
    Creates a coherent state |α⟩ with mean photon number n̄ = displacement_scale²
    (after node 32 calibration: scale=1 → 1 photon).
    Must be in (-2, +2) — the QUA hardware limit is ±(2 - 2^-16)."""
    displacement_alpha: float = 1.0
    """Multiplier applied on top of displacement_scale.  The actual QUA amplitude
    used is displacement_scale × displacement_alpha.  Typically an integer (1, 2, 3).
    The cavity thermalization wait is also scaled by this factor so that higher photon
    numbers fully decay between shots."""
    active_reset: bool = True
    """If True, apply a reverse displacement D(-α) after measurement to
    return the cavity to vacuum immediately, replacing passive thermalization.
    Only valid when using displacement (not saturation) to populate the cavity."""
    right_offset_mhz: float = 2.0
    """Frequency offset above the qubit ge frequency where the sweep starts [MHz].
    A small positive margin above the n=0 peak (which sits at the qubit frequency)."""
    left_span_mhz: float = 6.0
    """Frequency span below the qubit ge frequency to sweep [MHz].
    Should be large enough to cover all visible photon-number peaks: left_span > max_peaks * chi."""
    frequency_step_in_mhz: float = 0.04
    """Frequency step [MHz]."""
    max_peaks: int = 8
    """Maximum number of photon-number peaks to try when fitting."""
    chi2_threshold: float = 0.005
    """Reduced chi² threshold for stopping peak search.  The auto-fitter adds
    peaks until chi² drops below this value.  Lower values force more peaks to
    be tried before stopping; raise if over-fitting noise, lower if weak peaks
    are missed.  The 3% amplitude floor in the fitter already prevents purely
    spurious peaks from being accepted."""
    qubit_pulse: str = "selective_x180"
    """Qubit pulse operation to use for spectroscopy.
    Typical choices: 'selective_x180' (narrow-bandwidth, resolves photon-number peaks)
    or 'x180' (standard pi-pulse, faster but lower frequency resolution)."""
    cavity_reset_type: Literal["thermal", "active_sideband"] = "thermal"
    """How to reset the cavity between shots.
    'thermal'        — wait thermalization_time_factor × T1 (passive decay).
    'active_sideband'— drive f0g1 π-pulses to actively remove photons; requires a
                       calibrated f0g1_pi operation on the sideband_drive of the
                       corresponding CavityTransmonPair."""
    cavity_active_cooling_fock_n: int = 1
    """Starting Fock level for active sideband cooling (only used when
    cavity_reset_type='active_sideband').  Set to 1 for thermal state cooling;
    set higher if you know the cavity contains multiple photons."""
    f0g1_pulse_duration_ns: Optional[int] = None
    """Override the f0g1 sideband pulse duration [ns] during active cooling.
    When None (default), the calibrated f0g1_pi pulse length is used.
    Must be a multiple of 4 ns."""
    use_state_discrimination: bool = True
    """True -> measure qubit state. False -> measure raw I/Q."""
    normalize_plot: bool = False
    """When True and use_state_discrimination=False, normalize the plotted I signal
    to [0, 1] using min-max scaling.  Has no effect when use_state_discrimination=True."""

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
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
