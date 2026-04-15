from typing import Literal, Optional

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages per (amplitude, frequency) point."""
    mode_name: str = "alice"
    """Which cavity mode to calibrate: 'alice' or 'bob'."""
    cavity_reset_type: Literal["thermal", "active_sideband"] = "thermal"
    """How to reset the cavity between shots.
    'thermal'        — wait thermalization_time_factor × T1 (passive decay).
    'active_sideband'— drive f0g1 π-pulses to actively remove photons; requires a
                       calibrated f0g1_pi operation on the sideband_drive of the
                       corresponding CavityTransmonPair."""
    cavity_active_cooling_fock_n: int = 1
    """Starting Fock level for active sideband cooling (only used when
    cavity_reset_type='active_sideband').  The cooling loop removes photons from
    |fock_n⟩ down to vacuum by driving |n,g⟩→|n-1,f⟩ and waiting for qubit
    relaxation at each step.  Set to 1 for thermal state cooling; set higher if
    you know the cavity contains multiple photons."""
    f0g1_pulse_duration_ns: Optional[int] = None
    """Override the f0g1 sideband pulse duration [ns] during active cooling.
    When None (default), the calibrated f0g1_pi pulse length is used.
    Set to a longer value (e.g. several ms) to ensure the cavity photon
    decoheres fully during each cooling step, at the cost of longer reset time.
    Must be a multiple of 4 ns."""
    right_offset_mhz: float = 2.0
    """Frequency offset above the qubit ge frequency where the sweep starts [MHz].
    A small positive margin above the n=0 peak (which sits at the qubit frequency)."""
    left_span_mhz: float = 6.0
    """Frequency span below the qubit ge frequency to sweep [MHz].
    Should be large enough to cover all visible photon-number peaks: left_span > max_peaks * chi."""
    frequency_step_in_mhz: float = 0.04
    """Frequency step [MHz]."""
    max_peaks: int = 8
    """Maximum number of photon-number peaks to try when fitting.
    The algorithm tries 1, 2, ... peaks and stops when chi2 < chi2_threshold."""
    chi2_threshold: float = 2.0
    """Reduced chi² threshold for accepting a multi-Gaussian fit.
    chi2 = SS_res / ((N - P) * a²), same convention as power_rabi.
    chi2 ≤ 2 means residual RMS ≤ a/√2 (signal clearly above noise). Lower = stricter."""
    min_power_dbm: float = -30.0
    """Minimum displacement drive power [dBm] (outer sweep start)."""
    max_power_dbm: float = 0.0
    """Maximum displacement drive power [dBm] (outer sweep end, sets base amplitude via set_output_power)."""
    num_power_points: int = 21
    """Number of displacement power points (logarithmically spaced in amplitude)."""
    max_amp: float = 0.3
    """Maximum pulse amplitude [V] used to achieve max_power_dbm via set_output_power.
    Must be within (0, 0.5]."""
    displacement_pulse_duration_ns: Optional[int] = None
    """Override displacement pulse duration [ns]. None uses the operation default length."""
    selected_power_dbm: Optional[float] = None
    """If set, an additional 1D spectrum plot is produced at this displacement power [dBm].
    The nearest available power point in the sweep is used."""
    use_state_discrimination: bool = True
    """True -> measure qubit state. False -> measure raw I/Q."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
