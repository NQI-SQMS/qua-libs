from typing import Optional
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
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

    use_adaptive_span: bool = False
    """
    Enable adaptive calibration adjustments for subsequent runs.
    When enabled, the node will automatically:
    - No peak found: Increase frequency span (up to 800 MHz) AND power (+10 dBm)
    - Weak peak at high power: Increase power only (+10 dBm), keep frequency span
    - Over-saturation: Decrease power (-10 dBm)
      (over-saturation = all power points show excessive linewidth or high baseline)
    """

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
    Linewidth (FWHM) threshold in Hz above which the spectroscopy is considered
    power-broadened.  The highest drive power whose linewidth stays at or below
    this threshold is chosen as the operating point.  This power (before the
    safety buffer) is also used as the reference for scaling the saturation and
    x180 pulse amplitudes via the T_spec / T_pi_target ratio.
    """

    power_buffer_db: float = 3.0
    """
    Safety margin below the threshold power (in dB).
    """

    # ------------------------------------------------------------------
    # Rabi sweep targets (used to derive saturation / x180 amplitude)
    # ------------------------------------------------------------------
    rabi_target_periods: int = 3
    """
    Number of complete Rabi oscillation periods desired within the time Rabi sweep.
    Together with rabi_sweep_max_duration_ns this sets the target π-pulse duration
    (T_π = rabi_sweep_max_duration_ns / (2 × rabi_target_periods)) and from there
    the recommended saturation / x180 output power via the power-broadening fit.
    """

    rabi_sweep_max_duration_ns: float = 300.0
    """
    Upper bound (ns) of the planned time Rabi sweep.
    Used with rabi_target_periods to compute the target π-pulse duration.
    """

    peak_persistence_lookahead: int = 0
    """
    Number of consecutive higher-power levels that must also show a peak at a
    similar frequency for the current peak to be considered real.
    A peak that is absent from all of the next ``peak_persistence_lookahead``
    power levels is discarded as an isolated noise artefact.
    Set to 0 to disable persistence filtering entirely.
    """

    peak_persistence_freq_tolerance_hz: float = 5e6
    """
    Frequency tolerance (Hz) used when matching peaks across adjacent power
    levels during persistence filtering.  Two peaks at different power values
    are considered to be the same qubit transition if their detuning positions
    differ by less than this amount.
    """

    # ------------------------------------------------------------------
    # Signal selection
    # ------------------------------------------------------------------
    signal_source: str = "I_rot"
    """
    Which signal to use for peak detection and plotting.
      "I_rot"  – IQ-rotated I component (default). Maximises SNR by projecting
                 the 2D IQ trajectory onto the axis of largest qubit-induced
                 displacement. Sign ambiguity is resolved automatically.
      "IQ_abs" – Magnitude of the IQ vector. No rotation required; useful when
                 the IQ trajectory is poorly conditioned or the user wants a
                 rotation-free view. The qubit feature may appear as a peak or
                 dip depending on the device; the sign is resolved automatically.
      "phase"  – Unwrapped, slope-subtracted phase of the IQ vector (radians).
                 Useful when the qubit-induced response is more visible as a
                 phase shift than an amplitude change.
    """



class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
