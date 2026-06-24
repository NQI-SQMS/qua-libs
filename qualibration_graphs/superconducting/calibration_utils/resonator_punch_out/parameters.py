from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""

    frequency_span_in_mhz: float = 2
    """Span of frequencies to sweep in MHz. Default is 2 MHz."""

    frequency_step_in_mhz: float = 0.05
    """Step size for frequency sweep in MHz. Default is 0.05 MHz."""

    min_power_dbm: int = -50
    """Lower readout power (used as reference). Default is -50 dBm."""

    max_power_dbm: int = -25
    """Higher readout power (used to test Kerr shift). Default is -25 dBm."""

    num_power_points: int = 2
    """Number of power points.

    With exactly 2 points (default), uses the fast shift-based analysis: compares
    the Lorentzian dip position at low vs. high power, with adaptive retry support.

    With more than 2 points, runs a dense diagnostic sweep instead: locates the
    punch-out bifurcation directly from the resonance-vs-power curve (2D heatmap
    plot) and picks a safe operating power 2 steps before the bifurcation."""

    max_amp: float = 0.1
    """Maximum readout amplitude for the experiment. Default is 0.1."""

    frequency_shift_threshold_in_hz: float = 2e5
    """
    Minimum absolute frequency shift (in Hz) between low and high power
    required to declare a Kerr-induced shift.

    Typical values: 1e5 - 5e5 Hz, depending on resonator linewidth.
    """

    use_adaptive_span: bool = False
    """
    Enable adaptive span adjustment.

    When enabled, if no frequency shift is detected and the resonator frequency
    is at the bare frequency (from quam state) for both power points, the power
    span (max_power_dbm and min_power_dbm) is decreased by 10 dB for the next iteration.

    This helps find the optimal power range when the initial power is too high.
    """

    sweep_left_offset_mhz: float = 4.0
    """How far (MHz) to extend the frequency sweep to the LEFT of the bare resonator frequency.
    The sweep runs from  (f_bare - sweep_left_offset_mhz)  to  (f_bare - sweep_left_offset_mhz + frequency_span_in_mhz).
    A non-zero value ensures the dispersive-shifted low-power resonance, which sits
    below the bare frequency, is captured within the swept window."""

    chi2_threshold: float = 3.0
    """Residual chi-squared threshold for the Lorentzian dip fit at each power point.
    chi2 = SS_res / ((N - 4) * amp²). Both the low-power and high-power traces must
    satisfy chi2 ≤ threshold for the result to be considered valid.
    chi2 > threshold → the trace is too noisy to reliably fit a dip.
    Default 3.0. Lower (e.g. 1.5) to reject marginal fits."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
