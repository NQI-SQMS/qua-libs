"""Parameters for resonator spectroscopy versus power (IQ circles) calibration.

Self-contained copy of ``resonator_spectroscopy_vs_amplitude/parameters.py`` with one
extra knob (``num_iq_circles_to_plot``) for the per-power I/Q-circle overlay plot.
The original module is intentionally left untouched.
"""

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for resonator spectroscopy versus power (IQ circles)."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 15
    """Span of frequencies to sweep in MHz. Default is 15 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""
    max_power_dbm: int = -25
    """Maximum power level in dBm. Default is -25 dBm."""
    min_power_dbm: int = -50
    """Minimum power level in dBm. Default is -50 dBm."""
    num_power_points: int = 100
    """Number of points of the readout power axis. Default is 100."""
    max_amp: float = 0.1
    """Maximum readout amplitude for the experiment. Default is 0.1."""
    power_below_punchout_db: float = 3.0
    """The optimal readout power is set this many dB BELOW the punch-out edge (the highest
    power still on the dressed resonance). This is the only operating-point knob: lower it to
    push the chosen power closer to punch-out (more signal), raise it for a safer margin.
    Mirrors ``power_below_saturation_db`` in the qubit-spectroscopy-vs-power node. Default 3 dB.
    If raising it pushes the chosen power below the swept power range, that qubit is reported
    failed (not written) rather than having its readout silently disabled."""
    # NOTE: the former fitting knobs (derivative_crossing_threshold_in_hz_per_dbm,
    # derivative_smoothing_window_num_points, moving_average_filter_window_num_points,
    # buffer_from_crossing_threshold_in_dbm) were removed: the analysis now self-tunes the
    # optimal-power / resonance fit from the data (see analysis.fit_raw_data), so there is
    # nothing here for the user to set.
    num_iq_circles_to_plot: int = 12
    """Number of readout-power I/Q circles to overlay on a single axes per qubit.
    Powers are evenly sub-sampled from the full power axis. Default is 12."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameters for resonator spectroscopy versus power (IQ circles) calibration."""
