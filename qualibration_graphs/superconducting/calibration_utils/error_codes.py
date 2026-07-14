"""
Error codes and corrective actions for adaptive calibration nodes.

These codes provide standardized, machine-readable diagnostics for calibration
failures and the corrective actions taken.
"""
from enum import IntEnum


# =========================
# Qubit Spectroscopy vs Power Error Codes
# =========================

class QubitSpectroscopyErrorCode(IntEnum):
    """Error codes for qubit spectroscopy vs power calibration."""

    SUCCESS = 0
    """Calibration succeeded - valid peak found at acceptable power."""

    NO_PEAK_FOUND = 1
    """No peak detected at any power level within frequency span."""

    OVER_SATURATED = 3
    """Linewidth exceeds threshold or high baseline across all powers (overpowered)."""

    OVER_SATURATED_SUCCESS = 4
    """Valid peak found but showing over-saturation - success with warning."""


class QubitSpectroscopyCorrectiveAction(IntEnum):
    """Corrective actions for qubit spectroscopy vs power calibration."""

    NONE = 0
    """No action needed - calibration successful."""

    EXPAND_FREQUENCY_SPAN = 10
    """Expanding frequency search range (magnitude: expansion factor in %)."""

    INCREASE_POWER = 20
    """Increasing drive power sweep range (magnitude: power shift in dBm)."""

    DECREASE_POWER = 50
    """Decreasing power due to over-saturation (magnitude: power shift in dBm)."""

    RESET_ADAPTIVE_PARAMS = 99
    """Resetting all adaptive parameters after successful calibration."""


# =========================
# Resonator Spectroscopy Error Codes
# =========================

class ResonatorSpectroscopyErrorCode(IntEnum):
    """Error codes for resonator spectroscopy calibration."""

    SUCCESS = 0
    """Calibration succeeded - valid resonator dip found."""

    NO_DIP_FOUND = 1
    """No resonator dip detected (NaN position or insufficient contrast above noise)."""


class ResonatorSpectroscopyCorrectiveAction(IntEnum):
    """Corrective actions for resonator spectroscopy calibration."""

    NONE = 0
    """No action needed - calibration successful."""

    BLACKLIST_FREQUENCY = 10
    """Adding current resonator frequency to blacklist (magnitude: RF frequency in Hz)."""


# =========================
# Resonator Punch-Out Error Codes
# =========================

class ResonatorPunchOutErrorCode(IntEnum):
    """Error codes for resonator punch-out calibration."""

    SUCCESS = 0
    """Calibration succeeded - frequency shift detected above threshold."""

    NO_SHIFT_DETECTED = 1
    """No frequency shift detected between low and high power."""

    SHIFT_BELOW_THRESHOLD = 2
    """Frequency shift detected but below threshold."""

    AT_BARE_FREQUENCY = 3
    """Resonator at bare frequency with no shift (power too high)."""

    INVALID_DATA = 4
    """Invalid data - NaN values or frequency out of range."""

    WRONG_SHIFT_DIRECTION = 5
    """Resonator shifted to higher frequency at high power (unexpected direction).
    Physical expectation: freq_low - freq_high > 0 (Kerr shift lowers resonance at high power).
    No corrective action is taken for this error."""


class ResonatorPunchOutCorrectiveAction(IntEnum):
    """Corrective actions for resonator punch-out calibration."""

    NONE = 0
    """No action needed - calibration successful."""

    DECREASE_POWER_SPAN = 10
    """Decreasing power span (power too high) (magnitude: power shift in dBm)."""

    INCREASE_POWER_SPAN = 20
    """Increasing power span (power too low) (magnitude: power shift in dBm)."""

    RESET_ADAPTIVE_PARAMS = 99
    """Resetting all adaptive parameters after successful calibration."""


# =========================
# Power Rabi Error Codes
# =========================

class PowerRabiErrorCode(IntEnum):
    """Error codes for power Rabi calibration."""

    SUCCESS = 0
    """Calibration succeeded - valid Rabi oscillation with ~1 period found."""

    NO_OSCILLATION = 1
    """No Rabi oscillation detected (amplitude too small - noise or wrong qubit frequency)."""

    TOO_MANY_PERIODS = 2
    """Too many Rabi oscillation periods in the amplitude sweep (base amplitude too high)."""

    TOO_FEW_PERIODS = 3
    """Less than one Rabi oscillation period in the amplitude sweep (base amplitude too low)."""


class PowerRabiCorrectiveAction(IntEnum):
    """Corrective actions for power Rabi calibration."""

    NONE = 0
    """No action needed - calibration successful with acceptable period count."""

    BLACKLIST_FREQUENCY = 10
    """Adding current qubit frequency to blacklist (magnitude: RF frequency in Hz)."""

    REDUCE_AMPLITUDE = 20
    """Reducing base pulse amplitude to achieve ~1 period (magnitude: new amplitude in V)."""

    INCREASE_AMPLITUDE = 30
    """Increasing base pulse amplitude to achieve ~1 period (magnitude: new amplitude in V)."""

    INCREASE_OCTAVE_GAIN = 35
    """Increasing Octave RF upconversion gain when amplitude is at hardware limit
    (magnitude: new gain in dB)."""

    INCREASE_DURATION = 40
    """Increasing pulse duration to achieve ~1 period when amplitude and Octave gain are maxed
    (magnitude: new duration in ns)."""

    RESET_ADAPTIVE_PARAMS = 99
    """Resetting all adaptive parameters after successful calibration."""


# =========================
# Displacement Vacuum Calibration Error Codes
# =========================

class DisplacementVacuumErrorCode(IntEnum):
    """Error codes for the displacement vacuum-population calibration (node 22)."""

    SUCCESS = 0
    """Calibration succeeded - Gaussian fit converged and the swept amplitude
    range covered at least target_n_sigma * sigma_fit."""

    FIT_FAILED = 1
    """The Gaussian fit did not converge or gave a non-physical sigma/amplitude."""

    INSUFFICIENT_RANGE_COVERAGE = 2
    """Fit converged but the swept amplitude_scale range did not reach
    target_n_sigma * sigma_fit -- the Gaussian tail was not adequately sampled."""

    SPAN_TOO_LARGE = 3
    """Fit failed because the swept amplitude range is much larger than sigma:
    fewer than min_points_on_peak data points land on the Gaussian peak, so
    curve_fit cannot converge. The sweep must be shrunk before retrying."""


class DisplacementVacuumCorrectiveAction(IntEnum):
    """Corrective actions for the displacement vacuum-population calibration."""

    NONE = 0
    """No action needed - full coverage achieved, alpha_max calibrated normally."""

    INCREASE_AMPLITUDE_HEADROOM = 10
    """Raising the displacement operation's base amplitude (Volts) via the
    gain-locked set_locked_output_power helper so a given amplitude_scale
    reaches more photons, allowing target_n_sigma to be captured within the
    existing AMP_SCALE_LIMIT-bounded amplitude_scale range. Octave/FEM gain
    is never touched."""

    INCREASE_DURATION = 20
    """Increasing the displacement pulse duration once amplitude headroom
    (AMP_SCALE_LIMIT / DAC ceiling) is exhausted."""

    SHRINK_AMPLITUDE_SPAN = 30
    """Reducing amp_max (and amp_min symmetrically) because the current sweep
    range is so large that fewer than min_points_on_peak points landed on the
    Gaussian peak, preventing the fit from converging."""

    RESET_ADAPTIVE_PARAMS = 99
    """Resetting all adaptive parameters after successful calibration."""


# =========================
# Helper Functions
# =========================

def format_error_message(error_code: IntEnum, action_code: IntEnum,
                        magnitude: float = None) -> str:
    """
    Format a human-readable error and action message.

    Parameters
    ----------
    error_code : IntEnum
        The error code (from QubitSpectroscopyErrorCode or ResonatorPunchOutErrorCode)
    action_code : IntEnum
        The corrective action code
    magnitude : float, optional
        The magnitude of the correction (interpretation depends on action_code)

    Returns
    -------
    str
        Formatted message
    """
    msg = f"Error: {error_code.name} (code {error_code.value})\n"
    msg += f"Action: {action_code.name} (code {action_code.value})"

    if magnitude is not None:
        msg += f" - Magnitude: {magnitude}"

    return msg
