from typing import Literal, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    mode_name: str = "alice"
    """Which cavity mode to probe ('alice' or 'bob')."""

    num_shots: int = 1000
    """Number of averages per delay point."""

    min_delay_ns: int = 16
    """Minimum Ramsey wait time [ns]. Must be a multiple of 4."""

    max_delay_ns: int = 4000
    """Maximum Ramsey wait time [ns].
    Should cover at least two full oscillation periods: max_delay_ns > 2/χ_eff.
    For χ/(2π) â‰ˆ 600 kHz this is ~3 Âµs; for smaller χ increase accordingly."""

    delay_step_ns: int = 16
    """Step size between wait times [ns]. Must be a multiple of 4."""

    displacement_scale: float = 0.5
    """Cavity displacement amplitude_scale used during the measurement.
    Should deposit ~0.5â€“2 photons on average.  If displacement_k is calibrated,
    nÌ„ = displacement_k × displacement_scaleÂ².
    Keep below 0.5 (OPX amplitude limit)."""

    cavity_reset_type: Literal["thermal", "active_sideband"] = "thermal"
    """How to reset the cavity between shots.
    'thermal'         â€” passive decay (thermalization_time × T1).
    'active_sideband' â€” f0g1 π-pulse active cooling."""

    cavity_active_cooling_fock_n: int = 1
    """Starting Fock level for active sideband cooling
    (used only when cavity_reset_type='active_sideband')."""

    sideband_pulse_duration_ns: Optional[int] = None
    """Override the f0g1 sideband pulse duration [ns] during active cooling.
    None → use the calibrated f0g1_pi pulse length."""

    use_state_discrimination: bool = True
    """True → binary state discrimination via readout threshold.
    False → record raw I/Q."""
    use_confusion_matrix_correction: bool = False
    """Apply ge confusion matrix correction to averaged state probabilities."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
