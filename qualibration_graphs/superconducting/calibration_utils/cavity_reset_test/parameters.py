from typing import Literal, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 500
    """Number of averages per reset-duration point."""

    mode_name: str = "alice"
    """Which cavity mode to test: attribute name on the Cavity object (e.g. 'alice' or 'bob')."""

    reset_duration_start_in_ns: int = 100
    """Start of the reset flat-top duration sweep [ns]. Must be a multiple of 4."""

    reset_duration_end_in_ns: int = 5_000_000
    """End of the reset flat-top duration sweep [ns]. Must be a multiple of 4."""

    reset_duration_step_in_ns: int = 10_000
    """Step size of the reset flat-top duration sweep [ns]. Must be a multiple of 4."""

    use_state_discrimination: bool = True
    """True -> measure qubit state (recommended). False -> measure raw I/Q."""
    use_displaced_threshold: bool = False
    """When True and use_state_discrimination is True, use pair.ge_iq_threshold_displaced
    instead of the vacuum readout threshold (calibrated by node 26j)."""

    use_confusion_matrix_correction: bool = False
    """Apply ge confusion matrix correction to averaged state probabilities."""

    cavity_pre_reset_type: Literal["thermal", "active_sideband", "active_sideband_v2"] = "thermal"
    """How to reset the cavity before each shot (before the Fock prep).
    'thermal'           - wait thermalization_time_factor × T1.
    'active_sideband'   - cascade sideband pi-pulses to actively remove photons.
    'active_sideband_v2'- long SB pulse then N×(GEF reset → f0g1 π → GEF reset) per Fock level."""

    cavity_active_cooling_fock_n: int = 1
    """Starting Fock level for active sideband pre-reset (only used when
    cavity_pre_reset_type='active_sideband' or 'active_sideband_v2')."""

    cavity_active_cooling_n_repeats: int = 3
    """Number of (GEF-reset → f0g1-π → GEF-reset) cycles per Fock level.
    Only used when cavity_pre_reset_type='active_sideband_v2'."""

    sideband_pulse_duration_ns: Optional[int] = None
    """Override the sideband pulse flat-top duration [ns] for the Fock-prep step.
    When None, uses the calibrated pi_flat_top_length_ns from pair.transitions."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
