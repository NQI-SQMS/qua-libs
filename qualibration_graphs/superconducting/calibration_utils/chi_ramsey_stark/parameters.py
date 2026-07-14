from typing import Literal, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    IdleTimeNodeParameters,
    QubitsExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    mode_name: str = "alice"
    """Which cavity mode to probe ('alice' or 'bob')."""

    fock1_prep_method: Literal["sideband", "snap_displacement"] = "sideband"
    """How to prepare the cavity Fock |1> state before the Ramsey sequence.
    'sideband'          - ge pi -> ef pi -> f0g1 pi prepares |g,1>.
                          Requires calibrated f0g1 sideband (nodes 26, 26b).
    'snap_displacement' - D(alpha1) -> SNAP0 (selective_x180 x 2) -> D(alpha2)
                          prepares ~|1> without a calibrated sideband.
                          Requires displacement calibration and selective_x180."""

    fock1_alpha1: float = 1.0
    """First displacement amplitude [photons] for 'snap_displacement' prep.
    Ignored when fock1_prep_method='sideband'."""

    fock1_alpha2: float = -0.59
    """Correction displacement amplitude [photons] for 'snap_displacement' prep.
    Negative value displaces in the opposite direction.
    Ignored when fock1_prep_method='sideband'."""

    ramsey_pi_pulse_op: str = "x180"
    """Name of the calibrated pi pulse to use for both Ramsey arms.
    The pulse is played at amplitude_scale=0.5 to produce a pi/2 rotation.
    Use 'x180' for the standard broadband pi pulse, or 'selective_x180' for
    a number-selective pulse that drives the qubit only at the n=1 dressed
    frequency (better isolation when chi is small relative to the linewidth)."""

    artificial_detuning_hz: int = 200_000
    """Artificial detuning applied to the qubit XY drive during Ramsey [Hz].
    Shifts the oscillation so fringes are visible even without a photon.
    The measured oscillation frequency will be artificial_detuning_hz + chi.
    Recommended: 100-500 kHz."""

    cavity_reset_type: Literal["thermal", "active_sideband"] = "thermal"
    """How to reset the cavity between shots.
    'thermal'         - wait thermalization_time_factor x T1 for passive decay.
    'active_sideband' - drive f0g1 pi-pulses to actively remove photons."""

    cavity_active_cooling_fock_n: int = 1
    """Starting Fock level for active sideband cooling (only used when
    cavity_reset_type='active_sideband')."""

    sideband_pulse_duration_ns: Optional[int] = None
    """Override the f0g1 sideband pulse duration [ns] during active cooling.
    When None, the calibrated f0g1_pi pulse length is used."""

    use_state_discrimination: bool = True
    """True -> measure qubit state using IQ discrimination threshold.
    False -> measure raw I/Q signal."""

    use_confusion_matrix_correction: bool = False
    """Apply ge confusion matrix correction to averaged state probabilities."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    IdleTimeNodeParameters,
    QubitsExperimentNodeParameters,
):
    pass
