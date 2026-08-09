from typing import Literal, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 500
    """Number of averages per wait time point."""

    mode_name: str = "alice"
    """Which cavity mode to probe: attribute name on the Cavity object (e.g. 'alice' or 'bob')."""

    fock1_prep_method: Literal["sideband", "snap_displacement"] = "sideband"
    """How to create (and undo) the cavity (|0>+|1>)/sqrt(2) superposition.
    Both methods measure the same quantity: cavity Fock superposition T2ramsey.

    'sideband' (default):
        Opening: ge pi/2 -> ef pi -> f0g1 pi  (creates cavity superposition)
        Ramsey:  f0g1 pi -> wait tau -> f0g1 pi  (on sideband drive)
        Closing: ef pi -> ge pi/2  (back-converts to qubit observable)
        Requires calibrated f0g1 sideband (nodes 26, 26b) and EF_x180.

    'snap_displacement' (scaffold -- exact sequence TBD):
        Opening: SNAP+displacement sequence to create (|0>+|1>)/sqrt(2)
        Ramsey:  wait tau with frame rotation on cavity drive channel
        Closing: SNAP+displacement back-conversion to qubit observable
        Does not require a calibrated sideband; exact pulses to be filled in
        once the SNAP-based superposition sequence is characterised."""

    fock1_alpha1: float = 1.0
    """First displacement amplitude [photons] for 'snap_displacement' prep.
    Ignored when fock1_prep_method='sideband'."""

    fock1_alpha2: float = -0.59
    """Correction displacement amplitude [photons] for 'snap_displacement' prep.
    Negative value displaces in the opposite direction.
    Ignored when fock1_prep_method='sideband'."""

    ramsey_detuning_hz: float = 400.0
    """Artificial detuning applied as a frame rotation on the sideband drive [Hz].
    Creates Ramsey oscillation fringes in the cavity Fock superposition.
    Should be >> 1/T2 to resolve several oscillation periods."""

    cavity_reset_type: Literal["thermal", "active_sideband", "active_sideband_v2"] = "thermal"
    """How to reset the cavity before each shot.
    'thermal'           - wait thermalization_time_factor x T1 (passive decay).
    'active_sideband'   - cascade sideband pi-pulses to actively remove photons;
                          requires calibrated sideband operations on the
                          CavityTransmonPair sideband_drive (nodes 26 / 26b).
    'active_sideband_v2'- long SB pulse then N×(GEF reset → f0g1 π → GEF reset) per Fock level."""

    cavity_active_cooling_fock_n: int = 1
    """Starting Fock level for active sideband cooling (only used when
    cavity_reset_type='active_sideband' or 'active_sideband_v2').  Set to 1 for Fock |1> experiments."""

    cavity_active_cooling_n_repeats: int = 3
    """Number of (GEF-reset → f0g1-π → GEF-reset) cycles per Fock level.
    Only used when cavity_reset_type='active_sideband_v2'."""

    sideband_pulse_duration_ns: Optional[int] = None
    """Override the sideband pulse flat-top duration [ns] during active cooling.
    When None (default), the calibrated pi_flat_top_length_ns from pair.transitions is used.
    Set to a longer value (e.g. several ms) to ensure the cavity photon decoheres
    fully during each cooling step.  Must be a multiple of 4 ns."""

    use_state_discrimination: bool = True
    """True -> measure qubit state (recommended). False -> measure raw I/Q."""
    use_displaced_threshold: bool = False
    """When True and use_state_discrimination is True, use pair.ge_iq_threshold_displaced
    instead of the vacuum readout threshold (calibrated by node 26j)."""
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
