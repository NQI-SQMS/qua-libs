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
    """How to prepare the cavity Fock |1> state before the wait.
    'sideband'          - ge pi -> ef pi -> f0g1 sideband pi (sideband ladder).
                          Readout: inverse sideband (f0g1 pi -> ef pi -> measure ge).
                          Requires calibrated f0g1 sideband (nodes 26, 26b).
    'snap_displacement' - D(alpha1) -> SNAP₀ (selective_x180 × 2) -> D(alpha2).
                          Readout: selective_x180 at n=1 dressed qubit frequency (PNRS).
                          Requires calibrated displacement and selective_x180 pulses."""

    fock1_alpha1: float = 1.0
    """First displacement amplitude [photons] for 'snap_displacement' prep.
    Ignored when fock1_prep_method='sideband'."""

    fock1_alpha2: float = -0.59
    """Correction displacement amplitude [photons] for 'snap_displacement' prep.
    Negative value displaces in the opposite direction.
    Ignored when fock1_prep_method='sideband'."""

    cavity_reset_type: Literal["thermal", "active_sideband"] = "thermal"
    """How to reset the cavity before each shot.
    'thermal'         - wait thermalization_time_factor x T1 (passive decay).
    'active_sideband' - cascade sideband pi-pulses to actively remove photons;
                        requires calibrated sideband operations on the
                        CavityTransmonPair sideband_drive (nodes 26 / 26b)."""

    cavity_active_cooling_fock_n: int = 1
    """Starting Fock level for active sideband cooling (only used when
    cavity_reset_type='active_sideband').  Set to 1 for Fock |1> experiments."""

    sideband_pulse_duration_ns: Optional[int] = None
    """Override the sideband pulse flat-top duration [ns] during active cooling.
    When None (default), the calibrated pi_flat_top_length_ns from pair.transitions is used.
    Set to a longer value (e.g. several ms) to ensure the cavity photon decoheres
    fully during each cooling step.  Must be a multiple of 4 ns."""

    use_state_discrimination: bool = True
    """True -> measure qubit state (recommended). False -> measure raw I/Q."""
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
