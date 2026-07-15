from typing import Literal, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1000
    """Number of averages per time point."""

    mode_name: str = "alice"
    """Which cavity mode to probe: attribute name on the Cavity object (e.g. 'alice' or 'bob')."""

    cavity_reset_type: Literal["thermal", "active_sideband"] = "thermal"
    """How to reset the cavity before each displacement.
    'thermal'         — wait thermalization_time_factor × T1 (passive decay).
    'active_sideband' — drive f0g1 π-pulses to actively remove photons; requires a
                        calibrated f0g1_pi operation on the sideband_drive of the
                        corresponding CavityTransmonPair."""

    cavity_active_cooling_fock_n: int = 1
    """Starting Fock level for active sideband cooling (only used when
    cavity_reset_type='active_sideband').  Set to 1 for thermal state cooling."""

    sideband_pulse_duration_ns: Optional[int] = None
    """Override the f0g1 sideband pulse duration [ns] during active cooling.
    When None (default), the calibrated f0g1_pi pulse length is used.
    Must be a multiple of 4 ns."""

    ramsey_detuning_hz: float = 1000.0
    """Artificial detuning applied via frame rotation [Hz]. Creates Ramsey oscillation fringes
    needed to extract T2ramsey. Should be >> 1/T2 to resolve several oscillation periods."""

    displacement_alpha: float = 1.0
    """Desired coherent-state amplitude |α|.
    amplitude_scale = displacement_alpha / displacement_alpha_max
    (displacement_alpha_max is read from the CavityTransmonPair QuAM state,
    calibrated by node 22/32). With displacement_alpha=1 and a calibrated
    alpha_max, this corresponds to ~1 mean photon."""

    qubit_probe_operation: str = "x180"
    """Qubit operation to play after the reverse displacement.
    Use 'x180' (non-selective) for a standard pi-pulse, or 'selective_x180'
    for a narrow-bandwidth pulse sensitive to cavity photon number."""

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
