from typing import Literal, Optional
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    fock_level: int = 0
    """Which sideband transition (0=f0g1, 1=f1g2, …). Internally prepares Fock |fock_level+1⟩.
    Ignored when preparation_type='displacement'."""
    mode_name: str = "alice"
    """Which cavity mode to prepare: 'alice' or 'bob'."""
    num_shots: int = 2000
    """Number of single-shot measurements per blob (|g⟩ and |e⟩)."""
    operation: str = "readout"
    """Readout operation name on the resonator channel."""
    cavity_reset_type: Literal["thermal", "active_sideband"] = "thermal"
    """How to reset the cavity between shots.
    'thermal'         - passive decay (wait thermalization time).
    'active_sideband' - cascade sideband π-pulses to actively remove photons."""
    cavity_active_cooling_fock_n: int = 1
    """Starting Fock level for active sideband cooling (only used when
    cavity_reset_type='active_sideband')."""
    sideband_pulse_duration_ns: Optional[int] = None
    """Override the sideband pulse flat-top duration [ns] during active cooling.
    None → use the calibrated pi_flat_top_length_ns from pair.transitions."""
    cavity_thermalization_time_ns: Optional[int] = None
    """Explicit cavity thermal wait [ns]. None → delegate to cav_mode.reset."""
    preparation_type: Literal["sideband", "displacement"] = "sideband"
    """How to prepare the cavity state before measuring the IQ blobs.
    'sideband'    - Fock |fock_level+1⟩ via the standard sideband ladder (default).
    'displacement' - coherent state |α⟩ via a single displacement pulse."""
    displacement_alpha: float = 1.0
    """Desired coherent-state amplitude α for preparation_type='displacement'.
    amplitude_scale = displacement_alpha / displacement_alpha_max from QuAM state."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
