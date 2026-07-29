from typing import Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 400
    """Number of averages per amplitude point."""
    mode_name: str = "alice"
    """Which cavity mode to probe: 'alice' or 'bob'."""
    amp_min: float = -2.0
    """Minimum displacement amplitude in photon units (α, can be negative).
    The QUA amplitude_scale is computed as amp_min / displacement_alpha_max,
    so the physical sweep always covers [amp_min, amp_max] photons regardless
    of the current calibration."""
    amp_max: float = 2.0
    """Maximum displacement amplitude in photon units (α).
    See amp_min for details."""
    amp_points: int = 101
    """Number of amplitude points (linearly spaced from amp_min to amp_max)."""
    parity_time_ns: Optional[int] = None
    """Fixed Ramsey wait time for parity measurement [ns].
    If None, computed from chi_hz: t = 1 / (2 * abs(chi_hz)).
    Must be a multiple of 4 ns."""
    chi_hz: Optional[float] = None
    """Full per-photon qubit frequency shift [Hz] (negative for typical transmon-cavity
    systems), used to compute parity_time_ns when parity_time_ns is None.
    If also None, read from cavity_transmon_pairs in the machine state."""
    use_state_discrimination: bool = True
    """True → measure qubit state (0/1). False → measure raw I/Q."""
    use_displaced_threshold: bool = False
    """When True and use_state_discrimination is True, use pair.ge_iq_threshold_displaced
    instead of the vacuum readout threshold (calibrated by node 26j)."""
    use_confusion_matrix_correction: bool = False
    """Apply ge confusion matrix correction to averaged state probabilities."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
