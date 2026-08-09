"""Parameter definitions for SNZ conditional phase measurement.

This module defines the parameters for combining the SNZ t_phi_eff
scan with a conditional phase Ramsey measurement on the target qubit.
"""


from typing import ClassVar, Literal, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for the SNZ conditional phase scan."""

    num_shots: int = 200
    """Number of averages."""
    t_phi_eff_min: float = 0.0
    """Effective idle time sweep start (ns)."""
    t_phi_eff_max: float = 10.0
    """Effective idle time sweep end (ns)."""
    t_phi_eff_step: float = 0.1
    """Effective idle time sweep step (ns)."""
    amp_range: float = 0.03
    """Amplitude sweep half-range (relative, centered on 1.0)."""
    amp_step: float = 0.001
    """Amplitude sweep step (relative)."""
    num_frame_rotations: int = 10
    """Number of frame rotation points for phase tomography."""
    padding: int = 4
    """Zero-padding on each side of the baked waveform (samples)."""
    operation: Literal["cz_unipolar", "cz_SNZ"] = "cz_unipolar"
    """CZ macro used to derive the nominal amplitude A and flat duration."""
    use_state_discrimination: bool = True
    """Whether to use g/e/f state discrimination for readout."""
    leak_percentile: float = 20.0
    """Leakage percentile threshold for optimal point search (0-100).
    Points with leakage below this percentile are considered low-leakage
    candidates, among which the one closest to pi conditional phase is chosen."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Main parameters class for the SNZ conditional phase node."""

    targets_name: ClassVar[str] = "qubit_pairs"
