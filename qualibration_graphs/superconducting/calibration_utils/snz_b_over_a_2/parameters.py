"""Parameter definitions for SNZ t_phi_eff scan.

This module defines the parameters and waveform factory for the Di Carlo
Sudden Net-Zero (SNZ) experiment, scanning the effective idle time
(t_phi_eff) by combining integer t_phi with the B/A ratio.
"""


from typing import ClassVar, Literal, Optional, Tuple

import numpy as np
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


def snz_factory(A, ratio, length, t_phi, padding=10):
    """Construct an SNZ (Sudden Net-Zero) bipolar flux waveform.

    The waveform structure is::

        [padding | +A flat | +B | idle | -B | -A flat | padding]

    where B = A * ratio. The single B/-B samples sit between the flat
    sections and the idle period and correspond to the last/first sampling
    points of the positive/negative lobes in the Di Carlo SNZ protocol.

    Args:
        A: Amplitude of the flat sections (volts).
        ratio: B/A ratio for the transition samples.
        length: Total flat duration in samples (split equally between
            positive and negative halves, each ``length // 2`` samples).
        t_phi: Idle time in samples (ns) between the two lobes.
        padding: Zero-padding on each side of the pulse (samples).

    Returns:
        1-D numpy array with the complete SNZ waveform.
    """
    flat = np.ones(length // 2)
    B = np.array([A * ratio])
    idle = np.zeros(t_phi)
    pulse = np.concatenate([np.zeros(padding), A * flat, B, idle, -B, -A * flat, np.zeros(padding)])
    return pulse


def decompose_t_phi_eff(t_phi_eff: float) -> Tuple[int, float]:
    """Decompose an effective idle time into (t_phi, b_over_a).

    The mapping is::

        t_phi_eff = t_phi + 2 * (1 - B/A)

    so B/A = 1 gives t_phi_eff = t_phi and B/A = 0 gives t_phi_eff = t_phi + 2.

    For any ``t_phi_eff >= 0`` this function returns the unique pair where
    ``t_phi = floor(t_phi_eff / 2) * 2`` (even integers only) and B/A
    spans the full [0, 1] range over each 2 ns window.

    Args:
        t_phi_eff: Desired effective idle time (ns, float >= 0).

    Returns:
        (t_phi, b_over_a) tuple.
    """
    t_phi = int(np.floor(t_phi_eff / 2)) * 2
    b_over_a = 1.0 - (t_phi_eff - t_phi) / 2.0
    return t_phi, b_over_a


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for the SNZ t_phi_eff scan."""

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
    padding: int = 4
    """Zero-padding on each side of the baked waveform (samples)."""
    operation: Literal["cz_unipolar", "cz_SNZ"] = "cz_unipolar"
    """CZ macro used to derive the nominal amplitude A and flat duration."""
    use_state_discrimination: bool = True
    """Whether to use g/e/f state discrimination for readout."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Main parameters class for the SNZ t_phi_eff scan node."""

    targets_name: ClassVar[str] = "qubit_pairs"
