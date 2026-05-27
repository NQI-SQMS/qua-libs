from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    mode_name: str = "alice"
    """Which cavity mode to probe: attribute name on the Cavity object (e.g. 'alice' or 'bob')."""

    fock1_alpha1: float = 1.0
    """First displacement amplitude [photons] — D(α₁) step of the D-SNAP-D protocol.
    |α₁| = 1.0 maximises the initial P(n=1) ≈ 37%.
    amplitude_scale = fock1_alpha1 / displacement_k
    (displacement_k is read from the CavityTransmonPair QuAM state)."""

    fock1_alpha2: float = -0.59
    """Second (correction) displacement amplitude [photons] — D(α₂) step.
    Calibrated to maximise final P(n=1) after the SNAP₀ gate.
    amplitude_scale = fock1_alpha2 / displacement_k."""

    use_state_discrimination: bool = True
    """True -> measure qubit state (recommended). False -> measure raw I/Q."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    IdleTimeNodeParameters,
    QubitsExperimentNodeParameters,
):
    pass
