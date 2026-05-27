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
    amplitude_scale = fock1_alpha1 / displacement_k
    (displacement_k read from CavityTransmonPair QuAM state)."""

    fock1_alpha2: float = -0.59
    """Second (correction) displacement amplitude [photons] — D(α₂) step.
    Calibrated to maximise P(n=1) after the SNAP₀ gate.
    amplitude_scale = fock1_alpha2 / displacement_k."""

    ramsey_detuning_hz: float = 1000.0
    """Artificial detuning applied via frame rotation on the qubit [Hz].
    Creates Ramsey oscillation fringes needed to extract T2ramsey.
    Should be >> 1/T2 to resolve several oscillation periods."""

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
