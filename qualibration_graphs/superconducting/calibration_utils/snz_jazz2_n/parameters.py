"""Parameters module for the JAZZ2-N SNZ amplitude / t_phi_eff scan.

This node combines the JAZZ2-N protocol (arXiv:2402.18926v3, Fig. 13(b))
with the Sudden Net-Zero (SNZ) pulse shape (Negirneac et al., Phys. Rev.
Lett. 126, 220502 (2021); see also calibration_utils.snz_b_over_a).

Each "Z" inside the JAZZ2-N pulse train is replaced by a baked SNZ
waveform on the control qubit's flux line, parameterised by the effective
idle time ``t_phi_eff`` (which decomposes into integer ``t_phi`` and the
B/A transition-sample ratio).  The user picks a range of repetition counts
[N_min, N_max] (paper convention N = 2k; step 2) and the node sweeps the
3-D (amplitude_scale, t_phi_eff, N) volume measuring P_|00>. The map used
for the optimum search is the average of P_|00> over the N axis. Setting
``N_min == N_max`` reduces to a single-N scan (no averaging).

Because P_|00> is maximised when both the conditional phase is pi AND
leakage out of the computational subspace is suppressed, the (amp, t_phi_eff)
that maximises P_|00> simultaneously calibrates the SNZ angle and minimises
leakage. The node is exploratory (no state update); use it to read off
the optimum and apply it manually.
"""


from typing import ClassVar, Literal

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for the JAZZ2-N SNZ amplitude / t_phi_eff scan."""

    num_shots: int = 100
    """Number of shots to average over. Default is 100."""
    N_min: int = 0
    """Minimum repetition count of the JAZZ2-N sweep (paper convention N = 2k; auto-coerced to nearest even >= 0).
    Setting N_min == N_max disables averaging and recovers the single-N behaviour."""
    N_max: int = 8
    """Maximum repetition count of the JAZZ2-N sweep (paper convention N = 2k; auto-coerced to nearest even).
    The QUA inner loop steps by 2; i.e. all valid N values between N_min and N_max are visited.
    Multiplier of theta_CZ at N is m = N + 1; total X_pi echo count at N is 2N + 1."""
    amp_range: float = 0.030
    """Half-width of the amplitude-scale sweep around the stored CZ amplitude (center = 1.0). Default is 0.030."""
    amp_step: float = 0.001
    """Step of the amplitude-scale sweep. Default is 0.001."""
    t_phi_eff_min: float = 0.0
    """Effective idle time sweep start (ns)."""
    t_phi_eff_max: float = 5.0
    """Effective idle time sweep end (ns)."""
    t_phi_eff_step: float = 0.1
    """Effective idle time sweep step (ns)."""
    padding: int = 10
    """Zero-padding on each side of the baked SNZ waveform (samples)."""
    operation: Literal["cz_SNZ", "cz_unipolar"] = "cz_SNZ"
    """CZ macro used to derive the nominal SNZ amplitude A and flat duration. Default is 'cz_SNZ'."""
    use_state_discrimination: bool = True
    """JAZZ2-N reads the joint P_|00> of both qubits, which requires state discrimination. Setting this to False raises."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for the JAZZ2-N SNZ amplitude / t_phi_eff scan node."""

    targets_name: ClassVar[str] = "qubit_pairs"
