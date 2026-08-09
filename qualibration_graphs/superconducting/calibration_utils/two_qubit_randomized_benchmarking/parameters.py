from typing import ClassVar, List, Literal, Optional, Tuple

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitPairExperimentNodeParameters,
    QubitsExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    use_strict_timing: bool = False
    """Use strict timing in the QUA program. Default is False."""
    num_shots: int = 100
    """Number of random RB sequences. Default is 100."""
    num_random_sequences: int = 10
    """Number of random sequences. Default is 10."""
    max_circuit_depth: int = 20
    """Maximum circuit depth (number of Clifford gates). Default is 100."""
    num_intervals: int = 10
    """Number of intervals between 0 and max_circuit_depth. Default is 20."""
    interval: Literal["linear", "logarithmic"] = "logarithmic"
    """Interval type for circuit depths. Default is 'logarithmic'."""
    seed: Optional[int] = None
    """Seed for the random number generator. Default is None."""
    interleaved_CNOT: bool = False
    """Interleave CNOT. Default is False."""
    exclude_CNOT_from_clifford: bool = False
    """Exclude CNOT from the Clifford gates. Default is False."""

    cr_type: Literal["direct", "direct+cancel", "direct+echo", "direct+cancel+echo"] = "direct+echo"
    wf_type: Literal["square", "cosine", "gauss", "flattop"] = "flattop"
    forced_reverse: bool = False


class QubitPairExperimentNodeParametersCustom(QubitPairExperimentNodeParameters):
    use_state_discrimination: bool = True
    targets_name: ClassVar[str] = "qubit_pairs"


class Parameters(
    QubitPairExperimentNodeParametersCustom,
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    # QubitPairExperimentNodeParameters,
):
    pass


class ParametersOptimize(Parameters):
    # override
    num_shots: int = 100

    parameter1: str = "cross_resonance.target_qubit_RF_frequency"
    values1: str = "np.linspace(-5e6, 5e6, 10)"
    update_method1: Literal["absolute", "relative", "offset"] = "offset"
    parameter2: Optional[str] = "cross_resonance.drive_amplitude_scaling"
    values2: str = "np.linspace(0.9, 1.1, 5)"
    update_method2: Literal["absolute", "relative", "offset"] = "absolute"
