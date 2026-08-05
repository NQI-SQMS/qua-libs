"""Parameter definitions for Cross-Entropy Benchmarking (XEB) experiments.

This module defines the parameters used for XEB calibration of two-qubit gates,
including circuit depths, sequences, readout modes, and simulation options.
"""


from typing import ClassVar, List, Literal, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for XEB experiments."""

    n_sequences: int = 20
    """Number of random sequences to run per depth."""
    n_shots: int = 200
    """Number of averages per sequence. Note: The limit is around 64."""
    depth_min: int = 10
    """Minimum circuit depth."""
    depth_max: int = 200
    """Maximum circuit depth."""
    depth_step: int = 5
    """Step size for depth sweep."""
    baseline_gate: str = "x90"
    """Name of the baseline gate implementing a pi/2 rotation around the x-axis."""
    gate_set_choice_sw_or_t: Literal["sw", "t"] = "sw"
    """Choice of gate set for XEB (choose 'sw' or 't')."""
    discrimination_method: Literal["threshold", "gaussian"] = "threshold"
    """Method for state discrimination."""

    apply_two_qubit_gate: bool = True
    """Whether to apply the two-qubit gate in the circuit."""
    two_qubit_gate_idle_time_ns: int = 80
    """Idle time around two-qubit gate in nanoseconds."""
    reset_type: Literal["active", "thermal"] = "thermal"
    """Reset method for qubits."""
    estimate_2q_unitary: bool = True
    """Whether to run 2Q unitary estimation from measured probability."""

    hardware_simulate: bool = False
    """Run hardware pulse simulation."""
    simulation_duration_ns: int = 50000
    """Duration for hardware simulation."""

    control_readout_mode: Literal[2, 3] = 3
    """Readout mode for control qubit (2 or 3 states)."""
    target_readout_mode: Literal[2, 3] = 3
    """Readout mode for target qubit (2 or 3 states)."""
    analysis_only_path: Optional[str] = None
    """Path to load data for re-analysis only (skip experiment execution)."""
    cz_macro_name: Literal["cz_flattop", "cz_unipolar", "cz_bipolar", "cz_flattop_erf", "cz_SNZ"] = "cz_unipolar"
    """Name of the CZ macro to use from the qubit pair."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for XEB experiments."""

    targets_name: ClassVar[str] = "qubit_pairs"
    qubit_pairs: List[str] = []
    """List of qubit pair names to benchmark. XEB supports one pair at a time."""
