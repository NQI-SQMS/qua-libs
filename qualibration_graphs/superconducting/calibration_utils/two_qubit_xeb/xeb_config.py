"""XEB configuration dataclass and utilities."""

from dataclasses import dataclass, field
from typing import Literal, List, Union, Optional, Dict

import numpy as np
from qiskit.transpiler import CouplingMap
from quam_builder.architecture.superconducting.qubit import FluxTunableTransmon
from quam_builder.architecture.superconducting.qubit_pair import FluxTunableTransmonPair
from quam_config import Quam

from .gateset import QUAGateSet
from .macros import get_parallel_gate_combinations as gate_combinations
from .qua_gate import QUAGate


@dataclass
class XEBConfig:
    """
    Experiments parameters for running XEB

    Args:
        seqs: Number of random sequences to run per depth
        depths: Array of depths to iterate through
        n_shots: Number of averages per sequence
        qubits: List of active qubits for the experiment
        readout_qubits: List of qubits for readout (multiplexed readout)
        baseline_gate_name: Name of baseline X/2 gate (default "sx")
        gate_set_choice: Gate set "sw" or "t", or custom Dict[int, QUAGate]
        two_qb_gate: Two-qubit gate for the experiment
        qubit_pairs: List of qubit pairs (one pair supported)
        readout_pulse_name: Readout pulse name (default "readout")
        reset_method: "active" or "cooldown"
        reset_kwargs: Reset method kwargs
        save_dir: Directory for saved data
        should_save_data: Whether to save data
        generate_new_data: Whether to generate new data
        disjoint_processing: Process qubit states independently
        seed: Random seed


    """

    seqs: int
    depths: Union[np.ndarray, List[int]]
    n_shots: int
    qubits: List[FluxTunableTransmon]
    readout_qubits: Optional[List[FluxTunableTransmon]] = None
    baseline_gate_name: str = "sx"
    gate_set_choice: Union[Literal["sw", "t"], Dict[int, QUAGate]] = "sw"
    two_qb_gate: Optional[QUAGate] = None
    qubit_pairs: Optional[List[FluxTunableTransmonPair]] = field(default_factory=lambda: [])
    readout_pulse_name: str = "readout"
    reset_method: Literal["active", "cooldown"] = "cooldown"
    reset_kwargs: Optional[Dict[str, Union[float, str, int]]] = field(
        default_factory=lambda: {
            "cooldown_time": 20,
            "max_tries": None,
            "pi_pulse": None,
        }
    )
    save_dir: str = ""
    should_save_data: bool = True
    data_folder_name: Optional[str] = None
    generate_new_data: bool = True
    disjoint_processing: bool = False
    seed: int = 1234

    discrimination_method: Literal["threshold", "gaussian"] = "threshold"

    control_readout_mode: int = 2
    target_readout_mode: int = 2

    dim_c: int = 0
    dim_t: int = 0
    dim_k: int = 0  # 'k' for coupler

    # The total number of streams (e.g., 2*2*1=4 or 3*2*2=12)
    total_dim: int = 0

    enable_iq_snapshot: bool = False
    snapshot_sequence_index: int = 0
    snapshot_depth_index: int = 0
    two_qubit_gate_idle_time_ns: int = 0

    def __post_init__(self):
        if isinstance(self.depths, List):
            self.depths = np.array(self.depths)

        self.gate_set = QUAGateSet(self.gate_set_choice, self.baseline_gate_name)
        self.n_qubits = len(self.qubits)
        self.dim = 2**self.n_qubits
        self.available_combinations = None
        self.coupling_map = None

    def as_dict(self):
        """
        Return the XEBConfig object as a dictionary
        """
        config_dict = {
            "seqs": self.seqs,
            "depths": self.depths.tolist() if isinstance(self.depths, np.ndarray) else self.depths,
            "n_shots": self.n_shots,
            "qubits": [qubit.name if isinstance(qubit, FluxTunableTransmon) else qubit for qubit in self.qubits],
            "baseline_gate_name": self.baseline_gate_name,
            "gate_set_choice": self.gate_set_choice,
            "two_qb_gate": self.two_qb_gate.name if self.two_qb_gate else None,
            "qubit_pairs": [
                pair.name if isinstance(pair, FluxTunableTransmonPair) else pair for pair in self.qubit_pairs
            ],
            "coupling_map": list(self.coupling_map.get_edges()) if self.coupling_map else None,
            "available_combinations": self.available_combinations,
        }
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict, machine: Optional[Quam] = None):
        """
        Create an XEBConfig object from a dictionary that contains all relevant parameters.
        This method will usually be used to load a configuration from previously saved data from another run.

        Args:
            config_dict (Dict): Dictionary containing the configuration parameters
            machine (Optional[QuAM]): QuAM object containing the qubits and qubit pairs used in the experiment
        """
        qubits_names = config_dict["qubits"]
        qubits = [machine.qubits[qubit_name] if machine is not None else qubit_name for qubit_name in qubits_names]
        qubit_pairs_names = config_dict["qubit_pairs"]
        qubit_pairs = [
            machine.qubit_pairs[qubit_pair_name] if machine is not None else qubit_pair_name
            for qubit_pair_name in qubit_pairs_names
        ]
        two_qb_gate = QUAGate(config_dict["two_qb_gate"]) if config_dict["two_qb_gate"] else None
        if config_dict["gate_set_choice"] not in ["sw", "t"]:
            raise ValueError("Gate set choice must be either 'sw' or 't'")

        new_class = cls(
            seqs=config_dict["seqs"],
            depths=config_dict["depths"],
            n_shots=config_dict["n_shots"],
            qubits=qubits,
            baseline_gate_name=config_dict["baseline_gate_name"],
            gate_set_choice=config_dict["gate_set_choice"],
            two_qb_gate=two_qb_gate,
            qubit_pairs=qubit_pairs,
            # --- NEW: Load all other parameters from the dictionary ---
            # (Uses .get() to provide defaults if param is missing from old saves)
            readout_qubits=[
                machine.qubits[qn] if machine is not None else qn
                for qn in config_dict.get("readout_qubits", qubits_names)
            ],
            readout_pulse_name=config_dict.get("readout_pulse_name", "readout"),
            reset_method=config_dict.get("reset_method", "cooldown"),
            reset_kwargs=config_dict.get("reset_kwargs", {"cooldown_time": 20, "max_tries": None, "pi_pulse": None}),
            disjoint_processing=config_dict.get("disjoint_processing", False),
            seed=config_dict.get("seed", 1234),
            discrimination_method=config_dict.get("discrimination_method", "threshold"),
            control_readout_mode=config_dict.get("control_readout_mode", 2),
            target_readout_mode=config_dict.get("target_readout_mode", 2),
            enable_iq_snapshot=config_dict.get("enable_iq_snapshot", False),
            snapshot_sequence_index=config_dict.get("snapshot_sequence_index", 0),
            snapshot_depth_index=config_dict.get("snapshot_depth_index", 0),
            two_qubit_gate_idle_time_ns=config_dict.get("two_qubit_gate_idle_time_ns", 0),
        )

        # These are generated by XEB, so we load them separately
        new_class.coupling_map = CouplingMap(config_dict.get("coupling_map", None))
        new_class.available_combinations = config_dict.get("available_combinations", None)
        return new_class
