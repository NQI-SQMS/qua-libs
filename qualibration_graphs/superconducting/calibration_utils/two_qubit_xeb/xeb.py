"""XEB (Cross-Entropy Benchmarking) experiment implementation."""

import json
import copy
import warnings
from typing import Union, List, Optional, Dict, Tuple, Literal
import numpy as np
from qm.qua import *
from .macros import (
    qua_declaration,
    reset_qubit,
    binary,
    exponential_decay,
    fit_exponential_decay,
    get_parallel_gate_combinations as gate_combinations,
    align_transmon_pair,
    generate_circuits,
    generate_circuits_parameterized,
    compute_log_fidelity,
    evaluate_log_fidelity,
    update_record,
    update_data_frame,
    calc_ideal_probability_numpy,
    simulate_noisy_circuit_numpy,
    generate_gate_indices,
    cross_entropy,
)
import matplotlib.pyplot as plt
from qiskit_aer import AerJob
import pandas as pd
from .xeb_config import XEBConfig
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.providers import BackendV2
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import Statevector
from qualang_tools.results import DataHandler

from quam_config import Quam
from quam_builder.architecture.superconducting.qubit import FluxTunableTransmon
from qm import SimulationConfig, QuantumMachinesManager, generate_qua_script
from qm.jobs.running_qm_job import RunningQmJob
from qm.jobs.simulated_job import SimulatedJob
import seaborn as sns
from warnings import warn

from qualang_tools.units import unit
from qualang_tools.loops import from_array

from tqdm import tqdm
import time
from sklearn.mixture import GaussianMixture

u = unit(coerce_to_integer=True)


class XEB:
    def __init__(self, xeb_config: XEBConfig, machine: Quam, reset_type: str, cloud: bool = True):
        """
        Initialize the XEB experiment
        Args:
            xeb_config: XEBConfig object containing the parameters of the experiment
            machine: Machine object containing the Quantum Machine configuration
        """
        self.xeb_config = xeb_config
        self.machine = machine
        self.qubit_dict: Dict = {i: qubit for i, qubit in enumerate(self.qubits)}
        self.cloud = cloud
        self.reset_type = reset_type

        if len(self.qubit_pairs) == 0:
            warn("No qubit pairs provided. The experiment will run with single qubit gates only.")

        try:
            self.qubit_drive_channels = [qubit.xy for qubit in self.qubits]
            self.readout_channels = [qubit.resonator for qubit in self.qubits]
        except AttributeError:
            raise AttributeError(
                "Qubit objects must have 'xy' and 'resonator' attributes, "
                "Contact CS Team if your QuAM structure is different."
            )

        # Create CouplingMap from QuAM qubit pairs
        qubit_dict = {qubit: i for i, qubit in enumerate(self.qubits)}
        coupling_map = CouplingMap()
        for qubit in range(len(self.qubits)):
            coupling_map.add_physical_qubit(qubit)
        for qubit_pair in self.qubit_pairs:
            if qubit_pair.qubit_control not in self.qubits or qubit_pair.qubit_target not in self.qubits:
                raise ValueError("Qubit pairs must be formed by qubits present in the qubits list")
            coupling_map.add_edge(qubit_dict[qubit_pair.qubit_control], qubit_dict[qubit_pair.qubit_target])
        self._coupling_map = coupling_map
        self._available_combinations = gate_combinations(self.coupling_map)
        self.xeb_config.available_combinations = self.available_combinations
        self.xeb_config.coupling_map = self.coupling_map
        self.data_handler = DataHandler(name="XEB", root_data_folder=xeb_config.save_dir)

        # 1. Get individual dimensions from config
        dim_c = self.xeb_config.control_readout_mode
        dim_t = self.xeb_config.target_readout_mode
        # Coupler readout is disabled, its dimension is 1
        dim_k = 1

        # 2. Calculate total dimension
        total_dim = dim_c * dim_t * dim_k

        self.xeb_config.dim_c = dim_c
        self.xeb_config.dim_t = dim_t
        self.xeb_config.dim_k = dim_k
        self.xeb_config.total_dim = total_dim

        self.xeb_config.dim = total_dim

    @property
    def qubit_pairs(self):
        """
        Returns the qubit pairs for the XEB experiment
        """
        return self.xeb_config.qubit_pairs

    @property
    def qubits(self):
        """
        Returns the qubits for the XEB experiment
        """
        return self.xeb_config.qubits

    @property
    def readout_qubits(self):
        """
        Returns the readout qubits for the XEB experiment
        """
        return self.xeb_config.readout_qubits

    @property
    def available_combinations(self):
        """
        Returns the available combinations of qubit pairs for the XEB experiment
        """
        return self._available_combinations

    @property
    def coupling_map(self) -> CouplingMap:
        """
        Returns the coupling map for the XEB experiment
        """
        return self._coupling_map

    def _assign_amplitude_matrix(self, gate_idx, amp_matrix, amp_stream=None):
        """
        Assign the amplitude matrix of a gate based on the gate index

        Args:
            gate_idx (QUA int): Index of the gate
            amp_matrix (List): Amplitude matrix of the gate
            amp_stream (QUA stream): Stream to save the amplitude matrix
        """
        with switch_(gate_idx):
            for i in range(len(self.xeb_config.gate_set)):
                with case_(i):
                    for j in range(4):
                        assign(amp_matrix[j], self.xeb_config.gate_set[i].amp_matrix[j])
        if amp_stream is not None:  # Save the amplitude matrix to a stream
            for j in range(4):
                save(
                    amp_matrix[j],
                    amp_stream,
                )

    def _play_random_sq_gate(self, qubit: FluxTunableTransmon, gate_idx, amp_matrix: Optional[List] = None):
        """
        Play a random single qubit gate on a given qubit element.

        This macro plays a random single qubit gate on a given qubit element, by modulating
        the amplitude matrix of a baseline calibrated X/2 (SX) pulse if the gate set
        is set up to run through amplitude matrix modulation.
        Otherwise, it plays the gate through a switch case over the gate index.

        Args:
            qubit (Transmon): Qubit element on which to play the gate.
            gate_idx (QUA int): Index of the gate to play.
            amp_matrix (List): Amplitude matrix of the gate.
        """
        if self.xeb_config.gate_set.run_through_amp_matrix_modulation and amp_matrix is not None:
            # Play all gates through real-time amplitude matrix modulation
            # print(amp_matrix)
            # qubit.xy.play(self.xeb_config.baseline_gate_name, amplitude_scale=amp(*amp_matrix))
            play(self.xeb_config.baseline_gate_name * amp(*amp_matrix), qubit.xy.name)
            # qubit.xy.play('x180') # NOTE: for debugging purposes
            # qubit.xy.play('x90') # NOTE: for debugging purposes
            # pass # NOTE: for debugging purposes
        else:
            # Play all gates through switch case over the gate index
            with switch_(gate_idx, unsafe=True):
                for i in range(len(self.xeb_config.gate_set)):
                    with case_(i):
                        self.xeb_config.gate_set[i].gate_macro(qubit)

    def _xeb_prog(self, simulate: bool = False, stab_amp_in: float = 0.0):
        """
        Generate the QUA program for the XEB experiment
        Args:
            simulate: Indicate if output should be simulated or not
        Returns: QUA program for the XEB experiment
        """
        # --- 1. GET CONFIG & ELEMENT INFO ---
        n_qubits = self.xeb_config.n_qubits
        total_dim = self.xeb_config.total_dim
        random_gates = len(self.xeb_config.gate_set)

        # Get the element objects
        qubit_pair = self.qubit_pairs[0]
        qubit_c = qubit_pair.qubit_control
        qubit_t = qubit_pair.qubit_target
        qubit_k = self.machine.active_qubits[2]

        rst_readout = "readout"
        rst_pi_01 = "x180"
        rst_pi_12 = "EF_x180"

        # Get 2-state thresholds
        readout_op_name = self.xeb_config.readout_pulse_name
        threshold_c = qubit_c.resonator.operations[readout_op_name].threshold
        threshold_t = qubit_t.resonator.operations[readout_op_name].threshold
        threshold_k = 0

        with program() as xeb_prog:
            reset_global_phase()

            # --- 2. QUA DECLARATIONS ---
            I_raw, I_st_raw, Q_raw, Q_st_raw = qua_declaration(
                n_qubits=n_qubits, readout_elements=self.readout_channels
            )

            depth, depth_, n, s, tot_state_ = [declare(int) for _ in range(5)]

            # --- FIX 1: DEFINE CORRECT LENGTH (Max index + 1) ---
            max_gate_len = self.xeb_config.depths[-1] + 1

            gate = [declare(int, size=max_gate_len) for _ in range(n_qubits)]
            two_qubit_gate_pattern = declare(int, value=0)

            if self.xeb_config.gate_set.run_through_amp_matrix_modulation:
                amp_matrix = [[declare(fixed, size=max_gate_len) for _ in range(4)] for _ in range(n_qubits)]

            # --- DYNAMIC DECLARATIONS (Streams) ---
            counts_stab = declare(int, value=[0] * total_dim)
            counts_st_stab = [declare_stream() for _ in range(total_dim)]
            I_c_st_all_stab = declare_stream()
            Q_c_st_all_stab = declare_stream()
            I_t_st_all_stab = declare_stream()
            Q_t_st_all_stab = declare_stream()

            state_c = declare(int)
            state_t = declare(int)
            state_k = declare(int)
            I_k = declare(fixed)
            Q_k = declare(fixed)

            gate_st = [declare_stream() for _ in range(n_qubits)]
            amp_st = [declare_stream() for _ in range(n_qubits)]

            s_st = declare_stream()

            # --- 3. SETUP & PULSE DEFINITION ---

            if getattr(self.machine, "twpas", None):
                for twpa in self.machine.twpas.values():
                    twpa.initialize()
            self.machine.apply_all_flux_to_joint_idle()
            readout_operation_name = "readout"

            r = Random(seed=self.xeb_config.seed)

            # --- 4. GENERATE SEQUENCES ---
            with for_(s, 0, s < self.xeb_config.seqs, s + 1):
                save(s, s_st)

                # 1. First Gate (Index 0)
                for q in range(n_qubits):
                    assign(gate[q][0], r.rand_int(random_gates))
                    save(gate[q][0], gate_st[q])
                    if self.xeb_config.gate_set.run_through_amp_matrix_modulation:
                        self._assign_amplitude_matrix(
                            gate[q][0],
                            [amp_matrix[q][i][0] for i in range(4)],
                            amp_st[q],
                        )

                # 2. Subsequent Gates (Index 1 to End)
                # --- FIX 2: LOOP CONDITION (Use max_gate_len) ---
                with for_(depth_, 1, depth_ < max_gate_len, depth_ + 1):
                    for q in range(n_qubits):
                        assign(gate[q][depth_], r.rand_int(random_gates))
                        with while_(gate[q][depth_] == gate[q][depth_ - 1]):
                            assign(gate[q][depth_], r.rand_int(random_gates))
                        save(gate[q][depth_], gate_st[q])
                        if self.xeb_config.gate_set.run_through_amp_matrix_modulation:
                            self._assign_amplitude_matrix(
                                gate[q][depth_],
                                [amp_matrix[q][i][depth_] for i in range(4)],
                                amp_st[q],
                            )

                # --- 5. RUN EXPERIMENT LOOP ---
                with for_each_(depth, self.xeb_config.depths):
                    with for_(n, 0, n < self.xeb_config.n_shots, n + 1):
                        # qubit_c.xy.play("x180")
                        # --- A. Play circuit (Unchanged) ---
                        with for_(depth_, 0, depth_ < depth, depth_ + 1):
                            for q, qubit in enumerate(self.qubits):
                                self._play_random_sq_gate(
                                    qubit,
                                    gate[q][depth_],
                                    (
                                        [amp_matrix[q][i][depth_] for i in range(4)]
                                        if self.xeb_config.gate_set.run_through_amp_matrix_modulation
                                        else None
                                    ),
                                )

                            if self.xeb_config.two_qb_gate is not None and len(self.qubit_pairs) > 0:
                                if len(self.qubit_pairs) > 1:  # Multi-qubit XEB case
                                    with switch_(two_qubit_gate_pattern):
                                        for i, combination in enumerate(self.available_combinations):
                                            with case_(i):
                                                for pair in combination:
                                                    ctrl_idx, tgt_idx = pair
                                                    if tgt_idx < ctrl_idx:
                                                        ctrl_idx, tgt_idx = tgt_idx, ctrl_idx
                                                    qubit_pair = self.machine.qubit_pairs[
                                                        "coupler_q{}_q{}".format(ctrl_idx + 1, tgt_idx + 1)
                                                    ]
                                                    align_transmon_pair(qubit_pair)
                                                    self.xeb_config.two_qb_gate.gate_macro(qubit_pair)
                                                    align_transmon_pair(qubit_pair)

                                    with if_(two_qubit_gate_pattern == len(self.available_combinations) - 1):
                                        assign(two_qubit_gate_pattern, 0)
                                    with else_():
                                        assign(two_qubit_gate_pattern, two_qubit_gate_pattern + 1)
                                else:  # Two-qubit XEB case
                                    align()
                                    qubit_pair = self.qubit_pairs[0]
                                    self.xeb_config.two_qb_gate.gate_macro()
                                    align()

                            elif self.xeb_config.two_qubit_gate_idle_time_ns > 0:
                                align()
                                wait_cycles = self.xeb_config.two_qubit_gate_idle_time_ns // 4
                                for q in self.qubits:
                                    q.wait(wait_cycles)
                                align()

                        # --- B. DYNAMIC READOUT ---
                        align()
                        wait(25)

                        # 1. Readout Control Qubit
                        if self.xeb_config.control_readout_mode == 2:
                            qubit_c.readout_state(state_c)
                            # assign(state_c, Cast.to_int(I_raw[0] > threshold_c))  # 0 or 1
                            save(0.0, I_st_raw[0])
                            save(1.0, Q_st_raw[0])
                            save(0.0, I_c_st_all_stab)
                            save(1.0, Q_c_st_all_stab)

                        elif self.xeb_config.control_readout_mode == 3:
                            I_c_local, Q_c_local = declare(fixed), declare(fixed)
                            qubit_c.readout_state_gef(
                                state_c,
                                pulse_name="readout_GEF",
                            )
                            save(0.0, I_st_raw[0])
                            save(0.0, Q_st_raw[0])
                            save(0.0, I_c_st_all_stab)
                            save(0.0, Q_c_st_all_stab)

                        # 2. Readout Target Qubit
                        if self.xeb_config.target_readout_mode == 2:
                            qubit_t.readout_state(state_t)
                            # assign(state_t, Cast.to_int(I_raw[1] > threshold_t))  # 0 or 1
                            save(0.0, I_st_raw[1])
                            save(1.0, Q_st_raw[1])
                            save(0.0, I_t_st_all_stab)
                            save(1.0, Q_t_st_all_stab)

                        elif self.xeb_config.target_readout_mode == 3:
                            I_t_local, Q_t_local = declare(fixed), declare(fixed)
                            qubit_t.readout_state_gef(
                                state_t,
                                pulse_name="readout_GEF",
                            )
                            save(0.0, I_st_raw[1])
                            save(0.0, Q_st_raw[1])
                            save(0.0, I_t_st_all_stab)
                            save(0.0, Q_t_st_all_stab)

                        # 3. Coupler readout is disabled
                        assign(state_k, 0)  # Ignored state

                        # --- C. MIXED-RADIX STATE AGGREGATION ---
                        dim_c = self.xeb_config.dim_c  # Get Python var
                        dim_t = self.xeb_config.dim_t  # Get Python var

                        assign(
                            tot_state_,
                            state_c + (dim_c * state_t) + (dim_c * dim_t * state_k),
                        )

                        # --- D. Reset (Unchanged) ---
                        for q_idx, qubit in enumerate(self.qubits):
                            if not simulate:
                                if self.reset_type == "active":
                                    # active_reset_gef(qubit, readout_pulse_name=rst_readout, pi_01_pulse_name=rst_pi_01, pi_12_pulse_name=rst_pi_12)
                                    qubit.reset(reset_type="active")
                                else:
                                    qubit.wait(qubit.thermalization_time * u.ns)

                        if len(self.qubit_pairs) > 1:
                            assign(two_qubit_gate_pattern, 0)

                        # --- E. Dynamic Switch Case ---
                        with switch_(tot_state_):
                            for i in range(total_dim):
                                with case_(i):
                                    assign(counts_stab[i], counts_stab[i] + 1)
                        assign(tot_state_, 0)

                    # --- F. Dynamic Save Loop ---
                    for i in range(total_dim):
                        save(counts_stab[i], counts_st_stab[i])
                        assign(counts_stab[i], 0)

                # --- [END OF EXPERIMENT LOOP] ---

            with stream_processing():
                # --- FIX 3: BUFFER SIZE MATCHES ARRAY SIZE ---
                for q in range(n_qubits):
                    gate_st[q].buffer(max_gate_len).save_all(f"g{q}")
                    if self.xeb_config.gate_set.run_through_amp_matrix_modulation:
                        amp_st[q].buffer(max_gate_len, 2, 2).save_all(f"amp_matrix_q{q}")
                # ---------------------------------------------

                s_st.save_all("s_progress")

                for q in range(n_qubits):
                    I_st_raw[q].buffer(self.xeb_config.n_shots).map(FUNCTIONS.average()).buffer(
                        len(self.xeb_config.depths)
                    ).save_all(f"I{q}")
                    Q_st_raw[q].buffer(self.xeb_config.n_shots).map(FUNCTIONS.average()).buffer(
                        len(self.xeb_config.depths)
                    ).save_all(f"Q{q}")

                for i in range(total_dim):
                    counts_st_stab[i].buffer(len(self.xeb_config.depths)).save_all(f"s{i}")

                total_shots_per_seq = self.xeb_config.n_shots * len(self.xeb_config.depths)

                I_c_st_all_stab.buffer(total_shots_per_seq).save_all("I_c_all")
                Q_c_st_all_stab.buffer(total_shots_per_seq).save_all("Q_c_all")
                I_t_st_all_stab.buffer(total_shots_per_seq).save_all("I_t_all")
                Q_t_st_all_stab.buffer(total_shots_per_seq).save_all("Q_t_all")

        return xeb_prog

    def run(
        self,
        simulate: bool = False,
        simulation_config: Optional[SimulationConfig] = None,
        qmm_cloud_simulator: Optional[QuantumMachinesManager] = None,
        config=None,
        **simulate_kwargs,
    ):
        """
        Run QUA program for the XEB experiment
        Args:
            simulate: Indicate if output should be simulated or not
            simulation_config: SimulationConfig object containing the parameters of the simulation
            qmm_cloud_simulator: QuantumMachinesManager object to simulate the experiment
            simulate_kwargs: Optional additional keyword arguments passed to `qm.simulate`

        Returns: XEBJob object containing the information about the experiment (including results)

        """
        # Compile the QUA program
        if config is None:
            config = self.machine.generate_config()
        if simulation_config is None:
            simulation_config = SimulationConfig(duration=10_000)
        xeb_prog = self._xeb_prog(simulate=simulate)  # set simulate=True to get the amplitude matrix
        if simulate and qmm_cloud_simulator is not None:
            qmm = qmm_cloud_simulator
        else:
            qmm = self.machine.connect()
        qm = qmm.open_qm(config, close_other_machines=True)
        if simulate:
            with open("debug.py", "w+") as f:
                f.write(generate_qua_script(xeb_prog, config))
            job = qm.simulate(xeb_prog, simulate=simulation_config, **simulate_kwargs)
        elif self.xeb_config.generate_new_data:
            job = qm.execute(xeb_prog)
        else:
            warnings.warn(
                "Running deactivated. Set generate_new_data to True to run the experiment."
                "Use XEBResult.from_data() method to load data from a previous run."
            )
            return

        return XEBJob(
            running_job=job,
            xeb_config=self.xeb_config,
            data_handler=self.data_handler,
            available_combinations=self.available_combinations,
            simulate=False,
            hardware_simulate=simulate,
            cloud=self.cloud,
        )

    def simulate(self, backend: BackendV2):
        """
            Simulate the XEB experiment: To simulate it, you must provide an AerBackend object
             with a noise model corresponding to your experiments parameters.
            For instance,
        ```python
        from qiskit import Aer
        from qiskit.providers.aer import AerSimulator
        from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
        error1q = 0.07
        error2q = 0.03
        depol_error1q = depolarizing_error(error1q, 1)
        depol_error2q = depolarizing_error(error2q, 2)
        sq_gate_set = ["h", "t", "rx", "ry", "sw"] # Specify which gates are subject to noise
        noise_model = NoiseModel(basis_gates = sq_gate_set)
        noise_model.add_all_qubit_quantum_error(depol_error2q, ["cz"])
        noise_model.add_all_qubit_quantum_error(depol_error1q, sq_gate_set)
        backend = AerSimulator(noise_model=noise_model, method="density_matrix", basis_gates=noise_model.basis_gates)
        ```
            Args:
                backend: AerBackend object to simulate the experiment. Note that it should carry a noise model to see
                a fidelity decay.

            Returns: XEBJob object containing the information about the experiment (including results)

        """
        from qiskit_aer.backends.aerbackend import AerBackend

        assert isinstance(backend, AerBackend), "The backend should be an AerBackend object"
        num_qubits = len(self.qubits)
        random_gates = len(self.xeb_config.gate_set)
        sq_gates, counts_list, states_list, circuits_list = [], [], [], []
        two_qubit_gate_pattern = 0
        # Generate sequences
        for s in range(self.xeb_config.seqs):  # For each sequence
            circuits_list.append([])
            sq_gates.append(np.zeros((num_qubits, self.xeb_config.depths[-1]), dtype=int))
            for q in range(num_qubits):  # For each qubit
                # Generate random single qubit gates
                # Start the sequence with a random gate
                sq_gates[s][q][0] = np.random.randint(random_gates)
            for d_ in range(1, self.xeb_config.depths[-1]):  # Generate sequences of max_depth, to be truncated later
                for q in range(num_qubits):  # For each qubit
                    sq_gates[s][q][d_] = np.random.randint(random_gates)
                    # Make sure that the same gate is not applied twice in a row
                    while sq_gates[s][q][d_] == sq_gates[s][q][d_ - 1]:
                        sq_gates[s][q][d_] = np.random.randint(random_gates)
            for i, d in enumerate(self.xeb_config.depths):  # For each maximum depth
                # Define the circuit
                qc = QuantumCircuit(num_qubits)
                for d_ in range(d):  # Apply layers
                    for q in range(num_qubits):  # For each qubit, append single qubit gates
                        qc.append(self.xeb_config.gate_set[sq_gates[s][q][d_]].gate, [q])
                    qc.barrier()
                    # Apply two-qubit gates
                    if num_qubits >= 2 and self.xeb_config.two_qb_gate is not None and len(self.qubit_pairs) > 0:
                        for i, combination in enumerate(self.available_combinations):
                            if i == two_qubit_gate_pattern:
                                for pair in combination:
                                    qc.append(self.xeb_config.two_qb_gate.gate, pair)
                                qc.barrier()
                                break
                        if two_qubit_gate_pattern == len(self.available_combinations) - 1:
                            two_qubit_gate_pattern = 0
                        else:
                            two_qubit_gate_pattern += 1
                        # qc.append(self.xeb_config.two_qb_gate.gate, [0, 1])

                two_qubit_gate_pattern = 0
                qc.save_density_matrix()  # Actual state, subject to noise simulation
                circuits_list[s].append(qc)

                # Simulate the circuit
                # Execute circuit (transpiled) and store counts
        circ_list = [
            circuits_list[s][i].measure_all(inplace=False)
            for s in range(self.xeb_config.seqs)
            for i in range(len(self.xeb_config.depths))
        ]
        transpiled_circs = circ_list
        job = backend.run(transpiled_circs, shots=self.xeb_config.n_shots)
        gate_indices = np.array(sq_gates)

        return XEBJob(
            running_job=job,
            xeb_config=self.xeb_config,
            data_handler=self.data_handler,
            available_combinations=self.available_combinations,
            simulate=True,
            gate_indices=gate_indices,
        )


class XEBJob:
    def __init__(
        self,
        running_job: Union[SimulatedJob, RunningQmJob, AerJob],
        xeb_config: XEBConfig,
        data_handler: DataHandler,
        available_combinations: List[Tuple[Tuple[int, int]]],
        simulate=False,
        hardware_simulate=False,
        gate_indices=None,
        cloud=None,
    ):
        self.cloud = cloud
        self.job = running_job
        self.available_combinations = available_combinations
        self._simulate = simulate
        self._hardware_simulate = hardware_simulate
        self._result_handles = self.job.result() if isinstance(running_job, AerJob) else self.job.result_handles
        # self._result_handles.wait_for_all_values() # <-- This is correctly removed
        self.xeb_config = xeb_config
        self.data_handler = data_handler

        # --- MODIFIED LINES ---
        self._circuits = None  # Will be built later
        if self._simulate:
            self._gate_indices = gate_indices  # Save this from the simulation
        else:
            self._gate_indices = None  # Will be fetched later

    def _fetch_and_build_circuits(self):
        """
        Fetches gate data and builds the circuits list.
        This is called by result() AFTER the job is complete.
        """
        # If circuits are already built, do nothing.
        if self._circuits is not None:
            return

        circuits = []
        if self._simulate:
            # Simulation path: self._gate_indices was already set in __init__
            assert isinstance(self.job, AerJob), "The job should be an AerJob object"
            assert self._gate_indices is not None, "Gate indices missing for simulation"

            idx = 0
            for s in range(self.xeb_config.seqs):
                circuits.append([])
                for _ in range(len(self.xeb_config.depths)):
                    circuits[s].append(self.job.circuits()[idx].remove_final_measurements(inplace=False))
                    circuits[s][-1].data.pop(-1)  # Remove save_density_matrix instruction
                    circuits[s][-1].measure_all(inplace=True)
                    idx += 1

        else:
            # Hardware path: Fetch data and build _gate_indices

            # --- FIX: Match the size used in QUA (+1) ---
            max_depth = self.xeb_config.depths[-1] + 1
            # --------------------------------------------

            n_qubits = self.xeb_config.n_qubits

            # Initialize the array we'll fill
            self._gate_indices = np.zeros((self.xeb_config.seqs, n_qubits, max_depth), dtype=int)

            # Fetch the data
            # Cloud: nested structure g[q][s] = [[depth_array]] (list of 1 elem), so use g[q][s][0][d].
            # Local: g[q] has shape (seqs, max_gate_len), use g[q][s, d].
            if self.cloud:
                g = [self._result_handles.get(f"g{q}").fetch_all() for q in range(n_qubits)]
            else:
                g = [self._result_handles.get(f"g{q}").fetch_all()["value"] for q in range(n_qubits)]

            for s in range(self.xeb_config.seqs):
                for d in range(max_depth):
                    for q in range(n_qubits):
                        if self.cloud:
                            self._gate_indices[s, q, d] = g[q][s][0][d]
                        else:
                            self._gate_indices[s, q, d] = g[q][s, d]

            circuits = generate_circuits(self.xeb_config, self.gate_indices, self.available_combinations)

        self._circuits = circuits

    def result(self, disjoint_processing: Optional[bool] = None):
        """
        Returns the results of the XEB experiment
        ...
        """

        # --- NEW: ADD PROGRESS BAR ---
        # Only show progress for real hardware jobs
        if not self._simulate and not self._hardware_simulate:
            n_seqs = self.xeb_config.seqs

            # --- FIX STARTS HERE ---
            has_progress = False
            try:
                # Try to check if s_progress is in result handles
                if "s_progress" in self._result_handles:
                    has_progress = True
            except TypeError:
                # CloudResultHandles does not support 'in' operator
                # We can try to access it directly, or assume it's there if not simulated
                # However, count_so_far() might also behave differently on cloud.
                # For safety, we can skip the progress bar on Cloud or just try to get it.
                # Here we try to get it to see if it exists.
                try:
                    self._result_handles.get("s_progress")
                    # If get succeeds, we assume we can use it,
                    # BUT note that count_so_far might not be supported on all cloud versions.
                    # If you just want to fix the crash, setting has_progress=False is safest.
                    has_progress = True
                except Exception:
                    has_progress = False

            if has_progress:
                s_progress_handle = self._result_handles.get("s_progress")

                # Initialize tqdm progress bar
                print("Job is running...")
                try:
                    with tqdm(total=n_seqs, desc="Running XEB sequences") as pbar:
                        while self.job.is_running():
                            # Get the number of sequences completed
                            current_count = s_progress_handle.count_so_far()
                            # Update the progress bar
                            pbar.n = current_count
                            pbar.refresh()
                            time.sleep(0.5)  # Poll every 500ms

                        # Final update after job is done
                        current_count = s_progress_handle.count_so_far()
                        pbar.n = current_count
                        pbar.refresh()
                    print("Job finished, processing results...")
                except Exception as e:
                    # Fallback if count_so_far fails (e.g. on some cloud versions)
                    print(f"Progress bar failed ({e}), waiting for results...")
                    self._result_handles.wait_for_all_values()
            else:
                # Fallback if s_progress stream isn't found or check failed
                print("Job is running... (s_progress stream not found or cloud execution, cannot show progress)")
                self._result_handles.wait_for_all_values()
            # --- FIX ENDS HERE ---

        else:
            # For simulations or non-QM jobs, just wait
            self._result_handles.wait_for_all_values()

        self._fetch_and_build_circuits()

        if disjoint_processing is not None:
            assert isinstance(disjoint_processing, bool), "disjoint_processing should be a boolean"
            self.xeb_config.disjoint_processing = disjoint_processing

        # --- THIS ENTIRE BLOCK IS REPLACED ---
        if self._simulate:
            result = self.job.result()
            dms = np.array([result.data(i)["density_matrix"].data for i in range(len(result.get_counts()))])

            # --- New Dynamic Count Generation ---
            counts = {}
            dim_c = self.xeb_config.dim_c
            total_dim = self.xeb_config.total_dim
            n_seqs = self.xeb_config.seqs
            n_depths = len(self.xeb_config.depths)

            # Define the mapping from Qiskit bitstring to our new stream index
            # Qiskit "q1q0" -> (t, c). Our index = c + (dim_c * t) + (k=0)
            key_map = {
                "00": f"s{0 + (dim_c * 0)}",  # (t=0, c=0) -> s0
                "01": f"s{1 + (dim_c * 0)}",  # (t=0, c=1) -> s1
                "10": f"s{0 + (dim_c * 1)}",  # (t=1, c=0) -> s{dim_c}
                "11": f"s{1 + (dim_c * 1)}",  # (t=1, c=1) -> s{1 + dim_c}
            }

            # 1. Initialize all possible streams (s0..s{total_dim-1}) to zero arrays
            for i in range(total_dim):
                counts[f"s{i}"] = np.zeros((n_seqs, n_depths))

            # 2. Get the list of Qiskit count dicts
            qiskit_counts_list = result.get_counts()

            # 3. Iterate and populate the CS keys
            idx = 0
            for s in range(n_seqs):
                for d in range(n_depths):
                    qiskit_dict = qiskit_counts_list[idx]
                    for qiskit_key, stream_key in key_map.items():
                        if stream_key in counts:  # Ensure the key exists (it always should)
                            counts[stream_key][s, d] = qiskit_dict.get(qiskit_key, 0)
                    idx += 1

            # The 'states' dictionary is no longer needed
            saved_data = {"counts": counts, "density_matrices": dms, "gate_indices": self.gate_indices}

            return XEBResult(
                self.xeb_config,
                self.circuits,
                counts,
                # states, <-- REMOVED
                saved_data,
                self.data_handler if self.xeb_config.should_save_data else None,
            )
        # --- END OF REPLACEMENT ---

        else:
            # --- This is the QPU path ---
            gate_indices, quadratures, amp_st = {}, {}, {}
            result = self._result_handles

            for q, qubit in enumerate(self.xeb_config.qubits):
                if self.cloud:
                    gate_indices[f"g_{qubit.name}"] = result.get(f"g{q}").fetch_all()
                    quadratures[f"I_{qubit.name}"] = result.get(f"I{q}").fetch_all()
                    quadratures[f"Q_{qubit.name}"] = result.get(f"Q{q}").fetch_all()
                    if self.xeb_config.gate_set.run_through_amp_matrix_modulation:
                        amp_st[f"amp_matrix_{qubit.name}"] = result.get(f"amp_matrix_q{q}").fetch_all()
                else:
                    gate_indices[f"g_{qubit.name}"] = result.get(f"g{q}").fetch_all()["value"]
                    quadratures[f"I_{qubit.name}"] = result.get(f"I{q}").fetch_all()["value"]
                    quadratures[f"Q_{qubit.name}"] = result.get(f"Q{q}").fetch_all()["value"]
                    if self.xeb_config.gate_set.run_through_amp_matrix_modulation:
                        amp_st[f"amp_matrix_{qubit.name}"] = result.get(f"amp_matrix_q{q}").fetch_all()["value"]

            total_dim = self.xeb_config.total_dim
            seqs = self.xeb_config.seqs
            n_depths = len(self.xeb_config.depths)

            # --- 1. Fetch STABILIZED Data (using original names) ---
            counts, raw_iq_data = {}, {}
            for i in range(total_dim):
                stream_name = f"s{i}"
                if self.cloud:
                    raw = result.get(stream_name).fetch_all()
                    # Cloud: raw[s] = [[d0,d1,...]], convert to (seqs, n_depths) array
                    counts[stream_name] = np.array([[raw[s][0][d] for d in range(n_depths)] for s in range(seqs)])
                else:
                    counts[stream_name] = result.get(stream_name).fetch_all()["value"]

            raw_streams = ["I_c_all", "Q_c_all", "I_t_all", "Q_t_all"]

            for stream_name in raw_streams:
                if self.cloud:
                    raw_iq_data[stream_name] = result.get(stream_name).fetch_all()
                else:
                    raw_iq_data[stream_name] = result.get(stream_name).fetch_all()["value"]

            # --- 2. Create Result Object ---
            saved_data = {
                **quadratures,
                **counts,
                **amp_st,
                **raw_iq_data,
                "gate_indices": self.gate_indices,
                "gate_sequences": self.gate_sequences,
            }

            result_obj = XEBResult(
                self.xeb_config,
                self.circuits,
                counts,
                saved_data,
                self.data_handler if self.xeb_config.should_save_data else None,
            )

            return result_obj

    @property
    def circuits(self):
        """
        Returns the circuits generated by the XEB experiment
        Circuits are formatted as follows:
        - The first dimension corresponds to the sequence index
        - The second dimension corresponds to the depth index
        Returns:
            List of lists of QuantumCircuit objects representing the circuits generated by the XEB experiment.

        """
        return self._circuits

    @property
    def simulate(self):
        return self._simulate

    @property
    def hardware_simulate(self):
        return self._hardware_simulate

    @property
    def gate_indices(self):
        """
        Returns the gate indices of the XEB experiment in the form of a 3D numpy array (sequence, qubit, depth)
        """
        return self._gate_indices

    @property
    def gate_sequences(self):
        """
        Returns the gate sequences of the XEB experiment in the form of a 3D numpy array (sequence, qubit, depth)
        """
        gate_sequences = np.zeros(
            (self.xeb_config.seqs, self.xeb_config.n_qubits, self.xeb_config.depths[-1]), dtype=str
        )
        for s in range(self.xeb_config.seqs):
            for q in range(self.xeb_config.n_qubits):
                for d in range(self.xeb_config.depths[-1]):
                    gate_sequences[s, q, d] = self.xeb_config.gate_set[self.gate_indices[s, q, d]].name

        return gate_sequences

    def plot_simulated_samples(self):
        if self.hardware_simulate:
            samples = self.job.get_simulated_samples()
            plt.subplots(nrows=len(samples.keys()), sharex=True)
            for i, con in enumerate(samples.keys()):
                plt.subplot(len(samples.keys()), 1, i + 1)
                samples[con].plot()
                plt.title(con)
            plt.tight_layout()
            plt.show()
        else:
            warnings.warn("Simulated samples are not available because the job was run and not hardware-simulated.")


class XEBResult:
    def __init__(
        self,
        xeb_config: XEBConfig,
        circuits,
        counts: Dict,
        saved_data,
        data_handler: DataHandler = None,
        parameterize_circuit: bool = False,
    ):
        self.xeb_config = xeb_config
        self.circuits: List[List[QuantumCircuit]] = circuits
        self.circuits_parameter_assigned: List[List] = copy.deepcopy(circuits)
        self.counts = counts
        self.data = saved_data
        self.data_handler = data_handler
        self.parameterize_circuit = parameterize_circuit

        # --- [USER'S CODE] Leakage Analysis Structures ---
        self._leakage_probs_by_state = {}
        self._leakage_state_names = []
        self._leakage_probs = None
        self._disjoint_leakage_probs = None
        self._total_cs_probs = None

        # Retrieve Data (calculates fidelities, leakage, etc.)
        (
            self._joint_measured_probs,
            self._disjoint_measured_probs,
            self._joint_expected_probs,
            self._disjoint_expected_probs,
            self._records,
            self._log_fidelities,
            self._linear_fidelities,
            self._singularities,
            self._outliers,
        ) = self.retrieve_data()

        self.data.update(
            {
                "joint_measured_probs": self._joint_measured_probs,
                "disjoint_measured_probs": self._disjoint_measured_probs,
                "joint_expected_probs": self._joint_expected_probs,
                "disjoint_expected_probs": self._disjoint_expected_probs,
                "log_fidelities": self._log_fidelities,
                "linear_fidelities": (
                    np.array(self.linear_fidelities["fidelity"])
                    if not self.xeb_config.disjoint_processing
                    else np.array([fidelity["fidelity"] for fidelity in self.linear_fidelities])
                ),
                "singularities": self._singularities,
                "outliers": self._outliers,
                # User's leakage keys
                "leakage_probs": self._leakage_probs,
                "disjoint_leakage_probs": self._disjoint_leakage_probs,
                "total_cs_probs": self._total_cs_probs,
                "leakage_probs_by_state": self._leakage_probs_by_state,
                "leakage_state_names": self._leakage_state_names,
            }
        )

        if self.xeb_config.should_save_data and self.data_handler is not None:
            save_data = {}
            for key in self.data.keys():
                if "amp_matrix" in key:  # Remove amplitude matrices
                    continue
                save_data[key] = self.data[key]
            self.data_handler.save_data(
                saved_data, self.xeb_config.data_folder_name, metadata=self.xeb_config.as_dict()
            )

    @classmethod
    def from_data(
        cls,
        directory: str,
        disjoint_processing: Optional[bool] = None,
        data_handler: Optional[DataHandler] = None,
        parameterize_circuit=False,
    ):
        data: Dict = json.load(open(directory + "/data.json", "r"))
        arrays: Dict = np.load(directory + "/arrays.npz")
        node: Dict = json.load(open(directory + "/node.json", "r"))
        xeb_config = XEBConfig.from_dict(node["metadata"])
        if disjoint_processing is not None:
            assert isinstance(disjoint_processing, bool), "disjoint_processing should be a boolean"
            xeb_config.disjoint_processing = disjoint_processing
        gate_indices = arrays["gate_indices"]

        if parameterize_circuit:
            circuits, _, _ = generate_circuits_parameterized(
                xeb_config, gate_indices, xeb_config.available_combinations, en_measure=False
            )
        else:
            circuits = generate_circuits(xeb_config, gate_indices, xeb_config.available_combinations)

        new_data = {"states": {}, "counts": {}, "quadratures": {}, "amp_st": {}}

        for key, value in data.items():
            if "state" in key:
                for key2, value2 in value.items():
                    new_data["states"][key2] = arrays[value2[value2.find("#") + 1 :]]
            elif "count" in key:
                for key2, value2 in value.items():
                    new_data["counts"][key2] = arrays[value2[value2.find("#") + 1 :]]
            elif "amp_matrix" in key:
                new_data["amp_st"][key] = arrays[key]
            elif key.startswith("I") or key.startswith("Q"):
                new_data["quadratures"][key] = arrays[key]
            else:
                if key in arrays:
                    new_data[key] = arrays[key]
                else:
                    new_data[key] = value
        return cls(
            xeb_config, circuits, new_data["counts"], new_data["states"], new_data, data_handler, parameterize_circuit
        )

    @classmethod
    def from_data_qualibrate(
        cls,
        directory: str,
        disjoint_processing: Optional[bool] = None,
        data_handler: Optional[DataHandler] = None,
        parameterize_circuit=False,
    ):
        data: Dict = json.load(open(directory + "/data.json", "r"))
        arrays: Dict = np.load(directory + "/arrays.npz")
        xeb_config = XEBConfig.from_dict(data["xeb_config"])
        if disjoint_processing is not None:
            assert isinstance(disjoint_processing, bool), "disjoint_processing should be a boolean"
            xeb_config.disjoint_processing = disjoint_processing
        gate_indices = arrays["data.gate_indices"]
        if parameterize_circuit:
            circuits, _, _ = generate_circuits_parameterized(
                xeb_config, gate_indices, xeb_config.available_combinations, en_measure=False
            )
        else:
            circuits = generate_circuits(xeb_config, gate_indices, xeb_config.available_combinations)

        new_data = {"states": {}, "counts": {}, "quadratures": {}, "amp_st": {}}

        for key, value in data["data"].items():
            if "state" in key:
                new_data["states"][key] = arrays["data." + key]
            elif ("00" in key) or ("01" in key) or ("10" in key) or ("11" in key):
                new_data["counts"][key] = arrays["data." + key]
            elif "amp_matrix" in key:
                new_data["amp_st"][key] = arrays["data." + key]
        new_data["gate_indices"] = gate_indices

        return cls(
            xeb_config, circuits, new_data["counts"], new_data["states"], new_data, data_handler, parameterize_circuit
        )

    @classmethod
    def from_data_qualibrate_simulate(
        cls,
        directory: str,
        disjoint_processing: Optional[bool] = None,
        data_handler: Optional[DataHandler] = None,
        parameterize_circuit=False,
    ):
        data: Dict = json.load(open(directory + "/data.json", "r"))
        arrays: Dict = np.load(directory + "/arrays.npz")
        xeb_config = XEBConfig.from_dict(data["xeb_config"])
        if disjoint_processing is not None:
            assert isinstance(disjoint_processing, bool), "disjoint_processing should be a boolean"
            xeb_config.disjoint_processing = disjoint_processing
        gate_indices = arrays["data.gate_indices"]
        if parameterize_circuit:
            circuits, _, _ = generate_circuits_parameterized(
                xeb_config, gate_indices, xeb_config.available_combinations, en_measure=False
            )
        else:
            circuits = generate_circuits(xeb_config, gate_indices, xeb_config.available_combinations)

        new_data = {"states": {}, "counts": {}, "quadratures": {}, "amp_st": {}}
        for key, value in data["data"].items():
            if "state" in key:
                for key2, value2 in value.items():
                    new_data["states"][key2] = arrays["data.states." + key2]
            elif "count" in key:
                for key2, value2 in value.items():
                    new_data["counts"][key2] = arrays["data.counts." + key2]
        new_data["gate_indices"] = gate_indices

        return cls(
            xeb_config, circuits, new_data["counts"], new_data["states"], new_data, data_handler, parameterize_circuit
        )

    @classmethod
    def from_xarray_dataset(
        cls,
        ds,
        xeb_config: XEBConfig,
        data_handler: Optional[DataHandler] = None,
    ):
        """
        Create XEBResult from an xarray Dataset produced by XarrayDataFetcher.

        The dataset is expected to have variables matching QUA stream names:
        - s0, s1, ... (counts, shape sequence x depth)
        - g0, g1, ... (gate indices, shape sequence x max_gate_len)
        - I_c_all, Q_c_all, I_t_all, Q_t_all (optional, flattened IQ data)
        """
        # Resolve dimension names (fetcher may use different conventions)
        seq_dim = "sequence" if "sequence" in ds.dims else "seq"
        depth_dim = "depth" if "depth" in ds.dims else "depths"
        if seq_dim not in ds.dims:
            seq_dim = next((d for d in ds.dims if "seq" in d.lower()), list(ds.dims)[0])
        if depth_dim not in ds.dims:
            depth_dim = next((d for d in ds.dims if "depth" in d.lower()), list(ds.dims)[1])

        n_qubits = xeb_config.n_qubits
        total_dim = xeb_config.total_dim
        max_gate_len = xeb_config.depths[-1] + 1

        # Extract counts
        counts = {}
        for i in range(total_dim):
            var_name = f"s{i}"
            if var_name in ds:
                arr = ds[var_name]
                if hasattr(arr, "values"):
                    arr = arr.values
                counts[var_name] = np.asarray(arr)

        # Extract gate indices (shape: seqs x n_qubits x max_gate_len)
        gate_indices = np.zeros((xeb_config.seqs, n_qubits, max_gate_len), dtype=int)
        for q in range(n_qubits):
            var_name = f"g{q}"
            if var_name in ds:
                arr = ds[var_name]
                if hasattr(arr, "values"):
                    arr = arr.values
                arr = np.asarray(arr)
                n_seq = min(arr.shape[0], xeb_config.seqs)
                n_gate = min(arr.shape[-1] if arr.ndim > 1 else 1, max_gate_len)
                if arr.ndim >= 2:
                    gate_indices[:n_seq, q, :n_gate] = arr[:n_seq, :n_gate]

        circuits = generate_circuits(xeb_config, gate_indices, xeb_config.available_combinations)

        # Build saved_data
        saved_data = {"gate_indices": gate_indices, **counts}
        quadratures = {}
        for q, qubit in enumerate(xeb_config.qubits):
            for prefix in ["I", "Q"]:
                var_name = f"{prefix}{q}"
                if var_name in ds:
                    arr = ds[var_name]
                    if hasattr(arr, "values"):
                        arr = arr.values
                    quadratures[f"{prefix}_{qubit.name}"] = np.asarray(arr)
        saved_data.update(quadratures)

        raw_iq = {}
        for name in ["I_c_all", "Q_c_all", "I_t_all", "Q_t_all"]:
            if name in ds:
                arr = ds[name]
                if hasattr(arr, "values"):
                    arr = arr.values
                raw_iq[name] = np.asarray(arr)
        saved_data.update(raw_iq)

        return cls(xeb_config, circuits, counts, saved_data, data_handler)

    # --- [USER'S CODE] Helper for mixed-radix decoding ---
    @staticmethod
    def _decode_state_index(index, dim_c, dim_t):
        c_state = index % dim_c
        t_state = (index // dim_c) % dim_t
        k_state = index // (dim_c * dim_t)
        return (c_state, t_state, k_state)

    # --- [COLLEAGUE'S NEW METHODS] ---
    def compute_layer_fidelity_given_fSim_parameters(self, theta, phi):
        """
        Recalculate fidelity assuming specific fSim parameters.
        Calls retrieve_data with new angles to generate new 'Ideal' probs.
        """
        (
            self._joint_measured_probs,
            self._disjoint_measured_probs,
            self._joint_expected_probs,
            self._disjoint_expected_probs,
            self._records,
            self._log_fidelities,
            self._linear_fidelities,
            self._singularities,
            self._outliers,
        ) = self.retrieve_data(theta, phi)

        layer_fid_linear = self.get_layer_fidelity(fidelity_metric="linear")
        layer_fid_log = self.get_layer_fidelity(fidelity_metric="log")
        return layer_fid_linear, layer_fid_log

    def calc_fidelity_for_given_2q_unitary_params(self, theta_iswap=0, phi_cphase=np.pi, phi_rz1=0, phi_rz2=0):
        """
        Calculates the log fidelity for a given set of 2-qubit unitary parameters.
        Uses the gate indices and depths from the config.
        """
        ideal_probability_s = calc_ideal_probability_numpy(
            self.data["gate_indices"],
            self.xeb_config.depths,
            theta_iswap=theta_iswap,
            phi_cphase=phi_cphase,
            phi_rz1=phi_rz1,
            phi_rz2=phi_rz2,  # Using phi_rz1 variable for rz2 argument? Assuming typo in source, corrected here to use phi_rz2 if the func supports it, otherwise follow source.
        )

        f_xeb = [
            [
                compute_log_fidelity(
                    self.incoherent_distribution,
                    ideal_probability_s[s, d_, :],
                    self.data["joint_measured_probs"][s, d_],
                )
                for d_, depth in enumerate(self.xeb_config.depths)
            ]
            for s in range(self.xeb_config.seqs)
        ]
        singularity = []
        outlier = []
        log_fidelities = [
            [
                evaluate_log_fidelity(f_xeb[s][d_], singularity, outlier, s, int(depth))
                for d_, depth in enumerate(self.xeb_config.depths)
            ]
            for s in range(self.xeb_config.seqs)
        ]

        Fxeb = np.nanmean(log_fidelities, axis=0)
        a, layer_fid_log, *_ = fit_exponential_decay(self.xeb_config.depths, Fxeb)

        return layer_fid_log

    def retrieve_data(self, theta: float = 0, phi: float = np.pi):
        """
        Retrieve and post-process the data.
        Handles leakage, mixed-radix streams, and CS renormalization.
        """

        # --- MODIFICATION: Use Supervised Discrimination if requested ---
        if self.xeb_config.discrimination_method == "gaussian":
            try:
                self._perform_supervised_discrimination()
            except Exception as e:
                print(f"Supervised discrimination failed ({e}). Falling back to threshold counts.")
        # ----------------------------------------------------------------

        # --- 1. Get Config & Initialize ---
        dim_c = self.xeb_config.dim_c
        dim_t = self.xeb_config.dim_t
        dim_k = self.xeb_config.dim_k
        total_dim = self.xeb_config.total_dim

        n_qubits = self.xeb_config.n_qubits
        dim_2N = 2**n_qubits
        seqs = self.xeb_config.seqs
        depths = self.xeb_config.depths
        n_depths = len(depths)
        counts = self.counts

        # --- 2. Calculate Ideal Probabilities ---
        if self.xeb_config.two_qb_gate is None:
            use_phi = 0.0
            use_theta = 0.0
        else:
            use_phi = phi
            use_theta = theta

        self.ideal_probability_s = calc_ideal_probability_numpy(
            self.data["gate_indices"],
            self.xeb_config.depths,
            theta_iswap=use_theta,
            phi_cphase=use_phi,
            phi_rz1=0,
            phi_rz2=0,
        )

        # --- FIX: ROBUST SHAPE HANDLING ---
        # If ideal_probability_s is missing a depth (e.g. depth 0 or max depth), align them.
        n_ideal_depths = self.ideal_probability_s.shape[1]

        if n_ideal_depths != n_depths:
            print(f"Warning: Ideal Probabilities has {n_ideal_depths} depths, but Config has {n_depths}.")

            # Case A: Missing Depth 0 (common if sim skips identity)
            if n_ideal_depths == n_depths - 1 and depths[0] == 0:
                print(" -> Assuming missing Depth 0 (Identity). Padding ideal probs.")
                # Create Identity distribution (1.0 in state 0)
                ideal_identity = np.zeros((seqs, 1, dim_2N))
                ideal_identity[:, 0, 0] = 1.0
                self.ideal_probability_s = np.concatenate([ideal_identity, self.ideal_probability_s], axis=1)

            # Case B: Just truncated (e.g. gate_indices too small for max depth)
            else:
                print(f" -> Truncating analysis to first {n_ideal_depths} depths.")
                n_depths = n_ideal_depths
                depths = depths[:n_depths]
        # ----------------------------------

        # Parameterize circuits if needed
        for s in range(seqs):
            # FIX: Iterate only up to available n_depths
            for d_, depth in enumerate(depths):
                if self.parameterize_circuit:
                    sorted_params = sorted(self.circuits[s][d_].parameters, key=lambda p: p.name)
                    values = [use_theta, use_phi]
                    parameters = dict(zip(sorted_params, values))
                    self.circuits_parameter_assigned[s][d_] = self.circuits[s][d_].assign_parameters(
                        parameters=parameters
                    )

        # --- 3. Initialize Storage Arrays ---
        joint_expected_probs = self.ideal_probability_s
        joint_measured_probs = np.zeros((seqs, n_depths, dim_2N))

        disjoint_expected_probs = np.zeros((n_qubits, seqs, n_depths, 2))
        disjoint_measured_probs = np.zeros((n_qubits, seqs, n_depths, 2))

        # Leakage storage
        self._leakage_probs = np.zeros((seqs, n_depths))
        self._total_cs_probs = np.zeros((seqs, n_depths))
        self._disjoint_leakage_probs = np.zeros((n_qubits, seqs, n_depths))

        if not self.xeb_config.disjoint_processing:
            records, singularity, outlier = [], [], []
            incoherent_distribution = np.ones(dim_2N) / dim_2N
            log_fidelities = np.zeros((seqs, n_depths))
        else:
            records = [[] for _ in range(n_qubits)]
            singularity = [[] for _ in range(n_qubits)]
            outlier = [[] for _ in range(n_qubits)]
            incoherent_distribution = np.ones(2) / 2
            log_fidelities = np.zeros((n_qubits, seqs, n_depths))

        self.incoherent_distribution = incoherent_distribution

        # --- 4. Build CS/Leakage Map ---
        c_idx = 0
        t_idx = 1
        try:
            qs_names = [q.name if hasattr(q, "name") else q for q in self.xeb_config.qubits]
            pair = self.xeb_config.qubit_pairs[0]
            if hasattr(pair, "qubit_control"):
                c_name = pair.qubit_control.name
                t_name = pair.qubit_target.name
            else:
                c_name = pair.qubit_control
                t_name = pair.qubit_target

            if c_name in qs_names and t_name in qs_names:
                c_idx = qs_names.index(c_name)
                t_idx = qs_names.index(t_name)
        except Exception as e:
            print(f"Warning: Could not determine dynamic bitstring indices ({e}). Using defaults.")

        cs_index_map = {}
        for i in range(total_dim):
            c, t, k = self._decode_state_index(i, dim_c, dim_t)
            is_in_CS = (k == 0) and (c < 2) and (t < 2)
            if is_in_CS:
                bitstring_index = c * (2**c_idx) + t * (2**t_idx)
                cs_index_map[i] = bitstring_index
            else:
                state_name = f"{c}{t}{k}"
                self._leakage_state_names.append(state_name)
                self._leakage_probs_by_state[state_name] = np.zeros((seqs, n_depths))

        self._leakage_state_names = sorted(list(set(self._leakage_state_names)))

        # --- 5. Main Data Processing Loop ---
        for s in range(seqs):
            # FIX: Iterate only up to available n_depths
            for d_, depth in enumerate(depths):

                # Get Full Measured Distribution
                full_measured_probs = (
                    np.array([counts[f"s{i}"][s][d_] for i in range(total_dim)]) / self.xeb_config.n_shots
                )

                if not self.xeb_config.disjoint_processing:
                    # --- Joint Fidelity Post-Selection ---
                    total_cs_prob = 0.0
                    renormalized_cs_probs = np.zeros(dim_2N)

                    for i in range(total_dim):
                        prob = full_measured_probs[i]
                        if i in cs_index_map:
                            bitstring_index = cs_index_map[i]
                            renormalized_cs_probs[bitstring_index] = prob
                            total_cs_prob += prob
                        else:
                            c, t, k = self._decode_state_index(i, dim_c, dim_t)
                            state_name = f"{c}{t}{k}"
                            self._leakage_probs_by_state[state_name][s, d_] = prob

                    total_leakage_prob = 1.0 - total_cs_prob

                    if total_cs_prob > 1e-6:
                        renormalized_cs_probs = renormalized_cs_probs / total_cs_prob
                    else:
                        renormalized_cs_probs = incoherent_distribution

                    joint_measured_probs[s, d_] = renormalized_cs_probs
                    self._leakage_probs[s, d_] = total_leakage_prob
                    self._total_cs_probs[s, d_] = total_cs_prob

                else:
                    # --- Disjoint Fidelity Post-Selection ---
                    prob_c = np.zeros(dim_c)
                    prob_t = np.zeros(dim_t)

                    for i in range(total_dim):
                        prob = full_measured_probs[i]
                        c, t, k = self._decode_state_index(i, dim_c, dim_t)
                        prob_c[c] += prob
                        prob_t[t] += prob

                    prob_c_cs = prob_c[0] + prob_c[1]
                    c_renorm = [0.5, 0.5]
                    if prob_c_cs > 1e-6:
                        c_renorm = [prob_c[0] / prob_c_cs, prob_c[1] / prob_c_cs]

                    prob_t_cs = prob_t[0] + prob_t[1]
                    t_renorm = [0.5, 0.5]
                    if prob_t_cs > 1e-6:
                        t_renorm = [prob_t[0] / prob_t_cs, prob_t[1] / prob_t_cs]

                    disjoint_measured_probs[0, s, d_] = np.array(c_renorm)
                    disjoint_measured_probs[1, s, d_] = np.array(t_renorm)

                    self._disjoint_leakage_probs[0, s, d_] = prob_c[2] if dim_c == 3 else 0.0
                    self._disjoint_leakage_probs[1, s, d_] = prob_t[2] if dim_t == 3 else 0.0

                # --- Fidelity Calculation ---
                if not self.xeb_config.disjoint_processing:
                    f_xeb = compute_log_fidelity(
                        incoherent_distribution, joint_expected_probs[s, d_], joint_measured_probs[s, d_]
                    )
                    log_fidelities[s, d_] = evaluate_log_fidelity(f_xeb, singularity, outlier, s, int(depth))
                    records = update_record(
                        records, s, depth, joint_expected_probs[s, d_], joint_measured_probs[s, d_], dim_2N
                    )
                else:
                    qc = self.circuits[s][d_].remove_final_measurements(inplace=False)
                    statevector = Statevector(qc)
                    for q in range(n_qubits):
                        disjoint_expected_probs[q, s, d_] = statevector.probabilities([q], 5)

                    for q, qubit_name in enumerate(self.qubit_names):
                        f_xeb = compute_log_fidelity(
                            incoherent_distribution,
                            disjoint_expected_probs[q, s, d_],
                            disjoint_measured_probs[q, s, d_],
                        )
                        log_fidelities[q, s, d_] = evaluate_log_fidelity(
                            f_xeb, singularity[q], outlier[q], s, int(depth)
                        )
                        records[q] = update_record(
                            records[q],
                            s,
                            depth,
                            disjoint_expected_probs[q, s, d_],
                            disjoint_measured_probs[q, s, d_],
                            2,
                        )

        # --- 7. Linear Fidelity Fitting ---
        def per_cycle_depth(df):
            fid_lsq = df["numerator"].sum() / df["denominator"].sum()
            return pd.Series({"fidelity": fid_lsq})

        if not self.xeb_config.disjoint_processing:
            df = update_data_frame(pd.DataFrame(records))
            linear_fidelities = df.groupby("depth").apply(per_cycle_depth).reset_index()
        else:
            df, linear_fidelities = [], []
            for q in range(n_qubits):
                df_q = update_data_frame(pd.DataFrame(records[q]))
                linear_fidelities.append(df_q.groupby("depth").apply(per_cycle_depth).reset_index())
                df.append(df_q)

        if np.isnan(log_fidelities).all():
            warnings.warn("All fidelities computed from log-entropies are singularities.")

        return (
            joint_measured_probs,
            disjoint_measured_probs,
            joint_expected_probs,
            disjoint_expected_probs,
            df,
            log_fidelities,
            linear_fidelities,
            singularity,
            outlier,
        )

    def get_layer_fidelity(
        self, fidelity_metric: Literal["log", "linear"] = "linear", disjoint_processing: bool = None
    ):
        if disjoint_processing is not None:
            assert isinstance(disjoint_processing, bool), "disjoint_processing should be a boolean"
        else:
            disjoint_processing = self.xeb_config.disjoint_processing

        if disjoint_processing:
            if fidelity_metric == "log":
                Fxeb = np.nanmean(self.log_fidelities, axis=1)
            else:
                Fxeb = np.array([fidelity["fidelity"] for fidelity in self.linear_fidelities])

            a = [None] * len(self.qubit_names)
            layer_fid = [None] * len(self.qubit_names)
            for q, qubit in enumerate(self.qubit_names):
                a[q], layer_fid[q], *_ = fit_exponential_decay(self.xeb_config.depths, Fxeb[q])
        else:
            if fidelity_metric == "log":
                Fxeb = np.nanmean(self.log_fidelities, axis=0)
            else:
                Fxeb = np.array(self.linear_fidelities["fidelity"])
            a, layer_fid, *_ = fit_exponential_decay(self.xeb_config.depths, Fxeb)
        return layer_fid

    def plot_fidelities(self, fit_linear: bool = True, fit_log_entropy: bool = True, separate_plots: bool = False):
        figs = [plt.figure()]
        plt.rcParams["text.usetex"] = False

        def plot_fidelity_data(xx, depths, linear_fidelities, Fxeb, qubit_label=""):
            try:
                if fit_linear:
                    a_lin, layer_fid_lin, *_ = fit_exponential_decay(
                        linear_fidelities["depth"], linear_fidelities["fidelity"]
                    )
                    plt.plot(
                        xx,
                        exponential_decay(xx, a_lin, layer_fid_lin),
                        label=f"Fit (Linear XEB{qubit_label}), layer_fidelity={layer_fid_lin * 100:.2f}% (error: {1-layer_fid_lin:.1e})",
                    )
            except Exception:
                warnings.warn("Fit for Linear XEB data failed")

            try:
                if fit_log_entropy:
                    a_log, layer_fid_log, *_ = fit_exponential_decay(depths, Fxeb)
                    plt.plot(
                        xx,
                        exponential_decay(xx, a_log, layer_fid_log),
                        label=f"Fit (Log XEB{qubit_label}), layer_fidelity={layer_fid_log * 100:.2f}% (error: {1-layer_fid_log:.1e})",
                    )
            except Exception:
                warnings.warn("Fit for Log XEB data failed")

            if fit_linear:
                mask_lin = (linear_fidelities["fidelity"] > 0) & (linear_fidelities["fidelity"] <= 1)
                plt.scatter(
                    linear_fidelities["depth"][mask_lin],
                    linear_fidelities["fidelity"][mask_lin],
                    label=f"Linear XEB Data {qubit_label}",
                )

            if fit_log_entropy and not np.isnan(Fxeb).all():
                mask_log = (Fxeb > 0) & (Fxeb <= 1)
                plt.scatter(depths[mask_log], Fxeb[mask_log], label=f"Log XEB Data {qubit_label}", s=13.5, c="blue")
            else:
                warnings.warn(f"Log XEB data for {qubit_label} is a singularity.")

        if self.xeb_config.disjoint_processing:
            for q, qubit in enumerate(self.qubit_names):
                if separate_plots and q > 0:
                    figs.append(plt.figure())
                linear_fidelities = self.linear_fidelities[q]
                xx = np.linspace(0, linear_fidelities["depth"].max())
                Fxeb = np.nanmean(self.log_fidelities[q], axis=0)
                plot_fidelity_data(xx, self.xeb_config.depths, linear_fidelities, Fxeb, f" {qubit}")
                plt.ylabel("Circuit fidelity")
                plt.xlabel("Cycle Depth $d$")
                plt.title("XEB Fidelity")
                plt.legend(loc="best")
                if separate_plots:
                    plt.tight_layout()
                    plt.show()
        else:
            xx = np.linspace(0, self.linear_fidelities["depth"].max())
            Fxeb = np.nanmean(self.log_fidelities, axis=0)
            plot_fidelity_data(xx, self.xeb_config.depths, self.linear_fidelities, Fxeb)
            plt.ylabel("Circuit fidelity")
            plt.xlabel("Cycle Depth $d$")
            plt.title("XEB Fidelity")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.show()

        return figs

    def plot_records(self):
        depths = self.xeb_config.depths
        colors = sns.cubehelix_palette(n_colors=len(depths))
        colors = {k: colors[i] for i, k in enumerate(depths)}
        _lines = []

        def per_cycle_depth(df, _lines=None):
            fid_lsq = df["numerator"].sum() / df["denominator"].sum()
            cycle_depth = df.name
            xx = np.linspace(0, df["x"].max())
            (l,) = plt.plot(xx, fid_lsq * xx, color=colors[cycle_depth])
            plt.scatter(df["x"], df["y"], color=colors[cycle_depth])
            _lines += [l]
            return pd.Series({"fidelity": fid_lsq})

        if not self.xeb_config.disjoint_processing:
            plt.figure()
            fids = self.records.groupby("depth").apply(per_cycle_depth, _lines).reset_index()
            plt.xlabel(r"$e_U - u_U$", fontsize=18)
            plt.ylabel(r"$m_U - u_U$", fontsize=18)
            _lines = np.asarray(_lines)
            plt.legend(_lines[[0, -1]], depths[[0, -1]], loc="best", title="Cycle depth")
            title = "Fxeb_linear = %s" % [fids["fidelity"][x] for x in [0, 1]]
            plt.title(title)
            plt.tight_layout()
        else:
            fids = []
            for i, q in enumerate(self.qubit_names):
                _lines = []
                plt.figure()
                fids.append(self.records[i].groupby("depth").apply(per_cycle_depth, _lines).reset_index())
                plt.xlabel(r"$e_U - u_U$", fontsize=18)
                plt.ylabel(r"$m_U - u_U$", fontsize=18)
                _lines = np.asarray(_lines)
                plt.legend(_lines[[0, -1]], depths[[0, -1]], loc="best", title="Cycle depth")
                plt.title(
                    "q-%s: Fxeb_linear = %s"
                    % (
                        q,
                        [fids[i]["fidelity"][x] for x in [0, 1]],
                    )
                )
                plt.show()
        return plt.gcf()

    # --- [USER'S CODE] Plotting Logic (Leakage/Stabilization) ---
    def plot_state_heatmap(self):
        titles, data = [], []
        if not self.xeb_config.disjoint_processing:
            dim_2N = 2**self.xeb_config.n_qubits
            for i in range(dim_2N):
                state_str = binary(i, self.xeb_config.n_qubits)
                titles.append(f"Measured |{state_str}>")
                titles.append(f"Expected |{state_str}>")
                data.append(self.measured_probs[:, :, i])
                data.append(self.expected_probs[:, :, i])
        else:
            for i, q in enumerate(self.qubit_names):
                for j in range(2):
                    titles.append(f"{q} |{j}> Measured")
                    titles.append(f"{q} |{j}> Expected")
                    data.append(self.disjoint_measured_probs[i, :, :, j])
                    data.append(self.disjoint_expected_probs[i, :, :, j])

        num_plots = len(titles)
        if num_plots == 0:
            return plt.figure()

        plots_per_fig = 8
        num_figs = (num_plots + plots_per_fig - 1) // plots_per_fig
        figs = []

        for fig_idx in range(num_figs):
            fig, axs = plt.subplots(4, 2, figsize=(10, 8))
            figs.append(fig)
            axs = axs.flatten()
            start_idx = fig_idx * plots_per_fig
            end_idx = min(start_idx + plots_per_fig, num_plots)

            for plot_idx in range(start_idx, end_idx):
                ax_idx = plot_idx - start_idx
                ax = axs[ax_idx]

                plot_data = np.abs(data[plot_idx])
                finite_data = plot_data[np.isfinite(plot_data)]
                if finite_data.size > 0:
                    vmax = np.percentile(finite_data, 99)
                    if (vmax < 0.01) or (not np.isfinite(vmax)):
                        vmax = 0.01
                else:
                    vmax = 0.01

                pcm = ax.pcolor(
                    self.xeb_config.depths,
                    range(self.xeb_config.seqs),
                    plot_data,
                    vmin=0,
                    vmax=vmax,
                    cmap="viridis",
                )
                fig.colorbar(pcm, ax=ax, label="Probability")

                ax.set_title(titles[plot_idx])
                ax.set_xlabel("Circuit depth")
                ax.set_ylabel("Sequences")
                ax.set_xticks(self.xeb_config.depths)
                ax.set_yticks([0, self.xeb_config.seqs - 1], [1, self.xeb_config.seqs])

            for ax_idx in range(end_idx - start_idx, len(axs)):
                axs[ax_idx].set_visible(False)

            fig.suptitle("Computational State Heatmaps")
            plt.tight_layout()
            plt.show()

        return figs

    def plot_leakage(self):
        depths = self.xeb_config.depths
        if self.xeb_config.disjoint_processing:
            plt.figure()
            avg_leakage_c = np.mean(self._disjoint_leakage_probs[0], axis=0)
            avg_leakage_t = np.mean(self._disjoint_leakage_probs[1], axis=0)
            plt.plot(depths, avg_leakage_c, "o-", label="Control Qubit Leakage (P(c=2))")
            plt.plot(depths, avg_leakage_t, "s-", label="Target Qubit Leakage (P(t=2))")
            plt.title("Disjoint Leakage vs. Circuit Depth")
            plt.xlabel("Cycle Depth $d$")
            plt.ylabel("Leakage Probability")
            plt.legend()
            plt.grid(True, which="both")
            return plt.gcf()
        else:
            if not self._leakage_probs_by_state:
                return plt.figure()

            n_seqs = self.xeb_config.seqs
            n_depths = len(depths)
            control_leakage = np.zeros((n_seqs, n_depths))
            target_leakage = np.zeros((n_seqs, n_depths))
            coupler_leakage = np.zeros((n_seqs, n_depths))

            control_states_found = []
            target_states_found = []
            coupler_states_found = []

            for state_name, prob_array in self._leakage_probs_by_state.items():
                try:
                    c, t, k = int(state_name[0]), int(state_name[1]), int(state_name[2])
                except Exception:
                    continue
                if c >= 2:
                    control_leakage += prob_array
                    control_states_found.append(state_name)
                if t >= 2:
                    target_leakage += prob_array
                    target_states_found.append(state_name)
                if k >= 1:
                    coupler_leakage += prob_array
                    coupler_states_found.append(state_name)

            plot_list = []
            if np.sum(control_leakage) > 0:
                plot_list.append(("Control Leakage", control_leakage, control_states_found))
            if np.sum(target_leakage) > 0:
                plot_list.append(("Target Leakage", target_leakage, target_states_found))
            if np.sum(coupler_leakage) > 0:
                plot_list.append(("Coupler Leakage", coupler_leakage, coupler_states_found))

            if not plot_list:
                return plt.figure()

            n_plots = len(plot_list)
            fig, axs = plt.subplots(n_plots, 1, figsize=(8, n_plots * 3.5), squeeze=False)
            axs_flat = axs.flatten()

            # for i, (title, data, states_found) in enumerate(plot_list):
            #     ax = axs_flat[i]
            #     avg_data = np.mean(data, axis=0)
            #     ax.plot(depths, avg_data, "o-")
            #     states_str = ", ".join(sorted(list(set(states_found))))
            #     ax.set_title(f"{title}\n(Sum of states: |{states_str}>)")
            #     ax.set_xlabel("Cycle Depth $d$")
            #     ax.set_ylabel("Avg. Probability")
            #     ax.grid(True, which="both")

            # fig.suptitle("Component Leakage Probability vs. Circuit Depth", fontsize=16)
            # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            # plt.show()
            # return fig

    def plot_leakage_heatmap(self):
        if self.xeb_config.disjoint_processing:
            warnings.warn("Leakage heatmap is only supported for joint processing.")
            return plt.figure()

        titles, data = [], []
        for state_name in self._leakage_state_names:
            titles.append(f"Measured |{state_name}> (C,T,Cplr)")
            data.append(self._leakage_probs_by_state[state_name])

        num_plots = len(titles)
        if num_plots == 0:
            return plt.figure()

        plots_per_fig = 8
        num_figs = (num_plots + plots_per_fig - 1) // plots_per_fig
        figs = []

        for fig_idx in range(num_figs):
            fig, axs = plt.subplots(4, 2, figsize=(10, 8))
            figs.append(fig)
            axs = axs.flatten()
            start_idx = fig_idx * plots_per_fig
            end_idx = min(start_idx + plots_per_fig, num_plots)

            for plot_idx in range(start_idx, end_idx):
                ax_idx = plot_idx - start_idx
                ax = axs[ax_idx]

                plot_data = np.abs(data[plot_idx])
                vmax = np.percentile(plot_data[np.isfinite(plot_data)], 99)
                if (vmax < 0.01) or (not np.isfinite(vmax)):
                    vmax = 0.01

                pcm = ax.pcolor(
                    self.xeb_config.depths,
                    range(self.xeb_config.seqs),
                    plot_data,
                    vmin=0,
                    vmax=vmax,
                    cmap="Reds",
                )
                fig.colorbar(pcm, ax=ax, label="Probability")
                ax.set_title(titles[plot_idx])
                ax.set_xlabel("Circuit depth")
                ax.set_ylabel("Sequences")
                ax.set_xticks(self.xeb_config.depths)
                ax.set_yticks([0, self.xeb_config.seqs - 1], [1, self.xeb_config.seqs])

            for ax_idx in range(end_idx - start_idx, len(axs)):
                axs[ax_idx].set_visible(False)

            fig.suptitle("Leakage State Heatmaps")
            plt.tight_layout()
            plt.show()

        return figs

    def plot_leakage_vs_sequence(self):
        if self.xeb_config.disjoint_processing or self._leakage_probs is None:
            return plt.figure()

        fig, ax = plt.subplots(figsize=(10, 6))
        avg_leakage_vs_seq = np.mean(self._leakage_probs, axis=1)
        ax.plot(np.arange(self.xeb_config.seqs), avg_leakage_vs_seq, "o-")
        ax.set_xlabel("Sequence Index")
        ax.set_ylabel("Average Total Leakage Probability")
        ax.set_title("Total Leakage Drift vs. Sequence")
        ax.grid(True)
        plt.tight_layout()
        plt.show()
        return fig

    def calculate_purity(self):
        """
        Calculate Speckle Purity using F. Arute et al., Nature 574, 505 (2019) Eq. (49).
        Purity ~ Var(P) * D^2 * (D+1) / (D-1)
        """
        # 1. Get Dimensions
        n_qubits = self.xeb_config.n_qubits
        D = 2**n_qubits

        # 2. Prepare Data: We need the variance across sequences AND bitstrings per depth.
        # self.measured_probs has shape (seqs, depths, dim)
        # We transpose to (seqs, dim, depths) then flatten (seqs*dim, depths)
        measured_probs_flattened = self.measured_probs.transpose(0, 2, 1).reshape(-1, len(self.xeb_config.depths))

        # 3. Calculate Variance across the ensemble (axis 0)
        # This calculates Var(p) for each depth
        var_p = np.var(measured_probs_flattened, axis=0)

        # 4. Apply Eq. 49
        self.average_state_purity = var_p * (D**2 * (D + 1) / (D - 1))

        # 5. Fit Decay
        # Note: We fit sqrt(Purity) usually, but here we fit Purity directly to get error per cycle
        self.a_purity, self.purity_error_per_cycle, *_ = fit_exponential_decay(
            self.xeb_config.depths, self.average_state_purity
        )

        return self.average_state_purity

    def plot_fidelity_and_purity(self, log_yscale: bool = False):
        """
        Plots XEB Fidelity alongside sqrt(Purity) on a single figure.
        Robust to fitting failures.
        """
        from scipy.optimize import curve_fit

        # --- SAFEGUARDS INITIALIZATION ---
        e_xeb = None
        e_pur = None

        # Safely calculate purity
        if not hasattr(self, "average_state_purity") or self.average_state_purity is None:
            try:
                self.calculate_purity()
            except Exception as e:
                print(f"Warning: Purity calculation failed: {e}")
                self.average_state_purity = None

        xs = self.xeb_config.depths
        plt.figure(figsize=(8, 6))

        # --- 1. Prepare Data ---
        if isinstance(self.linear_fidelities, pd.DataFrame):
            xeb_data = self.linear_fidelities["fidelity"].values
        else:
            xeb_data = self.linear_fidelities

        # Check if we have valid purity data
        if self.average_state_purity is not None:
            purity_sqrt_data = np.sqrt(np.abs(self.average_state_purity))
        else:
            purity_sqrt_data = None

        # --- 2. Fit Mask (Depth >= 6) ---
        mask = xs >= 6
        if np.sum(mask) < 3:
            # print("Warning: Not enough points with depth >= 6. Fitting all data.")
            mask = np.ones_like(xs, dtype=bool)

        xs_fit = xs[mask]

        # --- 3. Fit Functions ---
        def exp_decay_zero(x, a, r):
            return a * (r**x)

        def exp_decay_offset(x, a, r, b):
            return a * (r**x) + b

        # --- 4. Perform Fits ---

        # A. Linear XEB Fit (Total Error)
        try:
            # Check for NaNs
            if np.any(np.isnan(xeb_data[mask])):
                raise ValueError("XEB data contains NaNs")

            p0 = [1.0, 0.95]
            popt_xeb, _ = curve_fit(exp_decay_zero, xs_fit, xeb_data[mask], p0=p0, bounds=([0, 0], [2, 1]))
            a_xeb, p_xeb = popt_xeb
            e_xeb = 1 - p_xeb

            plt.plot(xs, xeb_data, "o", color="C0", label="Linear XEB (Total)")
            plt.plot(
                xs, exp_decay_zero(xs, *popt_xeb), "-", color="C0", label=f"XEB Fit: $e_{{cyc}} = {e_xeb*100:.2f}\%$"
            )
        except Exception as e:
            print(f"XEB Fit failed: {e}")
            plt.plot(xs, xeb_data, "o", color="C0", label="Linear XEB (Fit Failed)")

        # B. Purity Fit (Incoherent Error)
        b_pur = 0.0  # Default asymptote for plotting
        if purity_sqrt_data is not None:
            try:
                y_fit = purity_sqrt_data[mask]

                # Check for NaNs or too few points
                if len(y_fit) < 3 or np.any(np.isnan(y_fit)):
                    raise ValueError("Insufficient or invalid purity data for fitting")

                # Guess: A=Range, r=0.95, B=Min
                p0 = [np.max(y_fit) - np.min(y_fit), 0.98, np.min(y_fit)]

                popt_pur, _ = curve_fit(exp_decay_offset, xs_fit, y_fit, p0=p0, bounds=([0, 0, 0], [2, 1, 1]))
                a_pur, p_pur, b_pur = popt_pur
                e_pur = 1 - p_pur

                # Plot
                if log_yscale:
                    y_plot_pur = purity_sqrt_data - b_pur
                    y_plot_pur[y_plot_pur <= 1e-5] = np.nan
                    label_pur = r"$\sqrt{\mathrm{Purity}} - B_{fit}$"
                    fit_plot = exp_decay_zero(xs, a_pur, p_pur)
                else:
                    y_plot_pur = purity_sqrt_data
                    label_pur = r"$\sqrt{\mathrm{Purity}}$"
                    fit_plot = exp_decay_offset(xs, *popt_pur)

                plt.plot(xs, y_plot_pur, "s", color="C1", label=label_pur)
                plt.plot(xs, fit_plot, "--", color="C1", label=f"Purity Fit: $e_{{cyc}} = {e_pur*100:.2f}\%$")

                print(f"Purity Fit Asymptote (B): {b_pur:.4f}")

            except Exception as e:
                print(f"Purity Fit failed: {e}")
                plt.plot(xs, purity_sqrt_data, "s", color="C1", label="sqrt(Purity) (Fit Failed)")
        else:
            print("Skipping Purity Plot (No Data)")

        # --- 5. Formatting ---
        plt.xlabel("Circuit Depth (Cycles)")

        if log_yscale:
            plt.ylabel(r"Signal Magnitude (Log Scale)")
            plt.yscale("log")
        else:
            plt.ylabel(r"Signal Magnitude (Linear Scale)")

        # Build Title safely
        title_str = "XEB Error Budget (Fit $d \\geq 6$)"
        if e_xeb is not None and e_pur is not None:
            # Only show subtraction if both valid
            ctrl_err = max(0, e_xeb - e_pur)  # Ensure non-negative
            title_str += f"\nControl Err $\\approx$ {ctrl_err*100:.2f}%"
        elif e_xeb is not None:
            title_str += f"\nTotal Err $\\approx$ {e_xeb*100:.2f}%"
        else:
            title_str += "\n(Fits Failed)"

        plt.title(title_str)
        plt.legend(loc="best", frameon=True, fontsize="small")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.tight_layout()

        return plt.gcf()

    def _perform_supervised_discrimination(self):
        """
        Performs Supervised Discrimination using pre-calibrated GEF centers and standard deviations.

        1. Fetches centers (gef_centers) and sigmas (extras['std_dev_X']) from the qubit objects.
        2. Calculates the likelihood P(IQ | State) for each shot.
        3. Normalizes to get posterior P(State | IQ).
        4. Rebuilds the joint counts dictionary using these probabilities.
        """
        print("Performing Supervised Discrimination (Pre-calibrated Centers)...")

        # 1. Setup Dimensions
        n_seqs = self.xeb_config.seqs
        n_depths = len(self.xeb_config.depths)
        n_shots = self.xeb_config.n_shots
        total_dim = self.xeb_config.total_dim

        # Prepare Count Storage (Floats allowed for soft counts)
        new_counts = {f"s{i}": np.zeros((n_seqs, n_depths), dtype=float) for i in range(total_dim)}

        def get_params(qubit, n_states):
            """Fetch centers and std_devs from qubit object."""
            if not hasattr(qubit.resonator, "gef_centers"):
                print(f"Warning: {qubit.name} has no 'gef_centers'. Cannot perform supervised discrimination.")
                return None, None

            # Retrieve centers
            centers = np.array(qubit.resonator.gef_centers)[:n_states]

            # Retrieve sigmas from extras
            sigmas = []
            if hasattr(qubit, "extras"):
                for k in range(n_states):
                    key = f"std_dev_{k}"
                    if key in qubit.extras:
                        sigmas.append(qubit.extras[key])
                    else:
                        print(f"Warning: {qubit.name} missing '{key}'. Assuming 0.1 (a.u.).")
                        sigmas.append(0.1)  # Fallback
            else:
                print(f"Warning: {qubit.name} has no 'extras'. Assuming 0.1 for all.")
                sigmas = [0.1] * n_states

            return centers, np.array(sigmas)

        def calculate_probs(i_data, q_data, centers, sigmas):
            """Calculate P(State | IQ) for a single channel."""
            # Flatten data
            i_flat = i_data.flatten()
            q_flat = q_data.flatten()
            points = np.column_stack((i_flat, q_flat))

            n_samples = len(points)
            n_states = len(centers)

            likelihoods = np.zeros((n_samples, n_states))

            for k in range(n_states):
                mu = centers[k]
                sigma = sigmas[k]
                var = sigma**2

                # Gaussian PDF (ignoring 2pi factor as it cancels in normalization)
                # P(z|k) ~ (1/sigma^2) * exp(-|z-mu|^2 / 2sigma^2)
                d2 = (points[:, 0] - mu[0]) ** 2 + (points[:, 1] - mu[1]) ** 2
                prefactor = 1.0 / (var + 1e-12)
                likelihoods[:, k] = prefactor * np.exp(-d2 / (2 * var))

            # Normalize to get Posterior P(k|z)
            total_likelihood = np.sum(likelihoods, axis=1, keepdims=True)
            # Avoid divide by zero
            total_likelihood[total_likelihood == 0] = 1e-12

            probs = likelihoods / total_likelihood

            # Reshape to (Seq, Depth, Shots, States)
            return probs.reshape(n_seqs, n_depths, n_shots, n_states)

        # 2. Process Channels
        pair = self.xeb_config.qubit_pairs[0]
        q_c = pair.qubit_control
        q_t = pair.qubit_target

        # --- Control ---
        dim_c = self.xeb_config.dim_c
        c_centers, c_sigmas = get_params(q_c, dim_c)
        if c_centers is None:
            return  # Abort if no calibration

        probs_c = calculate_probs(self.data["I_c_all"], self.data["Q_c_all"], c_centers, c_sigmas)

        # --- Target ---
        dim_t = self.xeb_config.dim_t
        t_centers, t_sigmas = get_params(q_t, dim_t)
        probs_t = calculate_probs(self.data["I_t_all"], self.data["Q_t_all"], t_centers, t_sigmas)

        # --- Coupler ---
        # Coupler readout is disabled, use dummy prob=1 for state 0
        probs_k = np.zeros((n_seqs, n_depths, n_shots, 1))
        probs_k[..., 0] = 1.0

        # Save params for visualization
        self.data["gmm_params_c"] = {"means": c_centers, "covs": [np.eye(2) * s**2 for s in c_sigmas]}
        self.data["gmm_params_t"] = {"means": t_centers, "covs": [np.eye(2) * s**2 for s in t_sigmas]}

        # 3. Calculate Joint Counts
        print("Calculating Joint Probabilities (Supervised)...")

        for i in range(total_dim):
            c_idx = i % dim_c
            t_idx = (i // dim_c) % dim_t
            k_idx = i // (dim_c * dim_t)

            # Joint Probability P(c,t,k) = P(c) * P(t) * P(k)
            p_shot = probs_c[..., c_idx] * probs_t[..., t_idx] * probs_k[..., k_idx]

            # Sum over shots to get soft counts
            new_counts[f"s{i}"] = np.sum(p_shot, axis=2)

        # 4. Update Results
        self.counts = new_counts
        self.data.update(new_counts)
        print("Supervised Discrimination complete.")

    def calc_populations_by_fitting(self, coupler_override=None):
        """
        Calculates state populations (P0, P1, P2) by fitting the aggregate 2D Histogram
        of the IQ data at each depth to a sum of fixed Gaussians.

        Modification: If the 3rd center (F-state) is [0,0], it is ignored in the fit,
        but the output array retains 3 columns (the 3rd column will be 0.0).

        Returns:
            dict: {
                "control": (n_depths, 3) array of populations,
                "target": (n_depths, 3) array of populations,
                "coupler": (n_depths, 3) array (if enabled)
            }
        """
        from scipy.optimize import nnls

        print("Calculating populations via 2D Histogram Amplitude Fitting...")

        # Configuration
        depths = self.xeb_config.depths
        n_depths = len(depths)
        n_seqs = self.xeb_config.seqs
        n_shots = self.xeb_config.n_shots
        bins = 50  # Resolution of the histogram grid

        results = {}

        # Helper to generate a 2D Gaussian Template on a grid
        def make_gaussian_template(xx, yy, mu, sigma):
            d2 = (xx - mu[0]) ** 2 + (yy - mu[1]) ** 2
            g = np.exp(-d2 / (2 * sigma**2))
            return g.flatten()

        # Helper to process one qubit channel
        def process_channel(I_all, Q_all, qubit, n_states):
            # Output storage: [Depth, State]
            # We always initialize with the requested n_states (e.g., 3)
            pops_vs_depth = np.zeros((n_depths, n_states))

            # 1. Validate Data & Reshape to (Seq, Depth, Shot)
            try:
                I_3d = I_all.reshape(n_seqs, n_depths, n_shots)
                Q_3d = Q_all.reshape(n_seqs, n_depths, n_shots)
            except ValueError:
                print(f"Error reshaping data for {getattr(qubit, 'name', 'Unknown')}. Check n_seqs/n_depths/n_shots.")
                return pops_vs_depth

            # 2. Get Reference Params
            if not hasattr(qubit, "resonator") or not hasattr(qubit.resonator, "gef_centers"):
                return pops_vs_depth

            # Retrieve centers and sigmas
            centers = np.array(qubit.resonator.gef_centers)[:n_states]
            sigmas = []
            for k in range(n_states):
                if hasattr(qubit, "extras"):
                    sigmas.append(qubit.extras.get(f"std_dev_{k}", 0.1))
                else:
                    sigmas.append(0.1)

            # --- MODIFICATION: Check for dummy F-center [0,0] ---
            # n_fit determines how many gaussians we actually try to fit.
            n_fit = n_states
            if n_states >= 3:
                # If the F-center (index 2) is at the origin, assume it's uncalibrated/dummy.
                if np.allclose(centers[2], [0, 0], atol=1e-6):
                    n_fit = 2
                    # print(f"Ignoring F-state for {qubit.name} (center is [0,0]). Fitting 2 states.")

            # Slice the parameters to only include valid states
            fit_centers = centers[:n_fit]
            fit_sigmas = sigmas[:n_fit]
            # ----------------------------------------------------

            # 3. Determine global grid bounds
            i_flat, q_flat = I_all.flatten(), Q_all.flatten()
            mask = np.isfinite(i_flat) & np.isfinite(q_flat)
            if not np.any(mask):
                return pops_vs_depth

            i_min, i_max = i_flat[mask].min(), i_flat[mask].max()
            q_min, q_max = q_flat[mask].min(), q_flat[mask].max()

            margin_i = (i_max - i_min) * 0.1
            margin_q = (q_max - q_min) * 0.1
            extent = [[i_min - margin_i, i_max + margin_i], [q_min - margin_q, q_max + margin_q]]

            # Create Grid
            x_edges = np.linspace(extent[0][0], extent[0][1], bins + 1)
            y_edges = np.linspace(extent[1][0], extent[1][1], bins + 1)
            xc = (x_edges[:-1] + x_edges[1:]) / 2
            yc = (y_edges[:-1] + y_edges[1:]) / 2
            xx, yy = np.meshgrid(xc, yc)

            # 4. Pre-compute Gaussian Templates (only for n_fit states)
            A = np.zeros((bins * bins, n_fit))
            for k in range(n_fit):
                A[:, k] = make_gaussian_template(xx, yy, fit_centers[k], fit_sigmas[k])

            # 5. Loop over Depths
            for d_idx in range(n_depths):
                # Slice the 3D array (Seq, Depth, Shot) -> Flatten Seqs/Shots
                i_d = I_3d[:, d_idx, :].flatten()
                q_d = Q_3d[:, d_idx, :].flatten()

                # Histogram
                hist, _, _ = np.histogram2d(i_d, q_d, bins=[x_edges, y_edges])
                b = hist.T.flatten()

                # Solve NNLS
                weights, _ = nnls(A, b)

                total = np.sum(weights)
                if total > 0:
                    # Fill only the fitted columns (e.g. 0 and 1).
                    # Column 2 (F) remains 0.0 if n_fit=2.
                    pops_vs_depth[d_idx, :n_fit] = weights / total

            return pops_vs_depth

        # Process Control
        results["control"] = process_channel(
            self.data["I_c_all"],
            self.data["Q_c_all"],
            self.xeb_config.qubit_pairs[0].qubit_control,
            self.xeb_config.dim_c,
        )

        # Process Target
        results["target"] = process_channel(
            self.data["I_t_all"],
            self.data["Q_t_all"],
            self.xeb_config.qubit_pairs[0].qubit_target,
            self.xeb_config.dim_t,
        )

        # Coupler readout is disabled

        return results

    def calculate_normalized_purity(self):
        """
        Calculates a modified Purity metric by:
        1. Normalizing each bitstring's probability trace to its own [min, max] range.
        2. Calculating the variance of that bitstring across sequences.
        3. Averaging these variances.

        This is robust to readout bias (static offsets) and readout inefficiency (attenuation).
        """
        from calibration_utils.two_qubit_xeb.macros import fit_exponential_decay

        # 1. Get Measured Probs: Shape (Seqs, Depths, Dim)
        probs = self.measured_probs
        n_seqs, n_depths, dim = probs.shape

        normalized_variances = []

        for k in range(dim):
            # Extract data for bitstring k -> Shape: (Seqs, Depths)
            p_k = probs[:, :, k]

            # Determine scaling factors (Global Max/Min per bitstring)
            # We must use global range to preserve the decay signal over depth.
            global_min = np.min(p_k)
            global_max = np.max(p_k)
            global_range = global_max - global_min

            if global_range > 1e-12:
                # Normalize this bitstring's data to [0, 1]
                p_k_norm = (p_k - global_min) / global_range

                # Calculate Variance across SEQUENCES (axis 0)
                # Result is a decay curve of shape (Depths,)
                var_k = np.var(p_k_norm, axis=0)

                normalized_variances.append(var_k)
            else:
                # Bitstring signal is constant (dead or zero), skip contribution
                pass

        if normalized_variances:
            # Average the variances over all valid bitstrings
            self.average_state_purity = np.mean(normalized_variances, axis=0)

            # Fit the decay of this new metric
            # Note: The absolute amplitude 'a_purity' is arbitrary due to normalization,
            # but 'purity_error_per_cycle' (decay rate) is the physically relevant quantity.
            self.a_purity, self.purity_error_per_cycle, *_ = fit_exponential_decay(
                self.xeb_config.depths, self.average_state_purity
            )
            return self.average_state_purity
        else:
            print("Warning: No valid bitstring data for normalized purity.")
            return None

    def calc_crosss_entropy(self, p, q):
        q = np.maximum(q, 1e-15)
        cross_entropy = -np.sum(p * np.log(q), axis=-1)
        return cross_entropy

    def erf_average_cross_entropy(self, prms):
        theta_iswap, phi_cphase, phi_rz1, phi_rz2 = prms

        ideal_probability_s = calc_ideal_probability_numpy(
            self.data["gate_indices"],  # NOTE: Use self.data['gate_indices'] for XEBResult
            self.xeb_config.depths,
            theta_iswap=theta_iswap,
            phi_cphase=phi_cphase,
            phi_rz1=phi_rz1,
            phi_rz2=phi_rz2,
        )

        # Match shapes
        n_meas = self.measured_probs.shape[1]
        n_exp = ideal_probability_s.shape[1]
        n_conf = len(self.xeb_config.depths)
        limit = min(n_meas, n_exp, n_conf)

        m_probs = self.measured_probs[:, :limit, :]
        e_probs = ideal_probability_s[:, :limit, :]

        cross_entropy_s = self.calc_crosss_entropy(m_probs, e_probs)
        total_loss = np.nanmean(cross_entropy_s)
        return total_loss

    def estimate_2q_unitary(
        self, en_plot=True, method: Literal["Nelder-Mead", "differential evolution"] = "differential evolution"
    ):
        from scipy.optimize import minimize, differential_evolution

        def record_loss(prms, convergence=None):  # callback func
            current_loss = self.erf_average_cross_entropy(prms)
            loss_history.append(current_loss)
            prms_history.append(prms.copy())

        loss_history = []
        prms_history = []

        if method == "Nelder-Mead":
            prms0 = [0, np.pi, 0, 0]  # assuming CZ gate
            res = minimize(
                self.erf_average_cross_entropy,
                prms0,
                method="Nelder-Mead",
                options={"disp": True},
                callback=record_loss,
            )
        elif method == "differential evolution":
            margin = 0.1
            bounds = [
                (-np.pi / 2 - margin, np.pi / 2 + margin),
                (-np.pi - margin, np.pi + margin),
                (-np.pi - margin, np.pi + margin),
                (-np.pi - margin, np.pi + margin),
            ]
            res = differential_evolution(
                self.erf_average_cross_entropy,
                bounds,
                strategy="best1bin",
                maxiter=1000,
                popsize=15,
                disp=True,
                callback=record_loss,
                tol=1e-4,
            )

        self.theta_iswap_opt, self.phi_cphase_opt, self.phi_rz1_opt, self.phi_rz2_opt = res.x

        fig = None
        if en_plot:
            prm_names = ["iSWAP", "CPhase", "RZ1", "RZ2"]
            fig = plt.figure()
            plt.subplot(211)
            plt.plot(loss_history)
            plt.text(0.98, 0.98, f"Final: {loss_history[-1]:.4f}", ha="right", va="top", transform=plt.gca().transAxes)
            plt.ylabel("Average Cross Entropy")
            plt.grid(True)
            plt.title("Convergence Plot of 2Q-Unitary Estimation")
            plt.subplot(212)
            for _i, _name in enumerate(prm_names):
                if len(prms_history) > 0:
                    plt.plot(np.array(prms_history)[:, _i] / np.pi, label=_name)
            plt.xlabel("Iteration")
            plt.ylabel("2Q Unitary Parameter (π rad)")
            plt.grid(True)
            plt.legend()

        # Update expected probs with optimized parameters
        self.expected_probs_opt = calc_ideal_probability_numpy(
            self.data["gate_indices"],
            self.xeb_config.depths,
            self.theta_iswap_opt,
            self.phi_cphase_opt,
            self.phi_rz1_opt,
            self.phi_rz2_opt,
        )

        # NOTE: You need to make sure calculate_linear_XEB_fidelity is available or copied too,
        # otherwise just skip the fidelity recalculation lines below if not needed.

        return fig

    def plot_fidelity_vs_depth_opt(self, data_to_plot="log", compare=True, plot_purity=True):
        """
        Plots the fidelity vs depth with optimized 2Q unitary parameters.
        Optionally overlays the Purity (Speckle) metric.
        """
        xs = self.xeb_config.depths
        i_color = 0
        zorder = 0
        plt.figure(figsize=(8, 6))

        # --- 1. Prepare INITIAL Data (Adaptation for XEBResult structure) ---
        # Calculate averages from the raw data stored in XEBResult
        if hasattr(self, "log_fidelities"):
            log_fid_avg = np.nanmean(self.log_fidelities, axis=0)
            log_fid_std = np.nanstd(self.log_fidelities, axis=0)
        else:
            log_fid_avg = np.zeros_like(xs)
            log_fid_std = np.zeros_like(xs)

        if isinstance(self.linear_fidelities, pd.DataFrame):
            lin_fid = np.array(self.linear_fidelities["fidelity"])
            lin_fid_std = np.zeros_like(xs)  # Std not usually stored in the DataFrame
        else:
            lin_fid = self.linear_fidelities
            lin_fid_std = getattr(self, "linear_XEB_fidelity_std", np.zeros_like(xs))

        # Perform fits on Initial Data (since XEBResult calculates these on the fly)
        try:
            a_log_init, log_fid_layer_init, *_ = fit_exponential_decay(xs, log_fid_avg)
            a_lin_init, lin_fid_layer_init, *_ = fit_exponential_decay(xs, lin_fid)
        except:
            a_log_init, log_fid_layer_init = 1.0, 0.0
            a_lin_init, lin_fid_layer_init = 1.0, 0.0

        # --- 2. Plotting ---

        # LOG FIDELITY
        if "log" in data_to_plot:
            if compare:
                # Plot Initial
                plt.errorbar(
                    xs,
                    log_fid_avg,
                    yerr=log_fid_std,
                    fmt="o",
                    label="Log XEB Data (Initial)",
                    color=f"C{i_color}",
                    zorder=zorder,
                    alpha=0.6,
                )
                plt.plot(
                    xs,
                    exponential_decay(xs, a_log_init, log_fid_layer_init),
                    label=f"Fit (Initial): {log_fid_layer_init*100:.2f}% (err: {1-log_fid_layer_init:.1e})",
                    color=f"C{i_color}",
                    linestyle="--",
                    zorder=zorder,
                )
                i_color += 1

            # Plot Optimized (These attributes come from estimate_2q_unitary)
            if hasattr(self, "log_XEB_fidelity_seq_avg_opt"):
                plt.errorbar(
                    xs,
                    self.log_XEB_fidelity_seq_avg_opt,
                    yerr=self.log_XEB_fidelity_seq_std_opt,
                    fmt="o",
                    label="Log XEB Data (Optimized)",
                    color=f"C{i_color}",
                    zorder=zorder,
                )
                plt.plot(
                    xs,
                    exponential_decay(xs, self.a_log_opt, self.log_XEB_layer_fidelity_opt),
                    label=f"Fit (Opt): {self.log_XEB_layer_fidelity_opt*100:.2f}% (err: {1-self.log_XEB_layer_fidelity_opt:.1e})",
                    color=f"C{i_color}",
                    zorder=zorder,
                )
                i_color += 1

        # LINEAR FIDELITY
        elif "linear" in data_to_plot:
            if compare:
                # Plot Initial
                plt.errorbar(
                    xs,
                    lin_fid,
                    yerr=lin_fid_std,
                    fmt="o",
                    label="Linear XEB Data (Initial)",
                    color=f"C{i_color}",
                    zorder=zorder,
                    alpha=0.6,
                )
                plt.plot(
                    xs,
                    exponential_decay(xs, a_lin_init, lin_fid_layer_init),
                    label=f"Fit (Initial): {lin_fid_layer_init*100:.2f}% (err: {1-lin_fid_layer_init:.1e})",
                    color=f"C{i_color}",
                    linestyle="--",
                    zorder=zorder,
                )
                i_color += 1

            # Plot Optimized
            if hasattr(self, "linear_XEB_fidelity_opt"):
                plt.errorbar(
                    xs,
                    self.linear_XEB_fidelity_opt,
                    yerr=self.linear_XEB_fidelity_std_opt,
                    fmt="o",
                    label="Linear XEB Data (Optimized)",
                    color=f"C{i_color}",
                    zorder=zorder,
                )
                plt.plot(
                    xs,
                    exponential_decay(xs, self.a_lin_opt, self.linear_XEB_layer_fidelity_opt),
                    label=f"Fit (Opt): {self.linear_XEB_layer_fidelity_opt*100:.2f}% (err: {1-self.linear_XEB_layer_fidelity_opt:.1e})",
                    color=f"C{i_color}",
                    zorder=zorder,
                )
                i_color += 1

        # PURITY (SPECKLE)
        if plot_purity:
            if not hasattr(self, "average_state_purity"):
                try:
                    self.calculate_purity()
                except:
                    pass

            if hasattr(self, "average_state_purity"):
                purity_sqrt = np.sqrt(np.abs(self.average_state_purity))
                plt.plot(
                    xs,
                    purity_sqrt,
                    "s",
                    label=r"$\sqrt{\mathrm{Purity}}$ (Speckle)",
                    color="gray",
                    zorder=-1,
                    alpha=0.5,
                )
                plt.plot(xs, purity_sqrt, "--", color="gray", linewidth=1, zorder=-1, alpha=0.5)
                plt.ylabel(r"$\sqrt{\mathrm{Purity}}$, Circuit Fidelity")
            else:
                plt.ylabel("Circuit Fidelity")
        else:
            plt.ylabel("Circuit Fidelity")

        # --- Formatting & Text Box ---
        ylim = plt.ylim()
        plt.ylim([np.max([ylim[0] - 0.2 * (ylim[1] - ylim[0]), -0.05]), np.min([ylim[1], 1.05])])
        plt.xlabel("Cycle Depth")
        plt.legend(loc="lower left", fontsize="small")

        # Add text box with optimized parameters if available
        if hasattr(self, "theta_iswap_opt"):
            txt0 = "Estimated Param.\n"
            txt1 = r"$\theta_\mathrm{iSWAP}=$" + f"{self.theta_iswap_opt*180/np.pi:.2f}" + r"$^\circ$" + "\n"
            txt2 = r"$\phi_\mathrm{CZ}=$" + f"{self.phi_cphase_opt*180/np.pi:.2f}" + r"$^\circ$" + "\n"
            txt3 = r"$\phi_\mathrm{RZ1}=$" + f"{self.phi_rz1_opt*180/np.pi:.2f}" + r"$^\circ$" + "\n"
            txt4 = r"$\phi_\mathrm{RZ2}=$" + f"{self.phi_rz2_opt*180/np.pi:.2f}" + r"$^\circ$"

            plt.text(
                0.98,
                0.98,
                txt0 + txt1 + txt2 + txt3 + txt4,
                ha="right",
                va="top",
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()

    def calculate_log_XEB_fidelity(self, expected_probs=None):
        if expected_probs is None:
            expected_probs = self.expected_probs

        # Ensure we have the incoherent distribution (1/D)
        n_qubits = self.xeb_config.n_qubits
        incoherent_distribution = np.ones(2**n_qubits) / 2**n_qubits

        # Calculate Fidelity per sequence/depth
        log_XEB_fidelity = np.array(
            [
                [
                    compute_log_fidelity(
                        incoherent_distribution, expected_probs[s, d_, :], self.measured_probs[s, d_, :]
                    )
                    for d_, depth in enumerate(self.xeb_config.depths)
                ]
                for s in range(self.xeb_config.seqs)
            ]
        )

        # Averages and Std
        log_XEB_fidelity_seq_avg = np.nanmean(log_XEB_fidelity, axis=0)
        log_XEB_fidelity_seq_std = np.nanstd(log_XEB_fidelity, axis=0)

        # Fit
        try:
            a_log, log_XEB_layer_fidelity, *_ = fit_exponential_decay(self.xeb_config.depths, log_XEB_fidelity_seq_avg)
        except:
            a_log, log_XEB_layer_fidelity = 1.0, 0.0

        return log_XEB_fidelity, log_XEB_fidelity_seq_avg, log_XEB_fidelity_seq_std, a_log, log_XEB_layer_fidelity

    def calculate_linear_XEB_fidelity(self, expected_probs=None, en_plot=False):
        if expected_probs is None:
            expected_probs = self.expected_probs

        n_qubits = self.xeb_config.n_qubits

        # Linear XEB Formula components
        e_u = np.sum(expected_probs**2, axis=-1)
        u_u = np.sum(expected_probs, axis=-1) / 2**n_qubits
        m_u = np.sum(self.measured_probs * expected_probs, axis=-1)

        x = e_u - u_u
        y = m_u - u_u

        numerator = x * y
        denominator = x**2

        # Sum over sequences to get fidelity per depth
        linear_XEB_fidelity = numerator.sum(axis=0) / denominator.sum(axis=0)

        # Fit
        try:
            a_lin, linear_XEB_layer_fidelity, *_ = fit_exponential_decay(self.xeb_config.depths, linear_XEB_fidelity)
        except:
            a_lin, linear_XEB_layer_fidelity = 1.0, 0.0

        # Calculate Error Bars (Standard Error of the Slope)
        std_errs = []
        for _i_depth, _ in enumerate(self.xeb_config.depths):
            try:
                residuals = y[:, _i_depth] - linear_XEB_fidelity[_i_depth] * x[:, _i_depth]
                residual_variance = np.sum(residuals**2) / (len(x[:, _i_depth]) - 1)
                std_err = np.sqrt(residual_variance / np.sum(x[:, _i_depth] ** 2))
                std_errs.append(std_err)
            except:
                std_errs.append(0.0)
        linear_XEB_fidelity_std = np.array(std_errs)

        fig = None
        # (Optional: Add plot logic here if you want the slope plots, but usually not needed for optimization view)

        return linear_XEB_fidelity, linear_XEB_fidelity_std, a_lin, linear_XEB_layer_fidelity, fig

    # --- ADD THIS METHOD TO CLASS XEBResult ---
    # def _perform_gaussian_discrimination(self):
    #     """
    #     Performs Global GMM Fit + Soft Counting.
    #     1. Aggregates ALL IQ data to fit the global Gaussian blobs (Means & Covariances).
    #     2. Uses this global model to calculate the posterior probability (Soft Count) for every shot.
    #     3. Rebuilds the count dictionary using these probabilities.
    #     """
    #     print("Performing Global GMM Fit & Soft Counting...")
    #     from sklearn.mixture import GaussianMixture

    #     # 1. Setup Dimensions
    #     n_seqs = self.xeb_config.seqs
    #     n_depths = len(self.xeb_config.depths)
    #     n_shots = self.xeb_config.n_shots
    #     total_dim = self.xeb_config.total_dim

    #     # Prepare Count Storage (Floats allowed for soft counts)
    #     new_counts = {f"s{i}": np.zeros((n_seqs, n_depths), dtype=float) for i in range(total_dim)}

    #     # 2. Helper: Fit & Predict
    #     def process_channel(i_data, q_data, n_states, name):
    #         if n_states < 2:
    #             # Trivial: 100% in state 0
    #             # Returns: (probs_3d, gmm_means, gmm_covariances)
    #             dummy_probs = np.zeros((n_seqs, n_depths, n_shots, n_states))
    #             dummy_probs[..., 0] = 1.0
    #             return dummy_probs, np.zeros((n_states, 2)), np.eye(2).reshape(1,2,2)

    #         # --- A. Flatten Data for Global Fit ---
    #         i_flat = i_data.flatten()
    #         q_flat = q_data.flatten()
    #         X_global = np.column_stack((i_flat, q_flat))

    #         # --- B. Fit Global GMM ---
    #         # We use 'full' covariance to allow elliptical blobs (e.g. T1 decay smear)
    #         gmm = GaussianMixture(n_components=n_states, covariance_type='full', random_state=42)
    #         gmm.fit(X_global)

    #         # --- C. Sort States (lowest I -> State 0) ---
    #         means = gmm.means_
    #         sorted_indices = np.argsort(means[:, 0])

    #         # Reorder parameters
    #         sorted_means = means[sorted_indices]
    #         sorted_covs = gmm.covariances_[sorted_indices]
    #         sorted_weights = gmm.weights_[sorted_indices]

    #         # Update GMM internal state to match sorted order (for prediction)
    #         gmm.means_ = sorted_means
    #         gmm.covariances_ = sorted_covs
    #         gmm.weights_ = sorted_weights
    #         gmm.precisions_cholesky_ = gmm.precisions_cholesky_[sorted_indices]

    #         # --- D. Soft Predict (Posterior Probabilities) ---
    #         # probs[i, k] = Probability that shot i belongs to state k
    #         probs_flat = gmm.predict_proba(X_global)

    #         # Reshape to (Seq, Depth, Shots, States)
    #         probs_3d = probs_flat.reshape(n_seqs, n_depths, n_shots, n_states)

    #         return probs_3d, sorted_means, sorted_covs

    #     # 3. Process Channels
    #     dim_c = self.xeb_config.dim_c
    #     dim_t = self.xeb_config.dim_t
    #     dim_k = self.xeb_config.coupler_readout_mode if self.xeb_config.coupler_readout_enable else 1

    #     print(f"Fitting Control (Mode {dim_c})...")
    #     probs_c, mu_c, cov_c = process_channel(self.data["I_c_all"], self.data["Q_c_all"], dim_c, "Control")

    #     print(f"Fitting Target (Mode {dim_t})...")
    #     probs_t, mu_t, cov_t = process_channel(self.data["I_t_all"], self.data["Q_t_all"], dim_t, "Target")

    #     if self.xeb_config.coupler_readout_enable:
    #         print(f"Fitting Coupler (Mode {dim_k})...")
    #         probs_k, mu_k, cov_k = process_channel(self.data["I_k_all"], self.data["Q_k_all"], dim_k, "Coupler")
    #     else:
    #         probs_k = np.zeros((n_seqs, n_depths, n_shots, 1))
    #         probs_k[..., 0] = 1.0
    #         mu_k, cov_k = np.zeros((1, 2)), np.eye(2).reshape(1,2,2)

    #     # 4. Save Fit Parameters for Plotting
    #     self.data["gmm_params_c"] = {"means": mu_c, "covs": cov_c}
    #     self.data["gmm_params_t"] = {"means": mu_t, "covs": cov_t}
    #     if self.xeb_config.coupler_readout_enable:
    #         self.data["gmm_params_k"] = {"means": mu_k, "covs": cov_k}

    #     # 5. Calculate Joint Counts (Vectorized)
    #     print("Calculating Joint Probabilities...")

    #     # We need P(S=s_idx) for each shot.
    #     # S = C + (Nc * T) + (Nc*Nt * K)

    #     # Iterate over all possible total states 'i'
    #     for i in range(total_dim):
    #         # Decode i -> (c, t, k) indices
    #         c_idx = i % dim_c
    #         t_idx = (i // dim_c) % dim_t
    #         k_idx = (i // (dim_c * dim_t))

    #         # Joint Probability P(c,t,k) = P(c) * P(t) * P(k) (Assuming uncorrelated readout)
    #         # Shape: (Seq, Depth, Shots)
    #         p_shot = probs_c[..., c_idx] * probs_t[..., t_idx] * probs_k[..., k_idx]

    #         # Sum probabilities over shots to get the "Soft Count"
    #         # Shape: (Seq, Depth)
    #         count_val = np.sum(p_shot, axis=2)
    #         new_counts[f"s{i}"] = count_val

    #     # 6. Update Results
    #     self.counts = new_counts
    #     self.data.update(new_counts)
    #     print("Global Soft Counting complete.")

    @property
    def measured_probs(self):
        return self._joint_measured_probs

    @property
    def disjoint_measured_probs(self):
        return self._disjoint_measured_probs

    @property
    def expected_probs(self):
        return self._joint_expected_probs

    @property
    def disjoint_expected_probs(self):
        return self._disjoint_expected_probs

    @property
    def records(self):
        return self._records

    @property
    def log_fidelities(self):
        return self._log_fidelities

    @property
    def linear_fidelities(self):
        if self.xeb_config.disjoint_processing:
            fidelities = [self._linear_fidelities[q] for q in range(self.xeb_config.n_qubits)]
        else:
            fidelities = self._linear_fidelities
        return fidelities

    @property
    def singularities(self):
        return self._singularities

    @property
    def outliers(self):
        return self._outliers

    @property
    def purities(self):
        var_pt = (2**self.xeb_config.n_qubits - 1) / (
            2 ** (2 * self.xeb_config.n_qubits)(2**self.xeb_config.n_qubits + 1)
        )
        purities = np.var(self.measured_probs, axis=-1) / var_pt
        return purities

    @property
    def qubit_names(self):
        return [qubit.name if isinstance(qubit, FluxTunableTransmon) else qubit for qubit in self.xeb_config.qubits]


class XEBResult_minimal:
    def __init__(self, xeb_config=None, gate_indices=None):
        if xeb_config is not None:
            self.xeb_config = xeb_config
        self.measured_probs = None
        self.expected_probs = None

    @property
    def _effective_depths(self):
        """
        Returns the subset of xeb_config.depths that matches the shape of measured_probs.
        """
        depths = self.xeb_config.depths
        if self.measured_probs is not None:
            n_meas_depths = self.measured_probs.shape[1]
            if len(depths) > n_meas_depths:
                return depths[:n_meas_depths]
        return depths

    def calculate_noisy_and_ideal_probability(self, parameters):
        if isinstance(parameters, dict):
            from types import SimpleNamespace

            parameters = SimpleNamespace(**parameters)

        self.gate_indices = generate_gate_indices(xeb_config=self.xeb_config)

        d_theta_iswap = parameters.numpy_simulate_iswap_angle_error_in_deg / 180 * np.pi
        d_phi_cz = parameters.numpy_simulate_cphase_angle_error_in_deg / 180 * np.pi
        d_phi_rz1 = parameters.numpy_simulate_rz1_angle_error_in_deg / 180 * np.pi
        d_phi_rz2 = parameters.numpy_simulate_rz2_angle_error_in_deg / 180 * np.pi

        theta_iswap = parameters.nominal_iswap_angle_in_deg / 180 * np.pi
        phi_cz = parameters.nominal_cphase_angle_in_deg / 180 * np.pi
        phi_rz1 = parameters.nominal_rz1_angle_in_deg / 180 * np.pi
        phi_rz2 = parameters.nominal_rz2_angle_in_deg / 180 * np.pi

        measured_probs_all = simulate_noisy_circuit_numpy(
            self.gate_indices,
            self.xeb_config.depths,
            theta_iswap=theta_iswap + d_theta_iswap,
            phi_cphase=phi_cz + d_phi_cz,
            phi_rz1=phi_rz1 + d_phi_rz1,
            phi_rz2=phi_rz2 + d_phi_rz2,
            one_over_f_amplitude_at_1Hz_GHz_per_rHz=parameters.numpy_simulate_one_over_f_amplitude_at_1Hz_GHz_per_rHz,
            white_noise_amplitude_GHz_per_rHz=parameters.numpy_simulate_white_noise_amplitude_GHz_per_rHz,
            gate_time_1q_ns=parameters.simulate_gate_time_1q_ns,
            gate_time_2q_ns=parameters.simulate_gate_time_2q_ns,
            n_noise_sample=parameters.numpy_simulate_n_noise_samples,
        )

        measured_probs = np.mean(measured_probs_all, axis=1)

        _insert_gate = self.xeb_config.two_qb_gate is not None

        expected_probs = calc_ideal_probability_numpy(
            self.gate_indices,
            self.xeb_config.depths,
            theta_iswap=0,
            phi_cphase=np.pi,
            phi_rz1=0,
            phi_rz2=0,
            insert_2q_gate=_insert_gate,
        )

        self.measured_probs = measured_probs

    def calc_expected_probs(self, theta_iswap=0, phi_cphase=np.pi, phi_rz1=0, phi_rz2=0, insert_2q_gate=True):
        expected_probs = calc_ideal_probability_numpy(
            self.gate_indices,
            self.xeb_config.depths,
            theta_iswap=theta_iswap,
            phi_cphase=phi_cphase,
            phi_rz1=phi_rz1,
            phi_rz2=phi_rz2,
            insert_2q_gate=insert_2q_gate,
        )
        self.expected_probs = expected_probs

    def calculate_log_XEB_fidelity(self, expected_probs=None):
        if expected_probs is None:
            expected_probs = self.expected_probs

        # --- FIX: Intersect shapes of Config, Measured, and Expected ---
        n_conf = len(self.xeb_config.depths)
        n_meas = self.measured_probs.shape[1]
        n_exp = expected_probs.shape[1]

        # The limit is the minimum of all available data
        limit = min(n_conf, n_meas, n_exp)

        # Slice everything to this safe limit
        depths = self.xeb_config.depths[:limit]
        m_probs = self.measured_probs[:, :limit, :]
        e_probs = expected_probs[:, :limit, :]
        # ---------------------------------------------------------------

        n_qubits = len(self.xeb_config.qubits)
        incoherent_distribution = np.ones(2**n_qubits) / 2**n_qubits

        log_XEB_fidelity = np.array(
            [
                [
                    compute_log_fidelity(incoherent_distribution, e_probs[s, d_, :], m_probs[s, d_, :])
                    for d_, depth in enumerate(depths)
                ]
                for s in range(self.xeb_config.seqs)
            ]
        )

        log_XEB_fidelity_seq_avg = np.nanmean(log_XEB_fidelity, axis=0)
        log_XEB_fidelity_seq_std = np.nanstd(log_XEB_fidelity, axis=0)

        a_log, log_XEB_layer_fidelity, *_ = fit_exponential_decay(depths, log_XEB_fidelity_seq_avg)

        return log_XEB_fidelity, log_XEB_fidelity_seq_avg, log_XEB_fidelity_seq_std, a_log, log_XEB_layer_fidelity

    def calculate_linear_XEB_fidelity(self, expected_probs=None, en_plot=False):
        if expected_probs is None:
            expected_probs = self.expected_probs

        # --- FIX: Intersect shapes ---
        n_conf = len(self.xeb_config.depths)
        n_meas = self.measured_probs.shape[1]
        n_exp = expected_probs.shape[1]

        limit = min(n_conf, n_meas, n_exp)

        depths = self.xeb_config.depths[:limit]
        m_probs = self.measured_probs[:, :limit, :]
        e_probs = expected_probs[:, :limit, :]
        # -----------------------------

        n_qubits = len(self.xeb_config.qubits)
        e_u = np.sum(e_probs**2, axis=-1)
        u_u = np.sum(e_probs, axis=-1) / 2**n_qubits

        m_u = np.sum(m_probs * e_probs, axis=-1)

        x = e_u - u_u
        y = m_u - u_u
        numerator = x * y
        denominator = x**2
        linear_XEB_fidelity = numerator.sum(axis=0) / denominator.sum(axis=0)

        # Layer Fidelity
        a_lin, linear_XEB_layer_fidelity, *_ = fit_exponential_decay(depths, linear_XEB_fidelity)

        # Error bar for linear XEB fidelity
        std_errs = []
        for _i_depth, _ in enumerate(depths):
            residuals = y[:, _i_depth] - linear_XEB_fidelity[_i_depth] * x[:, _i_depth]
            residual_variance = np.sum(residuals**2) / (len(x[:, _i_depth]) - 1)
            std_err = np.sqrt(residual_variance / np.sum(x[:, _i_depth] ** 2))
            std_errs.append(std_err)
        linear_XEB_fidelity_std = np.array(std_errs)

        if en_plot:
            fig = self._plot_linear_XEB_fidelity_slopes(e_u, u_u, m_u, linear_XEB_fidelity, depths)
        else:
            fig = None

        return linear_XEB_fidelity, linear_XEB_fidelity_std, a_lin, linear_XEB_layer_fidelity, fig

    def _plot_linear_XEB_fidelity_slopes(self, e_u, u_u, m_u, linear_XEB_fidelity, depths):
        y = m_u - u_u
        x = e_u - u_u
        colors = sns.cubehelix_palette(n_colors=len(depths))
        plt.figure()
        for _i_depth, _depth in enumerate(depths):
            xpl = np.linspace(0, x[:, _i_depth].max())
            label = f"{_depth}" if _i_depth in [0, len(depths) - 1] else None
            plt.plot(xpl, linear_XEB_fidelity[_i_depth] * xpl, color=colors[_i_depth], label=label)
            plt.scatter(x[:, _i_depth], y[:, _i_depth], color=colors[_i_depth])
            plt.xlabel(r"$e_U - u_U$")
            plt.ylabel(r"$m_U - u_U$")
            plt.legend(loc="best", title="Cycle Depth")
        return plt.gcf()

    def plot_fidelity_vs_depth(self, data_to_plot=["log", "linear"]):
        # Use existing effective logic, but recalculate to be safe or use property
        xs = self._effective_depths
        i = 0

    def plot_fidelity_vs_depth(self, data_to_plot=["log", "linear"], log_yscale=False, plot_purity=True):
        xs = self.xeb_config.depths
        i_color = 0
        zorder = 0
        plt.figure()
        if "log" in data_to_plot:
            # Slice xs to match data if needed (data might be smaller if computed before fix)
            limit = min(len(xs), len(self.log_XEB_fidelity_seq_avg))
            xs_log = xs[:limit]

            plt.errorbar(
                xs_log,
                self.log_XEB_fidelity_seq_avg[:limit],
                yerr=self.log_XEB_fidelity_seq_std[:limit],
                fmt="o",
                label="Log XEB Data",
                color=f"C{i}",
                zorder=zorder,
            )
            plt.errorbar(
                xs,
                self.log_XEB_fidelity_seq_avg,
                yerr=self.log_XEB_fidelity_seq_std,
                fmt="o",
                label="Log XEB Data",
                color=f"C{i_color}",
                zorder=zorder,
            )
            zorder += 1
            plt.plot(
                xs_log,
                exponential_decay(xs_log, self.a_log, self.log_XEB_layer_fidelity),
                label=f"Log XEB Fit, Layer Fidelity: {self.log_XEB_layer_fidelity*100:.2f}% (error: {1-self.log_XEB_layer_fidelity:.1e})",
                color=f"C{i}",
                zorder=zorder,
            )
            i += 1
            plt.plot(
                xs,
                exponential_decay(xs, self.a_log, self.log_XEB_layer_fidelity),
                label=f"Log XEB Fit, Layer Fidelity: {self.log_XEB_layer_fidelity*100:.2f}% (error: {1-self.log_XEB_layer_fidelity:.1e})",
                color=f"C{i_color}",
                zorder=zorder,
            )
            i_color += 1
            zorder += 1
        if "linear" in data_to_plot:
            limit = min(len(xs), len(self.linear_XEB_fidelity))
            xs_lin = xs[:limit]

            plt.errorbar(
                xs_lin,
                self.linear_XEB_fidelity[:limit],
                yerr=self.linear_XEB_fidelity_std[:limit],
                fmt="o",
                label="Linear XEB Data",
                color=f"C{i}",
                zorder=zorder,
            )
            plt.errorbar(
                xs,
                self.linear_XEB_fidelity,
                yerr=self.linear_XEB_fidelity_std,
                fmt="o",
                label="Linear XEB Data",
                color=f"C{i_color}",
                zorder=zorder,
            )
            zorder += 1
            plt.plot(
                xs_lin,
                exponential_decay(xs_lin, self.a_lin, self.linear_XEB_layer_fidelity),
                label=f"Linear XEB Fit, Layer Fidelity: {self.linear_XEB_layer_fidelity*100:.2f}% (error: {1-self.linear_XEB_layer_fidelity:.1e})",
                color=f"C{i}",
                zorder=zorder,
            )
            plt.plot(
                xs,
                exponential_decay(xs, self.a_lin, self.linear_XEB_layer_fidelity),
                label=f"Linear XEB Fit, Layer Fidelity: {self.linear_XEB_layer_fidelity*100:.2f}% (error: {1-self.linear_XEB_layer_fidelity:.1e})",
                color=f"C{i_color}",
                zorder=zorder,
            )
            i_color += 1
        if plot_purity:
            plt.plot(
                self.xeb_config.depths,
                self.average_state_purity**0.5,
                "o",
                label="Purity (speckle)",
                color=f"C{i_color}",
            )
            plt.ylabel(r"$\sqrt{\mathrm{Purity}}$, XEB Fidelity")
        else:
            plt.ylabel("XEB Fidelity")

        if log_yscale:
            plt.yscale("log")
        ylim = plt.ylim()
        plt.ylim([np.max([ylim[0] - 0.2 * (ylim[1] - ylim[0]), -0.19]), np.min([ylim[1], 1.1])])

        plt.xlabel("Cycle Depth")
        plt.legend(loc="lower left")
        self._order_legend(priority_text="Data")
        # plt.show()
        return plt.gcf()

    def plot_fidelity_vs_depth_opt(self, data_to_plot="log", compare=True, plot_purity=True):
        """
        Plots the fidelity vs depth with optimized 2Q unitary parameters.
        """
        xs = self._effective_depths
        i_color = 0
        zorder = 0
        plt.figure(figsize=(8, 6))

        # Helper to safely plot if lengths mismatch
        def safe_plot_errorbar(x_data, y_data, y_err, label, color, **kwargs):
            lim = min(len(x_data), len(y_data))
            plt.errorbar(
                x_data[:lim],
                y_data[:lim],
                yerr=y_err[:lim] if y_err is not None else None,
                label=label,
                color=color,
                **kwargs,
            )
            return x_data[:lim]

        # --- Plot Fidelities ---
        if "log" in data_to_plot:
            if compare:
                safe_plot_errorbar(
                    xs,
                    self.log_XEB_fidelity_seq_avg,
                    self.log_XEB_fidelity_seq_std,
                    fmt="o",
                    label="Log XEB Data (Initial)",
                    color=f"C{i_color}",
                    zorder=zorder,
                    alpha=0.6,
                )
                # Plot fit
                plt.plot(
                    xs,
                    exponential_decay(xs, self.a_log, self.log_XEB_layer_fidelity),
                    label=f"Fit (Initial): {self.log_XEB_layer_fidelity*100:.2f}% (err: {1-self.log_XEB_layer_fidelity:.1e})",
                    color=f"C{i_color}",
                    linestyle="--",
                    zorder=zorder,
                )
                i_color += 1

            safe_plot_errorbar(
                xs,
                self.log_XEB_fidelity_seq_avg_opt,
                self.log_XEB_fidelity_seq_std_opt,
                fmt="o",
                label="Log XEB Data (Optimized)",
                color=f"C{i_color}",
                zorder=zorder,
            )
            plt.plot(
                xs,
                exponential_decay(xs, self.a_log_opt, self.log_XEB_layer_fidelity_opt),
                label=f"Fit (Opt): {self.log_XEB_layer_fidelity_opt*100:.2f}% (err: {1-self.log_XEB_layer_fidelity_opt:.1e})",
                color=f"C{i_color}",
                zorder=zorder,
            )
            i_color += 1

        elif "linear" in data_to_plot:
            if compare:
                safe_plot_errorbar(
                    xs,
                    self.linear_XEB_fidelity,
                    self.linear_XEB_fidelity_std,
                    fmt="o",
                    label="Linear XEB Data (Initial)",
                    color=f"C{i_color}",
                    zorder=zorder,
                    alpha=0.6,
                )
                plt.plot(
                    xs,
                    exponential_decay(xs, self.a_lin, self.linear_XEB_layer_fidelity),
                    label=f"Fit (Initial): {self.linear_XEB_layer_fidelity*100:.2f}% (err: {1-self.linear_XEB_layer_fidelity:.1e})",
                    color=f"C{i_color}",
                    linestyle="--",
                    zorder=zorder,
                )
                i_color += 1

            safe_plot_errorbar(
                xs,
                self.linear_XEB_fidelity_opt,
                self.linear_XEB_fidelity_std_opt,
                fmt="o",
                label="Linear XEB Data (Optimized)",
                color=f"C{i_color}",
                zorder=zorder,
            )
            plt.plot(
                xs,
                exponential_decay(xs, self.a_lin_opt, self.linear_XEB_layer_fidelity_opt),
                label=f"Fit (Opt): {self.linear_XEB_layer_fidelity_opt*100:.2f}% (err: {1-self.linear_XEB_layer_fidelity_opt:.1e})",
                color=f"C{i_color}",
                zorder=zorder,
            )
            i_color += 1

        # --- Plot Purity (Speckle) ---
        if plot_purity:
            if not hasattr(self, "average_state_purity"):
                try:
                    self.calculate_purity()
                except Exception as e:
                    print(f"Could not calculate purity for plot: {e}")

            if hasattr(self, "average_state_purity"):
                purity_sqrt = np.sqrt(np.abs(self.average_state_purity))
                safe_plot_errorbar(
                    xs,
                    purity_sqrt,
                    None,
                    fmt="s",
                    label=r"$\sqrt{\mathrm{Purity}}$ (Speckle)",
                    color="gray",
                    zorder=-1,
                    alpha=0.5,
                )
                # Just draw a line for visual guide
                plt.plot(xs[: len(purity_sqrt)], purity_sqrt, "--", color="gray", linewidth=1, zorder=-1, alpha=0.5)
                plt.ylabel(r"$\sqrt{\mathrm{Purity}}$, Circuit Fidelity")
            else:
                plt.ylabel("Circuit Fidelity")
        else:
            plt.ylabel("Circuit Fidelity")

        ylim = plt.ylim()
        plt.ylim([np.max([ylim[0] - 0.2 * (ylim[1] - ylim[0]), -0.05]), np.min([ylim[1], 1.05])])
        plt.xlabel("Cycle Depth")
        plt.legend(loc="lower left", fontsize="small")

        if hasattr(self, "theta_iswap_opt"):
            txt0 = "Estimated Param.\n"
            txt1 = r"$\theta_\mathrm{iSWAP}=$" + f"{self.theta_iswap_opt*180/np.pi:.2f}" + r"$^\circ$" + "\n"
            txt2 = r"$\phi_\mathrm{CZ}=$" + f"{self.phi_cphase_opt*180/np.pi:.2f}" + r"$^\circ$" + "\n"
            txt3 = r"$\phi_\mathrm{RZ1}=$" + f"{self.phi_rz1_opt*180/np.pi:.2f}" + r"$^\circ$" + "\n"
            txt4 = r"$\phi_\mathrm{RZ2}=$" + f"{self.phi_rz2_opt*180/np.pi:.2f}" + r"$^\circ$"

            plt.text(
                0.98,
                0.98,
                txt0 + txt1 + txt2 + txt3 + txt4,
                ha="right",
                va="top",
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()

    def _order_legend(self, priority_text="error", loc="lower left"):
        """
        Manually fix the order of legend label for error bar plot (known issue in matplotlib)
        """
        handles, labels = plt.gca().get_legend_handles_labels()
        handles_ordered = []
        labels_ordered = []
        for _handle, _label in zip(handles, labels):
            if priority_text in _label:
                handles_ordered = [_handle] + handles_ordered
                labels_ordered = [_label] + labels_ordered
            else:
                handles_ordered = handles_ordered + [_handle]
                labels_ordered = labels_ordered + [_label]
        plt.legend(handles_ordered, labels_ordered, loc=loc)

    def load_xeb_config(self, data_path):
        data = json.load(open(data_path + "/data.json", "r"))
        self.xeb_config = XEBConfig.from_dict(data["xeb_config"])

    def calculate_purity(self):
        # FIX: Align measured_probs flattening with effective depths
        depths = self._effective_depths
        n_qubits = self.xeb_config.n_qubits
        D = 2**n_qubits

        # Ensure we only use valid measured_probs columns
        m_probs = self.measured_probs[:, : len(depths), :]

        measured_probs_flattened = m_probs.transpose(0, 2, 1).reshape(-1, len(depths))
        var_p = np.var(measured_probs_flattened, axis=0)
        self.average_state_purity = var_p * (D**2 * (D + 1) / (D - 1))

        self.a_purity, self.purity_error_per_cycle, *_ = fit_exponential_decay(depths, self.average_state_purity)
        return self.average_state_purity

    def plot_fidelity_and_purity(self, log_yscale=False):
        xs = self._effective_depths
        plt.figure(figsize=(8, 6))

        # --- SAFEGUARDS INITIALIZATION ---
        e_xeb = None  # Corresponds to linear fit error
        e_pur = None

        # 1. Plot XEB Data & Fit
        if not self.xeb_config.disjoint_processing:
            if isinstance(self.linear_fidelities, pd.DataFrame):
                fidelity_depths = self.linear_fidelities["depth"]
                fidelity_data = self.linear_fidelities["fidelity"]
            else:
                limit = min(len(xs), len(self.linear_XEB_fidelity))
                fidelity_depths = xs[:limit]
                fidelity_data = self.linear_fidelities[:limit]

            # Try to get or calculate e_xeb
            if hasattr(self, "linear_XEB_layer_fidelity") and self.linear_XEB_layer_fidelity is not None:
                layer_fid = self.linear_XEB_layer_fidelity
                a_lin = self.a_lin
                e_xeb = 1.0 - layer_fid
            else:
                try:
                    a_lin, layer_fid, *_ = fit_exponential_decay(fidelity_depths, fidelity_data)
                    e_xeb = 1.0 - layer_fid
                except Exception:
                    a_lin, layer_fid = 1.0, 0.0
                    e_xeb = None

            label_prefix = "Linear XEB"
            if hasattr(self, "linear_XEB_fidelity_std"):
                yerr = self.linear_XEB_fidelity_std[: len(fidelity_depths)]
            else:
                yerr = None

            plt.errorbar(fidelity_depths, fidelity_data, yerr=yerr, fmt="o", label=f"{label_prefix} Data", color="C0")

            # Plot Fit
            fit_err_str = f"{e_xeb*100:.2f}%" if e_xeb is not None else "NaN"
            plt.plot(
                xs,
                exponential_decay(xs, a_lin, layer_fid),
                label=f"{label_prefix} Fit (err: {fit_err_str})",
                color="C0",
            )

        # 2. Plot Purity
        # Safely try to calculate if missing
        if not hasattr(self, "average_state_purity") or self.average_state_purity is None:
            try:
                self.calculate_purity()
            except:
                self.average_state_purity = None

        if hasattr(self, "average_state_purity") and self.average_state_purity is not None:
            purity_sqrt = np.sqrt(np.abs(self.average_state_purity))
            limit = min(len(xs), len(purity_sqrt))

            plt.plot(xs[:limit], purity_sqrt[:limit], "s", label=r"$\sqrt{\mathrm{Purity}}$ (Speckle)", color="C1")
            plt.plot(xs[:limit], purity_sqrt[:limit], "--", color="C1", alpha=0.5)

            # Try to extract e_pur if available
            if hasattr(self, "purity_error_per_cycle"):
                e_pur = self.purity_error_per_cycle

        plt.xlabel("Cycle Depth")
        plt.ylabel(r"$\sqrt{\mathrm{Purity}}$, XEB Fidelity")
        if log_yscale:
            plt.yscale("log")
        plt.legend(loc="best")
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()

        # Safe Title
        title_str = "XEB Fidelity & Purity"
        if e_xeb is not None and e_pur is not None:
            ctrl_err = max(0, e_xeb - e_pur)
            title_str += f"\nControl Err $\\approx$ {ctrl_err*100:.2f}%"
        plt.title(title_str)

        return plt.gcf()

    def erf_average_infidelity(self, prms):
        theta_iswap, phi_cphase, phi_rz1, phi_rz2 = prms

        # Calculate Ideal
        ideal_probability_s = calc_ideal_probability_numpy(
            self.gate_indices,
            self.xeb_config.depths,
            theta_iswap=theta_iswap,
            phi_cphase=phi_cphase,
            phi_rz1=phi_rz1,
            phi_rz2=phi_rz2,
        )

        # --- FIX: Match Shapes ---
        n_meas = self.measured_probs.shape[1]
        n_exp = ideal_probability_s.shape[1]
        n_conf = len(self.xeb_config.depths)
        limit = min(n_meas, n_exp, n_conf)

        depths = self.xeb_config.depths[:limit]
        m_probs = self.measured_probs[:, :limit, :]
        e_probs = ideal_probability_s[:, :limit, :]
        # -------------------------

        f_xeb = [
            [
                compute_log_fidelity(self.incoherent_distribution, e_probs[s, d_, :], m_probs[s, d_, :])
                for d_, depth in enumerate(depths)
            ]
            for s in range(self.xeb_config.seqs)
        ]
        total_loss = 1 - np.nanmean(f_xeb)
        return total_loss

    def erf_average_cross_entropy(self, prms):
        theta_iswap, phi_cphase, phi_rz1, phi_rz2 = prms

        ideal_probability_s = calc_ideal_probability_numpy(
            self.gate_indices,
            self.xeb_config.depths,
            theta_iswap=theta_iswap,
            phi_cphase=phi_cphase,
            phi_rz1=phi_rz1,
            phi_rz2=phi_rz2,
        )

        # --- FIX: Match Shapes ---
        n_meas = self.measured_probs.shape[1]
        n_exp = ideal_probability_s.shape[1]
        n_conf = len(self.xeb_config.depths)
        limit = min(n_meas, n_exp, n_conf)

        m_probs = self.measured_probs[:, :limit, :]
        e_probs = ideal_probability_s[:, :limit, :]
        # -------------------------

        cross_entropy_s = self.calc_crosss_entropy(m_probs, e_probs)

        total_loss = np.nanmean(cross_entropy_s)
        return total_loss

    def calc_crosss_entropy(self, p, q):
        q = np.maximum(q, 1e-15)
        cross_entropy = -np.sum(p * np.log(q), axis=-1)
        return cross_entropy

    def estimate_2q_unitary(
        self, en_plot=True, method: Literal["Nelder-Mead", "differential evolution"] = "differential evolution"
    ):
        from scipy.optimize import minimize, differential_evolution

        def record_loss(prms, convergence=None):  # callback func
            current_loss = self.erf_average_cross_entropy(prms)
            loss_history.append(current_loss)
            prms_history.append(prms.copy())

        if method == "Nelder-Mead":
            prms0 = [0, np.pi, 0, 0]  # assiming CZ gate
            loss_history = [self.erf_average_cross_entropy(prms0)]
            prms_history = [prms0.copy()]
            res = minimize(
                self.erf_average_cross_entropy,
                prms0,
                method="Nelder-Mead",
                options={"disp": True},
                callback=record_loss,
            )
        elif method == "differential evolution":
            loss_history = []
            prms_history = []
            margin = 0.1
            bounds = [
                (-np.pi / 2 - margin, np.pi / 2 + margin),
                (-np.pi - margin, np.pi + margin),
                (-np.pi - margin, np.pi + margin),
                (-np.pi - margin, np.pi + margin),
            ]
            res = differential_evolution(
                self.erf_average_cross_entropy,
                bounds,
                strategy="best1bin",
                maxiter=100,
                popsize=15,
                disp=True,
                callback=record_loss,
                tol=1e-3,
            )

        self.theta_iswap_opt, self.phi_cphase_opt, self.phi_rz1_opt, self.phi_rz2_opt = res.x

        if en_plot:
            prm_names = ["iSWAP", "CPhase", "RZ1", "RZ2"]
            fig = plt.figure()
            plt.subplot(211)
            plt.plot(loss_history)
            plt.text(0.98, 0.98, f"Final: {loss_history[-1]:.4f}", ha="right", va="top", transform=plt.gca().transAxes)
            plt.ylabel("Average Cross Entropy")
            plt.gca().tick_params(axis="x", labelbottom=False)

            plt.grid(True)
            plt.title("Convergence Plot of 2Q-Unitary Estimation")
            plt.subplot(212)
            for _i, _name in enumerate(prm_names):
                plt.plot(np.array(prms_history)[:, _i] / np.pi, label=_name)
            plt.xlabel("Iteration")
            plt.ylabel("2Q Unitary Parameter (π rad)")
            plt.grid(True)
            plt.legend()
        else:
            fig = None

        # Calculate optimized ideal probabilities
        # (These might also be truncated, so we handle it below)
        self.expected_probs_opt = calc_ideal_probability_numpy(
            self.gate_indices,
            self.xeb_config.depths,
            self.theta_iswap_opt,
            self.phi_cphase_opt,
            self.phi_rz1_opt,
            self.phi_rz2_opt,
        )

        # Calculate linear XEB fidelity
        (
            self.linear_XEB_fidelity_opt,
            self.linear_XEB_fidelity_std_opt,
            self.a_lin_opt,
            self.linear_XEB_layer_fidelity_opt,
            _,
        ) = self.calculate_linear_XEB_fidelity(expected_probs=self.expected_probs_opt, en_plot=False)

        # Calculate log XEB fidelity
        (
            self.log_XEB_fidelity_opt,
            self.log_XEB_fidelity_seq_avg_opt,
            self.log_XEB_fidelity_seq_std_opt,
            self.a_log_opt,
            self.log_XEB_layer_fidelity_opt,
        ) = self.calculate_log_XEB_fidelity(expected_probs=self.expected_probs_opt)

        return fig

    def calc_expected_probs(self, theta_iswap=0, phi_cphase=np.pi, phi_rz1=0, phi_rz2=0, insert_2q_gate=True):
        """
        Calculates ideal probabilities and stores them in self.expected_probs.
        """
        expected_probs = calc_ideal_probability_numpy(
            self.gate_indices,
            self.xeb_config.depths,
            theta_iswap=theta_iswap,
            phi_cphase=phi_cphase,
            phi_rz1=phi_rz1,
            phi_rz2=phi_rz2,
            insert_2q_gate=insert_2q_gate,
        )
        # Safe slicing not strictly needed here as other functions handle it,
        # but storing the raw output is fine.
        self.expected_probs = expected_probs
