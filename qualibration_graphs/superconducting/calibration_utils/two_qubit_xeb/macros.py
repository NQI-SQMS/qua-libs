"""
This script contains useful QUA macros for the two-qubit cross-entropy benchmarking use case.

Author: Arthur Strauss - Quantum Machines
Last updated: 2024-12-08
"""

from typing import List, Dict
import json
from itertools import combinations
import numpy as np
from matplotlib import pyplot as plt
from qiskit.transpiler import CouplingMap
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter
from qm.qua import *
from qualang_tools.addons.variables import assign_variables_to_element
from quam_builder.architecture.superconducting.qubit_pair import FluxTunableTransmonPair
from quam_builder.architecture.superconducting.qubit import FluxTunableTransmon
from scipy import optimize
from scipy.stats import stats


def qua_declaration(n_qubits: int, readout_elements: list):
    """
    Macro to declare the necessary QUA variables

    :param n_qubits: Number of qubits used in this experiment
    :param readout_elements: List of readout elements
    :return:
    """
    I, Q = [[declare(fixed) for _ in range(n_qubits)] for _ in range(2)]
    I_st, Q_st = [[declare_stream() for _ in range(n_qubits)] for _ in range(2)]
    # Workaround to manually assign the results variables to the readout elements
    for i in range(n_qubits):
        assign_variables_to_element(readout_elements[i].name, I[i], Q[i])
    return I, I_st, Q, Q_st


def reset_qubit(method: str, qubit: FluxTunableTransmon, **kwargs):
    """
    Macro to reset the qubit state.

    If method is 'cooldown', then the variable cooldown_time (in clock cycles) must be provided as a python integer > 4.

    **Example**: reset_qubit('cooldown', cooldown_times=500)

    If method is 'active', then 3 parameters are available as listed below.

    **Example**: reset_qubit('active', threshold=-0.003, max_tries=3)

    :param method: Method the reset the qubit state. Can be either 'cooldown' or 'active'.
    :param qubit: The qubit to be addressed in QuAM
    :key cooldown_time: qubit relaxation time in clock cycle, needed if method is 'cooldown'. Must be an integer > 4.
    :key threshold: threshold to discriminate between the ground and excited state, needed if method is 'active'.
    :key max_tries: python integer for the maximum number of tries used to perform active reset,
        needed if method is 'active'. Must be an integer > 0 and default value is 1.
    :key Ig: A QUA variable for the information in the `I` quadrature used for active reset. If not given, a new
        variable will be created. Must be of type `Fixed`.
    :key pi_pulse: The pulse to play to get back to the ground state. Default is 'x180'.
    :return:
    """
    if method == "cooldown":
        # Check cooldown_time
        cooldown_time = kwargs.get("cooldown_time", None)
        if cooldown_time is None or cooldown_time < 4:
            raise Exception("'cooldown_time' must be an integer > 4 clock cycles")
        # Reset qubit state
        qubit.xy.wait(cooldown_time)
        return None
    if method == "active":
        # Check threshold
        threshold = kwargs.get("threshold", None)
        if threshold is None:
            raise Exception("'threshold' must be specified for active reset.")
        # Check max_tries
        max_tries = kwargs.get("max_tries", 1)
        if max_tries is None or (not float(max_tries).is_integer()) or max_tries < 1:
            raise Exception("'max_tries' must be an integer > 0.")
        # Check Ig
        Ig = kwargs.get("Ig", None)
        pi_pulse_name = kwargs.get("pi_pulse", "x180")
        # Reset qubit state
        return active_reset(threshold, qubit, max_tries=max_tries, Ig=Ig, pi_pulse=pi_pulse_name)
    raise ValueError(f"Unknown reset method: {method}. Use 'cooldown' or 'active'.")


# Macro for performing active reset until successful for a given number of tries.
def active_reset(threshold: float, qubit: FluxTunableTransmon, max_tries=1, Ig=None, pi_pulse: str = "x180"):
    """Macro for performing active reset until successful for a given number of tries.

    :param threshold: threshold for the 'I' quadrature discriminating between ground and excited state.
    :param qubit: The qubit element. Must be defined in the config.
    :param resonator: The resonator element. Must be defined in the config.
    :param max_tries: python integer for the maximum number of tries used to perform active reset. Must >= 1.
    :param Ig: A QUA variable for the information in the `I` quadrature. Should be of type `Fixed`. If not given, a new
        variable will be created
    :param pi_pulse: The pulse to play to get back to the ground state. Default is 'x180'.
    :return: A QUA variable for the information in the `I` quadrature and the number of tries after success.
    """
    if Ig is None:
        Ig = declare(fixed)
    if (max_tries < 1) or (not float(max_tries).is_integer()):
        raise Exception("max_count must be an integer >= 1.")
    # Initialize Ig to be > threshold
    assign(Ig, threshold + 2**-28)
    # Number of tries for active reset
    counter = declare(int)
    # Reset the number of tries
    assign(counter, 0)

    # Perform active feedback
    qubit.xy.align(qubit.resonator.name)
    # Use a while loop and counter for other protocols and tests
    with while_((Ig > threshold) & (counter < max_tries)):
        # Measure the resonator
        qubit.resonator.measure("readout")
        # Play a pi pulse to get back to the ground state (QUA requires parens)
        qubit.xy.play(pi_pulse, condition=(Ig > threshold))
        # Increment the number of tries
        assign(counter, counter + 1)
    return Ig, counter


def align_transmon(qubit: FluxTunableTransmon):
    """
    Macro to align all qubit drives with the associated resonator
    """
    qubit.xy.align(qubit.resonator.name, qubit.z.name)


def align_transmon_pair(qubit_pair: FluxTunableTransmonPair):
    """Align all qubit drives with the associated resonators for a given qubit pair."""
    all_channels = ["xy", "z", "resonator"]
    all_elements = []
    for qubit in [qubit_pair.qubit_control, qubit_pair.qubit_target]:
        for channel in all_channels:
            all_elements.append(getattr(qubit, channel).name)
    align(*all_elements)


def get_parallel_gate_combinations(
    coupling_map: CouplingMap, direction="forward"
):
    """
    Returns all possible combinations of qubit pairs for which a two-qubit gate can be applied in parallel,
    respecting the specified direction constraint.

    Parameters:
    - coupling_map: Qiskit CouplingMap object that represents the qubit connectivity.
    - direction: 'forward' or 'reverse' to indicate which direction of qubit pairs should be selected.

    Returns:
    - List of combinations where the maximum number of two-qubit gates can be applied in parallel.
    """
    # Get all possible two-qubit gate pairs
    qubit_pairs = coupling_map.get_edges()

    # Create a set to store unique pairs in the specified direction
    filtered_pairs = set()
    for q1, q2 in qubit_pairs:
        if direction == "forward":
            # Add the pair if it's in the forward direction
            if (q2, q1) not in filtered_pairs:
                filtered_pairs.add((q1, q2))
        elif direction == "reverse":
            # Add the reversed pair if the original forward pair exists
            if (q1, q2) not in filtered_pairs:
                filtered_pairs.add((q2, q1))

    # Convert the set back to a list for further processing
    qubit_pairs = list(filtered_pairs)

    max_parallel_combinations = []
    max_num_parallel_gates = 0

    # Check all possible combinations of the qubit pairs
    for r in range(1, len(qubit_pairs) + 1):
        for combo in combinations(qubit_pairs, r):
            # Check if all pairs in the combination can be applied in parallel
            used_qubits = set()
            valid = True
            for pair in combo:
                if pair[0] in used_qubits or pair[1] in used_qubits:
                    valid = False
                    break
                used_qubits.update(pair)

            if valid:
                if len(combo) > max_num_parallel_gates:
                    max_num_parallel_gates = len(combo)
                    max_parallel_combinations = [combo]
                elif len(combo) == max_num_parallel_gates:
                    max_parallel_combinations.append(combo)

    return max_parallel_combinations


def generate_circuits(
    xeb_config, gate_indices: np.ndarray, available_combinations
) -> List[List[QuantumCircuit]]:
    """Generate XEB circuits from gate indices."""
    two_qubit_gate_pattern = 0
    n_qubits = xeb_config.n_qubits
    circuits = []
    if all(isinstance(qubit, FluxTunableTransmon) for qubit in xeb_config.qubits):
        qubit_names = [qubit.name for qubit in xeb_config.qubits]
    else:
        qubit_names = xeb_config.qubits
    for s in range(xeb_config.seqs):
        circuits.append([])
        for d_, depth in enumerate(xeb_config.depths):
            q_regs = [QuantumRegister(1, qubit_name) for qubit_name in qubit_names]
            qc = QuantumCircuit(*q_regs)
            for d in range(depth):
                for q in range(n_qubits):
                    sq_gate = xeb_config.gate_set[gate_indices[s, q, d]].gate
                    qc.append(sq_gate, [q])
                qc.barrier()
                if xeb_config.two_qb_gate is not None:
                    for i, combination in enumerate(available_combinations):
                        if i == two_qubit_gate_pattern:
                            for pair in combination:
                                qc.append(xeb_config.two_qb_gate.gate, pair)
                            qc.barrier()
                            break
                    if two_qubit_gate_pattern == len(available_combinations) - 1:
                        two_qubit_gate_pattern = 0
                    else:
                        two_qubit_gate_pattern += 1

                    # qc.append(self.xeb_config.two_qb_gate.gate, [0, 1])
            qc.measure_all()
            circuits[s].append(qc)
            two_qubit_gate_pattern = 0
    return circuits


def generate_circuits_parameterized(
    xeb_config, gate_indices: np.ndarray, available_combinations, en_measure=False
) -> List[List[QuantumCircuit]]:
    """Generate parameterized XEB circuits from gate indices."""
    two_qubit_gate_pattern = 0
    n_qubits = xeb_config.n_qubits
    circuits = []
    if all(isinstance(qubit, FluxTunableTransmon) for qubit in xeb_config.qubits):
        qubit_names = [qubit.name for qubit in xeb_config.qubits]
    else:
        qubit_names = xeb_config.qubits
    for s in range(xeb_config.seqs):  # sequence
        circuits.append([])
        for d_, depth in enumerate(xeb_config.depths):  # depth
            q_regs = [QuantumRegister(1, qubit_name) for qubit_name in qubit_names]
            qc = QuantumCircuit(*q_regs)
            fSim_inst, theta, phi = generate_fSim_instruction()
            for d in range(depth):  # cycle
                for q in range(n_qubits):
                    sq_gate = xeb_config.gate_set[gate_indices[s, q, d]].gate
                    qc.append(sq_gate, [q])
                qc.barrier()

                for i, combination in enumerate(available_combinations):
                    if i == two_qubit_gate_pattern:
                        for pair in combination:
                            qc.append(fSim_inst, pair)
                        qc.barrier()
                        break
                if two_qubit_gate_pattern == len(available_combinations) - 1:
                    two_qubit_gate_pattern = 0
                else:
                    two_qubit_gate_pattern += 1

                    # qc.append(self.xeb_config.two_qb_gate.gate, [0, 1])
            if en_measure:
                qc.measure_all()
            circuits[s].append(qc)
            two_qubit_gate_pattern = 0
    return circuits, theta, phi


def generate_fSim_instruction():
    """Generate fSim gate instruction with theta and phi parameters."""
    theta = Parameter("θ")  # iSWAP angle
    phi = Parameter("φ")  # CZ angle

    qc = QuantumCircuit(2)

    qc.rxx(theta, 0, 1)
    qc.ryy(theta, 0, 1)
    qc.cp(phi, 0, 1)

    fSim_inst = qc.to_instruction(label="fSim")

    return fSim_inst, theta, phi


def prepare_gate_combination_LUT(
    theta_iswap, phi_cphase, phi_rz1, phi_rz2, separate_2q_gate=False
):
    """
    Prepare lookup tables (LUTs) for gate combinations used in quantum circuit construction.

    This function defines a set of single-qubit gates and combines them with a fixed two-qubit
    fSim gate to generate all possible combinations of gate layers. The resulting LUTs are used
    to construct quantum circuits with interleaved gate patterns.

    Parameters
    ----------
    theta_iswap : float
        The iSWAP angle parameter for the fSim gate.
    phi_cphase : float
        The controlled phase angle for the fSim gate.
    phi_rz1 : float
        The single-qubit Z-rotation phase applied to qubit 1.
    phi_rz2 : float
        The single-qubit Z-rotation phase applied to qubit 2.

    Returns
    -------
    LUT_1layer : list of list of ndarray
        A 3x3 list containing unitary matrices for one layer of gate combinations.
    LUT_2layer : list of list of list of list of ndarray
        A 3x3x3x3 list containing unitary matrices for two layers of gate combinations.

    Notes
    -----
    The gate definitions follow the conventions from:
    F. Arute et al., Nature 574, 505 (2019), Supplementary Equations (50)-(53).
    """
    # Define single-qubit gate unitaries
    SX = np.array([[1, -1j], [-1j, 1]], dtype=complex) / 2**0.5
    SY = np.array([[1, -1], [1, 1]], dtype=complex) / 2**0.5
    SW = np.array([[1, -(1j**0.5)], [(-1j) ** 0.5, 1]], dtype=complex) / 2**0.5
    I = np.eye(2, dtype=complex)

    # Define single-qubit gate unitaries on 2-qubit space via tensor product
    SX1 = np.kron(I, SX)
    SY1 = np.kron(I, SY)
    SW1 = np.kron(I, SW)
    SX2 = np.kron(SX, I)
    SY2 = np.kron(SY, I)
    SW2 = np.kron(SW, I)

    # Define the fSim gate unitary
    def fSim(theta_iswap=0, phi_cphase=np.pi):
        mat = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(theta_iswap), -1j * np.sin(theta_iswap), 0],
                [0, -1j * np.sin(theta_iswap), np.cos(theta_iswap), 0],
                [0, 0, 0, np.exp(-1j * (phi_cphase))],
            ]
        )
        return mat

    def RZ(phi_rz):
        return np.array([[1, 0], [0, np.exp(-1j * phi_rz)]], dtype=complex)

    def parallel_single_RZ(phi_rz1=0, phi_rz2=0):
        mat = np.kron(RZ(phi_rz2), RZ(phi_rz1))
        return mat

    gate2q = parallel_single_RZ(phi_rz1, phi_rz2) @ fSim(theta_iswap, phi_cphase)
    if separate_2q_gate:
        # Generate 1-layer LUT: all combinations of single-qubit gates
        LUT_1layer = [[_gate2 @ _gate1 for _gate2 in [SX2, SY2, SW2]] for _gate1 in [SX1, SY1, SW1]]
    else:
        # Generate 1-layer LUT: all combinations of single-qubit gates followed by fSim
        LUT_1layer = [[gate2q @ _gate2 @ _gate1 for _gate2 in [SX2, SY2, SW2]] for _gate1 in [SX1, SY1, SW1]]

    # if separate_2q_gate:
    #     LUT_2layer = None
    # else:
    #     # Generate 2-layer LUT: all combinations of two 1-layer gate sequences
    #     LUT_2layer = [
    #         [
    #             [
    #                 [
    #                     LUT_1layer[_i_l2q1][_i_l2q2] @ LUT_1layer[_i_l1q1][_i_l1q2]
    #                     for _i_l2q2 in range(3)
    #                 ]
    #                 for _i_l2q1 in range(3)
    #             ]
    #             for _i_l1q2 in range(3)
    #         ]
    #         for _i_l1q1 in range(3)
    #     ]

    return LUT_1layer, gate2q


def calc_ideal_probability_numpy(
    gate_indices, depths, theta_iswap, phi_cphase, phi_rz1, phi_rz2, insert_2q_gate=True
):
    """
    Calculate ideal output probabilities for a set of quantum circuits using NumPy.

    This function simulates quantum circuits composed of gate sequences defined by
    `gate_indices`, applies the corresponding unitary operations, and computes the
    probability of measuring the system in the |00⟩, |01⟩, |10⟩ and |11⟩ states at specified depths.

    Parameters
    ----------
    gate_indices : ndarray
        A 3D integer array of shape (n_sequences, 2, n_layers), where each entry specifies
        the gate index for qubit 1 and qubit 2 at each layer.
    depths : list of int
        A list of layer indices at which to record the output probabilities.
    theta_iswap : float
        The iSWAP angle parameter for the fSim gate.
    phi_cphase : float
        The controlled phase angle for the fSim gate.
    phi_rz1 : float
        The Z-rotation phase applied to qubit 1 in the fSim gate.
    phi_rz2 : float
        The Z-rotation phase applied to qubit 2 in the fSim gate.

    Returns
    -------
    ideal_probability_s : ndarray
        A 3D array of shape (n_sequences, len(depths), 4) containing the probabilities
        of measuring each basis state at the specified depths.
    """
    # Prepare gate lookup tables
    LUT_1layer, _ = prepare_gate_combination_LUT(
        theta_iswap, phi_cphase, phi_rz1, phi_rz2, separate_2q_gate=(not insert_2q_gate)
    )

    ideal_probability_s = []

    # Loop over each gate sequence
    for i_seq in range(gate_indices.shape[0]):
        circuit_unitary = np.eye(4, dtype=complex)
        ideal_probability = []

        # Apply gates layer by layer
        for _i_layer in range(gate_indices.shape[2]):
            i_q1 = gate_indices[i_seq, 0, _i_layer]
            i_q2 = gate_indices[i_seq, 1, _i_layer]
            circuit_unitary = LUT_1layer[i_q1][i_q2] @ circuit_unitary

            # Record probability at specified depth
            if _i_layer + 1 in depths:
                ideal_probability.append(np.abs(circuit_unitary[:, 0]) ** 2)

        ideal_probability_s.append(ideal_probability)

    ideal_probability_s = np.array(ideal_probability_s)
    return ideal_probability_s


def simulate_noisy_circuit_numpy(
    gate_indices,
    depths,
    theta_iswap,
    phi_cphase,
    phi_rz1,
    phi_rz2,
    one_over_f_amplitude_at_1Hz_GHz_per_rHz=5e-6,
    white_noise_amplitude_GHz_per_rHz=0,
    gate_time_1q_ns: int = 32,
    gate_time_2q_ns: int = 80,
    n_noise_sample=101,
):
    """
    Calculate ideal output probabilities for a set of quantum circuits using NumPy.

    This function simulates quantum circuits composed of gate sequences defined by
    `gate_indices`, applies the corresponding unitary operations, and computes the
    probability of measuring the system in the |00⟩, |01⟩, |10⟩ and |11⟩ states at specified depths.

    Parameters
    ----------
    gate_indices : ndarray
        A 3D integer array of shape (n_sequences, 2, n_layers), where each entry specifies
        the gate index for qubit 1 and qubit 2 at each layer.
    depths : list of int
        A list of layer indices at which to record the output probabilities.
    theta_iswap : float
        The iSWAP angle parameter for the fSim gate.
    phi_cphase : float
        The controlled phase angle for the fSim gate.
    phi_rz1 : float
        The Z-rotation phase applied to qubit 1 in the fSim gate.
    phi_rz2 : float
        The Z-rotation phase applied to qubit 2 in the fSim gate.

    Returns
    -------
    ideal_probability_s : ndarray
        A 3D array of shape (n_sequences, len(depths), 4) containing the probabilities
        of measuring each basis state at the specified depths.
    """
    # Prepare gate lookup tables
    LUT_1layer, gate2q = prepare_gate_combination_LUT(theta_iswap, phi_cphase, phi_rz1, phi_rz2, separate_2q_gate=True)
    n_layer = gate_indices.shape[2]

    total_circuit_time = n_layer * (gate_time_1q_ns + gate_time_2q_ns)
    dt = np.gcd(gate_time_1q_ns, gate_time_2q_ns)  # greatest common divider
    n_samples = total_circuit_time // dt
    if n_samples % 2 == 1:
        n_samples += 1
    sampling_time_index_1q = np.arange(n_layer) * (gate_time_1q_ns + gate_time_2q_ns) // dt
    sampling_time_index_2q = sampling_time_index_1q + gate_time_2q_ns // dt

    # Generate noise and PSD
    alpha = 1  # 1/f noise

    def RZ(phi_rz):
        return np.array([[1, 0], [0, np.exp(-1j * phi_rz)]], dtype=complex)

    def parallel_single_RZ(phi_rz1=0, phi_rz2=0):
        mat = np.kron(RZ(phi_rz2), RZ(phi_rz1))
        return mat

    ideal_probability_ss = []

    # Loop over each gate sequence
    for i_seq in range(gate_indices.shape[0]):
        ideal_probability_s = []
        for _noise_seed in range(n_noise_sample):
            circuit_unitary = np.eye(4, dtype=complex)
            ideal_probability = []
            q1_frequency_trace_GHz = generate_1_over_f_type_noise(
                amplitude=one_over_f_amplitude_at_1Hz_GHz_per_rHz,
                dt=dt * 1e-9,
                n_samples=n_samples,
                seed=2 * i_seq * _noise_seed,
                alpha=alpha,
                white_noise_amplitude=white_noise_amplitude_GHz_per_rHz,
            )
            q2_frequency_trace_GHz = generate_1_over_f_type_noise(
                amplitude=one_over_f_amplitude_at_1Hz_GHz_per_rHz,
                dt=dt * 1e-9,
                n_samples=n_samples,
                seed=(2 * i_seq + 1) * _noise_seed,
                alpha=alpha,
                white_noise_amplitude=white_noise_amplitude_GHz_per_rHz,
            )

            # Apply gates layer by layer
            for _i_layer in range(n_layer):

                i_q1 = gate_indices[i_seq, 0, _i_layer]
                i_q2 = gate_indices[i_seq, 1, _i_layer]

                phi_rz1_1q = (2 * np.pi) * q1_frequency_trace_GHz[sampling_time_index_1q[_i_layer]] * gate_time_1q_ns
                phi_rz2_1q = (2 * np.pi) * q2_frequency_trace_GHz[sampling_time_index_1q[_i_layer]] * gate_time_1q_ns
                phi_rz1_2q = (2 * np.pi) * q1_frequency_trace_GHz[sampling_time_index_2q[_i_layer]] * gate_time_2q_ns
                phi_rz2_2q = (2 * np.pi) * q2_frequency_trace_GHz[sampling_time_index_2q[_i_layer]] * gate_time_2q_ns

                phase_noise_after_1q = parallel_single_RZ(phi_rz1=phi_rz1_1q, phi_rz2=phi_rz2_1q)
                phase_noise_after_2q = parallel_single_RZ(phi_rz1=phi_rz1_2q, phi_rz2=phi_rz2_2q)

                layer_gate = LUT_1layer[i_q1][i_q2]
                circuit_unitary = phase_noise_after_2q @ gate2q @ phase_noise_after_1q @ layer_gate @ circuit_unitary

                # Record probability at specified depth
                if _i_layer + 1 in depths:
                    ideal_probability.append(np.abs(circuit_unitary[:, 0]) ** 2)
            ideal_probability_s.append(ideal_probability)
        ideal_probability_ss.append(ideal_probability_s)
    ideal_probability_ss = np.array(ideal_probability_ss)
    return ideal_probability_ss


def generate_1_over_f_type_noise(
    amplitude, dt, n_samples, seed=0, alpha=1, white_noise_amplitude=0
):
    """
    Generate 1/f^alpha noise using frequency-domain filtering.

    Parameters
    ----------
    amplitude : float
        amplitude spectral density of 1/f type noise at 1 Hz in 1/√Hz.
    dt : float
        Sampling interval in seconds.
    n_samples : int
        Number of samples in the time-domain signal. NEED to be even number
    seed : int, optional
        Random seed for reproducibility.
    alpha : float, optional
        Spectral exponent. alpha = 1: 1/f noise, alpha = 0: white noise. Default is 1.
    white_noise_amplitude : float, optional
        amplitude spectral density of white noise in 1/√Hz.

    Returns
    -------
    noise : ndarray
        Time-domain signal exhibiting 1/f^alpha noise characteristics.

    Notes
    -----
    This function generates colored noise by filtering white noise in the frequency domain.
    The amplitude of each frequency component is scaled by 1/f^alpha, and the result is
    transformed back to the time domain using an inverse FFT.

    The output noise has a spectral density that approximates the desired 1/f^alpha behavior,
    normalized to match the specified amplitude spectral density at 1 Hz.
    """
    np.random.seed(seed)
    sampling_rate = 1 / dt

    freqs = np.fft.rfftfreq(n_samples, d=dt)
    freqs[0] = freqs[1]  # Avoid division by zero at f=0

    # Generate white noise with the desired spectral density
    std_dev_base = amplitude * np.sqrt(sampling_rate / 2)
    white_noise_base = np.random.normal(loc=0, scale=std_dev_base, size=n_samples)

    FT_white_noise = np.fft.rfft(white_noise_base)
    FT_noise = FT_white_noise / freqs ** (alpha * 0.5)
    noise = np.fft.irfft(FT_noise)

    if white_noise_amplitude:
        std_dev_white = white_noise_amplitude * np.sqrt(sampling_rate / 2)
        white_noise = np.random.normal(loc=0, scale=std_dev_white, size=n_samples)
        noise = noise + white_noise

    return noise


def generate_gate_indices(xeb_config):
    """Generate random gate indices for XEB sequences."""
    num_qubits = len(xeb_config.qubits)
    num_gates = len(xeb_config.gate_set)
    gate_indices = []
    np.random.seed(xeb_config.seed)

    for _s in range(xeb_config.seqs):  # For each sequence
        gate_indices_tmp1 = []
        for _q in range(num_qubits):  # For each qubit
            gate_indices_tmp2 = []
            previous_gate = np.nan
            for _d in range(xeb_config.depths[-1]):
                next_gate = np.random.randint(num_gates)
                while next_gate == previous_gate:  # Make sure that the same gate is not applied twice in a row
                    next_gate = np.random.randint(num_gates)
                gate_indices_tmp2.append(next_gate)
                previous_gate = next_gate
            gate_indices_tmp1.append(gate_indices_tmp2)
        gate_indices.append(gate_indices_tmp1)
    gate_indices = np.array(gate_indices)
    return gate_indices


def binary(n, length):
    """
    Convert an integer to a binary string of a given length
    :param n: Integer to convert
    :param length: Length of the output string
    :return: Binary string corresponding to integer n
    """
    return bin(n)[2:].zfill(length)


def cross_entropy(p, q, epsilon=1e-15):
    """
    Calculate cross entropy between two probability distributions.

    Parameters:
    - p: numpy array, the true probability distribution
    - q: numpy array, the predicted probability distribution
    - epsilon: small value to avoid taking the logarithm of zero

    Returns:
    - Cross entropy between p and q
    """
    q = np.maximum(q, epsilon)  # Avoid taking the logarithm of zero
    x_entropy = -np.sum(p * np.log(q))
    return x_entropy


def compute_log_fidelity(incoherent_dist, expected_probs, measured_probs):
    """
    Compute the log fidelity between the expected and measured distributions.

    Parameters:
    - incoherent_dist: numpy array, the incoherent distribution
    - expected_probs: numpy array, the expected probabilities
    - measured_probs: numpy array, the measured probabilities

    Returns:
    - The log fidelity between the expected and measured distributions
    """
    # Compute the cross entropy between the incoherent distribution and the expected probabilities
    xe_incoherent = cross_entropy(incoherent_dist, expected_probs)
    xe_measured = cross_entropy(measured_probs, expected_probs)
    xe_expected = cross_entropy(expected_probs, expected_probs)

    f_xeb = (xe_incoherent - xe_measured) / (xe_incoherent - xe_expected)
    if np.isnan(f_xeb):
        print(f"[DEBUG] xe_incoherent: {xe_incoherent}, xe_measured: {xe_measured}, xe_expected: {xe_expected}")

    return f_xeb


def evaluate_log_fidelity(f_xeb, singularity, outlier, seq, depth):
    """
    Evaluate the log fidelity and return the corresponding value.
    """
    if np.isnan(f_xeb) or np.isinf(f_xeb):
        singularity.append((seq, depth))
        return np.nan
    if f_xeb < 0 or f_xeb > 1:
        outlier.append((seq, depth))
        return np.nan
    return f_xeb


def update_record(
    records, seq, depth, expected_probs, measured_probs, dim
):
    """
    Update the record to compute linear fidelities (Cirq like processing).
    """
    records += [
        {
            "sequence": seq,
            "depth": depth,
            "pure_probs": expected_probs,
            "measured_probs": measured_probs,
            "e_u": np.sum(expected_probs**2),
            "u_u": np.sum(expected_probs) / dim,
            "m_u": np.sum(measured_probs * expected_probs),
        }
    ]
    return records


def update_data_frame(df):
    """
    Update the data frame to compute linear fidelities (Cirq like processing).
    """
    try:
        df["y"] = df["m_u"] - df["u_u"]
        df["x"] = df["e_u"] - df["u_u"]
        df["numerator"] = df["x"] * df["y"]
        df["denominator"] = df["x"] ** 2
        return df

    except KeyError:
        raise ValueError("The records for linear XEB are empty. Please rerun the experiment.")


def create_subplot(data, subplot_number, title, depths, seqs):
    """Create a subplot for XEB data visualization."""
    print(title)
    print(f"data: {data}")
    print(subplot_number)
    plt.subplot(subplot_number)
    # plt.pcolor(depths, range(seqs), np.abs(data), vmin=0., vmax=1.)
    plt.pcolor(depths, range(seqs), np.abs(data))
    ax = plt.gca()
    ax.set_title(title)
    if subplot_number > 244:
        ax.set_xlabel("Circuit depth")
    ax.set_ylabel("Sequences")
    ax.set_xticks(depths)
    ax.set_yticks(np.arange(1, seqs + 1))
    plt.colorbar()


# Define Cirq functions for fitting (redefined here for avoiding additional dependencies)
# Those functions are slightly adapted to deal with possible singularities and outliers in the data
def exponential_decay(cycle_depths: np.ndarray, a: float, layer_fid: float) -> np.ndarray:
    """An exponential decay for fitting.

    This computes `a * layer_fid**cycle_depths`

    Args:
        cycle_depths: The various depths at which fidelity was estimated. This is the independent
            variable in the exponential function.
        a: A scale parameter in the exponential function.
        layer_fid: The base of the exponent in the exponential function.
    """
    return a * layer_fid**cycle_depths


def fit_exponential_decay(cycle_depths: np.ndarray, fidelities: np.ndarray) -> tuple[float, float, float, float]:
    """Fit an exponential model fidelity = a * layer_fid**x using nonlinear least squares.

    This uses `exponential_decay` as the function to fit with parameters `a` and `layer_fid`.
    This function is taken from Cirq: cirq-core/cirq/experiments/xeb_fitting.py

    Args:
        cycle_depths: The various depths at which fidelity was estimated. Each element is `x`
            in the fit expression.
        fidelities: The estimated fidelities for each cycle depth. Each element is `fidelity`
            in the fit expression.

    Returns:
        a: The first fit parameter that scales the exponential function, perhaps accounting for
            state prep and measurement (SPAM) error.
        layer_fid: The second fit parameters which serves as the base of the exponential.
        a_std: The standard deviation of the `a` parameter estimate.
        layer_fid_std: The standard deviation of the `layer_fid` parameter estimate.
    """
    cycle_depths = np.asarray(cycle_depths)
    fidelities = np.asarray(fidelities)
    mask = (fidelities > 0) & (fidelities <= 1)
    # mask = ~np.isnan(fidelities)
    print()
    masked_cycle_depths = cycle_depths[mask]
    masked_fidelities = fidelities[mask]

    log_fidelities = np.log(masked_fidelities)

    slope, intercept, _, _, _ = stats.linregress(masked_cycle_depths, log_fidelities)
    layer_fid_0 = np.clip(np.exp(slope), 0, 1)
    a_0 = np.clip(np.exp(intercept), 0, 1)

    try:
        (a, layer_fid), pcov = optimize.curve_fit(
            exponential_decay,
            masked_cycle_depths,
            masked_fidelities,
            p0=(a_0, layer_fid_0),
            bounds=((0, 0), (1, 1)),
            nan_policy="omit",
        )
    except ValueError:  # pragma: no cover
        return 0, 0, np.inf, np.inf

    a_std, layer_fid_std = np.sqrt(np.diag(pcov))
    return a, layer_fid, a_std, layer_fid_std


def load_XEB_data_qualibrate(directory, disjoint_processing=None, data_handler=None, parameterize_circuit=False):
    """
    Retrieve the XEBResult object from a saved data file (JSON format)

    Args:
        directory: Directory of the saved data files (should contain data.json and node.json)
        disjoint_processing: Indicate if disjoint processing should be applied to the results
        data_handler: DataHandler object to handle the data
    """
    from quam_libs.components.experiments.two_qubit_xeb.xeb import (
        XEBConfig,
    )  # Import XEBConfig here to avoid circular import

    with open(directory + "/data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    with np.load(directory + "/arrays.npz") as arrays_file:
        arrays = dict(arrays_file)
    # metadata: Dict = json.load(open(directory + "/node.json", "r"))
    xeb_config = XEBConfig.from_dict(data["xeb_config"])
    if disjoint_processing is not None:
        assert isinstance(disjoint_processing, bool), "disjoint_processing should be a boolean"
        xeb_config.disjoint_processing = disjoint_processing
    # print(arrays.files)
    gate_indices = arrays["data.gate_indices"]
    if parameterize_circuit:
        circuits, _, _ = generate_circuits_parameterized(
            xeb_config, gate_indices, xeb_config.available_combinations, en_measure=False
        )
    else:
        circuits = generate_circuits(xeb_config, gate_indices, xeb_config.available_combinations)

    new_data = {"states": {}, "counts": {}, "quadratures": {}, "amp_st": {}}
    print(f"[DEBUG] {list(arrays.keys())}")
    for key, value in data["data"].items():
        if "state" in key:  # state_q1 : 'data.state_q1'
            new_data["states"][key] = arrays["data." + key]
        elif ("00" in key) or ("01" in key) or ("10" in key) or ("11" in key):  # 00 :  'data.00'
            new_data["counts"][key] = arrays["data." + key]
        elif "amp_matrix" in key:
            new_data["amp_st"][key] = arrays["data." + key]
        # elif key.startswith("I") or key.startswith("Q"):
        #     new_data["quadratures"][key] = arrays['data.'+key]
        # else:
        #     if key in arrays:
        #         new_data[key] = arrays[key]
        #     else:
        #         new_data[key] = value
    new_data["gate_indices"] = gate_indices

    return xeb_config, circuits, new_data, data_handler, parameterize_circuit


def load_XEB_data_qualibrate_simulate(
    directory: str, disjoint_processing=None, data_handler=None, parameterize_circuit=False
):
    """
    Retrieve the XEBResult object from a saved data file (JSON format)

    Args:
        directory: Directory of the saved data files (should contain data.json and node.json)
        disjoint_processing: Indicate if disjoint processing should be applied to the results
        data_handler: DataHandler object to handle the data
    """
    from quam_libs.components.experiments.two_qubit_xeb.xeb import (
        XEBConfig,
    )  # Import XEBConfig here to avoid circular import

    with open(directory + "/data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    with np.load(directory + "/arrays.npz") as arrays_file:
        arrays = dict(arrays_file)
    # metadata: Dict = json.load(open(directory + "/node.json", "r"))
    xeb_config = XEBConfig.from_dict(data["xeb_config"])
    if disjoint_processing is not None:
        assert isinstance(disjoint_processing, bool), "disjoint_processing should be a boolean"
        xeb_config.disjoint_processing = disjoint_processing
    # print(arrays.files)
    gate_indices = arrays["data.gate_indices"]
    if parameterize_circuit:
        circuits, _, _ = generate_circuits_parameterized(
            xeb_config, gate_indices, xeb_config.available_combinations, en_measure=False
        )
    else:
        circuits = generate_circuits(xeb_config, gate_indices, xeb_config.available_combinations)

    new_data = {"states": {}, "counts": {}, "quadratures": {}, "amp_st": {}}
    print(f"[DEBUG 1] {list(arrays.keys())}")
    for key, value in data["data"].items():
        print(f"[DEBUG 1.1] {key}")
        if "state" in key:  #
            for key2, value2 in value.items():
                new_data["states"][key2] = arrays["data.states." + key2]
        elif "count" in key:
            for key2, value2 in value.items():
                print(f"[DEBUG 2] key : {key2}")
                new_data["counts"][key2] = arrays["data.counts." + key2]
        # elif "amp_matrix" in key:
        #     new_data["amp_st"][key] = arrays['data.'+key]
        # elif key.startswith("I") or key.startswith("Q"):
        #     new_data["quadratures"][key] = arrays['data.'+key]
        # else:
        #     if key in arrays:
        #         new_data[key] = arrays[key]
        #     else:
        #         new_data[key] = value
    new_data["gate_indices"] = gate_indices

    return xeb_config, circuits, new_data, data_handler, parameterize_circuit


# def retrieve_data(self, theta:float=0, phi:float=np.pi):
#         """
#         Retrieve the data from the XEB experiment

#         Returns:
#             measured_probs: Measured probabilities of the states
#             expected_probs: Expected probabilities of the states
#             records: Records of the experiment
#             log_fidelities: Logarithmic fidelities
#             linear_fidelities: Linear fidelities
#             singularities: Singularities
#             outliers: Outliers
#         """
#         dim = 2 ** len(self.xeb_config.qubits)
#         n_qubits = len(self.xeb_config.qubits)
#         seqs = self.xeb_config.seqs
#         depths = self.xeb_config.depths
#         counts = self.counts
#         states = self.states

#         self.ideal_probability_s = calc_ideal_probability_numpy(
#             self.data['gate_indices'], self.xeb_config.depths,
#             theta_iswap=theta, phi_cphase=phi, phi_rz1=0, phi_rz2=0)

#         existing_data = "joint_expected_probs" in self.data.keys()

#         for s in range(seqs):
#             for d_, depth in enumerate(depths):
#                 if self.parameterize_circuit:
#                     sorted_params = sorted(self.circuits[s][d_].parameters, key=lambda p: p.name)
#                     values = [theta, phi]
#                     parameters = dict(zip(sorted_params, values))
#                     self.circuits_parameter_assigned[s][d_] = (
#                         self.circuits[s][d_].assign_parameters(parameters=parameters))

#         if not existing_data or self.parameterize_circuit:
#             joint_expected_probs = np.zeros((seqs, len(depths), dim))
#             joint_measured_probs = np.zeros((seqs, len(depths), dim))

#             disjoint_expected_probs = np.zeros((n_qubits, seqs, len(depths), 2))
#             disjoint_measured_probs = np.zeros((n_qubits, seqs, len(depths), 2))
#         else:
#             joint_expected_probs = self.data["joint_expected_probs"]
#             joint_measured_probs = self.data["joint_measured_probs"]

#             disjoint_expected_probs = self.data["disjoint_expected_probs"]
#             disjoint_measured_probs = self.data["disjoint_measured_probs"]

#         if not self.xeb_config.disjoint_processing:
#             records, singularity, outlier = [], [], []
#             incoherent_distribution = np.ones(dim) / dim
#             log_fidelities = np.zeros((seqs, len(depths)))

#         else:
#             records = [[] for _ in range(n_qubits)]
#             singularity = [[] for _ in range(n_qubits)]
#             outlier = [[] for _ in range(n_qubits)]
#             incoherent_distribution = np.ones(2) / 2
#             log_fidelities = np.zeros((n_qubits, seqs, len(depths)))

#         self.incoherent_distribution = incoherent_distribution

#         for s in range(seqs):
#             for d_, depth in enumerate(depths):
#                 if self.parameterize_circuit:
#                     qc = self.circuits_parameter_assigned[s][d_]
#                 else:
#                     qc = self.circuits[s][d_]
#                     qc = qc.remove_final_measurements(inplace=False)
#                 if not existing_data or self.parameterize_circuit:
#                     statevector = Statevector(qc)
#                     joint_expected_probs[s, d_] = statevector.probabilities()#decimals=5)
#                     joint_measured_probs[s, d_] = (
#                         np.array([counts[binary(i, n_qubits)][s][d_] for i in range(dim)]) / self.xeb_config.n_shots
#                     )

#                     for q in range(n_qubits):
#                         disjoint_expected_probs[q, s, d_] = statevector.probabilities([q], 5)
#                         qubit_state = states[f"state_{self.qubit_names[q]}"][s, d_]
#                         disjoint_measured_probs[q, s, d_] = np.array([1 - qubit_state, qubit_state])

#                 if not self.xeb_config.disjoint_processing:
#                     # Calculate the cross-entropy fidelities (logarithmic)
#                     f_xeb = compute_log_fidelity(
#                         incoherent_distribution, joint_expected_probs[s, d_], joint_measured_probs[s, d_]
#                     )
#                     log_fidelities[s, d_] = evaluate_log_fidelity(f_xeb, singularity, outlier, s, int(depth))

#                     # Store records for linear XEB post-processing
#                     records = update_record(
#                         records, s, depth, joint_expected_probs[s, d_], joint_measured_probs[s, d_], dim
#                     )

#                 else:
#                     for q, qubit_name in enumerate(self.qubit_names):
#                         # Calculate the cross-entropy fidelities (logarithmic)
#                         f_xeb = compute_log_fidelity(
#                             incoherent_distribution,
#                             disjoint_expected_probs[q, s, d_],
#                             disjoint_measured_probs[q, s, d_],
#                         )
#                         log_fidelities[q, s, d_] = evaluate_log_fidelity(f_xeb, singularity[q], outlier[q],
#                                                                          s, int(depth))
#                         # Store records for linear XEB post-processing
#                         records[q] = update_record(
#                             records[q],
#                             s,
#                             depth,
#                             disjoint_expected_probs[q, s, d_],
#                             disjoint_measured_probs[q, s, d_],
#                             2,
#                         )

#         def per_cycle_depth(df):
#             fid_lsq = df["numerator"].sum() / df["denominator"].sum()
#             return pd.Series({"fidelity": fid_lsq})

#         if not self.xeb_config.disjoint_processing:
#             df = update_data_frame(pd.DataFrame(records))
#             linear_fidelities = df.groupby("depth").apply(per_cycle_depth).reset_index()
#         else:
#             df, linear_fidelities = [], []
#             for q in range(n_qubits):
#                 df_q = update_data_frame(pd.DataFrame(records[q]))
#                 linear_fidelities.append(df_q.groupby("depth").apply(per_cycle_depth).reset_index())
#                 df.append(df_q)

#         if np.isnan(log_fidelities).all():
#             warnings.warn("All fidelities computed from log-entropies are singularities.")

#         return (
#             joint_measured_probs,
#             disjoint_measured_probs,
#             joint_expected_probs,
#             disjoint_expected_probs,
#             df,
#             log_fidelities,
#             linear_fidelities,
#             singularity,
#             outlier,
#         )


def calc_measured_probability(counts, states, n_shots):
    """Compute joint and disjoint measured probabilities from counts and states."""
    n_qubits = 2
    dim = n_qubits**2
    counts_array = np.stack([counts[binary(i, n_qubits)] for i in range(dim)], axis=-1)
    joint_measured_probs = counts_array / n_shots

    state_array = np.array(list(states.values()))
    disjoint_measured_probs = np.stack([1 - state_array, state_array], axis=-1)

    return joint_measured_probs, disjoint_measured_probs


# def calc_fidelity_for_given_2q_unitary_params():
#     f_xeb = [
#         [compute_log_fidelity(self.incoherent_distribution, ideal_probability_s[s, d_, :],
#          self.data['joint_measured_probs'][s, d_]) for d_, depth in enumerate(self.xeb_config.depths)]
#         for s in range(self.xeb_config.seqs)
#     ]
#     # log_fidelities[s, d_] = evaluate_log_fidelity(f_xeb, singularity, outlier, s, int(depth))
#     f_xeb = np.array(f_xeb)
#     return f_xeb
