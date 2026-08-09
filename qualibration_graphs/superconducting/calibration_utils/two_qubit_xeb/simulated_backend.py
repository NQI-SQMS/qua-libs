"""Simulator backend utilities for XEB."""

from qiskit_aer.noise import depolarizing_error, thermal_relaxation_error, NoiseModel
from qiskit_aer import AerSimulator
from qiskit.transpiler import CouplingMap


def make_simulator_backend(num_qubits=2, error_1q=0.01, error_2q=0.05, gate_set_1q=None, gate_set_2q=None):
    """Create a depolarizing noise simulator backend."""
    if gate_set_1q is None:
        gate_set_1q = ["h", "t", "sx", "ry", "sw"]
    if gate_set_2q is None:
        gate_set_2q = ["cz", "cx"]
    cm = CouplingMap.from_line(num_qubits=num_qubits, bidirectional=False)
    depol_error1q = depolarizing_error(param=error_1q, num_qubits=1)
    depol_error2q = depolarizing_error(param=error_2q, num_qubits=2)

    noise_model = NoiseModel(basis_gates=gate_set_1q)
    if num_qubits == 2:
        noise_model.add_all_qubit_quantum_error(error=depol_error2q, instructions=gate_set_2q)
    noise_model.add_all_qubit_quantum_error(error=depol_error1q, instructions=gate_set_1q)
    # noise_model.add_all_qubit_quantum_error(depol_error1q, [ 'rx', 'sw', 'ry', 't'])
    simulator_backend = AerSimulator(
        coupling_map=cm,
        noise_model=noise_model,
        method="density_matrix",
    )
    return simulator_backend


def make_simulator_backend_coherence_time(
    T1_q1_us=100,
    T1_q2_us=100,
    T2_q1_us=10,
    T2_q2_us=10,
    gate_time_1q_ns=32,
    gate_time_2q_ns=100,
    gate_set_1q=None,
    gate_set_2q=None,
):
    """Create a simulator backend with coherence-time-based noise model."""
    if gate_set_1q is None:
        gate_set_1q = ["h", "t", "sx", "ry", "sw"]
    if gate_set_2q is None:
        gate_set_2q = ["cz", "cx"]
    cm = CouplingMap.from_line(num_qubits=2, bidirectional=False)

    error_q1 = thermal_relaxation_error(
        t1=T1_q1_us, t2=T2_q1_us, time=gate_time_1q_ns * 1e-3, excited_state_population=0
    )
    error_q2 = thermal_relaxation_error(
        t1=T1_q2_us, t2=T2_q2_us, time=gate_time_1q_ns * 1e-3, excited_state_population=0
    )
    error_2q = thermal_relaxation_error(t1=T1_q1_us, t2=T2_q1_us, time=gate_time_2q_ns * 1e-3).expand(
        thermal_relaxation_error(t1=T1_q2_us, t2=T2_q2_us, time=gate_time_2q_ns * 1e-3)
    )

    noise_thermal = NoiseModel()

    noise_thermal.add_quantum_error(error=error_q1, instructions=gate_set_1q, qubits=[0])
    noise_thermal.add_quantum_error(error=error_q2, instructions=gate_set_1q, qubits=[1])
    noise_thermal.add_quantum_error(error=error_2q, instructions=gate_set_2q, qubits=[1, 0])
    noise_thermal.add_quantum_error(error=error_2q, instructions=gate_set_2q, qubits=[0, 1])

    simulator_backend = AerSimulator(
        coupling_map=cm,
        noise_model=noise_thermal,
        method="density_matrix",
    )
    return simulator_backend


backend = make_simulator_backend(
    num_qubits=2, error_1q=0.01, error_2q=0.05, gate_set_1q=["h", "t", "sx", "ry", "sw"], gate_set_2q=["cz", "cx"]
)
