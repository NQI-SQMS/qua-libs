# %%
"""
Sideband Bring-Up Graph

Calibrates a single f|k>g|k+1> sideband transition selected by ``sideband_level``.
Setting ``sideband_level = N`` calibrates the f|N-1>g|N> transition (1-based):

  sideband_level = 1  →  f0g1
  sideband_level = 2  →  f1g2
  sideband_level = 3  →  f2g3
  …

The calibration chain for the selected level is:

  fNgN1_spectroscopy
  -> fNgN1_time_rabi
  -> fNgN1_ramsey          (sideband frequency fine-tuning)
  -> fNgN1_ge_spectroscopy (qubit GE frequency shift in Fock |k>)
  -> fNgN1_ef_spectroscopy (qubit EF frequency shift in Fock |k>)
"""

from typing import List

from qualibrate import GraphParameters, QualibrationGraph, QualibrationLibrary

library = QualibrationLibrary.get_active_library()


class SidebandBringUpParameters(GraphParameters):
    """Graph-flow parameters for the sideband bring-up graph."""

    qubits: List[str] = ["q1"]
    mode_name: str = "alice"
    """Cavity mode to calibrate: 'alice' or 'bob'."""
    sideband_level: int = 1
    """Sideband transition to calibrate (1-based): 1 → f0g1, 2 → f1g2, …
    Changing this requires reloading the library."""


with QualibrationGraph.build(
    "sideband_bringup_graph",
    parameters=SidebandBringUpParameters(),
) as graph:

    k = graph.parameters.sideband_level - 1
    qubits = graph.parameters.qubits
    mode = graph.parameters.mode_name

    n_spec = library.nodes["07_fNgN1_spectroscopy"].copy(
        name="fNgN1_spectroscopy",
        qubits=qubits,
        mode_name=mode,
        fock_level=k,
        frequency_span_in_mhz=10.0,
        frequency_step_in_mhz=0.05,
        operation_amplitude_factor=1.0,
        operation_len_in_ns=20_000,
        num_shots=500,
    )
    n_rabi = library.nodes["07b_fNgN1_time_rabi"].copy(
        name="fNgN1_time_rabi",
        qubits=qubits,
        mode_name=mode,
        fock_level=k,
        min_duration_ns=16,
        max_duration_ns=20_000,
        duration_step_ns=4,
        num_shots=100,
        cavity_thermalization_time_ns=200_000,
    )
    n_ramsey = library.nodes["07c_fNgN1_ramsey"].copy(
        name="fNgN1_ramsey",
        qubits=qubits,
        mode_name=mode,
        fock_level=k,
        min_wait_ns=16,
        max_wait_ns=5_000,
        num_wait_points=101,
        artificial_detuning_hz=1e6,
        num_shots=200,
    )
    n_ge_spec = library.nodes["07e_fNgN1_qubit_ge_spectroscopy"].copy(
        name="fNgN1_ge_spectroscopy",
        qubits=qubits,
        mode_name=mode,
        fock_level=k,
        frequency_span_in_mhz=5.0,
        frequency_step_in_mhz=0.05,
        operation_len_in_ns=20_000,
        num_shots=300,
        cavity_reset_type="thermal",
    )
    n_ef_spec = library.nodes["07g_fNgN1_qubit_ef_spectroscopy"].copy(
        name="fNgN1_ef_spectroscopy",
        qubits=qubits,
        mode_name=mode,
        fock_level=k,
        frequency_span_in_mhz=5.0,
        frequency_step_in_mhz=0.05,
        operation_len_in_ns=20_000,
        num_shots=300,
        cavity_reset_type="thermal",
    )
    for node in [n_spec, n_rabi, n_ramsey, n_ge_spec, n_ef_spec]:
        graph.add_node(node)

    graph.connect(n_spec, n_rabi)
    graph.connect(n_rabi, n_ramsey)
    graph.connect(n_ramsey, n_ge_spec)
    graph.connect(n_ge_spec, n_ef_spec)


graph.run()
