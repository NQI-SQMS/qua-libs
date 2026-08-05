"""CZ fine calibration graph."""

# %%
from typing import List

from qualibrate.core.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.core.parameters import GraphParameters
from qualibrate import QualibrationGraph
from qualibrate import QualibrationLibrary

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    """Graph-level parameters for the CZ fine calibration graph."""

    targets_name = "qubit_pairs"
    qubit_pairs: List[str] = ["q6-7"]
    operation: str = "cz_bipolar"


g = QualibrationGraph(
    name="CZ_bipolar_retune",
    parameters=Parameters(),
    nodes={
        "leakage_amplification": library.nodes["20d_cz_leakage_amplification_palea"].copy(
            name="leakage_amplification",
            num_shots=100,
            amp_range=0.05,
            amp_step=0.001,
            number_of_operations=20,
            use_state_discrimination=True,
            reset_type="thermal",
            operation="cz_bipolar",
        ),
        "conditional_phase_error_amp": library.nodes["33_cz_conditional_phase_error_amp"].copy(
            name="conditional_phase_error_amp",
            num_shots=100,
            amp_range=0.005,
            amp_step=0.0002,
            num_frame_rotations=10,
            number_of_operations=20,
            use_state_discrimination=True,
            reset_type="active",
            operation="cz_bipolar",
        ),
        "phase_compensation": library.nodes["35_cz_phase_compensation"].copy(
            name="phase_compensation",
            num_shots=100,
            num_frames=17,
            use_state_discrimination=True,
            reset_type="active",    
            operation="cz_bipolar",
        ),
        "phase_compensation_error_amp": library.nodes["35a_cz_phase_compensation_error_amp"].copy(
            name="phase_compensation_error_amp",
            num_shots=100,
            num_frames=201,
            number_of_operations=40,
            use_state_discrimination=True,
            reset_type="active",
            operation="cz_bipolar",
        ),
        "standard_rb": library.nodes["37_two_qubit_standard_rb"].copy(
            name="standard_rb",
            num_shots=100,
            use_state_discrimination=True,
            circuit_lengths=[1, 2, 5, 12, 27, 63, 150],
            num_circuits_per_length=5,
            seed=0,
            use_input_stream=False,
            reset_type="active",
            operation="cz_bipolar",
        ),
        "interleaved_cz_rb": library.nodes["37b_two_qubit_interleaved_cz_rb"].copy(
            name="interleaved_cz_rb",
            num_shots=100,
            use_state_discrimination=True,
            circuit_lengths=[1, 2, 5, 12, 27, 63, 150],
            num_circuits_per_length=5,
            seed=0,
            use_input_stream=False,
            reset_type="active",
            operation="cz_bipolar",
        ),
    },
    connectivity=[
        ("leakage_amplification", "conditional_phase_error_amp"),
        ("conditional_phase_error_amp", "phase_compensation"),
        ("phase_compensation", "phase_compensation_error_amp"),
        ("phase_compensation_error_amp", "standard_rb"),
        ("standard_rb", "interleaved_cz_rb"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run()

# %%
