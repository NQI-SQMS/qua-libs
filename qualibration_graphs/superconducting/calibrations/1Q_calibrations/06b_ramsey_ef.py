# %% {Imports}
import matplotlib.pyplot as plt
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.ramsey_ef import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.parameters import get_qubits, get_idle_times_in_clock_cycles
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Description}
description = """
        EF RAMSEY WITH VIRTUAL Z ROTATIONS
The program prepares the qubit in |e⟩ via a ge π-pulse, then performs a Ramsey sequence
on the e→f transition: x90_ef – idle_time – x90_ef (with virtual detuning applied via
frame rotation).  A final ge π-pulse is applied before readout to maximise readout contrast.

The EF Ramsey oscillation frequency is used to precisely determine the EF transition
frequency (i.e., correct the anharmonicity stored in the QUAM state), and the decay
envelope gives the EF coherence time T2*_ef.

The virtual detuning is applied symmetrically (± frequency_detuning_in_mhz) to
disambiguate the sign of the frequency correction.

Prerequisites:
    - Having calibrated the ge x180 and x90 pulses (nodes 03a, 04b/04c).
    - Having run qubit EF spectroscopy to set q.anharmonicity (node 12).
    - (optional) Having calibrated a dedicated x90_ef operation for better EF pi/2 pulses.

State update:
    - The EF anharmonicity: q.anharmonicity (corrected by the fitted freq_offset)
    - EF T2*: q.T2ramsey_ef
"""

node = QualibrationNode[Parameters, Quam](name="06b_ramsey_ef", description=description, parameters=Parameters())


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubits = ["q1"]
    # node.parameters.frequency_detuning_in_mhz = 1.0
    # node.parameters.ef_x180_operation = "EF_x180"
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the EF Ramsey sweep axes and QUA program."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    import dataclasses
    n_avg = node.parameters.num_shots
    idle_times = get_idle_times_in_clock_cycles(node.parameters)
    detuning = node.parameters.frequency_detuning_in_mhz * u.MHz
    ef_x180_op = node.parameters.ef_x180_operation

    # Ensure EF_x180 operation exists on each qubit (create from x180 if absent)
    for qubit in qubits:
        if ef_x180_op not in qubit.xy.operations:
            x180 = qubit.xy.operations["x180"]
            qubit.xy.operations[ef_x180_op] = (
                dataclasses.replace(x180, alpha=0.0) if hasattr(x180, "alpha") else dataclasses.replace(x180)
            )

    detuning_signs = [-1, 1]
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "idle_time": xr.DataArray(4 * idle_times, attrs={"long_name": "idle times", "units": "ns"}),
        "detuning_signs": xr.DataArray(detuning_signs, attrs={"long_name": "detuning signs"}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        idle_time = declare(int)
        detuning_sign = declare(int)
        virtual_detuning_phases = [declare(fixed) for _ in range(num_qubits)]

        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_each_(idle_time, idle_times):
                    with for_(*from_array(detuning_sign, detuning_signs)):
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            reset_frame(qubit.xy.name)
                            qubit.xy.wait(2 * qubit.thermalization_time // 4)
                        align()

                        # EF Ramsey sequence
                        for i, qubit in multiplexed_qubits.items():
                            # Compute virtual detuning phase for this idle_time
                            with if_(detuning_sign == 1):
                                assign(
                                    virtual_detuning_phases[i],
                                    Cast.mul_fixed_by_int(detuning * 1e-9, 4 * idle_time),
                                )
                            with else_():
                                assign(
                                    virtual_detuning_phases[i],
                                    Cast.mul_fixed_by_int(-detuning * 1e-9, 4 * idle_time),
                                )

                            # Step 1: prepare |e⟩ via ge π-pulse
                            qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                            qubit.xy.play("x180")

                            # Step 2: EF Ramsey — set IF to EF transition and play EF pi/2 pulses
                            # EF x90 = EF_x180 at half amplitude (same duration, half rotation)
                            qubit.xy.update_frequency(qubit.anharmonicity + qubit.xy.intermediate_frequency)
                            qubit.xy.play(ef_x180_op, amplitude_scale=0.5)
                            qubit.xy.frame_rotation_2pi(virtual_detuning_phases[i])
                            qubit.xy.wait(idle_time)
                            qubit.xy.play(ef_x180_op, amplitude_scale=0.5)

                            # Step 3: reset to ge frequency, apply ge π-pulse for readout
                            qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                            qubit.xy.play("x180")

                        align()

                        # Readout
                        for i, qubit in multiplexed_qubits.items():
                            qubit.readout_state(
                                state[i] if node.parameters.use_state_discrimination else None,
                                I=I[i], Q=Q[i], I_st=I_st[i], Q_st=Q_st[i],
                                state_st=state_st[i] if node.parameters.use_state_discrimination else None,
                            )
                        align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(detuning_signs)).buffer(len(idle_times)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(detuning_signs)).buffer(len(idle_times)).average().save(f"Q{i + 1}")
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(detuning_signs)).buffer(len(idle_times)).average().save(f"state{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the QUA program and fetch the raw dataset."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit the EF Ramsey oscillations and extract the anharmonicity correction and T2*_ef."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the EF Ramsey oscillations with fitted curves."""
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    plt.show()
    node.results["figures"] = {"amplitude": fig_raw_fit}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Correct q.anharmonicity by the fitted EF freq_offset and store EF T2*."""
    ef_x180_op = node.parameters.ef_x180_operation
    use_selective_update = node.parameters.selective_state_update and ef_x180_op != "EF_x180"
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if not node.results["fit_results"][q.name]["success"]:
                continue
            freq_offset = float(node.results["fit_results"][q.name]["freq_offset"])
            decay = float(node.results["fit_results"][q.name]["decay"])
            if use_selective_update:
                # Store correction in pulse.detuning; do NOT touch anharmonicity / T2ramsey_ef
                pulse = q.xy.operations.get(ef_x180_op)
                if pulse is not None and hasattr(pulse, "detuning"):
                    pulse.detuning -= freq_offset
            else:
                # The EF drive frequency = LO + IF_ge + anharmonicity.
                # freq_offset > 0 means the actual EF freq is above the drive, so increase anharmonicity.
                q.anharmonicity -= freq_offset
                q.extras["T2ramsey_ef"] = decay


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
