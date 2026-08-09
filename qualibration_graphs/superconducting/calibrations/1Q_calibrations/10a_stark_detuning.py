# %%
#!%load_ext autoreload
#!%autoreload 2
# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.core import tracked_updates
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

from calibration_utils.stark_detuning import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from calibration_utils.common_utils.plotting_tools import patch_fig_info
from quam_config import Quam

# %% {Node initialisation}
description = """
        AC STARK-SHIFT CALIBRATION WITH DRAG PULSES (GOOGLE METHOD)
The sequence consists in applying an increasing number of x180 and -x180 pulses successively for different DRAG
detunings.
After such a sequence, the qubit is expected to always be in the ground state if the AC Stark shift is
properly compensated by the DRAG detuning.
One can then take a line cut for a given number of pulse and fit the 1D trace with a parabola to get the optimum
detuning and update its value in the configuration.

This protocol is described in more details in https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.190503

Prerequisites:
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the qubit parameters precisely (nodes 04b_power_rabi.py and 06a_ramsey.py).
    - (optional) Having optimized the readout parameters (nodes 08a, 08b and 08c).
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

Next steps before going to the next node:
    - Update the DRAG detuning: qubit.xy.operations[operation].detuning.
    - (optional) Update the DRAG coefficient (alpha): qubit.xy.operations[operation].alpha.
"""


node = QualibrationNode[Parameters, Quam](
    name="10a_stark_detuning",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    pass


# Instantiate the QUAM class from the state file
# node.machine = Quam.load()


def check_sweep_within_bounds(qubits, dfs, base_detunings):
    invalid_qubits = []
    max_freq = 400e6

    for qubit in qubits:
        center = qubit.xy.intermediate_frequency + base_detunings[qubit.name]
        sweep_min = (dfs + center).min()
        sweep_max = (dfs + center).max()
        if sweep_min < -max_freq or sweep_max > max_freq:
            invalid_qubits.append((qubit.name, qubit.xy.intermediate_frequency, sweep_min, sweep_max))
    if invalid_qubits:
        msg = (
            f"The following qubits have intermediate frequencies that would cause the sweep to be "
            f"more than {max_freq * 1e-6} MHz away from the LO frequency:"
        )
        for q, f, sweep_min, sweep_max in invalid_qubits:
            msg += f"\n{q}: IF frequency {f * 1e-6} MHz, IF sweep {sweep_min * 1e-6} .. {sweep_max * 1e-6} MHz"
        raise ValueError(msg)


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    operation = node.parameters.operation

    # The frequency sweep is centered around the detuning already set on the operation (e.g. from a
    # previous stark-detuning calibration), not around the bare qubit intermediate frequency. That
    # pre-existing detuning is captured here before it gets reset to 0 for the duration of the sweep.
    node.namespace["tracked_qubits"] = []
    node.namespace["base_detunings"] = base_detunings = {}
    for q in qubits:
        qubit_name = q.name
        with tracked_updates(q, auto_revert=False, dont_assign_to_none=True) as q:
            cur_op = q.xy.operations[node.parameters.operation]
            base_detunings[qubit_name] = int(cur_op.detuning or 0)
            if node.parameters.alpha_setpoint is not None:
                cur_op.alpha = node.parameters.alpha_setpoint
            cur_op.detuning = 0
            node.namespace["tracked_qubits"].append(q)

    n_avg = node.parameters.num_shots
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, +span // 2, step, dtype=np.int32)
    check_sweep_within_bounds(qubits, dfs, base_detunings)

    N_pi = node.parameters.max_number_pulses_per_sweep
    N_pi_vec = np.linspace(1, N_pi, N_pi).astype("int")

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "nb_of_pulses": xr.DataArray(N_pi_vec, attrs={"long_name": "number of pulses"}),
        "detuning": xr.DataArray(
            dfs, attrs={"long_name": "pulse detuning relative to the pre-existing operation detuning", "units": "Hz"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        df = declare(int)
        npi = declare(int)
        count = declare(int)

        reset_global_phase()

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(npi, N_pi_vec)):
                    with for_(*from_array(df, dfs)):
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()

                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.update_frequency(
                                df + qubit.xy.intermediate_frequency + base_detunings[qubit.name]
                            )
                            with for_(count, 0, count < npi, count + 1):
                                if node.parameters.operation == "x180":
                                    qubit.xy.play(operation)
                                    qubit.xy.play(operation, amplitude_scale=-1.0)
                                elif node.parameters.operation == "x90":
                                    qubit.xy.play(operation)
                                    qubit.xy.play(operation)
                                    qubit.xy.play(operation, amplitude_scale=-1.0)
                                    qubit.xy.play(operation, amplitude_scale=-1.0)
                            qubit.xy.update_frequency(qubit.xy.intermediate_frequency)

                        align()
                        for i, qubit in multiplexed_qubits.items():
                            if node.parameters.use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])

        with stream_processing():
            n_st.save("n")
            for i, qubit in enumerate(qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(dfs)).buffer(N_pi).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(dfs)).buffer(N_pi).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(dfs)).buffer(N_pi).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher["n"],
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        node.log(job.execution_report())
    base_detunings = node.namespace["base_detunings"]
    dataset = dataset.assign_coords(
        base_detuning=("qubit", [base_detunings[name] for name in dataset.qubit.values])
    )
    dataset.base_detuning.attrs = {"long_name": "pre-existing operation detuning", "units": "Hz"}
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
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_proc"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_proc"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_proc"], node.namespace["qubits"], node.results["ds_fit"])
    patch_fig_info(node)
    plt.show()
    node.results["figures"] = {
        "detuning": fig_raw_fit,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    for qubit in node.namespace.get("tracked_qubits", []):
        qubit.revert_changes()

    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            fit_result = node.results["fit_results"][q.name]
            q.xy.operations[node.parameters.operation].detuning = fit_result["detuning"]
            if node.parameters.alpha_setpoint is not None:
                q.xy.operations[node.parameters.operation].alpha = node.parameters.alpha_setpoint


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
