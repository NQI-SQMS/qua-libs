# pylint: disable=duplicate-code
"""T2 echo coherence time versus flux-bias characterization."""

# %% {Imports}
import warnings
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
from quam_config import Quam
from calibration_utils.T2echo_vs_flux import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    plot_t2_echo_vs_flux,
)
from qualibration_libs.parameters import get_qubits, get_idle_times_in_clock_cycles
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher



# %% {Description}
description = """
        T2 ECHO VERSUS FLUX
The sequence plays an echo (x90 - idle/2 - x180 - idle/2 - -x90 - measurement) for several
idle times while a constant flux pulse biases the qubit during the two free-evolution
windows. Repeating the scan over several flux biases maps how the echo coherence time
depends on the qubit flux point. The single-qubit gates are played at the operating point
(flux baseline) so they stay calibrated; only the free-evolution windows are flux-biased.

For each flux bias an exponential decay is fitted along the idle time to extract
T2_echo(flux); the flux giving the longest T2 echo is reported.

Prerequisites:
    - Having calibrated the qubit frequency precisely (node 12_ramsey.py).
    - Having calibrated the x90/x180 pulses, and specified the flux point (qubit.z.flux_point).
    - A working z-line on every active qubit.

State update:
    - The flux giving the longest T2 echo and the corresponding T2 echo are stored in qubit.extras.
"""


node = QualibrationNode[Parameters, Quam](
    name="26b_echo_vs_flux",
    description=description,
    parameters=Parameters(),
    machine = Quam.load()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    pass


# Instantiate the QUAM class from the state file
# node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    # Check if the qubits have a z-line attached
    if any(q.z is None for q in qubits):
        warnings.warn("Found qubits without a flux line. T2 echo vs flux requires a z-line.")

    n_avg = node.parameters.num_shots  # The number of averages
    # Idle-time sweep (clock cycles; may be log-spaced and non-uniform). Each value is the
    # duration of ONE of the two echo half-delays, so the total free evolution is 2*t.
    idle_times = get_idle_times_in_clock_cycles(node.parameters)
    # Flux-bias sweep in V, centered on the qubit flux point
    fluxes = np.linspace(
        -node.parameters.flux_span / 2,
        +node.parameters.flux_span / 2,
        node.parameters.flux_num,
    )
    # Register the sweep axes to be added to the dataset when fetching data.
    # idle_time is reported as the total free-evolution time (2 * half-delay).
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "flux_bias": xr.DataArray(fluxes, attrs={"long_name": "flux bias", "units": "V"}),
        "idle_time": xr.DataArray(8 * idle_times, attrs={"long_name": "idle time", "units": "ns"}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        t = declare(int)  # QUA variable for the echo half-delay
        flux = declare(fixed)  # QUA variable for the flux dc level

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(flux, fluxes)):
                    with for_each_(t, idle_times):
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            reset_frame(qubit.xy.name)
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()

                        # Qubit manipulation: echo with the flux applied during the two idle windows.
                        # The z waits for each xy gate so the flux pulses align with the free-evolution
                        # windows and the gates themselves are played at the operating point (z baseline).
                        for i, qubit in multiplexed_qubits.items():
                            x90_len = qubit.xy.operations["x90"].length * u.ns // 4
                            x180_len = qubit.xy.operations["x180"].length * u.ns // 4
                            with strict_timing_():
                                qubit.xy.play("x90")
                                qubit.xy.wait(t)
                                qubit.xy.play("x180")
                                qubit.xy.wait(t)
                                qubit.xy.play("-x90")
                                # Flux on z, offset by the gate lengths to overlap the two idle windows
                                qubit.z.wait(x90_len)
                                qubit.z.play(
                                    "const",
                                    amplitude_scale=flux / qubit.z.operations["const"].amplitude,
                                    duration=t,
                                )
                                qubit.z.wait(x180_len)
                                qubit.z.play(
                                    "const",
                                    amplitude_scale=flux / qubit.z.operations["const"].amplitude,
                                    duration=t,
                                )
                        align()

                        # Qubit readout
                        for i, qubit in multiplexed_qubits.items():
                            if node.parameters.use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])

            # Measure sequentially
            if not node.parameters.multiplexed:
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(idle_times)).buffer(len(fluxes)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(idle_times)).buffer(len(fluxes)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(idle_times)).buffer(len(fluxes)).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    debug = False
    if debug:
        from pathlib import Path
        from qm import generate_qua_script
        file_name = Path(__file__).stem
        with open(Path(__file__).parent.parent / f"{file_name}_debug.py", 'w') as sourceFile:
            print(generate_qua_script(node.namespace["qua_program"], config), file=sourceFile)
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in
    a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit"
    and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_map = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    fig_curve = plot_t2_echo_vs_flux(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "echo_decay_map": fig_map,
        "t2_echo_vs_flux": fig_curve,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Store the best flux point and the corresponding T2 echo in qubit.extras if successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            fit_results = node.results["fit_results"][q.name]
            q.extras["T2echo_vs_flux_best_flux_V"] = float(fit_results["flux_at_max"])
            q.extras["T2echo_vs_flux_best_T2echo_s"] = float(fit_results["t2_echo_max"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save all node results and state updates."""
    node.save()
