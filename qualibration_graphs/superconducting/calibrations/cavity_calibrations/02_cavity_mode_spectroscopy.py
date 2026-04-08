# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.cavity_mode_spectroscopy import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Description}
description = """
        CAVITY MODE SPECTROSCOPY
Finds the bare resonance frequency of a storage cavity mode (e.g. alice or bob)
by sweeping the cavity drive frequency and using dispersive coupling to the qubit
as the photon detector.

Sequence (per cavity detuning df):
  1. Wait 2× thermalization time (qubit and cavity thermalise to |g,0⟩).
  2. Set qubit drive to bare ge frequency (no sweep on qubit).
  3. Sweep cavity drive to (IF_cavity + df) and play saturation / probe pulse.
  4. Apply selective_x180 on qubit at bare ge frequency.
     - Off resonance (no photons): selective pulse succeeds → qubit in |e⟩.
     - On resonance (photons present): dispersive shift detunes qubit → pulse
       fails → qubit stays in |g⟩.
  5. Measure qubit state.

The result is a DIP in the qubit excitation probability at the cavity resonance.
A Lorentzian dip fit extracts the cavity frequency.

Prerequisites:
    - Calibrated ge and ef pulses (nodes 04b, 13).
    - Calibrated selective_x180 pulse (node 04b with operation='selective_x180').

Parameters:
    - mode_name:                  'alice' or 'bob'
    - frequency_span_in_mhz:      frequency span around current cavity RF_frequency [MHz]
    - frequency_step_in_mhz:      frequency step [MHz]
    - operation:                  pulse to play on cavity_mode_drive (default 'saturation')
    - qubit_probe_operation:      qubit pulse for dispersive detection (default 'selective_x180')

State update:
    - cavity_mode.cavity_mode_drive.RF_frequency  [Hz]
"""

node = QualibrationNode[Parameters, Quam](name="02_cavity_mode_spectroscopy", description=description, parameters=Parameters())


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name = "alice"
    # node.parameters.frequency_span_in_mhz = 400.0
    pass


node.machine = Quam.load()


def _get_cavity_mode(node):
    mode_name = node.parameters.mode_name
    for cav in node.machine.cavities.values():
        mode = getattr(cav, mode_name, None)
        if mode is not None:
            return mode
    raise KeyError(f"Cavity mode '{mode_name}' not found in machine.cavities")


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the cavity mode spectroscopy sweep axes and QUA program."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, +span // 2, step)

    cavity_mode = _get_cavity_mode(node)
    op = node.parameters.operation
    op_len = node.parameters.operation_len_in_ns
    amp_factor = node.parameters.operation_amplitude_factor
    qubit_op = node.parameters.qubit_probe_operation

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(
            dfs, attrs={"long_name": "cavity detuning", "units": "Hz"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        f = declare(int)
        n_st = declare_stream()

        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        else:
            I, I_st, Q, Q_st, _, _ = node.machine.declare_qua_variables()

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                # Cavity thermalization in clock cycles.
                # Use explicit override if provided, otherwise fall back to T1 × factor.
                if node.parameters.cavity_thermalization_time_ns is not None:
                    therm_clk = int(min(max(node.parameters.cavity_thermalization_time_ns // 4, 4), 2_500_000_000))
                else:
                    therm_clk = int(min(max(cavity_mode.T1 * cavity_mode.thermalization_time_factor * 1e9 / 4, 4), 2_500_000_000))

                with for_(*from_array(f, dfs)):
                    for i, qubit in multiplexed_qubits.items():
                        # Thermalise cavity and qubit before each point.
                        cavity_mode.cavity_mode_drive.wait(therm_clk)
                        qubit.xy.wait(2 * qubit.thermalization_time * u.ns)

                        # Ensure qubit drive is at bare ge frequency
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency)

                        # Sweep cavity drive to (IF + f) and probe
                        cavity_mode.cavity_mode_drive.update_frequency(
                            cavity_mode.cavity_mode_drive.intermediate_frequency + f
                        )
                        if op_len is not None:
                            cavity_mode.cavity_mode_drive.play(
                                op,
                                amplitude_scale=amp_factor,
                                duration=op_len >> 2,
                            )
                        else:
                            cavity_mode.cavity_mode_drive.play(
                                op,
                                amplitude_scale=amp_factor,
                            )

                        # Dispersive detection: probe qubit at bare ge frequency.
                        # If photons are present the qubit is dispersively shifted
                        # and the selective probe pulse fails → dip in excitation.
                        align(cavity_mode.cavity_mode_drive.name, qubit.xy.name)
                        qubit.xy.play(qubit_op)

                        # Measure
                        align(qubit.xy.name, qubit.resonator.name)
                        if node.parameters.use_state_discrimination:
                            qubit.readout_state(state[i])
                            save(state[i], state_st[i])
                        else:
                            qubit.resonator.measure(
                                "readout", qua_vars=(I[i], Q[i])
                            )
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])

                        # Reverse the cavity displacement (de-excite residual photons).
                        align(qubit.resonator.name, cavity_mode.cavity_mode_drive.name)
                        cavity_mode.cavity_mode_drive.play(op, amplitude_scale=-amp_factor)

                        # Resonator depletion + qubit thermalization.
                        qubit.resonator.wait(node.machine.depletion_time * u.ns)
                        qubit.xy.wait(2 * qubit.thermalization_time * u.ns)

                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(dfs)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")


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
    """Fit the cavity resonance dip and extract the cavity frequency."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_raw"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        q_name: ("successful" if res["success"] else "failed")
        for q_name, res in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot qubit excitation vs cavity drive frequency with Lorentzian dip fit."""
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.namespace["qubits"],
        fit_results=node.results["fit_results"],
        mode_name=node.parameters.mode_name,
    )
    plt.show()
    node.results["figures"] = {"cavity_mode_spectroscopy": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the cavity mode RF frequency with the fitted resonance frequency."""
    mode_name = node.parameters.mode_name
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            freq_hz = node.results["fit_results"][q.name]["frequency_hz"]
            for cav in node.machine.cavities.values():
                mode = getattr(cav, mode_name, None)
                if mode is not None:
                    mode.cavity_mode_drive.RF_frequency = float(freq_hz)
                    break


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
