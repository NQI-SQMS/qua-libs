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
from calibration_utils.photon_number_splitting import (
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
        PHOTON NUMBER SPLITTING — CHI MEASUREMENT (29)

Displaces the selected cavity mode to a coherent state |α⟩ and sweeps the
qubit ge spectroscopy frequency.  The resulting spectrum shows photon-number-
split peaks:

    f_n = f_q - 2*chi*n   (n=0, 1, 2, ...)

separated by 2*chi.  The node auto-detects the number of peaks (1 → max_peaks)
by fitting successive multi-Gaussian models until the reduced chi² drops below
the threshold.  The mean spacing between adjacent peaks = 2*chi is reported and
saved to the machine state.

After measurement an optional active reset applies D(-α) to return the cavity
to vacuum immediately, replacing passive thermalization.

Prerequisites:
    - Calibrated qubit_pulse operation on qubit.xy (e.g. selective_x180 or x180).
    - A 'displacement' operation on cavity_mode_drive.

Parameters:
    - displacement_scale:  amplitude scale for the displacement pulse.
                           After node 32 calibration: scale=1 → 1 photon.
    - active_reset:        if True, apply D(-α) after measurement (default True).
    - qubit_pulse:         qubit operation for spectroscopy ('selective_x180' or 'x180').

State update:
    - cavity_mode.chi            [Hz]
    - cavity_transmon_pairs chi  [Hz]  (if the pair exists in the machine)
"""

node = QualibrationNode[Parameters, Quam](
    name="07_photon_number_splitting",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name = "alice"
    # node.parameters.displacement_alpha = 1.0
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
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    step_hz = int(node.parameters.frequency_step_in_mhz * u.MHz)
    left_hz = int(node.parameters.left_span_mhz * u.MHz)
    right_hz = int(node.parameters.right_offset_mhz * u.MHz)
    dfs = np.arange(-left_hz, right_hz, step_hz)

    displacement_scale = node.parameters.displacement_scale
    displacement_alpha = node.parameters.displacement_alpha
    # Actual QUA amplitude: scale × alpha (higher alpha → more photons)
    actual_displacement = displacement_scale * displacement_alpha
    cavity_mode = _get_cavity_mode(node)
    node.namespace["cavity_mode"] = cavity_mode

    # Resolve sideband_drive for active cavity cooling
    mode_name = node.parameters.mode_name
    sideband_drive = None
    pairs = getattr(node.machine, "cavity_transmon_pairs", {})
    for pair_key, pair in pairs.items():
        if pair_key.endswith(f"_{mode_name}") and getattr(pair, "sideband_drive", None) is not None:
            sideband_drive = pair.sideband_drive
            break
    node.namespace["sideband_drive"] = sideband_drive

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(
            dfs, attrs={"long_name": "qubit detuning", "units": "Hz"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        df = declare(int)
        n_st = declare_stream()

        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        else:
            I, I_st, Q, Q_st, _, _ = node.machine.declare_qua_variables()

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    # --- Reset cavity and qubit BEFORE displacement ---
                    sideband_drive = node.namespace["sideband_drive"]
                    for i, qubit in multiplexed_qubits.items():
                        cavity_mode.reset(
                            node.parameters.cavity_reset_type,
                            node.parameters.simulate,
                            log_callable=node.log,
                            sideband_drive=sideband_drive,
                            qubit_thermalization_time=qubit.thermalization_time,
                            fock_n=node.parameters.cavity_active_cooling_fock_n,
                            f0g1_pulse_duration_ns=node.parameters.f0g1_pulse_duration_ns,
                        )
                        qubit.reset(
                            node.parameters.reset_type,
                            node.parameters.simulate,
                            log_callable=node.log,
                        )

                    # Displace cavity to coherent state |α⟩ with actual_displacement = scale × alpha.
                    # align() with no args includes cavity_mode_drive, which is not part of qubit.align().
                    align()
                    cavity_mode.cavity_mode_drive.play(
                        "displacement",
                        amplitude_scale=actual_displacement,
                    )

                    for i, qubit in multiplexed_qubits.items():
                        # Qubit spectroscopy with selective pulse
                        align(cavity_mode.cavity_mode_drive.name, qubit.xy.name)
                        qubit.xy.update_frequency(
                            df + qubit.xy.intermediate_frequency
                        )
                        qubit.xy.play(node.parameters.qubit_pulse)

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

                        qubit.resonator.wait(node.machine.depletion_time * u.ns)

                    # Active reset: D(-α) returns cavity exactly to vacuum.
                    if node.parameters.active_reset:
                        align()
                        cavity_mode.cavity_mode_drive.play(
                            "displacement",
                            amplitude_scale=-actual_displacement,
                        )

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
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
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
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)
    node.namespace["cavity_mode"] = _get_cavity_mode(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_raw"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        q: ("successful" if res["success"] else "failed")
        for q, res in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.namespace["qubits"],
        fit_results=node.results["fit_results"],
        mode_name=node.parameters.mode_name,
        displacement_scale=node.parameters.displacement_scale,
        displacement_alpha=node.parameters.displacement_alpha,
        normalize_plot=node.parameters.normalize_plot,
    )
    plt.show()
    node.results["figures"] = {"photon_number_splitting": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Save the measured dispersive shift chi to the machine state."""
    mode_name = node.parameters.mode_name
    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            res = node.results["fit_results"].get(qubit.name)
            if res is None or not res["success"]:
                continue
            chi_hz = float(res["chi_hz"])

            # Write to cavity_mode.chi
            cavity_mode = node.namespace["cavity_mode"]
            cavity_mode.chi = chi_hz

            # Write to CavityTransmonPair if present
            pair_key = f"{qubit.name}_{mode_name}"
            pairs = getattr(node.machine, "cavity_transmon_pairs", None)
            if pairs is not None and pair_key in pairs:
                pairs[pair_key].chi = chi_hz


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
