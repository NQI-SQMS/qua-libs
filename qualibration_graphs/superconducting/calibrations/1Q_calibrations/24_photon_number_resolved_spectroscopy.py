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
from calibration_utils.shared import apply_confusion_matrix_correction, _get_cavity_mode
from quam_config import Quam
from calibration_utils.photon_number_resolved_spectroscopy import (
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
        PHOTON NUMBER RESOLVED SPECTROSCOPY — CHI MEASUREMENT (29)

Displaces the selected cavity mode to a coherent state |α⟩ and sweeps the
qubit ge spectroscopy frequency.  The resulting spectrum shows photon-number-
split peaks:

    f_n = f_q + chi*n   (n=0, 1, 2, ...)

separated by |chi|.  The node auto-detects the number of peaks (1 → max_peaks)
by fitting successive multi-Gaussian models until the reduced chi² drops below
the threshold.  chi = -(mean PNRS peak spacing) is saved to the machine state.

Convention: chi [Hz] is the full per-photon qubit frequency shift (negative for
typical transmon-cavity systems where more photons lower the qubit frequency).
chi = -(PNRS peak spacing) and |chi| = PNRS peak spacing.

After measurement an optional active reset applies D(-α) to return the cavity
to vacuum immediately, replacing passive thermalization.

Prerequisites:
    - Calibrated qubit_pulse operation on qubit.xy (e.g. selective_x180 or x180).
    - A 'displacement' operation on cavity_mode_drive.

Parameters:
    - displacement_alpha:  coherent state amplitude |α|; amplitude_scale is computed
                           as displacement_alpha / displacement_alpha_max from QuAM state.
    - qubit_pulse:         qubit operation for spectroscopy ('selective_x180' or 'x180').

State update:
    - cavity_transmon_pairs chi  [Hz]  (if the pair exists in the machine)
"""

node = QualibrationNode[Parameters, Quam](
    name="24_photon_number_resolved_spectroscopy",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name = "alice"
    # node.parameters.displacement_alpha = 1.0
    pass


node.machine = Quam.load()


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

    cavity_mode = _get_cavity_mode(node)
    node.namespace["cavity_mode"] = cavity_mode

    # Resolve sideband_drive and alpha_max from the CavityTransmonPair QuAM state
    mode_name = node.parameters.mode_name
    sideband_drive = None
    alpha_max = 1.0
    pairs = getattr(node.machine, "cavity_transmon_pairs", {})
    for pair_key, pair in pairs.items():
        if pair_key.endswith(f"_{mode_name}"):
            if getattr(pair, "sideband_drive", None) is not None:
                sideband_drive = pair.sideband_drive
            if getattr(pair, "displacement_alpha_max", None) is not None:
                alpha_max = float(pair.displacement_alpha_max)
            break

    amplitude_scale = node.parameters.displacement_alpha / alpha_max
    node.log(
        f"Displacement: alpha={node.parameters.displacement_alpha}, "
        f"alpha_max={alpha_max}, amplitude_scale={amplitude_scale:.4f}"
    )
    node.namespace["amplitude_scale"] = amplitude_scale
    node.namespace["sideband_drive"] = sideband_drive

    displaced_threshold = None
    if node.parameters.use_state_discrimination and node.parameters.use_displaced_threshold:
        for _pk, _pv in pairs.items():
            if _pk.endswith(f"_{mode_name}"):
                _t = getattr(_pv, "ge_iq_threshold_displaced", None)
                if _t is not None:
                    displaced_threshold = float(_t)
                break

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

        I, I_st, Q, Q_st, _, _ = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    # --- Reset cavity and qubit BEFORE displacement ---
                    sideband_drive = node.namespace["sideband_drive"]
                    for i, qubit in multiplexed_qubits.items():
                        _cavity_reset_kwargs = dict(
                            sideband_drive=sideband_drive,
                            qubit_thermalization_time=qubit.thermalization_time,
                            fock_n=node.parameters.cavity_active_cooling_fock_n,
                            sideband_pulse_duration_ns=node.parameters.sideband_pulse_duration_ns,
                        )
                        if node.parameters.cavity_reset_type == "active_sideband_v2":
                            _cavity_reset_kwargs.update(
                                qubit=qubit,
                                n_repeats=node.parameters.cavity_active_cooling_n_repeats,
                            )
                        cavity_mode.reset(
                            node.parameters.cavity_reset_type,
                            node.parameters.simulate,
                            log_callable=node.log,
                            **_cavity_reset_kwargs,
                        )
                        qubit.reset(
                            node.parameters.reset_type,
                            node.parameters.simulate,
                            log_callable=node.log,
                        )

                    # Displace cavity to coherent state |α⟩.
                    # align() with no args includes cavity_mode_drive, which is not part of qubit.align().
                    align()
                    cavity_mode.cavity_mode_drive.play(
                        "displacement",
                        amplitude_scale=node.namespace["amplitude_scale"],
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
                        qubit.readout_state(
                            state[i] if node.parameters.use_state_discrimination else None,
                            I=I[i], Q=Q[i], I_st=I_st[i], Q_st=Q_st[i],
                            state_st=state_st[i] if node.parameters.use_state_discrimination else None,
                            threshold=displaced_threshold,
                        )

                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(dfs)).average().save(f"state{i + 1}")


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
    if node.parameters.use_state_discrimination and node.parameters.use_confusion_matrix_correction:
        node.results["ds_raw"] = apply_confusion_matrix_correction(node.results["ds_raw"], node.namespace["qubits"])
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
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
        node.results["ds_fit"],
        node.namespace["qubits"],
        fit_results=node.results["fit_results"],
        mode_name=node.parameters.mode_name,
        displacement_alpha=node.parameters.displacement_alpha,
        normalize_plot=node.parameters.normalize_plot,
    )
    plt.show()
    node.results["figures"] = {"photon_number_resolved_spectroscopy": fig}


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

            # Write to CavityTransmonPair
            pair_key = f"{qubit.name}_{mode_name}"
            pairs = getattr(node.machine, "cavity_transmon_pairs", None)
            if pairs is not None and pair_key in pairs:
                pairs[pair_key].chi = chi_hz


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
