# %% {Imports}
import matplotlib.pyplot as plt
from dataclasses import asdict

import numpy as np
import xarray as xr

from qm.qua import *
from qm.qua.lib import Cast

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from calibration_utils.shared import (
    apply_confusion_matrix_correction,
    _get_cavity_mode,
    _get_pair,
    _get_transition_rf,
    _resolve_sb_op,
    _ef_if_at_fock,
    _ge_if_at_fock,
)
from qualibration_libs.parameters import (
    get_qubits,
    get_idle_times_in_clock_cycles,
)
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam
from calibration_utils.cavity_fock1_T1 import (
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.cavity_fock1_T1.parameters import Parameters

# %% {Node initialisation}
description = """
        CAVITY FOCK |1> T1
Measures the photon lifetime T1 of the cavity Fock |1> state.

Two preparation / readout methods are available via fock1_prep_method:

  'sideband' (default):
    Prep:    ge pi -> ef pi -> f0g1 pi  ->  |g,1>
    Readout: inverse sideband (f0g1 pi -> ef pi -> measure ge)
    Requires: calibrated f0g1 sideband (nodes 26, 26b) and EF_x180.

  'snap_displacement':
    Prep:    D(alpha1) -> selective_x180 × 2 (SNAP₀) -> D(alpha2)  ->  ~|1>
    Readout: selective_x180 at n=1 dressed qubit frequency (PNRS)
             maps P(n=1) -> P(|e>).
    Requires: displacement calibration (node 22/30), chi calibrated (node 25),
              selective_x180 tuned (node 33).

Sequence (per wait time tau):
  1. Reset cavity and qubit.
  2. Prepare Fock |1> (method selected by fock1_prep_method).
  3. Wait variable time tau.
  4. Readout (method-matched).
  5. Measure qubit state.

P(|e>) vs tau follows an exponential decay with time constant T1.

Parameters:
    - mode_name:          'alice' or 'bob'
    - fock1_prep_method:  'sideband' or 'snap_displacement'
    - fock1_alpha1/2:     displacement amplitudes (snap_displacement only)
    - idle_time_*:        range of wait times

State update:
    - cavity_mode.T1  (seconds)
"""



node = QualibrationNode[Parameters, Quam](
    name="33_cavity_fock1_T1",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name = "alice"
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    idle_times = get_idle_times_in_clock_cycles(node.parameters)  # clock cycles (4 ns each)

    cavity_mode = _get_cavity_mode(node)
    pair = _get_pair(node)
    prep_method = node.parameters.fock1_prep_method

    # Resolve sideband_drive and chi_hz from QuAM CavityTransmonPair
    sideband_drive = None
    chi_hz = 0.0
    if pair is not None:
        sideband_drive = getattr(pair, "sideband_drive", None)
        chi_hz = float(pair.chi) if getattr(pair, "chi", None) is not None else 0.0
    if prep_method == "sideband" and sideband_drive is None:
        raise ValueError(
            "No sideband_drive found in the CavityTransmonPair. "
            "Run the f0g1 sideband calibration nodes first, or set "
            "fock1_prep_method='snap_displacement'."
        )
    node.namespace["sideband_drive"] = sideband_drive

    # SNAP+displacement: resolve displacement scales and n=1 dressed qubit IF
    _AMP_MAX = 2.0 - 2**-16
    if prep_method == "snap_displacement":
        alpha_max = float(getattr(pair, "displacement_alpha_max", 1.0)) if pair is not None else 1.0
        amp_scale1 = node.parameters.fock1_alpha1 / alpha_max
        amp_scale2 = node.parameters.fock1_alpha2 / alpha_max
        for s, name in [(amp_scale1, "fock1_alpha1"), (amp_scale2, "fock1_alpha2")]:
            if abs(s) > _AMP_MAX:
                raise ValueError(
                    f"{name}={getattr(node.parameters, name)} / alpha_max={alpha_max} = "
                    f"{s:.4f} exceeds the QUA hardware limit ±{_AMP_MAX:.6f}."
                )
        node.namespace["amp_scale1"] = amp_scale1
        node.namespace["amp_scale2"] = amp_scale2
        node.log(
            f"Fock|1> T1: SNAP+displacement  "
            f"(alpha1={node.parameters.fock1_alpha1}, alpha2={node.parameters.fock1_alpha2})"
        )
    else:
        node.log("Fock|1> T1: sideband ladder  (ge pi -> ef pi -> f0g1 pi)")

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "idle_time": xr.DataArray(
            4 * idle_times, attrs={"long_name": "idle time", "units": "ns"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        t = declare(int)
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

                with for_each_(t, idle_times):
                    for i, qubit in multiplexed_qubits.items():
                        qubit_IF = int(qubit.xy.intermediate_frequency)

                        # -- 1. Reset cavity and qubit ------------------------
                        cavity_mode.reset(
                            node.parameters.cavity_reset_type,
                            node.parameters.simulate,
                            log_callable=node.log,
                            sideband_drive=sideband_drive,
                            qubit_thermalization_time=qubit.thermalization_time,
                            fock_n=node.parameters.cavity_active_cooling_fock_n,
                            sideband_pulse_duration_ns=node.parameters.sideband_pulse_duration_ns,
                            chi_hz=chi_hz,
                            pair=pair,
                        )
                        qubit.xy.wait(qubit.thermalization_time // 4)

                        if prep_method == "sideband":
                            # Sideband and ef frequencies at Fock |0> (cavity in vacuum)
                            ef_if_0 = _ef_if_at_fock(pair, qubit, 0)
                            op_0 = _resolve_sb_op(sideband_drive, 0)
                            sb_rf_0 = _get_transition_rf(pair, sideband_drive, 0)
                            target_if_0 = int(sideband_drive.intermediate_frequency) + int(sb_rf_0 - sideband_drive.RF_frequency)
                            tr_0 = pair.transitions.get("f0g1")
                            flat_top_clk_0 = (tr_0.pi_flat_top_length_ns // 4) if (tr_0 and tr_0.pi_flat_top_length_ns) else None

                            # -- 2a. Fock |1> prep: ge pi -> ef pi -> f0g1 pi ---
                            align(qubit.xy.name, sideband_drive.name, qubit.resonator.name)
                            qubit.xy.update_frequency(qubit_IF)
                            qubit.xy.play("x180")           # ge pi  ->  |e,0>
                            qubit.xy.update_frequency(ef_if_0)
                            qubit.xy.play("EF_x180")        # ef pi  ->  |f,0>
                            align(qubit.xy.name, sideband_drive.name)
                            sideband_drive.update_frequency(target_if_0)
                            if flat_top_clk_0 is not None:
                                sideband_drive.play(op_0, duration=flat_top_clk_0)  # f0g1 pi -> |g,1>
                            else:
                                sideband_drive.play(op_0)
                            align(sideband_drive.name, qubit.xy.name, qubit.resonator.name)

                            # -- 3a. Wait tau ----------------------------------
                            qubit.resonator.wait(t)

                            # -- 4a. Inverse sideband readout: f0g1 pi -> ef pi
                            # |g,1> -> |f,0> -> |e,0>  if photon survived
                            # |g,0> unchanged            if photon decayed
                            align(sideband_drive.name, qubit.xy.name, qubit.resonator.name)
                            sideband_drive.update_frequency(target_if_0)
                            if flat_top_clk_0 is not None:
                                sideband_drive.play(op_0, duration=flat_top_clk_0)
                            else:
                                sideband_drive.play(op_0)
                            align(sideband_drive.name, qubit.xy.name)
                            qubit.xy.update_frequency(ef_if_0)
                            qubit.xy.play("EF_x180")
                            qubit.xy.update_frequency(qubit_IF)

                        else:  # snap_displacement
                            # n=1 dressed qubit IF for PNRS readout
                            n1_if = _ge_if_at_fock(pair, qubit, 1)

                            # -- 2b. Fock |1> prep: D(alpha1) -> SNAP₀ -> D(alpha2)
                            align(qubit.xy.name, cavity_mode.cavity_mode_drive.name, qubit.resonator.name)
                            cavity_mode.cavity_mode_drive.play(
                                "displacement", amplitude_scale=node.namespace["amp_scale1"]
                            )
                            align(qubit.xy.name, cavity_mode.cavity_mode_drive.name)
                            qubit.xy.update_frequency(qubit_IF)
                            qubit.xy.play("selective_x180")  # SNAP₀: two selective pi pulses
                            qubit.xy.play("selective_x180")  # apply pi phase to |n=0> component
                            align(qubit.xy.name, cavity_mode.cavity_mode_drive.name)
                            cavity_mode.cavity_mode_drive.play(
                                "displacement", amplitude_scale=node.namespace["amp_scale2"]
                            )
                            align(qubit.xy.name, cavity_mode.cavity_mode_drive.name, qubit.resonator.name)

                            # -- 3b. Wait tau ----------------------------------
                            qubit.resonator.wait(t)

                            # -- 4b. PNRS readout: selective_x180 at n=1 dressed freq
                            # maps P(n=1) -> P(|e>)
                            align(qubit.xy.name, qubit.resonator.name)
                            qubit.xy.update_frequency(n1_if)
                            qubit.xy.play("selective_x180")
                            qubit.xy.update_frequency(qubit_IF)

                        # -- 5. Measure ----------------------------------------
                        align(qubit.xy.name, qubit.resonator.name)
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                        if node.parameters.use_state_discrimination:
                            assign(
                                state[i],
                                Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold),
                            )
                            save(state[i], state_st[i])
                        qubit.resonator.wait(qubit.resonator.depletion_time // 4)

                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(idle_times)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(idle_times)).average().save(f"Q{i + 1}")
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(idle_times)).average().save(f"state{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
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
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


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
        q_name: ("successful" if res["success"] else "failed")
        for q_name, res in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"],
        fit_results=node.results["fit_results"],
        mode_name=node.parameters.mode_name,
    )
    plt.show()
    node.results["figures"] = {"cavity_fock1_T1": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    mode_name = node.parameters.mode_name
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            T1_ns = node.results["fit_results"][q.name]["T1_ns"]
            T1_s = float(T1_ns) * 1e-9
            for cav in node.machine.cavities.values():
                mode = getattr(cav, mode_name, None)
                if mode is not None and hasattr(mode, "T1"):
                    mode.T1 = T1_s
                    break


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
