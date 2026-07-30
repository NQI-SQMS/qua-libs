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
)
from qualibration_libs.parameters import (
    get_qubits,
    get_idle_times_in_clock_cycles,
)
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam
from calibration_utils.cavity_fock1_T2 import (
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.cavity_fock1_T2.parameters import Parameters

# %% {Node initialisation}
description = """
        CAVITY FOCK |1> T2 RAMSEY
Measures the T2 Ramsey coherence time using one of two protocols selected by
fock1_prep_method.

  'sideband' (default) — cavity Fock superposition T2:
    Sequence:
      1. Reset cavity and qubit.
      2. Create (|0>+|1>)/sqrt(2) superposition:
           ge pi/2  ->  ef pi  ->  f0g1 pi
      3. Sideband Ramsey:
           wait tau, frame rotation (ramsey_detuning_hz), f0g1 pi (close)
      4. Back-conversion: ef pi  ->  ge pi/2
      5. Measure qubit.
    Requires: calibrated f0g1 sideband (nodes 26, 26b), EF_x180, x90.
    Extracts: T2ramsey of the cavity (|0>+|1>)/sqrt(2) state.

  'snap_displacement' — cavity Fock superposition T2 via SNAP+displacement:
    Sequence:
      1. Reset cavity and qubit.
      2. Create (|0>+|1>)/sqrt(2) in the cavity using a SNAP+displacement
         sequence (exact pulses TBD — scaffold in place).
      3. Cavity Ramsey:
           wait tau  ->  frame_rotation(detuning × tau) on cavity drive
      4. Back-conversion: undo superposition using SNAP+displacement (TBD).
      5. Measure qubit.
    Extracts: T2ramsey of the cavity (|0>+|1>)/sqrt(2) state, same quantity
              as the 'sideband' method but without requiring a calibrated
              f0g1 sideband.

Parameters:
    - mode_name:          'alice' or 'bob'
    - fock1_prep_method:  'sideband' or 'snap_displacement'
    - fock1_alpha1/2:     displacement amplitudes (snap_displacement only)
    - ramsey_detuning_hz: artificial detuning [Hz]
    - idle_time_*:        range of wait times

State update:
    - cavity_mode.T2ramsey  (seconds)
"""



node = QualibrationNode[Parameters, Quam](
    name="34_cavity_fock1_T2",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name = "alice"
    # node.parameters.ramsey_detuning_hz = 400.0
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

    displaced_threshold = None
    if node.parameters.use_state_discrimination and node.parameters.use_displaced_threshold:
        _t = getattr(pair, "ge_iq_threshold_displaced", None) if pair is not None else None
        if _t is not None:
            displaced_threshold = float(_t)

    # Phase increment per clock cycle for the Ramsey frame rotation:
    # detuning_hz * 1e-9 * 4 ns/cc = turns per clock cycle
    detuning_turns_per_cc = node.parameters.ramsey_detuning_hz * 1e-9 * 4

    if prep_method == "snap_displacement":
        node.log(
            f"Fock|1> T2: SNAP+displacement cavity Ramsey (scaffold — sequence TBD), "
            f"detuning={node.parameters.ramsey_detuning_hz:.1f} Hz"
        )
    else:
        node.log(
            f"Fock|1> T2: sideband Ramsey, detuning={node.parameters.ramsey_detuning_hz:.1f} Hz"
        )

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "idle_time": xr.DataArray(
            4 * idle_times, attrs={"long_name": "idle time", "units": "ns"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        t = declare(int)
        phase = declare(fixed)
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
                            ef_if_0 = pair.ef_if_at_fock(qubit, 0)
                            sb_rf_0 = pair.get_transition_rf(0)
                            target_if_0 = int(sideband_drive.intermediate_frequency) + int(sb_rf_0 - sideband_drive.RF_frequency)
                            tr_0 = pair.transitions.get("f0g1")
                            flat_top_clk_0 = (tr_0.pi_flat_top_length_ns // 4) if (tr_0 and tr_0.pi_flat_top_length_ns) else None

                            # -- 2a. Create cavity superposition (|0>+|1>)/sqrt(2) --
                            # ge pi/2 -> ef pi -> f0g1 pi  [opening arm of sideband Ramsey]
                            align(qubit.xy.name, sideband_drive.name, qubit.resonator.name)
                            qubit.xy.update_frequency(qubit_IF)
                            qubit.xy.play("x90")                    # ge pi/2
                            qubit.xy.update_frequency(ef_if_0)
                            qubit.xy.play("EF_x180")                # ef pi

                            # Set sideband frequency and compute phase before strict_timing_
                            align(qubit.xy.name, sideband_drive.name)
                            sideband_drive.update_frequency(target_if_0)
                            assign(phase, Cast.mul_fixed_by_int(detuning_turns_per_cc, t))

                            # -- 3a. Sideband Ramsey ---------------------------
                            # Opening f0g1 pi, wait tau with frame rotation, closing f0g1 pi.
                            with strict_timing_():
                                sideband_drive.play("sideband_ramp_up")
                                if flat_top_clk_0 is not None:
                                    sideband_drive.play("sideband_square", duration=flat_top_clk_0)
                                else:
                                    sideband_drive.play("sideband_square")
                                sideband_drive.play("sideband_ramp_down")
                                sideband_drive.wait(t)              # Ramsey wait
                                frame_rotation_2pi(phase, sideband_drive.name)
                                sideband_drive.play("sideband_ramp_up")
                                if flat_top_clk_0 is not None:
                                    sideband_drive.play("sideband_square", duration=flat_top_clk_0)
                                else:
                                    sideband_drive.play("sideband_square")
                                sideband_drive.play("sideband_ramp_down")
                            reset_frame(sideband_drive.name)

                            # -- 4a. Back-conversion: ef pi -> ge pi/2 ---------
                            align(sideband_drive.name, qubit.xy.name)
                            qubit.xy.update_frequency(ef_if_0)
                            qubit.xy.play("EF_x180")                # ef pi: |f,0> -> |e,0>
                            qubit.xy.update_frequency(qubit_IF)
                            qubit.xy.play("x90")                    # ge pi/2: closes Ramsey

                        else:  # snap_displacement
                            # -- 2b. Create cavity (|0>+|1>)/sqrt(2) using SNAP+displacement
                            # TODO: insert calibrated SNAP+displacement superposition creation
                            # sequence here (equivalent to the sideband opening arm:
                            # ge pi/2 -> ef pi -> f0g1 pi, but realised without a sideband).
                            align(qubit.xy.name, cavity_mode.cavity_mode_drive.name, qubit.resonator.name)

                            # -- 3b. Cavity Ramsey: wait tau with frame rotation
                            assign(phase, Cast.mul_fixed_by_int(detuning_turns_per_cc, t))
                            with strict_timing_():
                                # TODO: insert opening arm (first SNAP/cavity pi/2 equivalent)
                                cavity_mode.cavity_mode_drive.wait(t)   # Ramsey wait
                                frame_rotation_2pi(phase, cavity_mode.cavity_mode_drive.name)
                                # TODO: insert closing arm (second SNAP/cavity pi/2 equivalent)
                            reset_frame(cavity_mode.cavity_mode_drive.name)

                            # -- 4b. Undo superposition using SNAP+displacement
                            # TODO: insert calibrated SNAP+displacement back-conversion sequence
                            # here (equivalent to the sideband closing arm:
                            # f0g1 pi -> ef pi -> ge pi/2, but realised without a sideband).
                            align(cavity_mode.cavity_mode_drive.name, qubit.xy.name)

                        # -- 5. Measure ----------------------------------------
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
    node.results["figures"] = {"cavity_fock1_T2": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    mode_name = node.parameters.mode_name
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            T2ramsey_ns = node.results["fit_results"][q.name]["T2ramsey_ns"]
            T2ramsey_s = float(T2ramsey_ns) * 1e-9
            for cav in node.machine.cavities.values():
                mode = getattr(cav, mode_name, None)
                if mode is not None and hasattr(mode, "T2ramsey"):
                    mode.T2ramsey = T2ramsey_s
                    break


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
