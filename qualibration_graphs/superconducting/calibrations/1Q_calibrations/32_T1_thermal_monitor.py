# %% {Imports}
import time
import matplotlib.pyplot as plt

import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from calibration_utils.shared import apply_confusion_matrix_correction
from quam_config import Quam
from qualibration_libs.data import XarrayDataFetcher, convert_IQ_to_V
from qualibration_libs.parameters import get_qubits, get_idle_times_in_clock_cycles
from qualibration_libs.runtime import simulate_and_plot
from calibration_utils.T1_thermal_monitor import (
    Parameters,
    IterResult,
    fit_T1_dataset,
    fit_ramsey_dataset,
    fit_echo_dataset,
    fit_rpm_datasets,
    build_dataset,
    reconstruct_iter_results,
    log_iteration,
    log_summary,
    plot_T1_thermal_monitor,
)

# %% {Node initialisation}
description = """
        T1, T2*, T2 ECHO, QUBIT FREQUENCY & THERMAL POPULATION MONITOR
Repeatedly runs four experiments per iteration to monitor long-timescale
coherence fluctuations and qubit frequency drift:

  1. T1 sweep       — x180 → wait(t) → measure
                      → T1 [µs]
  2. Ramsey sweep   — x90 → wait(t) → virtual-Z(φ) → x90 → measure (±detuning)
                      → T2* [µs] + frequency offset Δf [Hz]
  3. Echo sweep     — x90 → wait(t) → x180 → wait(t) → -x90 → measure
                      → T2 echo [µs]
  4. RPM 'g' sweep  — ge_π → ef(a) → ef_π → ge_π → measure  (start from |g⟩)
  5. RPM 'e' sweep  — ef(a) → ge_π → measure  (start from thermal state)
                      → P_th + effective qubit temperature T_eff

All results are accumulated and saved as an xr.Dataset (.h5 via node.save()).
Raw I/Q datasets for every iteration and every experiment are also saved.

Prerequisites:
    - Calibrated x90 and x180 pulses (nodes 03/04).
    - Calibrated EF_x180 pulse (node 13).
    - Calibrated readout (nodes 02a/02b/02c).

State update: none (monitoring only).
"""

node = QualibrationNode[Parameters, Quam](
    name="32_qubit_coherence_monitor",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubits = ["q1"]
    # node.parameters.n_iter = 60
    # node.parameters.num_shots = 200
    # node.parameters.ramsey_frequency_detuning_in_mhz = 1.0
    pass


node.machine = Quam.load()


def _rpm_program(qubits, amp_factors, num_qubits, n_avg, ef_op, u, start_from_g: bool):
    """Build a QUA program for one RPM sweep direction."""
    with program() as qua_prog:
        n = declare(int)
        a = declare(fixed)
        n_st = declare_stream()

        I, I_st, Q, Q_st, _, _ = node.machine.declare_qua_variables()
        state    = [declare(int)     for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(a, amp_factors)):
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.wait(2 * qubit.thermalization_time // 4)

                        if start_from_g:
                            qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                            qubit.xy.play("x180")

                        qubit.xy.update_frequency(
                            qubit.xy.intermediate_frequency + qubit.anharmonicity
                        )
                        qubit.xy.play(ef_op, amplitude_scale=a)

                        if start_from_g:
                            qubit.xy.play(ef_op)

                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                        qubit.xy.play("x180")

                        align(qubit.xy.name, qubit.resonator.name)
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                        assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                        save(state[i], state_st[i])
                        qubit.resonator.wait(node.machine.depletion_time // 4)

                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(amp_factors)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(amp_factors)).average().save(f"Q{i + 1}")
                state_st[i].buffer(len(amp_factors)).average().save(f"state{i + 1}")

    return qua_prog


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the T1, Ramsey, echo, and two RPM QUA programs."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    n_avg  = node.parameters.num_shots
    ef_op  = node.parameters.ef_operation
    x180_op = node.parameters.x180_operation

    # ── T1 sweep axes ────────────────────────────────────────────────────────
    t1_idle_times = get_idle_times_in_clock_cycles(node.parameters)
    node.namespace["t1_sweep_axes"] = {
        "qubit":     xr.DataArray(qubits.get_names()),
        "idle_time": xr.DataArray(
            4 * t1_idle_times,
            attrs={"long_name": "idle time", "units": "ns"},
        ),
    }

    # ── Ramsey sweep axes ────────────────────────────────────────────────────
    _ramsey_space = np.geomspace if node.parameters.ramsey_log_or_linear_sweep == "log" else np.linspace
    ramsey_times_ns = np.unique(
        _ramsey_space(
            node.parameters.ramsey_min_wait_time_in_ns,
            node.parameters.ramsey_max_wait_time_in_ns,
            node.parameters.ramsey_wait_time_num_points,
        ).astype(int) // 4 * 4
    )
    ramsey_idle_times = np.maximum(ramsey_times_ns // 4, 4).astype(int)
    detuning_signs = [-1, 1]
    detuning = node.parameters.ramsey_frequency_detuning_in_mhz * u.MHz

    node.namespace["ramsey_idle_times"] = ramsey_idle_times
    node.namespace["ramsey_sweep_axes"] = {
        "qubit":          xr.DataArray(qubits.get_names()),
        "idle_time":      xr.DataArray(
            4 * ramsey_idle_times,
            attrs={"long_name": "idle time", "units": "ns"},
        ),
        "detuning_signs": xr.DataArray(
            detuning_signs,
            attrs={"long_name": "detuning signs"},
        ),
    }

    # ── Echo sweep axes ──────────────────────────────────────────────────────
    _echo_space = np.geomspace if node.parameters.echo_log_or_linear_sweep == "log" else np.linspace
    echo_times_ns = np.unique(
        _echo_space(
            node.parameters.echo_min_wait_time_in_ns,
            node.parameters.echo_max_wait_time_in_ns,
            node.parameters.echo_wait_time_num_points,
        ).astype(int) // 4 * 4
    )
    echo_idle_times = np.maximum(echo_times_ns // 4, 4).astype(int)

    node.namespace["echo_idle_times"] = echo_idle_times
    node.namespace["echo_sweep_axes"] = {
        "qubit":     xr.DataArray(qubits.get_names()),
        "idle_time": xr.DataArray(
            8 * echo_idle_times,   # total free-evolution time = 2 × t_half
            attrs={"long_name": "idle time", "units": "ns"},
        ),
    }

    # ── RPM sweep axes ───────────────────────────────────────────────────────
    amp_factors = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
    )
    node.namespace["rpm_sweep_axes"] = {
        "qubit":      xr.DataArray(qubits.get_names()),
        "amp_factor": xr.DataArray(amp_factors, attrs={"long_name": "EF amplitude factor"}),
    }

    # ── T1 QUA program ───────────────────────────────────────────────────────
    with program() as t1_prog:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        t = declare(int)
        if node.parameters.use_state_discrimination:
            state    = [declare(int)     for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_each_(t, t1_idle_times):
                    # --- Reset ---
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(
                            node.parameters.reset_type,
                            node.parameters.simulate,
                            log_callable=node.log,
                        )
                    align()

                    # --- Qubit drive ---
                    for i, qubit in multiplexed_qubits.items():
                        qubit.align()
                        qubit.xy.play("x180")
                        qubit.align()
                        qubit.resonator.wait(t)

                    # --- Readout ---
                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                        if node.parameters.use_state_discrimination:
                            assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                            save(state[i], state_st[i])
                        qubit.resonator.wait(qubit.resonator.depletion_time // 4)

                    align()
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(t1_idle_times)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(t1_idle_times)).average().save(f"Q{i + 1}")
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(t1_idle_times)).average().save(f"state{i + 1}")

    node.namespace["qua_program_t1"] = t1_prog

    # ── Ramsey QUA program ───────────────────────────────────────────────────
    # Per-qubit IF correction derived from the chosen pulse's detuning field (compile-time)
    ramsey_corrections = {}
    if node.parameters.correct_with_pulse_detuning:
        for qubit in qubits:
            pulse = qubit.xy.operations.get(x180_op)
            ramsey_corrections[qubit.name] = int(getattr(pulse, "detuning", 0)) if pulse is not None else 0
    else:
        for qubit in qubits:
            ramsey_corrections[qubit.name] = 0

    with program() as ramsey_prog:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        idle_time     = declare(int)
        detuning_sign = declare(int)
        phi           = [declare(fixed) for _ in range(num_qubits)]

        if node.parameters.use_state_discrimination:
            state    = [declare(int)     for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            # Apply element IF correction once before all averages (compile-time constant)
            for qubit in multiplexed_qubits.values():
                if ramsey_corrections[qubit.name] != 0:
                    update_frequency(
                        qubit.xy.name,
                        int(qubit.xy.intermediate_frequency) + ramsey_corrections[qubit.name],
                    )

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_each_(idle_time, ramsey_idle_times):
                    with for_(*from_array(detuning_sign, detuning_signs)):
                        # --- State initialization ---
                        for i, qubit in multiplexed_qubits.items():
                            reset_frame(qubit.xy.name)

                        # --- Reset ---
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(
                                node.parameters.reset_type,
                                node.parameters.simulate,
                            )
                        align()

                        # --- Qubit drive ---
                        for i, qubit in multiplexed_qubits.items():
                            with if_(detuning_sign == 1):
                                assign(phi[i], Cast.mul_fixed_by_int(detuning * 1e-9, 4 * idle_time))
                            with else_():
                                assign(phi[i], Cast.mul_fixed_by_int(-detuning * 1e-9, 4 * idle_time))
                            qubit.xy.play(x180_op, amplitude_scale=0.5)
                            qubit.xy.frame_rotation_2pi(phi[i])
                            qubit.xy.wait(idle_time)
                            qubit.xy.play(x180_op, amplitude_scale=0.5)
                        align()

                        # --- Readout ---
                        for i, qubit in multiplexed_qubits.items():
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                            if node.parameters.use_state_discrimination:
                                assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                                save(state[i], state_st[i])
                            qubit.resonator.wait(qubit.resonator.depletion_time // 4)
                        align()

            # Restore element IF to nominal value after all averages
            for qubit in multiplexed_qubits.values():
                if ramsey_corrections[qubit.name] != 0:
                    update_frequency(qubit.xy.name, int(qubit.xy.intermediate_frequency))

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(detuning_signs)).buffer(len(ramsey_idle_times)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(detuning_signs)).buffer(len(ramsey_idle_times)).average().save(f"Q{i + 1}")
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(detuning_signs)).buffer(len(ramsey_idle_times)).average().save(f"state{i + 1}")

    node.namespace["qua_program_ramsey"] = ramsey_prog

    # ── Echo QUA program ─────────────────────────────────────────────────────
    with program() as echo_prog:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        t = declare(int)
        if node.parameters.use_state_discrimination:
            state    = [declare(int)     for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_each_(t, echo_idle_times):
                    # --- State initialization ---
                    for i, qubit in multiplexed_qubits.items():
                        reset_frame(qubit.xy.name)

                    # --- Reset ---
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(
                            node.parameters.reset_type,
                            node.parameters.simulate,
                        )
                    align()

                    # --- Qubit drive ---
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.play("x90")
                        qubit.xy.wait(t)
                        qubit.xy.play("x180")
                        qubit.xy.wait(t)
                        qubit.xy.play("-x90")
                        qubit.align()
                    align()

                    # --- Readout ---
                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                        if node.parameters.use_state_discrimination:
                            assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                            save(state[i], state_st[i])
                        qubit.resonator.wait(qubit.resonator.depletion_time // 4)
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(echo_idle_times)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(echo_idle_times)).average().save(f"Q{i + 1}")
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(echo_idle_times)).average().save(f"state{i + 1}")

    node.namespace["qua_program_echo"] = echo_prog

    # ── RPM QUA programs ─────────────────────────────────────────────────────
    node.namespace["qua_program_g"] = _rpm_program(
        qubits, amp_factors, num_qubits, n_avg, ef_op, u, start_from_g=True
    )
    node.namespace["qua_program_e"] = _rpm_program(
        qubits, amp_factors, num_qubits, n_avg, ef_op, u, start_from_g=False
    )


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Simulate only the T1 program for quick waveform inspection."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program_t1"], node.parameters
    )
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Run T1 + Ramsey + echo + RPM programs n_iter times and accumulate results."""
    qmm    = node.machine.connect()
    config = node.machine.generate_config()
    qubits = node.namespace["qubits"]
    n_iter = node.parameters.n_iter
    use_sd = node.parameters.use_state_discrimination
    detuning_mhz = node.parameters.ramsey_frequency_detuning_in_mhz

    iter_results  = {q.name: IterResult(qubit=q.name) for q in qubits}
    raw_t1_list:     list = []
    raw_ramsey_list: list = []
    raw_echo_list:   list = []
    raw_g_list:      list = []
    raw_e_list:      list = []

    t0 = time.time()

    for iteration in range(n_iter):
        # ── T1 sweep ─────────────────────────────────────────────────────────
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(node.namespace["qua_program_t1"])
            fetcher = XarrayDataFetcher(job, node.namespace["t1_sweep_axes"])
            for ds_t1 in fetcher:
                pass

        if not use_sd:
            ds_t1 = convert_IQ_to_V(ds_t1, qubits)
        T1_fits = fit_T1_dataset(ds_t1, use_sd)
        raw_t1_list.append(ds_t1.assign_coords(iteration=iteration))

        # ── Ramsey sweep ─────────────────────────────────────────────────────
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(node.namespace["qua_program_ramsey"])
            fetcher = XarrayDataFetcher(job, node.namespace["ramsey_sweep_axes"])
            for ds_ramsey in fetcher:
                pass

        if not use_sd:
            ds_ramsey = convert_IQ_to_V(ds_ramsey, qubits)
        ramsey_fits = fit_ramsey_dataset(ds_ramsey, use_sd, qubits, detuning_mhz)
        raw_ramsey_list.append(ds_ramsey.assign_coords(iteration=iteration))

        # ── Echo sweep ───────────────────────────────────────────────────────
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(node.namespace["qua_program_echo"])
            fetcher = XarrayDataFetcher(job, node.namespace["echo_sweep_axes"])
            for ds_echo in fetcher:
                pass

        if not use_sd:
            ds_echo = convert_IQ_to_V(ds_echo, qubits)
        echo_fits = fit_echo_dataset(ds_echo, use_sd)
        raw_echo_list.append(ds_echo.assign_coords(iteration=iteration))

        # ── RPM sweeps ───────────────────────────────────────────────────────
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(node.namespace["qua_program_g"])
            fetcher = XarrayDataFetcher(job, node.namespace["rpm_sweep_axes"])
            for ds_g in fetcher:
                pass

        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(node.namespace["qua_program_e"])
            fetcher = XarrayDataFetcher(job, node.namespace["rpm_sweep_axes"])
            for ds_e in fetcher:
                pass

        rpm_fits = fit_rpm_datasets(ds_g, ds_e, use_sd, qubits)
        raw_g_list.append(ds_g.assign_coords(iteration=iteration))
        raw_e_list.append(ds_e.assign_coords(iteration=iteration))

        # ── Accumulate ───────────────────────────────────────────────────────
        t_sec = time.time() - t0
        for q in qubits:
            qn = q.name
            iter_results[qn].t_sec.append(t_sec)
            iter_results[qn].T1_us.append(T1_fits[qn]["T1_us"])
            iter_results[qn].T1_error_us.append(T1_fits[qn]["T1_error_us"])
            iter_results[qn].T2star_us.append(ramsey_fits[qn]["T2star_us"])
            iter_results[qn].T2star_error_us.append(ramsey_fits[qn]["T2star_error_us"])
            iter_results[qn].freq_offset_hz.append(ramsey_fits[qn]["freq_offset_hz"])
            iter_results[qn].T2echo_us.append(echo_fits[qn]["T2echo_us"])
            iter_results[qn].T2echo_error_us.append(echo_fits[qn]["T2echo_error_us"])
            iter_results[qn].P_th.append(rpm_fits[qn]["P_th"])
            iter_results[qn].P_th_err.append(rpm_fits[qn]["P_th_err"])
            iter_results[qn].T_eff_mk.append(rpm_fits[qn]["T_eff_mk"])
            iter_results[qn].T_eff_mk_err.append(rpm_fits[qn]["T_eff_mk_err"])

        log_iteration(
            iteration, n_iter, t_sec,
            T1_fits, ramsey_fits, echo_fits, rpm_fits,
            log_callable=node.log,
        )

        # ── Checkpoint: persist all data collected so far ────────────────────
        # node.namespace is kept in sync so iter_results survives a crash in
        # interactive mode (accessible via node.namespace["iter_results"]).
        node.namespace["iter_results"]    = iter_results
        node.namespace["raw_t1_list"]     = raw_t1_list
        node.namespace["raw_ramsey_list"] = raw_ramsey_list
        node.namespace["raw_echo_list"]   = raw_echo_list
        node.namespace["raw_g_list"]      = raw_g_list
        node.namespace["raw_e_list"]      = raw_e_list

        # Write a single overwriting .h5 checkpoint so data survives a crash.
        # NOTE: node.save() is intentionally NOT called here — each call would
        # create a new, incrementing snapshot in the qualibrate database.
        # The official node.save() remains in analyse_data.
        _partial_path = (
            node._get_storage_manager().data_handler.root_data_folder
            / f"partial_{node.name}.h5"
        )
        build_dataset(iter_results).to_netcdf(_partial_path)


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset and reconstruct IterResult objects."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"]       = get_qubits(node)
    node.namespace["iter_results"] = reconstruct_iter_results(node.results["ds"])


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Assemble xr.Dataset from accumulated IterResult objects and log a summary."""
    iter_results = node.namespace["iter_results"]
    node.results["ds"] = build_dataset(iter_results)
    log_summary(iter_results, log_callable=node.log)

    # Concatenate per-iteration raw I/Q datasets and save for archival
    for key, list_name in [
        ("ds_raw_t1",     "raw_t1_list"),
        ("ds_raw_ramsey", "raw_ramsey_list"),
        ("ds_raw_echo",   "raw_echo_list"),
        ("ds_raw_g",      "raw_g_list"),
        ("ds_raw_e",      "raw_e_list"),
    ]:
        raw = node.namespace.get(list_name)
        if raw:
            node.results[key] = xr.concat(raw, dim="iteration")

    if node.parameters.use_state_discrimination and node.parameters.use_confusion_matrix_correction:
        qubits = node.namespace["qubits"]
        for key in ["ds_raw_t1", "ds_raw_ramsey", "ds_raw_echo", "ds_raw_g", "ds_raw_e"]:
            if key in node.results:
                node.results[key] = apply_confusion_matrix_correction(node.results[key], qubits)

    # Save immediately so all experimental data is on disk before plotting.
    # This guarantees no data loss if plot_data raises an exception.
    node.save()


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot T1, T2*, T2 echo, qubit frequency, and thermal population time traces."""
    try:
        fig = plot_T1_thermal_monitor(
            node.namespace["iter_results"],
            node.namespace["qubits"],
        )
        plt.show()
        node.results["figures"] = {"monitor": fig}
    except Exception as e:
        node.log(f"WARNING: plot_data failed ({e}); data already saved by analyse_data.")


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
