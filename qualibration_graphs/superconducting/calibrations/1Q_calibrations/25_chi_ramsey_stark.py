# %% {Imports}
import logging
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

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
    _fock_prep_qua,
)
from quam_config import Quam
from qualibration_libs.parameters import get_qubits, get_idle_times_in_clock_cycles
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from calibration_utils.chi_ramsey_stark import (
    FockChiFit,
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_ramsey_stark,
)

logger = logging.getLogger(__name__)


# %% {Description}
description = """
        CHI CALIBRATION — FOCK |1> QUBIT RAMSEY (25)

Measures the dispersive shift χ between a transmon qubit and a storage cavity
mode by preparing exactly one photon (Fock |1>) in the cavity and performing
a qubit Ramsey experiment.

Physics
-------
In the dispersive regime the qubit frequency shifts by χ per cavity photon:

    f_qubit(n=1) = f_qubit(n=0) + χ

Driving the qubit at its bare ge frequency with an artificial detuning δ, the
Ramsey oscillation frequency is:

    f_osc = δ + χ

so χ is extracted directly:

    χ = f_osc − δ

Experiment sequence (per Ramsey delay τ)
-----------------------------------------
  1. Reset cavity (thermal or active sideband) and qubit.
  2. Prepare Fock |1> (method selected by fock1_prep_method):
       'sideband'          — ge pi -> ef pi -> f0g1 pi  ->  |g,1>
       'snap_displacement' — D(alpha1) -> SNAP0 -> D(alpha2)  ->  ~|1>
  3. Qubit Ramsey at bare ge frequency (no frequency update):
       x90  —  wait τ  —  frame_rotation(δ × τ)  —  x90
  4. Measure qubit state.

Analysis
--------
  • Fit decaying cosine to P(|e>) vs τ  →  f_osc, T2*.
  • χ = f_osc − artificial_detuning_hz.

State updates
-------------
  • cavity_transmon_pairs[key].chi  [Hz]
"""

node = QualibrationNode[Parameters, Quam](
    name="25_chi_ramsey_stark",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name = "alice"
    # node.parameters.fock1_prep_method = "sideband"
    # node.parameters.artificial_detuning_hz = 200_000
    # node.parameters.max_wait_time_in_ns = 5000
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

    # SNAP+displacement: resolve amplitude scales
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
            f"Chi Ramsey: SNAP+displacement Fock|1> prep, "
            f"detuning={node.parameters.artificial_detuning_hz:.0f} Hz"
        )
    else:
        node.log(
            f"Chi Ramsey: sideband Fock|1> prep, "
            f"detuning={node.parameters.artificial_detuning_hz:.0f} Hz"
        )

    # Phase increment per clock cycle for the Ramsey frame rotation:
    # detuning_hz * 1e-9 * 4 ns/cc = turns per clock cycle
    detuning_turns_per_cc = node.parameters.artificial_detuning_hz * 1e-9 * 4

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

                        # -- 1. Reset cavity and qubit -------------------------
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
                            # -- 2a. Fock |1> prep via sideband ladder ---------
                            # ge pi -> ef pi -> f0g1 pi  ->  |g,1>
                            align(qubit.xy.name, sideband_drive.name, qubit.resonator.name)
                            _fock_prep_qua(1, pair, qubit, sideband_drive)

                        else:  # snap_displacement
                            # -- 2b. Fock |1> prep: D(alpha1) -> SNAP0 -> D(alpha2)
                            align(qubit.xy.name, cavity_mode.cavity_mode_drive.name, qubit.resonator.name)
                            cavity_mode.cavity_mode_drive.play(
                                "displacement", amplitude_scale=node.namespace["amp_scale1"]
                            )
                            align(qubit.xy.name, cavity_mode.cavity_mode_drive.name)
                            qubit.xy.update_frequency(qubit_IF)
                            qubit.xy.play("selective_x180")  # SNAP0: two selective pi pulses
                            qubit.xy.play("selective_x180")  # apply pi phase to |n=0> component
                            align(qubit.xy.name, cavity_mode.cavity_mode_drive.name)
                            cavity_mode.cavity_mode_drive.play(
                                "displacement", amplitude_scale=node.namespace["amp_scale2"]
                            )

                        # -- 3. Qubit Ramsey at bare ge frequency --------------
                        # Drive at qubit_IF (no update_frequency).
                        # The cavity photon shifts the qubit by chi, so the
                        # Ramsey oscillates at artificial_detuning + chi.
                        pi_op = node.parameters.ramsey_pi_pulse_op
                        align(qubit.xy.name, qubit.resonator.name)
                        qubit.xy.update_frequency(qubit_IF)
                        assign(phase, Cast.mul_fixed_by_int(detuning_turns_per_cc, t))
                        reset_frame(qubit.xy.name)
                        with strict_timing_():
                            qubit.xy.play(pi_op, amplitude_scale=0.5)   # pi/2 opening arm
                            qubit.xy.wait(t)                             # Ramsey wait
                            frame_rotation_2pi(phase, qubit.xy.name)
                            qubit.xy.play(pi_op, amplitude_scale=0.5)   # pi/2 closing arm
                        reset_frame(qubit.xy.name)

                        # -- 4. Measure ----------------------------------------
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
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    node.results["simulation"] = {
        "figure": fig, "wf_report": wf_report.to_dict(), "samples": samples
    }


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


# %% {Load_data}
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
        node.results["ds_raw"] = apply_confusion_matrix_correction(
            node.results["ds_raw"], node.namespace["qubits"]
        )
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {
        q: ("successful" if res.success else "failed")
        for q, res in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    fit_results = {
        k: FockChiFit(**v) for k, v in node.results["fit_results"].items()
    }
    fig = plot_ramsey_stark(
        node.results["ds_raw"],
        node.namespace["qubits"],
        fit_results=fit_results,
        mode_name=node.parameters.mode_name,
    )
    plt.show()
    node.results["figures"] = {"chi_ramsey_stark": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Write chi to the corresponding CavityTransmonPair."""
    mode_name = node.parameters.mode_name
    fit_results = {
        k: FockChiFit(**v) for k, v in node.results["fit_results"].items()
    }

    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            res = fit_results.get(qubit.name)
            if res is None or not res.success or not np.isfinite(res.chi_hz):
                continue

            chi_hz = float(res.chi_hz)

            pair_key = f"{qubit.name}_{mode_name}"
            pairs = getattr(node.machine, "cavity_transmon_pairs", None)
            if pairs is not None and pair_key in pairs:
                pairs[pair_key].chi = chi_hz

            break  # single cavity mode per run


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
