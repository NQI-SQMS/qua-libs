# %% {Imports}
import logging
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from calibration_utils.shared import apply_confusion_matrix_correction, _get_cavity_mode
from quam_config import Quam
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from calibration_utils.parity_time_measurement import (
    Parameters,
    ParityTimeFit,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_parity_time,
)

logger = logging.getLogger(__name__)


# %% {Description}
description = """
        PARITY-TIME CALIBRATION â€” WIGNER TOMOGRAPHY (30)

Experimentally calibrates the dispersive Ramsey wait time τ_parity required
for Wigner tomography.  τ_parity is the duration for which the qubit
accumulates phase nπ when the cavity contains n photons:

    χ_eff Â· τ_parity = π   →   τ_parity = 1 / (2 Â· f_χ)

This differs from the analytical estimate 1/(2χ) because:
  â€¢ The pulsed displacement (same hardware path as the actual Wigner experiment)
    includes AC Stark shifts not present in the CW chi_ramsey_stark calibration.
  â€¢ Higher-order dispersive terms (χ') shift the effective coupling.
  â€¢ Finite π/2 pulse durations contribute additional phase.

Experiment sequence
-------------------
For each delay τ:

  1. Reset cavity and qubit.
  2. Apply a short displacement pulse (displacement_scale â‰ˆ 0.5, nÌ„ â‰ˆ 1 photon).
  3. Ramsey:  selective_y90 → wait(τ) → selective_y90
     (selective_y90 = frame_rotation(+π/2) Â· selective_x180/2 Â· frame_rotation(-π/2))
  4. Measure qubit state.

P(e) oscillates primarily at f_χ = χ_eff/(2π) (n=1 term dominates for nÌ„ â‰ˆ 1).
A damped-cosine fit extracts f_χ and τ_parity = 1/(2Â·f_χ).

State updates
-------------
  â€¢ cavity_transmon_pairs["{qubit}_{mode}"].parity_time   [seconds]
  â€¢ cavity_transmon_pairs["{qubit}_{mode}"].chi           [Hz]
    chi = -chi_eff_hz  (full per-photon shift, negative for typical transmon-cavity)
    chi_eff_hz is the Ramsey oscillation frequency â‰ˆ PNRS peak spacing (n=0→1).
"""

node = QualibrationNode[Parameters, Quam](
    name="28_parity_time_measurement",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name         = "alice"
    # node.parameters.displacement_scale = 0.5
    # node.parameters.max_delay_ns       = 4000
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the QUA parity-time Ramsey program."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    cavity_mode = _get_cavity_mode(node)
    node.namespace["cavity_mode"] = cavity_mode

    # Resolve sideband_drive for optional active cavity cooling
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
    node.namespace["sideband_drive"] = sideband_drive

    # -- Delay sweep -----------------------------------------------------------
    min_ns  = node.parameters.min_delay_ns
    max_ns  = node.parameters.max_delay_ns
    step_ns = node.parameters.delay_step_ns

    tau_ns  = np.arange(min_ns, max_ns + step_ns, step_ns)
    tau_clk = np.maximum((tau_ns // 4).astype(int), 4)
    tau_ns_actual = (tau_clk * 4).astype(int)
    n_tau = len(tau_clk)

    node.namespace["tau_ns"] = tau_ns_actual

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "delay": xr.DataArray(
            tau_ns_actual,
            attrs={"long_name": "Ramsey wait time", "units": "ns"},
        ),
    }

    disp_scale = float(node.parameters.displacement_scale) / alpha_max
    n_avg = node.parameters.num_shots

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_stream()
        tau_clk_v = declare(int)

        I, I_st, Q, Q_st, _, _ = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state    = [declare(int)     for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_each_(tau_clk_v, tau_clk.tolist()):

                    # -- Reset -------------------------------------------------
                    sd = node.namespace["sideband_drive"]
                    for i, qubit in multiplexed_qubits.items():
                        pair_key = f"{qubit.name}_{node.parameters.mode_name}"
                        _pairs = getattr(node.machine, "cavity_transmon_pairs", None)
                        _chi = float(_pairs[pair_key].chi) if (_pairs and pair_key in _pairs and _pairs[pair_key].chi is not None) else None
                        cavity_mode.reset(
                            node.parameters.cavity_reset_type,
                            node.parameters.simulate,
                            log_callable=node.log,
                            sideband_drive=sd,
                            qubit_thermalization_time=qubit.thermalization_time,
                            fock_n=node.parameters.cavity_active_cooling_fock_n,
                            sideband_pulse_duration_ns=node.parameters.sideband_pulse_duration_ns,
                            chi_hz=_chi,
                        )
                        qubit.reset(
                            node.parameters.reset_type,
                            node.parameters.simulate,
                            log_callable=node.log,
                        )

                    # -- Cavity displacement (~nÌ„ photons) ----------------------
                    align()
                    cavity_mode.cavity_mode_drive.play(
                        "displacement", amplitude_scale=disp_scale
                    )

                    # -- Parity Ramsey: selective_y90 → wait(τ) → selective_y90 --
                    # selective_y90 = frame_rotation(+π/2) Â· selective_x180/2 Â· frame_rotation(-π/2)
                    # Restoring the frame after each half-pulse preserves phase coherence.
                    for i, qubit in multiplexed_qubits.items():
                        align(cavity_mode.cavity_mode_drive.name, qubit.xy.name)
                        qubit.xy.frame_rotation_2pi(0.25)
                        qubit.xy.play("selective_x180", amplitude_scale=0.5)
                        qubit.xy.frame_rotation_2pi(-0.25)
                    for i, qubit in multiplexed_qubits.items():
                        wait(tau_clk_v, qubit.xy.name)
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.frame_rotation_2pi(0.25)
                        qubit.xy.play("selective_x180", amplitude_scale=0.5)
                        qubit.xy.frame_rotation_2pi(-0.25)

                    # -- Measure -----------------------------------------------
                    for i, qubit in multiplexed_qubits.items():
                        align(qubit.xy.name, qubit.resonator.name)
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                        if node.parameters.use_state_discrimination:
                            assign(
                                state[i],
                                Cast.to_int(
                                    I[i] > qubit.resonator.operations["readout"].threshold
                                ),
                            )
                            save(state[i], state_st[i])
                        qubit.resonator.wait(node.machine.depletion_time // 4)

                    align()
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(n_tau).average().save(f"I{i + 1}")
                Q_st[i].buffer(n_tau).average().save(f"Q{i + 1}")
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(n_tau).average().save(f"state{i + 1}")


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


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"]      = get_qubits(node)
    node.namespace["cavity_mode"] = _get_cavity_mode(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    if node.parameters.use_state_discrimination and node.parameters.use_confusion_matrix_correction:
        node.results["ds_raw"] = apply_confusion_matrix_correction(node.results["ds_raw"], node.namespace["qubits"])
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
        k: ParityTimeFit(**v) for k, v in node.results["fit_results"].items()
    }
    fig = plot_parity_time(
        node.results["ds_fit"],
        node.namespace["qubits"],
        fit_results=fit_results,
        mode_name=node.parameters.mode_name,
    )
    plt.show()
    node.results["figures"] = {"parity_time_ramsey": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Write τ_parity to cavity_transmon_pairs[key].parity_time [seconds]."""
    mode_name = node.parameters.mode_name
    fit_results = {
        k: ParityTimeFit(**v) for k, v in node.results["fit_results"].items()
    }

    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            res = fit_results.get(qubit.name)
            if res is None or not res.success:
                continue

            pair_key = f"{qubit.name}_{mode_name}"
            pairs = getattr(node.machine, "cavity_transmon_pairs", None)
            if pairs is not None and pair_key in pairs:
                pairs[pair_key].parity_time = float(res.parity_time_s)
                # chi_eff_hz is the Ramsey oscillation frequency = PNRS peak spacing (n=0→1)
                # pair.chi stores the full per-photon shift with sign: chi = -chi_eff_hz
                pairs[pair_key].chi = -float(res.chi_eff_hz)
                node.log(
                    f"Updated {pair_key}.parity_time = {res.parity_time_s * 1e9:.0f} ns  |  "
                    f"chi = {-res.chi_eff_hz / 1e3:.2f} kHz"
                )
            else:
                logger.warning(
                    "cavity_transmon_pairs[%s] not found â€” "
                    "parity_time not persisted.", pair_key
                )

            break  # single cavity mode per run


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
