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
from quam_config import Quam
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from calibration_utils.chi_ramsey_stark import (
    Parameters,
    RamseyStarkFit,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_ramsey_stark,
)

logger = logging.getLogger(__name__)


# %% {Description}
description = """
        RAMSEY STARK-SHIFT — CAVITY-TRANSMON χ CALIBRATION (25)

Measures the dispersive shift χ between a transmon qubit and a storage cavity
mode using Ramsey interferometry under a continuous-wave (CW) cavity drive.

Physics
-------
In the dispersive regime:

    H/ħ = ω_r a†a  +  (ω_q/2) σ_z  +  χ a†a σ_z

The qubit frequency shifts by

    Δω_q = 2χ n̄_ss

when the cavity is populated with a steady-state photon number n̄_ss ∝ A²,
where A is the (dimensionless) cavity drive amplitude.

Experiment sequence
-------------------
For each (A, τ) pair:

  1. Reset cavity (thermal or active sideband) and qubit.
  2. Apply CW cavity drive at amplitude A for a fixed duration
       T_total = ring_up_ns + max_delay_ns + 2 × t_x90 + buffer
     The cavity reaches steady state n̄_ss ∝ A² after ring_up_ns.
  3. Wait ring_up_ns on the qubit channel (cavity still being driven).
  4. Ramsey with artificial detuning:
       x90  –  wait τ  –  x90
     The qubit oscillates at f_R(A) = f_artificial + 2χ n̄_ss(A).
  5. Measure qubit state.

Analysis
--------
  • For each amplitude A, fit f_R(A) from the Ramsey oscillation.
  • Δf(A) = f_R(A) − f_R(0).
  • Fit Δf vs A²  →  slope = 2χ · (dn̄_ss/dA²).
  • If displacement_k is calibrated: χ = slope / (2 × displacement_k).

State updates
-------------
  • cavity_transmon_pairs[key].chi     [Hz]
"""

node = QualibrationNode[Parameters, Quam](
    name="25_chi_ramsey_stark",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name = "alice"
    # node.parameters.cavity_amplitudes = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
    # node.parameters.max_delay_ns = 3000
    # node.parameters.ring_up_ns = 2000
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
    """Build the QUA program for the Ramsey Stark chi measurement."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    n_avg = node.parameters.num_shots

    cavity_mode = _get_cavity_mode(node)
    node.namespace["cavity_mode"] = cavity_mode

    # Resolve sideband_drive for optional active cavity cooling
    mode_name = node.parameters.mode_name
    sideband_drive = None
    pairs = getattr(node.machine, "cavity_transmon_pairs", {})
    for pair_key, pair in pairs.items():
        if pair_key.endswith(f"_{mode_name}") and getattr(pair, "sideband_drive", None) is not None:
            sideband_drive = pair.sideband_drive
            break
    node.namespace["sideband_drive"] = sideband_drive

    # ── Sweep arrays ──────────────────────────────────────────────────────────
    min_delay_ns  = node.parameters.min_delay_ns
    max_delay_ns  = node.parameters.max_delay_ns
    delay_step_ns = node.parameters.delay_step_ns
    ring_up_ns    = node.parameters.ring_up_ns
    art_det_hz    = node.parameters.artificial_detuning_hz
    amps          = np.array(node.parameters.cavity_amplitudes, dtype=float)

    # Ramsey delay array (ns, then clock cycles, minimum 4 clk = 16 ns)
    tau_ns  = np.arange(min_delay_ns, max_delay_ns + delay_step_ns, delay_step_ns)
    tau_clk = np.maximum((tau_ns // 4).astype(int), 4)
    tau_ns_actual = (tau_clk * 4).astype(int)

    n_amp = len(amps)
    n_tau = len(tau_clk)

    # x90 pulse length (for cavity drive total-duration calculation)
    first_qubit = next(iter(qubits))
    x90_ns = int(first_qubit.xy.operations["x90"].length)  # in ns

    # Cavity CW drive duration: covers ring-up + max Ramsey delay + 2×x90 + 200 ns buffer.
    # This is FIXED regardless of the actual τ in each iteration, so the cavity is
    # always still being driven during the Ramsey wait.
    cavity_total_ns  = ring_up_ns + int(tau_ns_actual[-1]) + 2 * x90_ns + 200
    cavity_total_clk = cavity_total_ns // 4
    ring_up_clk      = ring_up_ns // 4

    node.namespace["tau_ns"]           = tau_ns_actual
    node.namespace["amps"]             = amps
    node.namespace["cavity_total_clk"] = cavity_total_clk

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "drive_amplitude": xr.DataArray(
            amps,
            attrs={"long_name": "cavity drive amplitude", "units": ""},
        ),
        "delay": xr.DataArray(
            tau_ns_actual,
            attrs={"long_name": "Ramsey delay", "units": "ns"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        n        = declare(int)
        n_st     = declare_stream()
        drive_amp = declare(fixed)
        tau_clk_v = declare(int)

        I, I_st, Q, Q_st, _, _ = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state    = [declare(int)         for _ in range(num_qubits)]
            state_st = [declare_stream()     for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                # Outer sweep: cavity drive amplitude
                # for_each_ is used instead of from_array because the amplitude
                # list may not satisfy from_array's even-spacing requirement.
                with for_each_(drive_amp, amps.tolist()):

                    # Inner sweep: Ramsey delay
                    with for_each_(tau_clk_v, tau_clk.tolist()):

                        # ── Reset cavity and qubit ────────────────────────────
                        sd = node.namespace["sideband_drive"]
                        for i, qubit in multiplexed_qubits.items():
                            cavity_mode.reset(
                                node.parameters.cavity_reset_type,
                                node.parameters.simulate,
                                log_callable=node.log,
                                sideband_drive=sd,
                                qubit_thermalization_time=qubit.thermalization_time,
                                fock_n=node.parameters.cavity_active_cooling_fock_n,
                                f0g1_pulse_duration_ns=node.parameters.f0g1_pulse_duration_ns,
                            )
                            qubit.reset(
                                node.parameters.reset_type,
                                node.parameters.simulate,
                                log_callable=node.log,
                            )

                        # ── Start CW cavity drive (fixed total duration) ──────
                        # align() ensures cavity_mode_drive and qubit channels
                        # start from the same time point.
                        align()
                        cavity_mode.cavity_mode_drive.play(
                            "displacement",
                            amplitude_scale=drive_amp,
                            duration=cavity_total_clk,
                        )

                        # ── Wait for cavity ring-up on qubit channels ─────────
                        # After ring_up_clk the cavity has reached steady state.
                        for i, qubit in multiplexed_qubits.items():
                            wait(ring_up_clk, qubit.xy.name)

                        # ── Ramsey sequence with detuning ─────────────────────
                        # Shift IF by +art_det_hz so the Ramsey oscillation is
                        # visible at A=0.  The additional 2χ n̄_ss shift is what
                        # we extract from the amplitude dependence.
                        for i, qubit in multiplexed_qubits.items():
                            update_frequency(
                                qubit.xy.name,
                                int(qubit.xy.intermediate_frequency) + art_det_hz,
                            )

                        # align()
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.play("x90")
                        for i, qubit in multiplexed_qubits.items():
                            wait(tau_clk_v, qubit.xy.name)
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.play("x90")

                        # Restore qubit IF immediately after Ramsey
                        for i, qubit in multiplexed_qubits.items():
                            update_frequency(
                                qubit.xy.name,
                                int(qubit.xy.intermediate_frequency),
                            )

                        # ── Measure ───────────────────────────────────────────
                        for i, qubit in multiplexed_qubits.items():
                            align(qubit.xy.name, qubit.resonator.name)
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                            if node.parameters.use_state_discrimination:
                                assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                                save(state[i], state_st[i])
                                wait(qubit.resonator.depletion_time // 4, qubit.resonator.name)
                            qubit.resonator.wait(node.machine.depletion_time * u.ns)

                        # ── Wait for cavity CW drive to finish ───────────────
                        # align() prevents the next-shot reset from starting
                        # before the cavity drive pulse has fully completed.
                        align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # Buffer order must match loop order: inner=tau, outer=amplitude
                I_st[i].buffer(n_tau).buffer(n_amp).average().save(f"I{i + 1}")
                Q_st[i].buffer(n_tau).buffer(n_amp).average().save(f"Q{i + 1}")
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(n_tau).buffer(n_amp).average().save(f"state{i + 1}")


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
        k: RamseyStarkFit(**v) for k, v in node.results["fit_results"].items()
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
    """Write χ to the corresponding CavityTransmonPair."""
    mode_name   = node.parameters.mode_name
    fit_results = {
        k: RamseyStarkFit(**v) for k, v in node.results["fit_results"].items()
    }

    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            res = fit_results.get(qubit.name)
            if res is None or not res.success or res.chi_hz is None:
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
