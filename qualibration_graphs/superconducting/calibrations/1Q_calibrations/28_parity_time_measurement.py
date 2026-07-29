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
from calibration_utils.shared import (
    apply_confusion_matrix_correction,
    _get_cavity_mode,
    _get_pair_components,
)
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
        PARITY-TIME CALIBRATION — WIGNER TOMOGRAPHY (30)

Experimentally calibrates the dispersive Ramsey wait time t_parity required
for Wigner tomography.  t_parity is the duration for which the qubit
accumulates phase n*pi when the cavity contains n photons:

    chi_eff * t_parity = pi   ->   t_parity = 1 / (2 * f_chi)

Experiment sequence
-------------------
For each delay tau:

  1. Reset cavity and qubit.
  2. Prepare Fock |1> via the f0g1 sideband ladder
     (ge pi -> ef pi -> f0g1 pi, identical to the Fock-state T1/T2 nodes).
  3. Reset qubit frequency to the bare GE IF (n=0 photons).
  4. Standard Ramsey:  x90 -> wait(tau) -> x90
  5. Measure qubit state.

With 1 photon in the cavity the qubit is detuned from the bare GE frequency
by chi_eff, so P(e) oscillates at f_chi = chi_eff / (2*pi).
A damped-cosine fit extracts f_chi and t_parity = 1 / (2 * f_chi).

Prerequisites: calibrated f0g1 sideband (nodes 26, 26b, 26e, 26f) and EF_x180.

State updates
-------------
  - cavity_transmon_pairs["{qubit}_{mode}"].parity_time   [seconds]  (always)
  - cavity_transmon_pairs["{qubit}_{mode}"].chi           [Hz]       (only if currently None)
    chi = -chi_eff_hz  (full per-photon shift, negative for typical transmon-cavity)
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

    mode_name = node.parameters.mode_name
    pair, qubit_pair, sb_drive, _ = _get_pair_components(node)
    node.namespace["sb_drive"] = sb_drive
    node.namespace["pair"] = pair
    base_qubit_if = int(qubit_pair.xy.intermediate_frequency)
    node.namespace["base_qubit_if"] = base_qubit_if
    node.namespace["sideband_drive"] = sb_drive

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

                    # -- Fock |1> preparation via f0g1 sideband ladder --------
                    align()
                    node.namespace["pair"].fock_prep_qua(1, qubit)

                    # -- Reset qubit to bare GE frequency (n=0) ---------------
                    align()
                    qubit.xy.update_frequency(node.namespace["base_qubit_if"])

                    # -- Standard Ramsey: x90 -> wait(tau) -> x90 -------------
                    for i, qubit in multiplexed_qubits.items():
                        align(qubit.xy.name)
                    with strict_timing_():
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.play("x90")
                        for i, qubit in multiplexed_qubits.items():
                            wait(tau_clk_v, qubit.xy.name)
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.play("x90")

                    # -- Measure -----------------------------------------------
                    for i, qubit in multiplexed_qubits.items():
                        align(qubit.xy.name, qubit.resonator.name)
                        qubit.readout_state(
                            state[i] if node.parameters.use_state_discrimination else None,
                            I=I[i], Q=Q[i], I_st=I_st[i], Q_st=Q_st[i],
                            state_st=state_st[i] if node.parameters.use_state_discrimination else None,
                        )

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
                # Only write chi if not yet calibrated by another node (e.g. chi_ramsey_stark)
                if pairs[pair_key].chi is None:
                    pairs[pair_key].chi = -float(res.chi_eff_hz)
                    node.log(f"  chi written: {-res.chi_eff_hz / 1e3:.2f} kHz")
                else:
                    node.log(f"  chi kept: {pairs[pair_key].chi / 1e3:.2f} kHz (not overwritten)")
                node.log(
                    f"Updated {pair_key}.parity_time = {res.parity_time_s * 1e9:.0f} ns  |  "
                    f"chi = {-res.chi_eff_hz / 1e3:.2f} kHz"
                )
            else:
                logger.warning(
                    "cavity_transmon_pairs[%s] not found — "
                    "parity_time not persisted.", pair_key
                )

            break  # single cavity mode per run


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
