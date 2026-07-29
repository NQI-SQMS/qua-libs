# %% {Imports}
import logging
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
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from calibration_utils.displacement_wigner_calibration import (
    Parameters,
    FitParameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    resolve_parity_time_ns,
    plot_wigner_1d,
)

logger = logging.getLogger(__name__)


# %% {Node initialisation}
description = """
        DISPLACEMENT WIGNER CALIBRATION (32)

Calibrates the unit displacement amplitude by measuring a 1D slice of the
cavity Wigner function at zero phase.

Sequence (per displacement amplitude scale a):
  1. Thermalize cavity (wait based on cavity T1).
  2. Apply displacement pulse at amplitude_scale = a.
  3. Parity Ramsey with strict timing:
       x90 → wait(t_parity) → x90
     where t_parity = 1 / (2·chi_hz) so each photon accumulates a phase of π.
  4. Measure qubit state (parity measurement).
  5. Reverse displacement: amplitude_scale = -a.

The measured signal P_e(a) is a 1D slice of the Wigner function W(α):

    P_e(a) ≈ offset + amplitude · exp(-2·(a / sigma)²)

The fitted sigma is the amplitude_scale for exactly 1 zero-point fluctuation
(= 1 photon for a coherent state) — the unit displacement scale A₁ph.

Prerequisites:
    - Calibrated x90 pulse on the qubit (nodes 03/04).
    - Known chi_hz (from node 28 or set via parameters.chi_hz), OR set
      parameters.parity_time_ns directly.
    - Displacement pulse operation on cavity_mode_drive.

State updates:
    - cavity_mode.cavity_mode_drive.operations["displacement"].amplitude
      (multiplied by sigma so that amplitude_scale=1 → 1 photon)
"""

node = QualibrationNode[Parameters, Quam](
    name="30_displacement_wigner_calibration",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name = "alice"
    # node.parameters.amp_min = -2.0
    # node.parameters.amp_max = 2.0
    # node.parameters.amp_points = 101
    # node.parameters.parity_time_ns = 7140   # set directly, OR
    # node.parameters.chi_hz = 70e3           # let node compute t_parity
    # node.parameters.num_shots = 400
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the QUA program for the 1D Wigner parity sweep."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    n_avg = node.parameters.num_shots

    cavity_mode = _get_cavity_mode(node)
    node.namespace["cavity_mode"] = cavity_mode

    # Parity time (clock cycles)
    parity_time_ns = resolve_parity_time_ns(node)
    node.namespace["parity_time_ns"] = parity_time_ns
    parity_clk = parity_time_ns // 4  # QUA clock cycles (4 ns each)

    # Thermalization: cavity T1 × factor, clamped to [16 ns, 10 s]
    t1_s = getattr(cavity_mode, "T1", 100e-6)
    factor = getattr(cavity_mode, "thermalization_time_factor", 5.0)
    therm_clk = int(np.clip(t1_s * factor * 1e9 / 4, 4, 2_500_000_000))

    # amp_min/amp_max are in photon units (alpha). Convert to amplitude_scale using
    # current alpha_max so the sweep covers the same physical range on re-runs.
    mode_name = node.parameters.mode_name
    _pairs = getattr(node.machine, "cavity_transmon_pairs", {})
    current_alpha_max = 1.0
    for _pk, _pv in _pairs.items():
        if _pk.endswith(f"_{mode_name}") and getattr(_pv, "displacement_alpha_max", None) is not None:
            current_alpha_max = float(_pv.displacement_alpha_max)
            break
    node.namespace["current_alpha_max"] = current_alpha_max

    # Amplitude sweep (includes negative values for full Wigner slice)
    amp_array = np.linspace(
        node.parameters.amp_min / current_alpha_max,
        node.parameters.amp_max / current_alpha_max,
        node.parameters.amp_points,
    )
    node.namespace["amp_array"] = amp_array

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "amp": xr.DataArray(
            amp_array,
            attrs={"long_name": "displacement amplitude scale", "units": "a.u."},
        ),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        a = declare(fixed)
        a_neg = declare(fixed)

        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(a, amp_array)):
                    assign(a_neg, -a)

                    # --- Thermalize cavity + reset qubit BEFORE displacement ---
                    # Reset must happen before displacing so that qubit drives do
                    # not disturb the displaced cavity state.
                    cavity_mode.cavity_mode_drive.wait(therm_clk)
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(
                            node.parameters.reset_type,
                            node.parameters.simulate,
                            log_callable=node.log,
                        )

                    # --- Displace ---
                    # align() with no args includes cavity_mode_drive, which is not part of qubit.align().
                    align()
                    cavity_mode.cavity_mode_drive.play("displacement", amplitude_scale=a)

                    # --- Parity Ramsey with strict timing ---
                    # strict_timing_ prevents QUA from inserting extra delays between
                    # the align, x90, wait, x90 — keeping t_parity exact.
                    # Align cavity drive with qubit xy before strict_timing_ so alignment
                    # overhead does not perturb the parity wait time.
                    for i, qubit in multiplexed_qubits.items():
                        align(cavity_mode.cavity_mode_drive.name, qubit.xy.name)
                        with strict_timing_():
                            qubit.xy.play("x90")
                            qubit.xy.wait(parity_clk)
                            qubit.xy.play("x90")

                    # --- Measure ---
                    align()
                    for i, qubit in multiplexed_qubits.items():
                        qubit.readout_state(
                            state[i] if node.parameters.use_state_discrimination else None,
                            I=I[i], Q=Q[i], I_st=I_st[i], Q_st=Q_st[i],
                            state_st=state_st[i] if node.parameters.use_state_discrimination else None,
                        )

                    align()
                    # --- Reverse displacement (cavity back toward vacuum) ---
                    align()
                    cavity_mode.cavity_mode_drive.play("displacement", amplitude_scale=a_neg)

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(amp_array)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(amp_array)).average().save(f"Q{i + 1}")
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(amp_array)).average().save(f"state{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}


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
    node.namespace["parity_time_ns"] = node.results.get("parity_time_ns", resolve_parity_time_ns(node))


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    if node.parameters.use_state_discrimination and node.parameters.use_confusion_matrix_correction:
        node.results["ds_raw"] = apply_confusion_matrix_correction(node.results["ds_raw"], node.namespace["qubits"])
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    node.results["parity_time_ns"] = node.namespace.get("parity_time_ns")

    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        q: ("successful" if res["success"] else "failed")
        for q, res in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    cavity_mode = node.namespace["cavity_mode"]
    base_amp = float(cavity_mode.cavity_mode_drive.operations["displacement"].amplitude)
    fig = plot_wigner_1d(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["fit_results"],
        parity_time_ns=node.results.get("parity_time_ns", 0),
        base_amplitude=base_amp,
        current_alpha_max=node.namespace.get("current_alpha_max", 1.0),
    )
    plt.show()
    node.results["figures"] = {"wigner_1d": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    cavity_mode = node.namespace["cavity_mode"]
    mode_name = node.parameters.mode_name
    base_amp = float(cavity_mode.cavity_mode_drive.operations["displacement"].amplitude)

    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            res = node.results["fit_results"].get(qubit.name)
            if res is None or not res["success"]:
                continue

            sigma = res["sigma"]

            MAX_VOLTAGE = 0.5      # OPX+ DAC ceiling [V]
            AMP_SCALE_LIMIT = 1.9  # firmware headroom

            alpha_max = MAX_VOLTAGE / (base_amp * sigma * AMP_SCALE_LIMIT)
            cal_amplitude = base_amp * sigma * alpha_max  # ≈ 0.263 V

            cavity_mode.cavity_mode_drive.operations["displacement"].amplitude = float(cal_amplitude)

            pair_key = f"{qubit.name}_{mode_name}"
            pairs = getattr(node.machine, "cavity_transmon_pairs", None)
            if pairs is not None and pair_key in pairs:
                k_fit = 1.0 / (sigma ** 2)
                if hasattr(pairs[pair_key], "displacement_k"):
                    pairs[pair_key].displacement_k = float(k_fit)
                if hasattr(pairs[pair_key], "displacement_alpha_max"):
                    pairs[pair_key].displacement_alpha_max = float(alpha_max)

            node.log(
                f"Displacement (Wigner) calibration: sigma={sigma:.4f}, alpha_max={alpha_max:.3f}, "
                f"stored amplitude={cal_amplitude:.6f} V  "
                f"(amplitude_scale=1 -> {alpha_max:.2f} photons, "
                f"max safe alpha={alpha_max * AMP_SCALE_LIMIT:.2f})"
            )

            break  # one cavity mode shared across all qubits in this run


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
