# %% {Imports}
import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

from qualibration_libs.analysis import fit_oscillation_decay_exp
from calibration_utils.ramsey.analysis import calculate_fit_results
from calibration_utils.fNgN1_ramsey import Parameters

from quam_builder.architecture.superconducting.qubit_pair.cavity_transmon_pair import SidebandTransition
from calibration_utils.shared import (
    apply_confusion_matrix_correction,
    _get_pair_components,
    _get_transition_rf,
    _fock_prep_qua,
    _ge_if_at_fock,
    _ef_if_at_fock,
)

# %% {Node initialisation}
description = """
        SIDEBAND RAMSEY - precise frequency calibration for any |n⟩ → |n+1⟩ transition

Performs a Ramsey fringe experiment on the sideband drive to extract the resonance
frequency of the |f, n⟩ ↔ |g, n+1⟩ transition with high precision.

An artificial detuning δ is added so that fringes are visible even at zero drive
detuning.  The fitted oscillation frequency f_obs satisfies:
    f_obs = |f_drive - f_sideband + δ|
from which the corrected sideband frequency is:
    f_sideband = f_drive + δ - f_obs  (using the sign that minimises |correction|)

Sequence:
  0. Thermalize cavity and qubit.
  1. [Fock prep] For j = 0 … fock_level-1:
       π_ge → π_ef → sideband_pi(f{j}g{j+1}) → cavity in |j+1⟩, qubit in |g⟩.
  2. π_ge  →  |e⟩
  3. π_ef  →  |f⟩
  4. π/2 sideband pulse  →  superposition (|f,n⟩ + |g,n+1⟩).
  5. Wait τ  +  virtual frame rotation (artificial detuning δ).
  6. π/2 sideband pulse  →  interference.
  7. π_ef  (back-swap: maps |f⟩ → |e⟩ for readout).
  8. Measure qubit state  →  oscillation vs τ.

Prerequisites:
    - Calibrated sideband frequency (node 26, fock_level=k).
    - Calibrated π-pulse length (node 26b, fock_level=k) → stored in pair.extras
      or sideband_drive.operations["f{k}g{k+1}_pi"].length.

State update:
    - cavity_transmon_pairs["{qubit}_{mode}"].extras["f{k}g{k+1}_RF_frequency"]
      (fine-tuned from Ramsey correction).
"""


@dataclass
class RamseyFitParameters:
    """Fit results for a single qubit's sideband Ramsey experiment."""
    oscillation_frequency_hz: float
    """Observed Ramsey oscillation frequency [Hz]."""
    frequency_correction_hz: float
    """Correction to add to the sideband RF frequency [Hz].  freq_correction = δ - f_obs."""
    T2_ramsey_ns: float
    """Fitted T2* of the sideband [ns]."""
    success: bool


node = QualibrationNode[Parameters, Quam](
    name="26c_fNgN1_ramsey",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Debugging / local overrides."""
    # node.parameters.fock_level = 1
    pass


node.machine = Quam.load()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _fit_ramsey(ds: xr.Dataset, node) -> Tuple[xr.Dataset, Dict[str, RamseyFitParameters]]:
    """Fit Ramsey oscillations using the same approach as the ge Ramsey node.

    Uses fit_oscillation_decay_exp along the wait_ns dimension (keeping the
    detuning_signs dimension intact), then extracts the unambiguous frequency
    correction via calculate_fit_results:
        freq_offset = (f_obs_+ - f_obs_-) / 2  →  applied as  rf -= freq_offset
    """
    signal_name = "state" if node.parameters.use_state_discrimination else "I"
    signal = getattr(ds, signal_name)

    # Fit decaying cosine along wait_ns for each (qubit, detuning_signs) slice
    fit = fit_oscillation_decay_exp(signal, "wait_ns")

    ds_fit = xr.merge([ds, fit.rename("fit")])

    frequency  = fit.sel(fit_vals="f")
    decay      = fit.sel(fit_vals="decay")
    decay_res  = fit.sel(fit_vals="decay_decay")
    tau        = 1 / decay                            # in ns⁻¹ → T2 in ns
    tau_error  = tau * (np.sqrt(decay_res) / decay)

    detuning_hz = float(node.parameters.artificial_detuning_hz)
    freq_offset, decay_out, decay_error_out = calculate_fit_results(
        frequency, tau, tau_error, fit, detuning_hz
    )

    nan_fail = np.isnan(freq_offset) | np.isnan(decay_out)
    fit_results = {}
    for q in ds.qubit.values:
        success = bool(~nan_fail.sel(qubit=q).values)
        # freq_offset is Î" in GHz; convert to Hz and negate → correction to ADD to rf
        freq_off_hz = 1e9 * float(freq_offset.sel(qubit=q))
        T2_ns = 1e9 * float(decay_out.sel(qubit=q))  # decay is in ns⁻¹ → T2 in ns
        f_avg_hz = 1e9 * float(frequency.mean(dim="detuning_signs").sel(qubit=q).values)
        fit_results[q] = RamseyFitParameters(
            oscillation_frequency_hz=f_avg_hz,
            frequency_correction_hz=-freq_off_hz,
            T2_ramsey_ns=T2_ns,
            success=success,
        )

    return ds_fit, fit_results


def _log_ramsey_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        if isinstance(res, dict):
            res = type("FR", (), res)()
        status = "SUCCESS" if res.success else "FAIL"
        log_callable(
            f"Results for qubit {q}: {status}\n"
            f"\tf_osc: {res.oscillation_frequency_hz * 1e-3:.1f} kHz | "
            f"correction: {res.frequency_correction_hz * 1e-3:.1f} kHz | "
            f"T2*: {res.T2_ramsey_ns * 1e-3:.1f} us"
        )


def _plot_ramsey(ds_fit: xr.Dataset, qubits, fit_results: Dict, k: int, mode_name: str):
    from qualibration_libs.plotting import QubitGrid, grid_iter
    from qualibration_libs.analysis.models import oscillation_decay_exp
    sign_colors = {1: "C0", -1: "C2"}
    sign_labels = {1: "+δ", -1: "−δ"}
    detuning_signs = ds_fit.detuning_signs.values.tolist()
    signal_name = "state" if "state" in ds_fit.data_vars else "I"

    grid = QubitGrid(ds_fit, [q.grid_location for q in qubits], size=6)
    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        ds_q = ds_fit.sel(qubit=q_name)
        x = ds_q.wait_ns.values  # nanoseconds

        for sign in detuning_signs:
            color = sign_colors.get(int(sign), "C0")
            label = sign_labels.get(int(sign), str(sign))
            y = getattr(ds_q, signal_name).sel(detuning_signs=sign).values
            ax.plot(x * 1e-3, y, ".", ms=3, color=color, label=label)
            if "fit" in ds_fit.data_vars:
                p = ds_q.fit.sel(detuning_signs=sign)
                fit_y = oscillation_decay_exp(
                    x,
                    float(p.sel(fit_vals="a")),
                    float(p.sel(fit_vals="f")),
                    float(p.sel(fit_vals="phi")),
                    float(p.sel(fit_vals="offset")),
                    float(p.sel(fit_vals="decay")),
                )
                if np.any(np.isfinite(fit_y)):
                    ax.plot(x * 1e-3, fit_y, "-", lw=1.5, color=color)

        res = fit_results.get(q_name)
        if res and (res["success"] if isinstance(res, dict) else res.success):
            corr = res["frequency_correction_hz"] if isinstance(res, dict) else res.frequency_correction_hz
            ax.set_title(f"Δf = {corr * 1e-3:.1f} kHz", fontsize=9)
        ax.set_xlabel("Wait time (µs)")
        ax.set_ylabel("State")
        ax.legend(fontsize=7)
    grid.fig.suptitle(f"Sideband Ramsey f{k}g{k+1} — {mode_name}")
    grid.fig.tight_layout()
    return grid.fig


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    wait_times_cc = (
        np.linspace(
            node.parameters.min_wait_ns,
            node.parameters.max_wait_ns,
            node.parameters.num_wait_points,
        )
        // 4
    ).astype(int)
    wait_times_ns = wait_times_cc * 4

    k = node.parameters.fock_level
    pair, pair_qubit, sideband_drive, cav_mode = _get_pair_components(node)

    # Calibrated frequency for this transition
    centre_rf = _get_transition_rf(pair, sideband_drive, k)
    if_offset = int(centre_rf - sideband_drive.RF_frequency)
    target_if = int(sideband_drive.intermediate_frequency) + if_offset

    # π/2 duration in clock cycles (half the calibrated pi pulse).
    # π/2 flat duration: half the calibrated flat-top pi duration.
    tr_k = pair.transitions.get(f"f{k}g{k+1}")
    if tr_k is not None and tr_k.pi_flat_top_length_ns:
        pi_ns = int(tr_k.pi_flat_top_length_ns)
    else:
        pi_ns = int(sideband_drive.operations["sideband_square"].length) * 4
    pi2_cc = max(pi_ns // 8, 4)  # flat portion cc: pi_ns//4 for pi, //2 for half → //8; min 4 cc

    # Artificial detuning factor:  phi = detuning_hz * 1e-9 * (4 * wait_cc) [cycles]
    detuning_factor     = float(node.parameters.artificial_detuning_hz) * 1e-9
    detuning_factor_neg = -detuning_factor

    detuning_signs = [-1, 1]  # inner loop; sign=-1 → −δ, sign=+1 → +δ

    ge_if_k = _ge_if_at_fock(pair, pair_qubit, k)
    ef_if_k = _ef_if_at_fock(pair, pair_qubit, k)

    chi_hz = float(pair.chi) if (pair is not None and getattr(pair, "chi", None) is not None) else 0.0

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "wait_ns": xr.DataArray(
            wait_times_ns,
            attrs={"long_name": "Ramsey wait time", "units": "ns"},
        ),
        "detuning_signs": xr.DataArray(
            detuning_signs,
            attrs={"long_name": "detuning sign"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        n            = declare(int)
        t            = declare(int)
        phi          = declare(fixed)
        detuning_sign = declare(int)
        n_st         = declare_stream()

        I, I_st, Q, Q_st, _, _ = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_each_(t, wait_times_cc):
                    with for_(*from_array(detuning_sign, detuning_signs)):
                        for i, qubit in multiplexed_qubits.items():
                            # -- Fock state preparation --------------------------
                            _fock_prep_qua(k, pair, qubit, sideband_drive)

                            # -- Prepare qubit in |f⟩ at Fock-k-shifted frequencies -
                            qubit.xy.update_frequency(ge_if_k)
                            qubit.xy.play("x180")
                            qubit.xy.update_frequency(ef_if_k)
                            qubit.xy.play("EF_x180")

                            # -- Set sideband to transition frequency ------------
                            sideband_drive.update_frequency(target_if)
                            align(qubit.xy.name, sideband_drive.name)

                            # -- First π/2 sideband pulse ------------------------
                            reset_frame(sideband_drive.name)
                            with strict_timing_():
                                sideband_drive.play("sideband_ramp_up")
                                sideband_drive.play("sideband_square", duration=pi2_cc)
                                sideband_drive.play("sideband_ramp_down")

                            # -- Wait + signed artificial detuning --------------
                            sideband_drive.wait(t)
                            with if_(detuning_sign == 1):
                                assign(phi, Cast.mul_fixed_by_int(detuning_factor, 4 * t))
                            with else_():
                                assign(phi, Cast.mul_fixed_by_int(detuning_factor_neg, 4 * t))
                            frame_rotation_2pi(phi, sideband_drive.name)

                            # -- Second π/2 sideband pulse -----------------------
                            with strict_timing_():
                                sideband_drive.play("sideband_ramp_up")
                                sideband_drive.play("sideband_square", duration=pi2_cc)
                                sideband_drive.play("sideband_ramp_down")

                            # -- Back-swap: π_ef converts |f⟩ → |e⟩ for readout -
                            align(sideband_drive.name, qubit.xy.name)
                            qubit.xy.update_frequency(ef_if_k)
                            qubit.xy.play("EF_x180")

                            # -- Readout ----------------------------------------
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

                            # -- Reset cavity and qubit -------------------------
                            cav_mode.reset(
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
                            qubit.xy.wait(2 * qubit.thermalization_time // 4)

                        align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(detuning_signs)).buffer(len(wait_times_ns)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(detuning_signs)).buffer(len(wait_times_ns)).average().save(f"Q{i + 1}")
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(detuning_signs)).buffer(len(wait_times_ns)).average().save(f"state{i + 1}")


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
    from qualibration_libs.data import convert_IQ_to_V
    ds = node.results["ds_raw"]

    # Convert raw I/Q if not using state discrimination
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])

    if node.parameters.use_state_discrimination and node.parameters.use_confusion_matrix_correction:
        ds = apply_confusion_matrix_correction(ds, node.namespace["qubits"])
    node.results["ds_raw"] = ds

    ds_fit, fit_results = _fit_ramsey(ds, node)
    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = {q: asdict(v) for q, v in fit_results.items()}

    _log_ramsey_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        q: ("successful" if res["success"] else "failed")
        for q, res in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    k = node.parameters.fock_level
    fig = _plot_ramsey(
        node.results["ds_fit"],
        node.namespace["qubits"],
        node.results["fit_results"],
        k=k,
        mode_name=node.parameters.mode_name,
    )
    plt.show()
    node.results["figures"] = {f"f{k}g{k+1}_ramsey": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    k = node.parameters.fock_level
    tr_key = f"f{k}g{k+1}"
    pair, _, sideband_drive, _ = _get_pair_components(node)

    with node.record_state_updates():
        for q_name, res in node.results["fit_results"].items():
            if not res["success"]:
                continue
            current_rf = _get_transition_rf(pair, sideband_drive, k)
            corrected_rf = current_rf + res["frequency_correction_hz"]
            if tr_key not in pair.transitions:
                pair.transitions[tr_key] = SidebandTransition()
            pair.transitions[tr_key].RF_frequency = corrected_rf
            pair.transitions[tr_key].T2_star_ns = res["T2_ramsey_ns"]
            if k == 0:
                sideband_drive.RF_frequency = corrected_rf
            break


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
