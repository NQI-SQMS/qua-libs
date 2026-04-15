# %% {Imports}
"""
03e_qubit_bo_bootstrap.py — Gaussian-Process Bayesian Optimisation qubit bootstrap node.

Replaces the nested FSM qubit-discovery sequence
(spec_vs_power → qubit_spec → power_rabi with multi-level retry loops) with a
single measurement (time-domain Rabi) and a GP-BO optimizer that searches the
drive frequency and amplitude simultaneously.

Algorithm overview (Wolff et al. APS 2026, Pfafflab / UIUC)
─────────────────────────────────────────────────────────────
1. Build a time-Rabi QUA program template (fixed pulse shape, variable duration).
2. Phase 1a — Broad LHS: evaluate the cost function at n_initial_lhs quasi-random
   Latin Hypercube points spread across ±freq_search_radius_mhz.
3. Phase 1b — Zoom LHS: evaluate n_zoom_lhs points within ±zoom_radius_mhz of
   the best Phase-1a point.
4. Phase 2  — BO acquisition: maximize EI (or UCB) to suggest the next point;
   register the measured cost; stop when the GP-predicted optimum frequency
   moves less than convergence_tolerance_mhz between consecutive iterations.
5. Write the best (ω_q, V_π) to the QUAM state.

What runs on the FPGA vs off the FPGA
───────────────────────────────────────
 On FPGA  : time-Rabi pulse sequence, num_shots averaging, stream_processing
 Off FPGA : frequency/amplitude update between BO iterations, GP fitting,
             EI maximization, cost function evaluation, convergence check

Each BO iteration = one fresh qm.execute() call with a regenerated config
(QUAM Python-side frequency/amplitude set → generate_config() → execute).
Overhead ~200 ms per config regen is acceptable for ~50 total evaluations.

Prerequisites
─────────────
- Mixer / Octave calibration (node 01a)
- Resonator bringup — readout must be working (node 02x)
- The qubit.xy QUAM prior frequency does not need to be accurate —
  the BO searches ±freq_search_radius_mhz (default ±200 MHz) around it.

State updates
─────────────
- qubit.xy.RF_frequency               (qubit drive frequency)
- qubit.xy.operations[operation].amplitude  (OPX IF amplitude for the π-pulse)
- qubit.xy.frequency_converter_up.gain     (Octave gain, if amplitude > max_amplitude_opx)
- Optionally (4D mode): qubit.resonator.RF_frequency, readout amplitude
"""
import logging
import math
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import asdict

from qm.qua import (
    align,
    declare,
    declare_stream,
    for_,
    program,
    save,
    stream_processing,
)
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam

from calibration_utils.qubit_bo_bootstrap import (
    Parameters,
    BOOptimizer,
    fit_rabi_trace,
    compute_cost,
)
from calibration_utils.qubit_bo_bootstrap.plotting import (
    plot_bo_trajectory,
    plot_convergence,
    plot_rabi_with_fit,
    plot_cost_landscape_1d,
)
from calibration_utils.qubit_bo_bootstrap.analysis import _sinusoid as _eval_sinusoid
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

logger = logging.getLogger(__name__)
u = unit(coerce_to_integer=True)

# ── Constant: minimum OPX pulse duration (1 clock cycle = 4 ns) ───────────────
_MIN_DURATION_CC = 4  # clock cycles


# %% {Node initialisation}
description = """
        QUBIT BO BOOTSTRAP (03e)

Discover the qubit drive frequency and π-pulse amplitude using Gaussian-Process
Bayesian Optimisation over time-domain Rabi oscillations (Wolff et al. APS 2026).

Replaces the nested FSM: spec_vs_power → qubit_spec → power_rabi.

Three-phase strategy:
  Phase 1a — Broad LHS: n_initial_lhs points over ±freq_search_radius_mhz
  Phase 1b — Zoom LHS:  n_zoom_lhs points in ±zoom_radius_mhz around Phase-1a best
  Phase 2  — BO (EI):   up to n_bo_iterations guided acquisitions; stops when
              the GP-predicted optimum moves < convergence_tolerance_mhz

Prerequisites:
    - Resonator bringup complete (calibrated readout, depletion time known)
    - Mixer/Octave calibration done (node 01a)

State updates:
    - qubit.xy.RF_frequency                    (best drive frequency)
    - qubit.xy.operations[operation].amplitude (best OPX amplitude)
    - qubit.xy.frequency_converter_up.gain     (Octave gain if needed for wide range)
    - If optimize_readout_jointly=True:
        qubit.resonator.RF_frequency
        qubit.resonator.operations["readout"].amplitude
"""

node = QualibrationNode[Parameters, Quam](
    name="03e_qubit_bo_bootstrap",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """
    Override parameters for local / debug runs.

    This action is skipped when the node is run from a QualibrationGraph
    (external mode). Uncomment lines below to pin specific values.
    """
    # node.parameters.qubits = ["q1"]
    # node.parameters.n_initial_lhs = 6
    # node.parameters.n_bo_iterations = 10
    # node.parameters.freq_search_radius_mhz = 100.0
    # node.parameters.simulate = True
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """
    Build the time-Rabi QUA program template.

    This program is compiled ONCE and reused for every BO iteration.  Between
    iterations, the QUAM Python state (RF_frequency, amplitude) is updated and
    generate_config() is called — the resulting config carries the new waveform
    and LO/IF settings so each qm.execute() sees the updated parameters.

    Program structure
    ─────────────────
    Outer loop: n_avg (num_shots) averages
    Inner loop: duration sweep (durations_cc array, clock cycles = 4 ns each)
      For each duration:
        1. Reset qubit (active reset or thermalization wait)
        2. Play drive pulse at variable duration (current QUAM amplitude from config)
        3. Measure resonator, save I/Q
    Stream processing: buffer(n_pts).average() → shape (n_pts,) per qubit

    Note: the QUA program is built for ONE qubit at a time (sequential per-qubit
    BO loop). A separate program is stored in node.namespace["qua_programs"][q.name].
    """
    node.namespace["qubits"] = qubits = get_qubits(node)
    params = node.parameters
    operation = params.operation

    # Duration array in clock cycles (must be multiples of 4 ns = 1 cc)
    min_cc = max(_MIN_DURATION_CC, params.min_duration_ns // 4)
    max_cc = max(min_cc + 1, params.max_duration_ns // 4)
    step_cc = max(1, params.duration_step_ns // 4)
    durations_cc = np.arange(min_cc, max_cc, step_cc, dtype=int)
    n_pts = len(durations_cc)
    t_ns = durations_cc * 4.0  # in nanoseconds

    node.namespace["durations_cc"] = durations_cc
    node.namespace["t_ns"] = t_ns
    node.namespace["n_pts"] = n_pts

    node.log(
        f"Time-Rabi sweep: {t_ns[0]:.0f}–{t_ns[-1]:.0f} ns "
        f"({n_pts} points, {params.duration_step_ns} ns step)."
    )

    # Build one QUA program per qubit (sequential BO: each qubit gets its own program)
    node.namespace["qua_programs"] = {}
    for qubit in qubits:
        with program() as qua_prog:
            I = declare(float)   # type: ignore[assignment]
            Q = declare(float)   # type: ignore[assignment]
            I_st = declare_stream()
            Q_st = declare_stream()
            n = declare(int)
            t = declare(int)  # duration in clock cycles

            with for_(n, 0, n < params.num_shots, n + 1):
                with for_(*from_array(t, durations_cc)):
                    qubit.reset(params.reset_type, params.simulate)
                    align()

                    # Play the drive pulse at variable duration.
                    # amplitude_scale=1.0: the waveform amplitude comes directly from
                    # the QUAM config (updated Python-side before each config regen).
                    qubit.xy.play(operation, duration=t, amplitude_scale=1.0)
                    align()

                    # Readout
                    qubit.resonator.measure("readout", qua_vars=(I, Q))
                    save(I, I_st)
                    save(Q, Q_st)
                    qubit.resonator.wait(node.machine.depletion_time * u.ns)

            with stream_processing():
                # Average over shots → shape (n_pts,) per qubit
                I_st.buffer(n_pts).average().save("I")
                Q_st.buffer(n_pts).average().save("Q")

        node.namespace["qua_programs"][qubit.name] = qua_prog

    node.log(
        f"Built {len(qubits)} time-Rabi QUA program(s) for: "
        f"{[q.name for q in qubits]}"
    )


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """
    Simulate one time-Rabi evaluation at the QUAM prior (no BO loop).

    This is for debugging the QUA program structure only — it runs the program
    for the first qubit at the current QUAM state (prior frequency/amplitude).
    The full BO loop is NOT run in simulation mode.
    """
    qubits = node.namespace["qubits"]
    if not qubits:
        return
    first_qubit = qubits[0]
    qua_prog = node.namespace["qua_programs"][first_qubit.name]
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, qua_prog, node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}
    node.log("Simulation complete. Inspect the pulse sequence above.")


# %% {Run_BO_loop}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def run_bo_loop(node: QualibrationNode[Parameters, Quam]):
    """
    Execute the three-phase GP-BO qubit discovery loop.

    For each qubit (sequentially):
      Phase 1a — Broad LHS: n_initial_lhs quasi-random points
      Phase 1b — Zoom LHS:  n_zoom_lhs points around Phase-1a best
      Phase 2  — BO EI/UCB: guided acquisition until convergence

    Each BO evaluation:
      1. Apply candidate (ω_d, V_d) to QUAM Python state
      2. Call generate_config() to bake state into OPX config
      3. Execute the pre-compiled QUA program
      4. Fetch averaged I quadrature array (shape: n_pts)
      5. Fit sinusoidal Rabi trace → extract Ω_R, amplitude, SNR
      6. Evaluate Wolff cost function
      7. Register (x, cost) with the GP optimizer
      8. Restore QUAM to prior state (frequency + amplitude)

    At the end, writes best (ω_q, V_π) to node.results for update_state.
    """
    params = node.parameters
    qubits = node.namespace["qubits"]
    durations_cc = node.namespace["durations_cc"]
    t_ns = node.namespace["t_ns"]
    qmm = node.machine.connect()

    history: dict = {}        # per-qubit list of evaluation dicts
    best_results: dict = {}   # per-qubit best parameters found

    for qubit in qubits:
        q_name = qubit.name
        node.log(f"\n{'─'*60}")
        node.log(f"Starting BO bootstrap for qubit {q_name}")
        node.log(f"{'─'*60}")

        q_history = []
        qua_prog = node.namespace["qua_programs"][q_name]

        # ── Prior state (save for restoration between BO evals) ───────────────
        prior_freq_hz = float(qubit.xy.RF_frequency)
        prior_amp_v = float(qubit.xy.operations[params.operation].amplitude)
        prior_octave_gain = float(qubit.xy.frequency_converter_up.gain)

        # Compute prior output power in dBm (used to scale amplitude candidates)
        try:
            prior_power_dbm = float(qubit.xy.get_output_power(params.operation))
        except Exception:
            # Fallback: estimate from OPX amplitude only (no Octave scaling info)
            prior_power_dbm = 20.0 * math.log10(max(prior_amp_v, 1e-9))
            logger.warning(
                "[%s] Could not get_output_power; using OPX-amplitude-only estimate "
                "(%.1f dBm). Octave gain adjustment may be inaccurate.",
                q_name, prior_power_dbm,
            )

        node.log(
            f"[{q_name}] Prior: f_drive={prior_freq_hz/1e6:.3f} MHz  "
            f"V_d={prior_amp_v:.4f} V  P_out={prior_power_dbm:.1f} dBm"
        )

        # ── Build BO search bounds ─────────────────────────────────────────────
        lo_freq_hz = float(qubit.xy.LO_frequency)
        opx_bandwidth_hz = 250e6

        # Drive frequency bounds: ±search_radius around prior, clamped so that
        # the OPX IF (= f_drive − f_LO) stays within ±250 MHz bandwidth
        f_lo_mhz = prior_freq_hz / 1e6
        r = params.freq_search_radius_mhz
        f_min_hz = max(prior_freq_hz - r * 1e6, lo_freq_hz - opx_bandwidth_hz)
        f_max_hz = min(prior_freq_hz + r * 1e6, lo_freq_hz + opx_bandwidth_hz)
        node.log(
            f"[{q_name}] Drive frequency search: "
            f"[{f_min_hz/1e6:.1f}, {f_max_hz/1e6:.1f}] MHz"
        )

        # Drive amplitude bounds: ±dB range relative to prior, clamped by OPX max.
        v_d_min = prior_amp_v * 10 ** (params.amp_search_min_db / 20.0)
        v_d_max = prior_amp_v * 10 ** (params.amp_search_max_db / 20.0)
        v_d_min = max(v_d_min, 1e-4)
        # v_d_max may exceed max_amplitude_opx — that is handled by set_output_power
        # (Octave gain adjustment), so we do NOT clamp v_d_max here.
        node.log(
            f"[{q_name}] Drive amplitude search: "
            f"[{v_d_min:.4f}, {v_d_max:.4f}] V  "
            f"(= [{params.amp_search_min_db:+.0f}, {params.amp_search_max_db:+.0f}] dB "
            f"relative to prior {prior_amp_v:.4f} V)"
        )

        # 2D bounds: (omega_d, v_d)
        bounds_2d = [(f_min_hz, f_max_hz), (v_d_min, v_d_max)]

        # Optional 4D bounds: also (omega_r, v_r)
        if params.optimize_readout_jointly:
            prior_rf_hz = float(qubit.resonator.RF_frequency)
            prior_r_amp = float(qubit.resonator.operations["readout"].amplitude)
            r_r = params.readout_freq_radius_mhz
            r_f_min = prior_rf_hz - r_r * 1e6
            r_f_max = prior_rf_hz + r_r * 1e6
            r_v_min = prior_r_amp * 10 ** (params.readout_amp_search_min_db / 20.0)
            r_v_max = prior_r_amp * 10 ** (params.readout_amp_search_max_db / 20.0)
            bounds = bounds_2d + [(r_f_min, r_f_max), (r_v_min, r_v_max)]
            node.log(f"[{q_name}] 4D joint readout search enabled.")
        else:
            bounds = bounds_2d

        # ── Initialize optimizer ───────────────────────────────────────────────
        rng_seed = params.bo_seed if params.bo_seed is not None else 42
        optimizer = BOOptimizer(bounds=bounds, acq=params.acq, random_state=rng_seed)

        def _evaluate(x: np.ndarray, phase: str) -> dict:
            """
            Run one time-Rabi experiment at parameter vector x and return an evaluation dict.

            Parameters
            ──────────
            x     : parameter vector in physical units.
                    2D: [omega_d_hz, v_d_V]
                    4D: [omega_d_hz, v_d_V, omega_r_hz, v_r_V]
            phase : 'lhs_broad', 'lhs_zoom', or 'bo'

            Returns a dict with cost, fit metrics, and raw trace data for plotting.
            """
            omega_d_hz = float(x[0])
            v_d_V = float(x[1])
            omega_d_mhz = omega_d_hz / 1e6

            # IF frequency check: must be within OPX bandwidth
            if_hz = omega_d_hz - lo_freq_hz
            if abs(if_hz) > opx_bandwidth_hz:
                node.log(
                    f"[{q_name}] SKIP: IF={if_hz/1e6:.1f} MHz exceeds ±250 MHz OPX bandwidth."
                )
                return {
                    "phase": phase,
                    "omega_d_mhz": omega_d_mhz,
                    "v_d": v_d_V,
                    "cost": 1000.0,
                    "success": False,
                }

            try:
                # ── Apply QUAM state ──────────────────────────────────────────
                qubit.xy.RF_frequency = omega_d_hz

                if v_d_V <= params.max_amplitude_opx:
                    # Within OPX range: set OPX amplitude directly.
                    qubit.xy.operations[params.operation].amplitude = v_d_V
                else:
                    # Beyond OPX range: use set_output_power to shift excess to Octave gain.
                    delta_db = 20.0 * math.log10(v_d_V / max(prior_amp_v, 1e-9))
                    new_power_dbm = prior_power_dbm + delta_db
                    qubit.xy.set_output_power(
                        power_in_dbm=new_power_dbm,
                        max_amplitude=params.max_amplitude_opx,
                        operation=params.operation,
                    )

                if params.optimize_readout_jointly and len(x) >= 4:
                    qubit.resonator.RF_frequency = float(x[2])
                    qubit.resonator.operations["readout"].amplitude = float(x[3])

                # ── Regenerate config and execute ─────────────────────────────
                config = node.machine.generate_config()
                with qm_session(qmm, config, timeout=params.timeout) as qm:
                    job = qm.execute(qua_prog)
                    result_handles = job.result_handles
                    result_handles.wait_for_all_values()
                    I_arr = result_handles.get("I").fetch_all()
                    Q_arr = result_handles.get("Q").fetch_all()

                if I_arr is None or len(I_arr) == 0:
                    raise RuntimeError("Empty I array fetched from QM job.")

                # ── Fit and cost ──────────────────────────────────────────────
                fit = fit_rabi_trace(
                    I_arr=I_arr,
                    t_ns=t_ns,
                    target_rabi_freq_mhz=params.target_rabi_freq_mhz,
                )
                cost = compute_cost(
                    fit=fit,
                    target_rabi_freq_mhz=params.target_rabi_freq_mhz,
                    w_rabi=params.w_rabi,
                    w_amp=params.w_amp,
                    w_snr=params.w_snr,
                )

                # Fitted sinusoid for plotting
                if fit.success:
                    I_fit = _eval_sinusoid(
                        t_ns, fit.fit_a, fit.fit_f_mhz, fit.fit_phi, fit.fit_offset
                    )
                else:
                    I_fit = np.zeros_like(I_arr)

                eval_dict = {
                    "phase": phase,
                    "omega_d_mhz": omega_d_mhz,
                    "v_d": v_d_V,
                    "cost": cost,
                    "success": fit.success,
                    "rabi_freq_mhz": fit.rabi_freq_mhz,
                    "amplitude": fit.amplitude,
                    "snr": fit.snr,
                    "t_ns": t_ns.tolist(),
                    "I_arr": I_arr.tolist(),
                    "I_fit": I_fit.tolist(),
                }
                if params.optimize_readout_jointly and len(x) >= 4:
                    eval_dict["omega_r_mhz"] = float(x[2]) / 1e6
                    eval_dict["v_r"] = float(x[3])

                node.log(
                    f"[{q_name}] {phase:10s}  "
                    f"f={omega_d_mhz:.3f} MHz  V={v_d_V:.4f} V  "
                    f"cost={cost:.3f}  "
                    f"Omega_R={fit.rabi_freq_mhz:.2f} MHz  "
                    f"A={fit.amplitude:.3f}  SNR={fit.snr:.1f}"
                )
                return eval_dict

            except Exception as exc:
                logger.exception("[%s] Evaluation failed: %s", q_name, exc)
                return {
                    "phase": phase,
                    "omega_d_mhz": omega_d_mhz,
                    "v_d": v_d_V,
                    "cost": 1000.0,
                    "success": False,
                }
            finally:
                # ── Always restore QUAM prior state ───────────────────────────
                # This ensures that QUAM does not accumulate stale BO candidate
                # values between iterations.  The best result is committed in
                # update_state only after the full BO loop completes.
                qubit.xy.RF_frequency = prior_freq_hz
                qubit.xy.operations[params.operation].amplitude = prior_amp_v
                qubit.xy.frequency_converter_up.gain = prior_octave_gain
                if params.optimize_readout_jointly:
                    qubit.resonator.RF_frequency = prior_rf_hz  # noqa: F821
                    qubit.resonator.operations["readout"].amplitude = prior_r_amp  # noqa: F821

        # ── Phase 1a — Broad LHS ──────────────────────────────────────────────
        node.log(f"\n[{q_name}] Phase 1a — Broad LHS ({params.n_initial_lhs} points)")
        lhs_broad = optimizer.lhs_samples(params.n_initial_lhs)
        for i, x in enumerate(lhs_broad):
            node.log(f"  LHS broad [{i+1}/{params.n_initial_lhs}]")
            ev = _evaluate(x, "lhs_broad")
            q_history.append(ev)
            optimizer.register(x, ev["cost"])

        # ── Phase 1b — Zoom LHS ───────────────────────────────────────────────
        node.log(f"\n[{q_name}] Phase 1b — Zoom LHS ({params.n_zoom_lhs} points)")

        # Build a new BOOptimizer instance for the zoomed search space
        # (reuses all Phase 1a observations via register() calls below)
        best_x_broad = optimizer.best_x
        if best_x_broad is None:
            best_x_broad = lhs_broad[0]  # fallback: use first LHS point

        best_f_hz = float(best_x_broad[0])
        zoom_r = params.zoom_radius_mhz * 1e6
        f_zoom_min = max(best_f_hz - zoom_r, f_min_hz)
        f_zoom_max = min(best_f_hz + zoom_r, f_max_hz)

        # Restrict amplitude to the [v_d_min, v_d_max] range (unchanged for zoom)
        bounds_zoom = [(f_zoom_min, f_zoom_max)] + list(bounds[1:])
        zoom_optimizer = BOOptimizer(
            bounds=bounds_zoom, acq=params.acq, random_state=rng_seed + 1
        )
        # Seed zoom optimizer with Phase 1a observations that fall within zoom window
        for ev in q_history:
            x_ev = np.array([ev["omega_d_mhz"] * 1e6, ev["v_d"]])
            if params.optimize_readout_jointly:
                x_ev = np.append(x_ev, [ev.get("omega_r_mhz", 0) * 1e6, ev.get("v_r", 0)])
            zoom_optimizer.register(x_ev, ev["cost"])

        lhs_zoom = zoom_optimizer.lhs_samples(params.n_zoom_lhs)
        for i, x in enumerate(lhs_zoom):
            node.log(f"  LHS zoom [{i+1}/{params.n_zoom_lhs}]")
            ev = _evaluate(x, "lhs_zoom")
            q_history.append(ev)
            zoom_optimizer.register(x, ev["cost"])
            optimizer.register(x, ev["cost"])  # keep the broad optimizer updated too

        # ── Phase 2 — BO acquisition ──────────────────────────────────────────
        node.log(f"\n[{q_name}] Phase 2 — BO acquisition (max {params.n_bo_iterations} iter)")

        # Use the zoom optimizer for BO Phase 2 (tighter search space)
        x_opt_prev, _ = zoom_optimizer.predict_optimum()
        converged = False

        for bo_iter in range(params.n_bo_iterations):
            x_next = zoom_optimizer.suggest()
            ev = _evaluate(x_next, "bo")
            ev["predicted_freq_mhz"] = float(x_opt_prev[0]) / 1e6  # before update
            q_history.append(ev)
            zoom_optimizer.register(x_next, ev["cost"])
            optimizer.register(x_next, ev["cost"])

            x_opt_new, mu_opt = zoom_optimizer.predict_optimum()
            ev["predicted_freq_mhz"] = float(x_opt_new[0]) / 1e6  # after update

            shift_mhz = abs(x_opt_new[0] - x_opt_prev[0]) / 1e6
            node.log(
                f"  BO [{bo_iter+1}/{params.n_bo_iterations}]  "
                f"next: f={x_next[0]/1e6:.3f} MHz  "
                f"predicted opt: {x_opt_new[0]/1e6:.3f} MHz  "
                f"shift: {shift_mhz:.3f} MHz"
            )

            if shift_mhz < params.convergence_tolerance_mhz:
                node.log(
                    f"[{q_name}] BO converged after {bo_iter+1} iterations "
                    f"(shift {shift_mhz:.3f} < {params.convergence_tolerance_mhz} MHz)."
                )
                converged = True
                x_opt_prev = x_opt_new
                break

            x_opt_prev = x_opt_new

        if not converged:
            node.log(
                f"[{q_name}] BO reached max iterations ({params.n_bo_iterations}) "
                f"without full convergence. Using best observed result."
            )

        # ── Determine best result ─────────────────────────────────────────────
        # Use GP-predicted optimum if the GP converged; otherwise observed best.
        best_x_final, _ = zoom_optimizer.predict_optimum()
        best_observed_x = zoom_optimizer.best_x
        best_observed_cost = zoom_optimizer.best_cost

        # Pick the evaluation with the lowest observed cost for the trace plot
        valid_evals = [e for e in q_history if e.get("success", False)]
        if valid_evals:
            best_eval_idx = min(range(len(q_history)), key=lambda i: q_history[i]["cost"])
        else:
            best_eval_idx = 0

        best_omega_d_hz = float(best_x_final[0])
        best_v_d = float(best_x_final[1])
        if params.optimize_readout_jointly and len(best_x_final) >= 4:
            best_omega_r_hz = float(best_x_final[2])
            best_v_r = float(best_x_final[3])
        else:
            best_omega_r_hz = None
            best_v_r = None

        node.log(
            f"\n[{q_name}] RESULT: "
            f"f_q = {best_omega_d_hz/1e6:.4f} MHz  "
            f"V_d = {best_v_d:.4f} V  "
            f"(observed best cost = {best_observed_cost:.3f})"
        )

        history[q_name] = q_history
        best_results[q_name] = {
            "omega_d_hz": best_omega_d_hz,
            "v_d": best_v_d,
            "omega_r_hz": best_omega_r_hz,
            "v_r": best_v_r,
            "best_eval_idx": best_eval_idx,
            "converged": converged,
            "n_evaluations": len(q_history),
        }

        # Outcome: successful if at least one valid (fit succeeded) evaluation was found
        if valid_evals:
            node.outcomes[q_name] = "successful"
        else:
            node.outcomes[q_name] = "failed"

    node.results["bo_history"] = history
    node.results["best_results"] = best_results
    node.log(
        f"\nBO loop complete. Outcomes: "
        + ", ".join(f"{k}: {v}" for k, v in node.outcomes.items())
    )


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously saved BO run for re-plotting or re-analysis."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """
    Generate BO diagnostic plots.

    Four panels:
    1. Cost trajectory (scatter + running best, coloured by phase)
    2. Convergence curve (GP-predicted optimum frequency vs iteration)
    3. Best Rabi trace + sinusoidal fit
    4. Cost landscape 1D projection (cost vs drive frequency)
    """
    history = node.results.get("bo_history", {})
    best_results = node.results.get("best_results", {})
    qubits = node.namespace.get("qubits", [])
    qubit_names = [q.name for q in qubits]

    if not history:
        node.log("No BO history found — skipping plots.")
        return

    figures = {}

    fig_traj = plot_bo_trajectory(history, qubit_names)
    plt.show()
    figures["bo_trajectory"] = fig_traj

    fig_conv = plot_convergence(
        history, qubit_names,
        convergence_tolerance_mhz=node.parameters.convergence_tolerance_mhz,
    )
    plt.show()
    figures["bo_convergence"] = fig_conv

    best_eval_idx = {
        q: best_results[q]["best_eval_idx"]
        for q in qubit_names
        if q in best_results
    }
    fig_rabi = plot_rabi_with_fit(history, qubit_names, best_eval_idx)
    plt.show()
    figures["best_rabi_trace"] = fig_rabi

    fig_landscape = plot_cost_landscape_1d(history, qubit_names)
    plt.show()
    figures["cost_landscape"] = fig_landscape

    node.results["figures"] = figures


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """
    Write the best BO result to the QUAM state.

    For each qubit:
    - If the BO outcome was 'successful': commit the best (ω_q, V_d) to QUAM.
      - qubit.xy.RF_frequency ← best drive frequency
      - qubit.xy.operations[operation].amplitude ← best OPX IF amplitude
      - qubit.xy.frequency_converter_up.gain  ← Octave gain (if set_output_power used)
      - If 4D mode: also commit readout frequency and amplitude.
    - If the BO outcome was 'failed': log a warning and leave QUAM unchanged.

    Power handling
    ──────────────
    We use set_output_power() with the best candidate power level in dBm and
    max_amplitude_opx as the constraint.  This replicates the logic used in
    04b_power_rabi.py and 04c_time_rabi.py for wide-range amplitude updates.
    """
    qubits = node.namespace["qubits"]
    best_results = node.results.get("best_results", {})
    params = node.parameters

    with node.record_state_updates():
        for qubit in qubits:
            q_name = qubit.name
            outcome = node.outcomes.get(q_name, "failed")

            if outcome != "successful":
                node.log(
                    f"[{q_name}] BO outcome was '{outcome}' — "
                    f"QUAM state NOT updated. Check the cost landscape plot."
                )
                continue

            result = best_results.get(q_name)
            if result is None:
                node.log(f"[{q_name}] No result dict found — skipping state update.")
                continue

            best_omega_d_hz = result["omega_d_hz"]
            best_v_d = result["v_d"]

            # Set drive frequency
            qubit.xy.RF_frequency = best_omega_d_hz
            node.log(
                f"[{q_name}] Set qubit drive frequency: "
                f"{best_omega_d_hz/1e6:.4f} MHz"
            )

            # Set drive amplitude (with Octave gain adjustment if needed)
            if best_v_d <= params.max_amplitude_opx:
                qubit.xy.operations[params.operation].amplitude = best_v_d
                node.log(
                    f"[{q_name}] Set {params.operation} OPX amplitude: "
                    f"{best_v_d:.4f} V (within OPX range)"
                )
            else:
                try:
                    prior_amp = float(qubit.xy.operations[params.operation].amplitude)
                    prior_power = float(qubit.xy.get_output_power(params.operation))
                    delta_db = 20.0 * math.log10(best_v_d / max(prior_amp, 1e-9))
                    new_power_dbm = prior_power + delta_db
                    qubit.xy.set_output_power(
                        power_in_dbm=new_power_dbm,
                        max_amplitude=params.max_amplitude_opx,
                        operation=params.operation,
                    )
                    node.log(
                        f"[{q_name}] Set output power via Octave+OPX: "
                        f"{new_power_dbm:.1f} dBm "
                        f"(amplitude capped at {params.max_amplitude_opx} V)"
                    )
                except Exception as exc:
                    node.log(
                        f"[{q_name}] WARNING: Could not call set_output_power ({exc}). "
                        f"Clamping OPX amplitude to {params.max_amplitude_opx} V."
                    )
                    qubit.xy.operations[params.operation].amplitude = params.max_amplitude_opx

            # 4D: update readout parameters
            if params.optimize_readout_jointly and result.get("omega_r_hz") is not None:
                qubit.resonator.RF_frequency = result["omega_r_hz"]
                node.log(
                    f"[{q_name}] Set resonator frequency: "
                    f"{result['omega_r_hz']/1e6:.4f} MHz"
                )
                if result.get("v_r") is not None:
                    qubit.resonator.operations["readout"].amplitude = result["v_r"]
                    node.log(
                        f"[{q_name}] Set readout amplitude: {result['v_r']:.4f} V"
                    )

            node.log(
                f"[{q_name}] QUAM state update complete. "
                f"(BO ran {result['n_evaluations']} evaluations, "
                f"converged={result['converged']})"
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save all results (history, figures, QUAM state diff) to the qualibrate data store."""
    node.save()
