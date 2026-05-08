# %%
"""
03e_qubit_bo_bootstrap.py
═══════════════════════════════════════════════════════════════════════════════
QUBIT BO BOOTSTRAP — autonomous discovery of ω_q and V_π via GP-BO.

Replaces the nested retry FSM:
    spec_vs_power [loop] → qubit_spec → power_rabi [loop] [outer loop]

with a single elegant optimization loop:
    LHS seeding → GP-guided EI acquisition → convergence check → write QUAM

Prerequisites
─────────────
- Resonator bringup complete: QUAM has valid resonator.RF_frequency and
  resonator readout amplitude (run 02f_resonator_bringup_graph.py first).
- Mixer calibration done (01a_mixer_calibration.py).
- QUAM has a 'square_drive' operation on qubit.xy — see TODO below.

State updates
─────────────
- qubit.xy.RF_frequency              → ω_d (≈ ω_q after convergence)
- qubit.f_01                         → ω_q / (2π), in Hz
- qubit.xy.operations["x180"].amplitude  → V_π (π-pulse amplitude)

TODO before first run
─────────────────────
1. Add a 'square_drive' operation to QUAM Transmon:
       qubit.xy.operations["square_drive"] = SquarePulse(amplitude=1.0, length=200)
   The length doesn't matter — it's overridden by duration= in play().
   amplitude=1.0 so that amplitude_scale=v_d gives the correct physical amplitude.

2. Verify IF constraint: omega_d_hz − qubit.xy.LO_frequency must be within
   ±250 MHz (OPX bandwidth). The node clips search bounds but check LO placement.

3. If using Octave: ensure the Octave RF upconversion is enabled and gain is set
   before running. The node does not modify Octave gain.
"""

import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from typing import Dict, Optional

from qm.qua import *
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from qualibration_libs.parameters import get_qubits

from calibration_utils.time_rabi_bo.parameters import TimeRabiBoParameters
from calibration_utils.time_rabi_bo.analysis import (
    fit_time_rabi,
    compute_cost,
    fit_ramsey_t2star,
    rabi_fit_curve,
    TimeRabiFitResult,
)
from calibration_utils.time_rabi_bo.bo_optimizer import BOOptimizer
from calibration_utils.time_rabi_bo.plotting import plot_bo_results

logger = logging.getLogger(__name__)

# ── Node declaration ───────────────────────────────────────────────────────────

description = """
QUBIT BO BOOTSTRAP

Autonomously discovers the qubit drive frequency (ω_q) and π-pulse amplitude
(V_π) by running a Gaussian-process Bayesian optimisation loop over the
Wolff et al. (APS 2026) time-Rabi cost function:

    C = 10 × |Ω_R − Ω_T| / Ω_T  −  3·log(A)  −  log(SNR)

Phase 1 (LHS): quasi-random samples across the search volume seed the GP.
Phase 2 (BO):  Expected Improvement acquisition guides sampling toward ω_q, V_π.

Unlike gradient descent (Wolff), EI is non-local: it explores uncertain regions
even after finding a local minimum, preventing spurious-mode convergence.

Optional post-convergence Ramsey sanity check flags spurious modes (short T2*).
"""

node = QualibrationNode[TimeRabiBoParameters, Quam](
    name="03e_qubit_bo_bootstrap",
    description=description,
    parameters=TimeRabiBoParameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[TimeRabiBoParameters, Quam]):
    """Override parameters for local debugging."""
    # node.parameters.qubits = ["q0"]
    # node.parameters.target_rabi_freq_mhz = 20.0
    # node.parameters.freq_search_radius_mhz = 200.0
    # node.parameters.n_initial_lhs = 4       # reduce for quick debug
    # node.parameters.n_bo_iterations = 10    # reduce for quick debug
    pass


node.machine = Quam.load()


# ── QUA program builder ────────────────────────────────────────────────────────

def _build_time_rabi_program(qubit, t_arr_clk_cycles: np.ndarray, num_shots: int):
    """
    Build a QUA program that sweeps the drive pulse duration across t_arr_clk_cycles.

    The qubit's xy RF_frequency and x180 amplitude are baked into the config
    at program-generation time (via QUAM → generate_config()). Both are updated
    before each call to this function.

    Parameters
    ──────────
    qubit           : QUAM Transmon object
    t_arr_clk_cycles: pulse durations in OPX clock cycles (4 ns each)
                      shape (n_pts,), dtype int, all values ≥ 4 (= 16 ns minimum)
    num_shots       : number of averages

    Returns
    ───────
    qua_program : compiled QUA program
    """
    with program() as prog:
        n    = declare(int)
        t    = declare(int)
        I    = declare(fixed)
        Q    = declare(fixed)
        I_st = declare_stream()
        Q_st = declare_stream()
        n_st = declare_stream()

        with for_(n, 0, n < num_shots, n + 1):
            save(n, n_st)

            with for_each_(t, t_arr_clk_cycles.tolist()):
                # --- Reset ---
                qubit.reset("thermal")
                align()

                # --- Qubit drive ---
                qubit.xy.play(
                    node.parameters.pulse_operation,
                    duration=t,   # QUA int variable, OPX clock cycles (4 ns each)
                )
                align()

                # --- Readout ---
                qubit.resonator.measure("readout", qua_vars=(I, Q))
                save(I, I_st)
                save(Q, Q_st)

                wait(qubit.resonator.depletion_time // 4 + 4)
                align()

        with stream_processing():
            n_st.save("n")
            I_st.buffer(len(t_arr_clk_cycles)).average().save("I")
            Q_st.buffer(len(t_arr_clk_cycles)).average().save("Q")

    return prog


def _time_array(params: TimeRabiBoParameters) -> np.ndarray:
    """
    Build the time axis for the time-Rabi sweep.

    Returns t_clk_cycles (int array) and t_ns (float array).
    All durations are rounded to 4 ns (1 OPX clock cycle) and ≥ 16 ns.
    """
    omega_T_hz  = params.target_rabi_freq_mhz * 1e6
    t_max_ns    = params.num_periods / omega_T_hz * 1e9
    n_pts       = int(params.pts_per_period * params.num_periods)
    t_ns_raw    = np.linspace(16.0, t_max_ns, n_pts)

    # Round to 4 ns granularity, clip to minimum 16 ns (4 clock cycles)
    t_clk = np.maximum(4, np.round(t_ns_raw / 4).astype(int))
    t_ns  = t_clk * 4.0   # actual times after rounding

    return t_clk, t_ns


# ── Single time-Rabi evaluation ────────────────────────────────────────────────

def _evaluate_time_rabi(
    node: QualibrationNode,
    qubit,
    omega_d_hz: float,
    power_dbm: float,
    params: TimeRabiBoParameters,
) -> TimeRabiFitResult:
    """
    Update QUAM with (ω_d, power_dbm), regenerate config, run one time-Rabi, fit, return result.

    This is the inner loop of the BO: one hardware evaluation = one call here.
    """
    # ── 1. Validate and set drive frequency ───────────────────────────────────
    # Check IF stays within OPX bandwidth (±250 MHz from LO)
    lo_freq = getattr(qubit.xy, "LO_frequency", None) or getattr(qubit.xy, "lo_frequency", None)
    if lo_freq is not None:
        if_freq = omega_d_hz - lo_freq
        if abs(if_freq) > 400e6:
            logger.warning(
                f"Requested IF {if_freq/1e6:.1f} MHz exceeds ±400 MHz OPX limit. "
                f"Clamping omega_d."
            )
            omega_d_hz = float(np.clip(omega_d_hz, lo_freq - 400e6, lo_freq + 400e6))

    # ── 2. Update QUAM state ──────────────────────────────────────────────────
    orig_rf_freq = float(qubit.xy.RF_frequency)
    orig_amp     = float(qubit.xy.operations[params.pulse_operation].amplitude)

    qubit.xy.RF_frequency = omega_d_hz
    qubit.xy.set_output_power(
        power_in_dbm=power_dbm,
        max_amplitude=params.max_amplitude_opx,
        operation=params.pulse_operation,
    )

    # ── 3. Build time array and QUA program ───────────────────────────────────
    t_clk, t_ns = _time_array(params)
    qua_prog    = _build_time_rabi_program(qubit, t_clk, params.num_shots)

    # ── 4. Execute ────────────────────────────────────────────────────────────
    import traceback as _tb
    qmm    = node.machine.connect()
    config = node.machine.generate_config()

    qm = None
    try:
        qm = qmm.open_qm(config, close_other_machines=True)
        job = qm.execute(qua_prog)
        job.result_handles.wait_for_all_values()
        I_avg = job.result_handles.get("I").fetch_all()  # shape (n_pts,)
        Q_avg = job.result_handles.get("Q").fetch_all()
    except Exception as exc:
        logger.warning(f"QUA execution failed:\n{_tb.format_exc()}")
        # Restore QUAM frequency; amplitude will be corrected on the next set_output_power call
        qubit.xy.RF_frequency = orig_rf_freq
        qubit.xy.operations[params.pulse_operation].amplitude = orig_amp
        result = TimeRabiFitResult(
            rabi_freq_mhz=params.target_rabi_freq_mhz * 20,
            amplitude=1e-6, snr=1e-6, cost=1e6,
            fit_success=False, raw_trace=np.zeros(len(t_clk)),
        )
        result.omega_d_hz = omega_d_hz
        result.power_dbm = power_dbm
        return result
    finally:
        if qm is not None:
            try:
                qm.close()
            except Exception:
                pass

    # ── 5. Fit ────────────────────────────────────────────────────────────────
    # Use whichever quadrature has more oscillation — readout phase is not
    # guaranteed to align with I. Peak-to-peak selects the better axis.
    i_range = float(I_avg.max() - I_avg.min())
    q_range = float(Q_avg.max() - Q_avg.min())
    use_i   = i_range >= q_range
    signal  = I_avg if use_i else Q_avg

    print(
        f"  ω_d={omega_d_hz/1e9:.4f} GHz  P={power_dbm:.1f} dBm  "
        f"I_pp={i_range:.4f}  Q_pp={q_range:.4f}  using={'I' if use_i else 'Q'}"
    )

    result = fit_time_rabi(
        t_ns, signal, params.target_rabi_freq_mhz,
        chi2_threshold=2.0,
        min_snr=params.min_snr,
    )
    result.cost = compute_cost(
        result, params.target_rabi_freq_mhz,
        w_rabi=params.rabi_freq_weight,
        w_amp =params.log_amp_weight,
        w_snr =params.log_snr_weight,
        w_chi2=params.log_chi2_weight,
    )
    result.omega_d_hz = omega_d_hz
    result.power_dbm  = power_dbm
    result.raw_I      = I_avg.copy()
    result.raw_Q      = Q_avg.copy()

    print(
        f"    → Ω_R={result.rabi_freq_mhz:.2f} MHz  A={result.amplitude:.4f}  "
        f"SNR={result.snr:.2f}  χ²={result.chi2:.2f}  C={result.cost:.3f}  ok={result.fit_success}"
    )
    return result


# ── Post-convergence Ramsey sanity check ───────────────────────────────────────

def _ramsey_sanity_check(
    node: QualibrationNode,
    qubit,
    params: TimeRabiBoParameters,
) -> Optional[float]:
    """
    Run a quick Ramsey at the current QUAM ω_q to verify T2* is physical.

    Returns T2star_ns if fit succeeded, None otherwise.
    A very short T2* (< min_t2star_sanity_ns) indicates a spurious mode.
    """
    from qm.qua import program, declare, declare_stream, for_, for_each_, fixed, save, stream_processing, wait

    t_arr_ns  = np.linspace(16, params.ramsey_sanity_max_wait_ns, params.ramsey_sanity_num_pts)
    t_arr_clk = np.maximum(4, np.round(t_arr_ns / 4).astype(int))
    detuning_hz = params.ramsey_sanity_detuning_mhz * 1e6

    # Temporarily shift drive frequency by detuning for Ramsey
    original_rf = float(qubit.xy.RF_frequency)
    qubit.xy.RF_frequency = original_rf + detuning_hz

    with program() as ramsey_prog:
        n   = declare(int)
        t   = declare(int)
        I   = declare(fixed)
        Q   = declare(fixed)
        I_st = declare_stream()
        n_st = declare_stream()

        with for_(n, 0, n < params.ramsey_sanity_num_shots, n + 1):
            save(n, n_st)
            with for_each_(t, t_arr_clk.tolist()):
                # --- Reset ---
                qubit.reset("thermal")
                align()

                # --- Qubit drive ---
                qubit.xy.play("x90")       # π/2 pulse
                wait(t)
                qubit.xy.play("x90")       # π/2 pulse
                align()

                # --- Readout ---
                qubit.resonator.measure("readout", qua_vars=(I, Q))
                save(I, I_st)
                wait(qubit.resonator.depletion_time // 4 + 4)
                align()

        with stream_processing():
            n_st.save("n")
            I_st.buffer(len(t_arr_clk)).average().save("I")

    qubit.xy.RF_frequency = original_rf  # restore before executing

    qmm    = node.machine.connect()
    config = node.machine.generate_config()

    qm = None
    try:
        qm = qmm.open_qm(config, close_other_machines=True)
        job = qm.execute(ramsey_prog)
        job.result_handles.wait_for_all_values()
        I_avg = job.result_handles.get("I").fetch_all()
    except Exception as exc:
        import traceback as _tb
        logger.warning(f"Ramsey sanity check execution failed:\n{_tb.format_exc()}")
        return None
    finally:
        if qm is not None:
            try:
                qm.close()
            except Exception:
                pass

    t2star_ns, success = fit_ramsey_t2star(t_arr_clk * 4.0, I_avg, params.ramsey_sanity_detuning_mhz)
    if success:
        logger.info(f"Ramsey sanity check: T2* = {t2star_ns:.0f} ns")
    else:
        logger.warning("Ramsey sanity check: fit failed.")
    return t2star_ns if success else None


# ── Main BO loop run_action ────────────────────────────────────────────────────

@node.run_action(skip_if=node.parameters.load_data_id is not None)
def run_bo_bootstrap(node: QualibrationNode[TimeRabiBoParameters, Quam]):
    """
    Full BO pipeline for each qubit:
        Phase 1 (LHS) → Phase 2 (BO) → convergence → [Ramsey check] → QUAM write.
    """
    params = node.parameters
    qubits = get_qubits(node)

    node.results["bo_history"]   = {}
    node.results["fit_results"]  = {}
    node.results["ds_raw"]       = {}
    node.results["ds_best_trace"] = {}

    for qubit in qubits:
        qubit_name = qubit.name
        logger.info(f"\n{'='*64}\nBO Bootstrap: qubit {qubit_name}\n{'='*64}")

        # ── Build search bounds ────────────────────────────────────────────────
        f_center = float(qubit.xy.RF_frequency)
        f_lo     = f_center - params.freq_search_radius_mhz * 1e6
        f_hi     = f_center + params.freq_search_radius_mhz * 1e6

        # Clip to OPX IF bandwidth (±400 MHz from LO) so the optimizer never
        # proposes frequencies the hardware cannot reach. Without this, out-of-
        # range proposals get clamped to the same boundary frequency, which feeds
        # the GP contradictory data and causes the length-scale to diverge.
        lo_freq = getattr(qubit.xy, "LO_frequency", None) or getattr(qubit.xy, "lo_frequency", None)
        if lo_freq is not None:
            lo_freq = float(lo_freq)
            hw_lo = lo_freq - 400e6
            hw_hi = lo_freq + 400e6
            f_lo_clipped = max(f_lo, hw_lo)
            f_hi_clipped = min(f_hi, hw_hi)
            if f_lo_clipped >= f_hi_clipped:
                logger.error(
                    f"{qubit_name}: search range [{f_lo/1e9:.3f}, {f_hi/1e9:.3f}] GHz is "
                    f"entirely outside OPX bandwidth [{hw_lo/1e9:.3f}, {hw_hi/1e9:.3f}] GHz. "
                    f"Move the LO or adjust QUAM RF_frequency prior."
                )
                node.outcomes[qubit_name] = "failed"
                continue
            if f_lo_clipped > f_lo or f_hi_clipped < f_hi:
                logger.warning(
                    f"{qubit_name}: search range clipped from "
                    f"[{f_lo/1e9:.3f}, {f_hi/1e9:.3f}] GHz to "
                    f"[{f_lo_clipped/1e9:.3f}, {f_hi_clipped/1e9:.3f}] GHz "
                    f"(OPX IF bandwidth ±400 MHz from LO={lo_freq/1e9:.3f} GHz)."
                )
            f_lo, f_hi = f_lo_clipped, f_hi_clipped

        if params.optimize_readout_jointly:
            r_f_center = float(qubit.resonator.RF_frequency)
            r_amp_center = float(qubit.resonator.operations["readout"].amplitude)
            bounds = [
                (f_lo, f_hi),
                (params.min_power_dbm, params.max_power_dbm),
                (r_f_center - params.readout_freq_radius_mhz * 1e6,
                 r_f_center + params.readout_freq_radius_mhz * 1e6),
                (max(r_amp_center * 10 ** (params.readout_amp_search_min_db / 20), 1e-4),
                 min(r_amp_center * 10 ** (params.readout_amp_search_max_db / 20), 0.49)),
            ]
        else:
            bounds = [(f_lo, f_hi), (params.min_power_dbm, params.max_power_dbm)]

        logger.info(
            f"Search bounds:\n"
            f"  ω_d   : [{f_lo/1e9:.4f}, {f_hi/1e9:.4f}] GHz\n"
            f"  Power : [{params.min_power_dbm:.1f}, {params.max_power_dbm:.1f}] dBm"
        )

        optimizer = BOOptimizer(
            bounds=bounds,
            acq=params.acq_function,
            kappa=params.ucb_kappa,
            noise_level=params.gp_noise_level,
        )

        history = []
        _best_cost_seen = np.inf
        _best_result: Optional[TimeRabiFitResult] = None  # result with lowest cost so far

        def _record(phase: str, global_idx: int, x: np.ndarray, result: TimeRabiFitResult):
            # Use result.omega_d_hz (post-clamp) so the GP trains on actual hardware
            # frequencies, not the optimizer's (possibly out-of-range) proposal.
            history.append({
                "phase":          phase,
                "global_idx":     global_idx,
                "omega_d_hz":     result.omega_d_hz,
                "power_dbm":      result.power_dbm,
                "rabi_freq_mhz":  result.rabi_freq_mhz,
                "amplitude":      result.amplitude,
                "snr":            result.snr,
                "chi2":           result.chi2,
                "cost":           result.cost,
                "fit_success":    result.fit_success,
            })

        global_idx = 0

        # ── Phase 1: Latin Hypercube Seeding ───────────────────────────────────
        logger.info(f"Phase 1: {params.n_initial_lhs} LHS samples")
        lhs_pts = optimizer.lhs_samples(params.n_initial_lhs)

        for i, x in enumerate(lhs_pts):
            omega_d    = x[0]
            power_dbm  = x[1]
            if params.optimize_readout_jointly:
                qubit.resonator.RF_frequency = x[2]
                qubit.resonator.operations["readout"].amplitude = x[3]

            result = _evaluate_time_rabi(node, qubit, omega_d, power_dbm, params)
            optimizer.register(x, result.cost)
            _record("lhs", global_idx, x, result)
            global_idx += 1
            if result.cost < _best_cost_seen:
                _best_cost_seen = result.cost
                _best_result = result

        # ── Phase 2: BO-guided acquisition ─────────────────────────────────────
        logger.info(f"Phase 2: up to {params.n_bo_iterations} BO iterations")
        prev_x_opt = None

        for i in range(params.n_bo_iterations):
            x_next     = optimizer.suggest()
            omega_d    = x_next[0]
            power_dbm  = x_next[1]
            if params.optimize_readout_jointly:
                qubit.resonator.RF_frequency = x_next[2]
                qubit.resonator.operations["readout"].amplitude = x_next[3]

            result = _evaluate_time_rabi(node, qubit, omega_d, power_dbm, params)
            optimizer.register(x_next, result.cost)
            _record("bo", global_idx, x_next, result)
            global_idx += 1
            if result.cost < _best_cost_seen:
                _best_cost_seen = result.cost
                _best_result = result

            # Convergence check on GP-predicted optimum
            if params.convergence_tolerance_mhz > 0:
                x_opt, mu_opt = optimizer.predict_optimum()
                if prev_x_opt is not None:
                    delta_mhz = abs(x_opt[0] - prev_x_opt[0]) / 1e6
                    logger.info(
                        f"  BO iter {i+1}: predicted optimum Δ = {delta_mhz:.3f} MHz "
                        f"(threshold = {params.convergence_tolerance_mhz} MHz)"
                    )
                    if delta_mhz < params.convergence_tolerance_mhz:
                        logger.info(f"Converged at BO iteration {i+1}.")
                        break
                prev_x_opt = x_opt.copy()

        # ── Extract best result ────────────────────────────────────────────────
        if optimizer.best_x is None:
            logger.error(f"{qubit_name}: BO found no valid point — outcome=failed.")
            node.outcomes[qubit_name] = "failed"
            continue

        best_omega_d   = float(optimizer.best_x[0])
        best_power_dbm = float(optimizer.best_x[1])

        logger.info(
            f"\n[{qubit_name}] BO converged:\n"
            f"  ω_q ≈ {best_omega_d/1e9:.6f} GHz\n"
            f"  Power ≈ {best_power_dbm:.1f} dBm\n"
            f"  Best C = {optimizer.best_cost:.3f}\n"
            f"  Total evaluations = {optimizer.n_observations}"
        )

        # ── Build xarray datasets ──────────────────────────────────────────────
        n_total = len(history)
        node.results["ds_raw"][qubit_name] = xr.Dataset(
            {
                "omega_d_ghz":   (["iteration"], [h["omega_d_hz"] / 1e9 for h in history]),
                "power_dbm":     (["iteration"], [h["power_dbm"]        for h in history]),
                "rabi_freq_mhz": (["iteration"], [h["rabi_freq_mhz"]    for h in history]),
                "amplitude":     (["iteration"], [h["amplitude"]         for h in history]),
                "snr":           (["iteration"], [h["snr"]               for h in history]),
                "chi2":          (["iteration"], [h["chi2"]              for h in history]),
                "cost":          (["iteration"], [h["cost"]              for h in history]),
                "fit_success":   (["iteration"], [h["fit_success"]       for h in history]),
                "phase":         (["iteration"], [h["phase"]             for h in history]),
            },
            coords={"iteration": np.arange(n_total)},
            attrs={
                "qubit":              qubit_name,
                "target_rabi_mhz":   params.target_rabi_freq_mhz,
                "freq_center_ghz":   float(qubit.xy.RF_frequency) / 1e9,
                "n_lhs":             params.n_initial_lhs,
                "n_bo_iterations":   params.n_bo_iterations,
            },
        )

        t_clk, t_ns = _time_array(params)
        if _best_result is not None and len(_best_result.raw_trace) > 0:
            fit_curve = rabi_fit_curve(t_ns, _best_result)
            node.results["ds_best_trace"][qubit_name] = xr.Dataset(
                {
                    "signal": (["time_ns"], _best_result.raw_trace),
                    "I":      (["time_ns"], _best_result.raw_I if len(_best_result.raw_I) == len(t_ns) else _best_result.raw_trace),
                    "Q":      (["time_ns"], _best_result.raw_Q if len(_best_result.raw_Q) == len(t_ns) else np.zeros(len(t_ns))),
                    "fit":    (["time_ns"], fit_curve if fit_curve is not None else np.full(len(t_ns), np.nan)),
                },
                coords={"time_ns": t_ns},
                attrs={
                    "qubit":       qubit_name,
                    "omega_d_ghz": _best_result.omega_d_hz / 1e9,
                    "power_dbm":   _best_result.power_dbm,
                    "cost":        _best_result.cost,
                },
            )

        node.results["bo_history"][qubit_name] = history

        # ── Phase 3: Ramsey sanity check (optional) ────────────────────────────
        if params.run_ramsey_sanity_check:
            # Set QUAM to the BO result so Ramsey runs at the right frequency and power
            qubit.xy.RF_frequency = best_omega_d
            qubit.xy.set_output_power(
                power_in_dbm=best_power_dbm,
                max_amplitude=params.max_amplitude_opx,
                operation=params.pulse_operation,
            )
            qubit.xy.set_output_power(
                power_in_dbm=best_power_dbm,
                max_amplitude=params.max_amplitude_opx,
                operation="x180",
            )

            t2star_ns = _ramsey_sanity_check(node, qubit, params)

            if t2star_ns is not None and t2star_ns < params.min_t2star_sanity_ns:
                logger.warning(
                    f"{qubit_name}: T2* = {t2star_ns:.0f} ns < "
                    f"{params.min_t2star_sanity_ns:.0f} ns threshold. "
                    f"Possible spurious mode at {best_omega_d/1e9:.6f} GHz. "
                    f"Setting outcome=failed."
                )
                # TODO: blacklist best_omega_d in QUAM temp_calibration here,
                # then the graph can re-run BO with updated bounds excluding this region.
                node.outcomes[qubit_name] = "failed"
                node.results["fit_results"][qubit_name] = {
                    "outcome":    "spurious_mode",
                    "omega_q_hz": best_omega_d,
                    "power_dbm":  best_power_dbm,
                    "best_cost":  optimizer.best_cost,
                    "t2star_ns":  t2star_ns,
                    "n_evaluations": optimizer.n_observations,
                }
                continue

            node.results["fit_results"][qubit_name] = {
                **node.results.get("fit_results", {}).get(qubit_name, {}),
                "t2star_ns": t2star_ns,
            }

        # ── Write to QUAM ──────────────────────────────────────────────────────
        qubit.xy.RF_frequency = best_omega_d
        qubit.f_01            = best_omega_d   # approximate; Ramsey refines in x180_fine_cal
        qubit.xy.set_output_power(
            power_in_dbm=best_power_dbm,
            max_amplitude=params.max_amplitude_opx,
            operation=params.pulse_operation,
        )
        qubit.xy.set_output_power(
            power_in_dbm=best_power_dbm,
            max_amplitude=params.max_amplitude_opx,
            operation="x180",
        )

        if params.optimize_readout_jointly and optimizer.best_x is not None and len(optimizer.best_x) == 4:
            qubit.resonator.RF_frequency = float(optimizer.best_x[2])
            qubit.resonator.operations["readout"].amplitude = float(optimizer.best_x[3])

        node.results["bo_history"][qubit_name]  = history
        node.results["fit_results"][qubit_name] = {
            "omega_q_hz":    best_omega_d,
            "power_dbm":     best_power_dbm,
            "best_cost":     optimizer.best_cost,
            "n_evaluations": optimizer.n_observations,
            "outcome":       "successful",
        }
        node.outcomes[qubit_name] = "successful"


# ── Plot results ──────────────────────────────────────────────────────────────

@node.run_action(skip_if=node.parameters.load_data_id is not None)
def plot_data(node: QualibrationNode[TimeRabiBoParameters, Quam]):
    """
    Three-panel figure per qubit:
      1. Convergence — cost vs. iteration (LHS grey, BO blue, best starred)
      2. Landscape   — (ω_d, power) scatter coloured by log10(cost)
      3. Best trace  — time-Rabi signal at converged point with fit overlay
    """
    ds_raw        = node.results.get("ds_raw", {})
    ds_best_trace = node.results.get("ds_best_trace", {})
    fit_results   = node.results.get("fit_results", {})

    if not ds_raw:
        logger.info("plot_data: no BO data to plot (node may have loaded from id or no qubits ran).")
        return

    fig = plot_bo_results(ds_raw, ds_best_trace, fit_results)
    plt.show()
    node.results["figures"] = {"bo_bootstrap": fig}


# ── State persistence ──────────────────────────────────────────────────────────

@node.run_action
def update_state(node: QualibrationNode[TimeRabiBoParameters, Quam]):
    """
    Save calibrated QUAM state to disk and log summary.

    QUAM writes happen inside run_bo_bootstrap; this action persists them.
    """
    succeeded = [q for q, outcome in node.outcomes.items() if outcome == "successful"]
    failed    = [q for q, outcome in node.outcomes.items() if outcome != "successful"]

    if succeeded:
        node.machine.save()
        for q in succeeded:
            r = node.results["fit_results"].get(q, {})
            logger.info(
                f"[{q}] SAVED → f_01={r.get('omega_q_hz',0)/1e9:.6f} GHz  "
                f"V_pi={r.get('v_pi',0)*1e3:.2f} mV"
            )

    if failed:
        logger.warning(f"Qubits with failed BO: {failed}")


# ── Save results ──────────────────────────────────────────────────────────────

@node.run_action()
def save_results(node: QualibrationNode[TimeRabiBoParameters, Quam]):
    """Persist raw data, fit results, and figures to the QUAlibrate storage."""
    node.save()


if __name__ == "__main__":
    node.run()
