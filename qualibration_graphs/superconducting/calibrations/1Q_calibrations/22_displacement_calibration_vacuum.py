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
from quam_config import Quam
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from calibration_utils.displacement_calibration_vacuum import (
    Parameters,
    FitParameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_vacuum_calibration,
)

logger = logging.getLogger(__name__)


# %% {Node initialisation}
description = """
        DISPLACEMENT VACUUM-POPULATION CALIBRATION (35) — dual-sequence with baseline

Calibrates the unit displacement amplitude by sweeping the cavity displacement
amplitude and measuring the vacuum-state population with a selective qubit π-pulse.

Because of cross-Kerr coupling between the cavity mode and the readout resonator, the
resonator IQ response shifts as a function of displacement amplitude even when the qubit
is untouched.  To remove this spurious baseline, each averaging iteration now runs TWO
sub-sequences for every amplitude point:

  PART 1 — Baseline (no qubit π-pulse):
    1a. Reset cavity + qubit.
    1b. Displace cavity to |α = a · A_unit⟩.
    1c. Measure readout IQ  →  I_base, Q_base  (cross-Kerr offset only).
    1d. Apply D(-a) if active_reset = True.

  PART 2 — Signal (full protocol):
    2a. Reset cavity + qubit (independent reset, same conditions as Part 1).
    2b. Displace cavity identically.
    2c. Apply selective_x180 (or x180) on qubit — flips qubit only when cavity is in |0⟩.
    2d. Measure readout IQ  →  I, Q  (cross-Kerr offset + vacuum-population signal).
    2e. Apply D(-a) if active_reset = True.

Post-processing (in analysis.py, before state discrimination):
    I_corr(a) = I(a) - I_base(a)    ≈  P_vacuum(a) · (I_e - I_g)
    Q_corr(a) = Q(a) - Q_base(a)

This subtraction is performed in the IQ (Volt) domain, BEFORE any threshold or
state-discrimination logic is applied, so that the cross-Kerr offset does not bias
the vacuum-population estimate.

The measured signal after subtraction:
    I_corr(a) = amplitude · exp(-(a / A_1ph)²) + offset

where A_1ph = sigma is the amplitude_scale that produces exactly 1 photon on average.

Parameters:
  - mode_name:       Cavity mode to calibrate ('alice' or 'bob').
  - qubit_pulse:     'selective_x180' (spectrally selective, recommended) or 'x180'.
  - amp_min/max:     Amplitude sweep range (amp_min=0 for half-Gaussian).
  - active_reset:    Play D(-a) after each sub-sequence to speed up thermalization.

State updates:
  - cavity_mode.cavity_mode_drive.operations["displacement"].amplitude
    (set to base_amp × sigma, so amplitude_scale=1 → 1 photon)
"""

node = QualibrationNode[Parameters, Quam](
    name="22_displacement_calibration_vacuum",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name = "alice"
    # node.parameters.qubit_pulse = "selective_x180"  # or "x180"
    # node.parameters.amp_min = 0.0
    # node.parameters.amp_max = 2.0
    # node.parameters.amp_points = 51
    # node.parameters.active_reset = True
    # node.parameters.num_shots = 1000
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
    """Build the QUA program for the vacuum-population displacement calibration."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    n_avg = node.parameters.num_shots

    cavity_mode = _get_cavity_mode(node)
    node.namespace["cavity_mode"] = cavity_mode

    # Resolve sideband_drive for active cavity cooling (used if cavity_reset_type='active_sideband')
    mode_name = node.parameters.mode_name
    sideband_drive = None
    pairs = getattr(node.machine, "cavity_transmon_pairs", {})
    for pair_key, pair in pairs.items():
        if pair_key.endswith(f"_{mode_name}") and getattr(pair, "sideband_drive", None) is not None:
            sideband_drive = pair.sideband_drive
            break
    node.namespace["sideband_drive"] = sideband_drive

    # amp_min/amp_max are in photon units (alpha). Convert to amplitude_scale using the
    # current alpha_max so the sweep always covers the same physical range on re-runs.
    # On first run alpha_max defaults to 1.0 (amplitude_scale == photon number).
    current_alpha_max = 1.0
    for _pk, _pv in pairs.items():
        if _pk.endswith(f"_{mode_name}") and getattr(_pv, "displacement_alpha_max", None) is not None:
            current_alpha_max = float(_pv.displacement_alpha_max)
            break
    node.namespace["current_alpha_max"] = current_alpha_max

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

    # Whether to run the dual-sequence baseline protocol is decided here at
    # program-build time (Python if, not a QUA conditional) so that the compiled
    # QUA program contains only the instructions that will actually be executed.
    subtract_baseline = node.parameters.subtract_baseline

    with program() as node.namespace["qua_program"]:
        # --- QUA variable declarations ---
        # I/Q: signal sequence (with qubit π-pulse) — always present.
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()

        # I_base/Q_base: baseline sequence (no π-pulse).
        # Declared only when subtract_baseline=True; the compiled program will not
        # contain any reference to these variables when the flag is False.
        if subtract_baseline:
            I_base = [declare(fixed) for _ in range(num_qubits)]   # baseline in-phase
            Q_base = [declare(fixed) for _ in range(num_qubits)]   # baseline quadrature
            I_base_st = [declare_stream() for _ in range(num_qubits)]
            Q_base_st = [declare_stream() for _ in range(num_qubits)]
            # When subtracting the baseline, state discrimination cannot be computed
            # per-shot inside QUA: the threshold would have to be applied to
            # (I_signal - I_baseline), but those come from separate shots and are
            # only both available after averaging.  State discrimination is therefore
            # deferred to Python (process_raw_dataset).

        if not subtract_baseline and node.parameters.use_state_discrimination:
            # Original single-sequence mode: compute the binary state per-shot inside
            # QUA so that averaging gives a proper probability (not a thresholded mean).
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        a = declare(fixed)      # current displacement amplitude scale
        a_neg = declare(fixed)  # negated amplitude for the D(-a) active-reset pulse

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(a, amp_array)):
                    assign(a_neg, -a)   # pre-compute negated amplitude once per point

                    # ============================================================
                    # PART 1 — BASELINE measurement (only when subtract_baseline=True)
                    # ============================================================
                    # Purpose: capture the bare readout-resonator IQ response at each
                    # displacement amplitude WITHOUT any qubit drive.  The cavity-
                    # resonator cross-Kerr coupling shifts the resonator frequency
                    # (and hence its IQ response) as a function of photon number, so
                    # this baseline is amplitude-dependent and must be measured
                    # independently rather than assumed to be a fixed offset.
                    if subtract_baseline:
                        # Reset the cavity to vacuum and the qubit to |g⟩ before the
                        # baseline pulse sequence.
                        sideband_drive = node.namespace["sideband_drive"]
                        for i, qubit in multiplexed_qubits.items():
                            cavity_mode.reset(
                                node.parameters.cavity_reset_type,
                                node.parameters.simulate,
                                log_callable=node.log,
                                sideband_drive=sideband_drive,
                                qubit_thermalization_time=qubit.thermalization_time,
                                fock_n=node.parameters.cavity_active_cooling_fock_n,
                                f0g1_pulse_duration_ns=node.parameters.f0g1_pulse_duration_ns,
                            )
                            qubit.reset(
                                node.parameters.reset_type,
                                node.parameters.simulate,
                                log_callable=node.log,
                            )

                        # Displace the cavity to the coherent state |α = a · A_unit⟩.
                        # align() with no args synchronises ALL QUA elements (including
                        # cavity_mode_drive, which is on a separate channel from xy/resonator).
                        align()
                        cavity_mode.cavity_mode_drive.play("displacement", amplitude_scale=a)

                        # NO qubit π-pulse here — the qubit stays in |g⟩.
                        # The readout captures only the cross-Kerr-induced IQ shift
                        # from the displaced cavity, with no vacuum-population signal.

                        # Measure the readout IQ for each qubit (baseline).
                        # Align to the cavity drive channel first to ensure the
                        # displacement pulse has finished before readout begins.
                        for i, qubit in multiplexed_qubits.items():
                            align(cavity_mode.cavity_mode_drive.name, qubit.resonator.name)
                            qubit.resonator.measure("readout", qua_vars=(I_base[i], Q_base[i]))
                            save(I_base[i], I_base_st[i])
                            save(Q_base[i], Q_base_st[i])
                            # Wait for the resonator ring-down before the next sequence.
                            qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)

                        # Optional D(-a): return the cavity toward vacuum to speed up
                        # thermalization before the signal sequence.
                        align()
                        if node.parameters.active_reset:
                            align()
                            cavity_mode.cavity_mode_drive.play("displacement", amplitude_scale=a_neg)
                        align()

                    # ============================================================
                    # PART 2 — SIGNAL measurement (full protocol, WITH π-pulse)
                    # ============================================================
                    # This block is always executed regardless of subtract_baseline.
                    # When subtract_baseline=True it is preceded by Part 1 (above),
                    # so it constitutes the second sub-sequence of the dual protocol.
                    # When subtract_baseline=False it is the only sequence (original
                    # single-sequence behaviour).

                    # Reset the cavity and qubit.  When subtract_baseline=True this is
                    # the second independent reset, ensuring the cross-Kerr environment
                    # is the same as in the baseline sub-sequence.
                    sideband_drive = node.namespace["sideband_drive"]
                    for i, qubit in multiplexed_qubits.items():
                        cavity_mode.reset(
                            node.parameters.cavity_reset_type,
                            node.parameters.simulate,
                            log_callable=node.log,
                            sideband_drive=sideband_drive,
                            qubit_thermalization_time=qubit.thermalization_time,
                            fock_n=node.parameters.cavity_active_cooling_fock_n,
                            f0g1_pulse_duration_ns=node.parameters.f0g1_pulse_duration_ns,
                        )
                        qubit.reset(
                            node.parameters.reset_type,
                            node.parameters.simulate,
                            log_callable=node.log,
                        )

                    # Displace the cavity (identical to Part 1 when subtract_baseline=True,
                    # so the cross-Kerr offset seen by the resonator is the same in both
                    # sub-sequences and cancels exactly upon subtraction).
                    align()
                    cavity_mode.cavity_mode_drive.play("displacement", amplitude_scale=a)

                    # Apply the qubit π-pulse AFTER the displacement.
                    # selective_x180: spectrally narrow, flips the qubit only when the
                    #   cavity is in |0⟩ (n=0 Fock state).
                    # x180: broadband π-pulse, flips unconditionally (use for comparison
                    #   or when selective_x180 is not yet calibrated).
                    for i, qubit in multiplexed_qubits.items():
                        align(cavity_mode.cavity_mode_drive.name, qubit.xy.name)
                        qubit.xy.play(node.parameters.qubit_pulse)

                    # Measure the readout IQ for each qubit (signal).
                    for i, qubit in multiplexed_qubits.items():
                        align(qubit.xy.name, qubit.resonator.name)
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                        # Original single-sequence mode: apply the discrimination
                        # threshold per-shot so that averaging gives a proper
                        # probability estimate (average of binary outcomes).
                        if not subtract_baseline and node.parameters.use_state_discrimination:
                            assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                            save(state[i], state_st[i])
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)

                    # Optional active reset after the signal sequence.
                    align()
                    if node.parameters.active_reset:
                        align()
                        cavity_mode.cavity_mode_drive.play("displacement", amplitude_scale=a_neg)
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # Signal I/Q: always saved.
                # Shape after buffering: (num_shots, num_amp_points) → average over
                # shots → final shape (num_amp_points,).
                I_st[i].buffer(len(amp_array)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(amp_array)).average().save(f"Q{i + 1}")

                if subtract_baseline:
                    # Baseline I/Q: saved only in dual-sequence mode.
                    # Named Ib{i+1} / Qb{i+1} so that XarrayDataFetcher groups them
                    # into dataset variables 'Ib' and 'Qb' (via the base-name regex),
                    # stacked along the qubit axis exactly like 'I' and 'Q'.
                    # process_raw_dataset subtracts these from I/Q in Volt space,
                    # BEFORE any state-discrimination threshold is applied.
                    I_base_st[i].buffer(len(amp_array)).average().save(f"Ib{i + 1}")
                    Q_base_st[i].buffer(len(amp_array)).average().save(f"Qb{i + 1}")

                elif node.parameters.use_state_discrimination:
                    # Original single-sequence mode with per-shot state computation.
                    # Averaging binary (0/1) outcomes gives a proper probability,
                    # which is more accurate than thresholding the averaged I signal.
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


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
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
    fig = plot_vacuum_calibration(
        node.results["ds_raw"],
        node.results["fit_results"],
        mode_name=node.parameters.mode_name,
        qubit_pulse=node.parameters.qubit_pulse,
        normalize_plot=node.parameters.normalize_plot,
        base_amplitude=base_amp,
    )
    plt.show()
    node.results["figures"] = {"vacuum_calibration": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    cavity_mode = node.namespace["cavity_mode"]
    base_amp = float(cavity_mode.cavity_mode_drive.operations["displacement"].amplitude)

    MAX_VOLTAGE = 0.5      # OPX+ DAC ceiling [V]
    AMP_SCALE_LIMIT = 1.9  # firmware headroom (max safe amplitude_scale < 2.0)

    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            res = node.results["fit_results"].get(qubit.name)
            if res is None or not res["success"]:
                continue

            sigma = res["sigma"]

            # Auto-compute alpha_max: fill DAC to MAX_VOLTAGE / AMP_SCALE_LIMIT ≈ 0.263 V.
            # amplitude_scale=1 → alpha_max photons; firmware limit AMP_SCALE_LIMIT gives
            # max accessible alpha = alpha_max * AMP_SCALE_LIMIT.
            alpha_max = MAX_VOLTAGE / (base_amp * sigma * AMP_SCALE_LIMIT)
            cal_amplitude = base_amp * sigma * alpha_max  # = MAX_VOLTAGE / AMP_SCALE_LIMIT

            cavity_mode.cavity_mode_drive.operations["displacement"].amplitude = float(cal_amplitude)

            mode_name = node.parameters.mode_name
            pair_key = f"{qubit.name}_{mode_name}"
            pairs = getattr(node.machine, "cavity_transmon_pairs", None)
            if pairs is not None and pair_key in pairs:
                if hasattr(pairs[pair_key], "displacement_alpha_max"):
                    pairs[pair_key].displacement_alpha_max = float(alpha_max)
                k_fit = 1.0 / (sigma ** 2)
                if hasattr(pairs[pair_key], "displacement_k"):
                    pairs[pair_key].displacement_k = float(k_fit)

            node.log(
                f"Displacement calibration: sigma={sigma:.4f}, alpha_max={alpha_max:.3f}, "
                f"stored amplitude={cal_amplitude:.6f} V  "
                f"(amplitude_scale=1 -> {alpha_max:.2f} photons, "
                f"max safe alpha={alpha_max * AMP_SCALE_LIMIT:.2f})"
            )

            break  # one cavity mode shared across all qubits in this run


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
