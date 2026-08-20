# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np

from qm.qua import *
from qm.qua.lib import Cast

from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter

from qualibrate import QualibrationNode
from calibration_utils.active_gef_reset_test import Parameters
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Node initialisation}
description = """
        ACTIVE GEF RESET TEST (38)

Characterises the efficiency of qubit.reset_qubit_active_gef() by running six
experiments (Taeyoon 3_6_transmon_active_reset_test style):

  Prepared state  |  Reset applied  |  Expected outcome
  ─────────────────────────────────────────────────────
      |g⟩         |  No             |  ~thermal Pe (baseline)
      |g⟩         |  Yes            |  Pe ≈ 0  (no spurious flips)
      |e⟩         |  No             |  Pe ≈ 1  (pure excited)
      |e⟩         |  Yes            |  Pe ≈ 0  (correctly reset)
      |f⟩         |  No             |  Pe ≈ ?  (leakage, not in ge subspace)
      |f⟩         |  Yes            |  Pe ≈ 0  (f→e→g corrected)

For each case the raw IQ is collected (at the GEF-optimised readout frequency)
and classified using the nearest-centre rule from qubit.resonator.gef_centers.

Sequence per shot (6 back-to-back sub-experiments):
  1. Thermalize qubit.
  2. Optionally prepare |e⟩ (x180) or |f⟩ (x180 + EF_x180).
  3. Optionally call reset_qubit_active_gef().
  4. Shift resonator to GEF_frequency_shift, measure I/Q, restore frequency.
  5. Wait resonator depletion.

Analysis:
  Histogram the classified state (0=g, 1=e, 2=f) for each of the 6 cases.
  Plot: 6 IQ scatter panels + bar-chart summary of P(g/e/f).

No state updates — this is a diagnostic/verification node.

Prerequisites:
  - Node 14/15 (GEF IQ blobs): qubit.resonator.gef_centers and GEF_frequency_shift calibrated.
  - EF_x180 operation calibrated on qubit.xy.
"""

node = QualibrationNode[Parameters, Quam](
    name="16_active_gef_reset_test",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.num_shots = 5000
    pass


node.machine = Quam.load()

# Case labels — order must match QUA program stream order
CASES = [
    ("g", False),
    ("g", True),
    ("e", False),
    ("e", True),
    ("f", False),
    ("f", True),
]
CASE_LABELS = [
    "|g⟩  no reset",
    "|g⟩  reset",
    "|e⟩  no reset",
    "|e⟩  reset",
    "|f⟩  no reset",
    "|f⟩  reset",
]
N_CASES = len(CASES)


# %% {Create QUA program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    node.namespace["qubits"] = qubits = get_qubits(node)
    qubit_list = list(qubits)
    if len(qubit_list) != 1:
        raise ValueError(
            f"16_active_gef_reset_test expects exactly one qubit, got {len(qubit_list)}."
        )
    qubit = qubit_list[0]
    node.namespace["qubit"] = qubit

    n_avg = node.parameters.num_shots

    # Frequencies resolved at Python time
    ge_if = int(qubit.xy.intermediate_frequency)
    ef_if = ge_if + int(qubit.anharmonicity)
    gef_shift = int(getattr(qubit.resonator, "GEF_frequency_shift", 0) or 0)
    std_rr_if = int(qubit.resonator.intermediate_frequency)
    gef_rr_if = std_rr_if + gef_shift

    node.namespace["ge_if"] = ge_if
    node.namespace["ef_if"] = ef_if
    node.namespace["gef_rr_if"] = gef_rr_if
    node.namespace["std_rr_if"] = std_rr_if

    gef_centers = getattr(qubit.resonator, "gef_centers", None)
    if not gef_centers:
        raise ValueError(
            "qubit.resonator.gef_centers not found. "
            "Run node 15 (iq_blobs_gef) first."
        )

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        I = declare(fixed)
        Q = declare(fixed)
        n_st = declare_stream()

        # One pair of (I_st, Q_st) per case
        I_sts = [declare_stream() for _ in range(N_CASES)]
        Q_sts = [declare_stream() for _ in range(N_CASES)]

        node.machine.initialize_qpu(target=qubit)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            for case_idx, (prep_state, do_reset) in enumerate(CASES):
                # ── 1. Thermalize ──────────────────────────────────────────
                qubit.reset(
                    node.parameters.reset_type,
                    node.parameters.simulate,
                    log_callable=node.log,
                )
                # Extra wait so |f>→|e>→|g> decay chain fully thermalizes
                qubit.xy.wait(qubit.thermalization_time // 4)
                align()

                # ── 2. Prepare target state ────────────────────────────────
                if prep_state == "e":
                    qubit.xy.update_frequency(ge_if)
                    qubit.xy.play("x180")
                    align()
                elif prep_state == "f":
                    qubit.xy.update_frequency(ge_if)
                    qubit.xy.play("x180")
                    qubit.xy.update_frequency(ef_if)
                    qubit.xy.play("EF_x180")
                    qubit.xy.update_frequency(ge_if)
                    align()

                # ── 3. Optional GEF active reset ───────────────────────────
                if do_reset:
                    qubit.reset_qubit_active_gef()
                    align()

                # ── 4. GEF single-shot readout ─────────────────────────────
                # Shift to GEF frequency for optimal g/e/f SNR, then restore.
                qubit.resonator.update_frequency(gef_rr_if)
                qubit.resonator.measure("readout", qua_vars=(I, Q))
                qubit.resonator.update_frequency(std_rr_if)
                qubit.resonator.wait(qubit.resonator.depletion_time // 4)

                save(I, I_sts[case_idx])
                save(Q, Q_sts[case_idx])
                align()

        with stream_processing():
            n_st.save("n")
            for ci in range(N_CASES):
                I_sts[ci].save_all(f"I_{ci}")
                Q_sts[ci].save_all(f"Q_{ci}")


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
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    n_avg = node.parameters.num_shots
    data_keys = ["n"] + [f"I_{ci}" for ci in range(N_CASES)] + [f"Q_{ci}" for ci in range(N_CASES)]

    node.log(f"Running active GEF reset test — {N_CASES} cases × {n_avg} shots …")
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(node.namespace["qua_program"])
        results = fetching_tool(job, data_list=data_keys, mode="live")
        while results.is_processing():
            fetched = results.fetch_all()
            progress_counter(fetched[0], n_avg, start_time=results.get_start_time())
        fetched = results.fetch_all()
        node.log(job.execution_report())

    n_val, *iq_arrays = fetched
    for ci in range(N_CASES):
        node.results[f"I_{ci}"] = iq_arrays[ci].tolist()
        node.results[f"Q_{ci}"] = iq_arrays[N_CASES + ci].tolist()


# %% {Load historical data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_data_id = node.parameters.load_data_id
    node.load_from_id(load_data_id)
    node.parameters.load_data_id = load_data_id


# %% {Analyse data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    qubit = node.namespace.get("qubit")
    if qubit is None:
        from qualibration_libs.parameters import get_qubits
        qubit = list(get_qubits(node))[0]

    gef_centers = np.array(qubit.resonator.gef_centers)  # shape (3, 2): [g, e, f] × [I, Q]

    def classify_nearest(I_arr, Q_arr):
        """Classify each shot as 0/1/2 (g/e/f) by nearest blob centre (L1)."""
        IQ = np.stack([I_arr, Q_arr], axis=-1)  # (N, 2)
        dists = np.sum(np.abs(IQ[:, None, :] - gef_centers[None, :, :]), axis=-1)  # (N, 3)
        return np.argmin(dists, axis=-1)

    results_summary = {}
    for ci, label in enumerate(CASE_LABELS):
        I_arr = np.asarray(node.results[f"I_{ci}"])
        Q_arr = np.asarray(node.results[f"Q_{ci}"])
        states = classify_nearest(I_arr, Q_arr)
        pg = float(np.mean(states == 0))
        pe = float(np.mean(states == 1))
        pf = float(np.mean(states == 2))
        results_summary[label] = {"P_g": pg, "P_e": pe, "P_f": pf}
        node.log(f"{label:20s}  P(g)={pg*100:5.1f}%  P(e)={pe*100:5.1f}%  P(f)={pf*100:5.1f}%")

    node.results["classification"] = results_summary
    node.outcomes = {"reset_test": "successful"}


# %% {Plot data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    qubit = node.namespace.get("qubit")
    if qubit is None:
        from qualibration_libs.parameters import get_qubits
        qubit = list(get_qubits(node))[0]

    gef_centers = np.array(qubit.resonator.gef_centers)
    summary = node.results["classification"]

    # LDA thresholds from node 37 (optional — shown when available)
    g_ef_thr = getattr(qubit, "gef_g_ef_threshold", None)
    ge_f_thr = getattr(qubit, "gef_ge_f_threshold", None)
    f_is_below = getattr(qubit, "gef_f_is_below_ge_f", True)
    has_thresholds = g_ef_thr is not None and ge_f_thr is not None

    # ── IQ scatter: 3 rows × 2 cols (no-reset | reset) ────────────────────
    fig_iq, axes = plt.subplots(3, 2, figsize=(11, 14))
    fig_iq.suptitle("GEF Active Reset Test — IQ scatter", fontsize=14, fontweight="bold")

    state_colors = ["#4586E6", "#FFA600", "#51A351"]  # g=blue, e=orange, f=green
    state_names = ["|g⟩", "|e⟩", "|f⟩"]

    all_I = np.concatenate([np.asarray(node.results[f"I_{ci}"]) for ci in range(N_CASES)])
    all_Q = np.concatenate([np.asarray(node.results[f"Q_{ci}"]) for ci in range(N_CASES)])
    xlim = (all_I.min() * 1.1, all_I.max() * 1.1)
    ylim = (all_Q.min() * 1.1, all_Q.max() * 1.1)

    for ci, (label, ax) in enumerate(zip(CASE_LABELS, axes.flatten())):
        I_arr = np.asarray(node.results[f"I_{ci}"])
        Q_arr = np.asarray(node.results[f"Q_{ci}"])

        # ── Background decision zones (LDA thresholds from node 37) ────────
        # Decision tree:
        #   1. I side of g_ef_thr → |g⟩  (full height on g's side)
        #   2. Other side, f's Q side of ge_f_thr → |f⟩
        #   3. Other side, remaining Q side → |e⟩
        # The "g side" is determined from gef_centers, not hard-coded as left.
        if has_thresholds:
            xL, xR = xlim
            yB, yT = ylim
            Q_height = yT - yB

            g_center_I = float(gef_centers[0][0])
            g_is_left = g_center_I <= g_ef_thr
            g_x0, g_x1   = (xL, g_ef_thr) if g_is_left else (g_ef_thr, xR)
            ef_x0, ef_x1 = (g_ef_thr, xR) if g_is_left else (xL, g_ef_thr)

            # |g⟩ zone: full height on g's side of the vertical threshold
            ax.axvspan(g_x0, g_x1, ymin=0, ymax=1,
                       color=state_colors[0], alpha=0.13, zorder=0)

            # |f⟩ and |e⟩ zones: on the {e,f} side, split by ge_f_thr in Q
            f_ymax = max(0.0, min(1.0, (ge_f_thr - yB) / Q_height)) if f_is_below else 1.0
            f_ymin = 0.0 if f_is_below else max(0.0, min(1.0, (ge_f_thr - yB) / Q_height))
            e_ymin = f_ymax if f_is_below else 0.0
            e_ymax = 1.0 if f_is_below else f_ymin
            ax.axvspan(ef_x0, ef_x1, ymin=f_ymin, ymax=f_ymax,
                       color=state_colors[2], alpha=0.13, zorder=0)
            ax.axvspan(ef_x0, ef_x1, ymin=e_ymin, ymax=e_ymax,
                       color=state_colors[1], alpha=0.13, zorder=0)

            # Decision boundary lines (gray dashed)
            ax.axvline(g_ef_thr, color="gray", linestyle="--", linewidth=1.0, zorder=5)
            ax.axhline(ge_f_thr, color="gray", linestyle="--", linewidth=1.0, zorder=5)

        ax.scatter(I_arr, Q_arr, s=3, alpha=0.15, color="#1a1a3e", zorder=3)
        for k, (cx, cy) in enumerate(gef_centers):
            ax.plot(cx, cy, marker="*", markersize=20, color=state_colors[k],
                    markeredgecolor="white", markeredgewidth=0.8, zorder=10,
                    label=state_names[k])

        cls = summary[label]
        info = f"P(g)={cls['P_g']*100:.1f}%  P(e)={cls['P_e']*100:.1f}%  P(f)={cls['P_f']*100:.1f}%"
        ax.set_title(f"{label}\n{info}", fontsize=9, fontweight="bold")
        ax.set_xlabel("I quadrature [V]", fontsize=8)
        ax.set_ylabel("Q quadrature [V]", fontsize=8)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        legend = ax.legend(loc="lower left", fontsize=8, markerscale=0.75,
                           title="Regions & Centers", title_fontsize=8,
                           framealpha=0.85)
        legend._legend_box.align = "left"

    plt.tight_layout()
    plt.show()

    # ── Bar chart: P(g/e/f) for each case ─────────────────────────────────
    fig_bar, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(N_CASES)
    w = 0.28
    pg_vals = [summary[l]["P_g"] * 100 for l in CASE_LABELS]
    pe_vals = [summary[l]["P_e"] * 100 for l in CASE_LABELS]
    pf_vals = [summary[l]["P_f"] * 100 for l in CASE_LABELS]

    ax.bar(x - w, pg_vals, w, label="P(g)", color="#4586E6")
    ax.bar(x,     pe_vals, w, label="P(e)", color="#FFA600")
    ax.bar(x + w, pf_vals, w, label="P(f)", color="#51A351")
    ax.set_xticks(x)
    ax.set_xticklabels(CASE_LABELS, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Population (%)")
    ax.set_title("GEF Active Reset Test — state populations", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.show()

    node.results["figures"] = {"iq_scatter": fig_iq, "bar_chart": fig_bar}


# %% {Save results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
