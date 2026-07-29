# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np

from qm.qua import *
from qm.qua.lib import Cast

from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter

from qualibrate import QualibrationNode
from calibration_utils.shared import _get_pair
from calibration_utils.qubit_gef_thresholds import Parameters
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Node initialisation}
description = """
        QUBIT g/e/f READOUT THRESHOLD CALIBRATION (37)

NOTE: If your system already has a GEF IQ blobs calibration node (node 10 or
similar) that stores qubit.gef_g_ef_threshold, qubit.gef_ge_f_threshold and
qubit.gef_f_is_below_ge_f, you do NOT need this node — _fock_prep_qua_sfp reads
those fields directly.  Run this node only if those fields are absent.

Calibrates the two IQ-plane thresholds needed for the 3-state (|g⟩, |e⟩, |f⟩)
single-shot discrimination used by the SFP feedforward protocol:

  g_ef_threshold  — I threshold: I ≤ threshold → |g⟩ (sideband succeeded).
  ge_f_threshold  — Q threshold separating |e⟩ from |f⟩ once |g⟩ is excluded.
  f_is_below_ge_f — True if |f⟩ gives Q below ge_f_threshold (check your system).

Sequence (num_shots single shots per state):
  1. Reset qubit.  Measure →  g blob  (I_g, Q_g)
  2. Reset + x180.  Measure →  e blob  (I_e, Q_e)
  3. Reset + x180 + EF_x180.  Measure →  f blob  (I_f, Q_f)

Analysis:
  - Finds optimal binary I-threshold (g vs {e,f}) and Q-threshold ({g,e} vs f)
    by minimising false-detection count.
  - Results stored as qubit.gef_g_ef_threshold, qubit.gef_ge_f_threshold,
    qubit.gef_f_is_below_ge_f (same field names as the existing GEF node).

Prerequisites:
  - Qubit ge and ef transitions calibrated (nodes 01–10 and ef spectroscopy/Rabi).
"""

node = QualibrationNode[Parameters, Quam](
    name="37_qubit_gef_thresholds",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name = "alice"
    # node.parameters.num_shots = 10000
    pass


node.machine = Quam.load()


# %% {Create QUA program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    node.namespace["qubits"] = qubits = get_qubits(node)
    qubit_list = list(qubits)
    if len(qubit_list) != 1:
        raise ValueError(
            f"37_qubit_gef_thresholds expects exactly one qubit, got {len(qubit_list)}."
        )
    qubit = qubit_list[0]
    node.namespace["qubit"] = qubit

    pair = _get_pair(node)
    node.namespace["pair"] = pair
    if pair is None:
        raise ValueError(
            f"No CavityTransmonPair found for mode '{node.parameters.mode_name}'. "
            "Check mode_name parameter."
        )

    n_avg = node.parameters.num_shots
    # EF frequency at Fock |0⟩ (no photons during qubit calibration)
    ef_if_0 = pair.ef_if_at_fock(qubit, 0)
    ge_if_0 = pair.ge_if_at_fock(qubit, 0)
    node.namespace["ge_if_0"] = ge_if_0
    node.namespace["ef_if_0"] = ef_if_0

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        Ig = declare(fixed)
        Qg = declare(fixed)
        Ie = declare(fixed)
        Qe = declare(fixed)
        If_ = declare(fixed)
        Qf = declare(fixed)

        Ig_st = declare_stream()
        Qg_st = declare_stream()
        Ie_st = declare_stream()
        Qe_st = declare_stream()
        If_st = declare_stream()
        Qf_st = declare_stream()
        n_st = declare_stream()

        node.machine.initialize_qpu(target=qubit)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            # --- Ground state |g⟩ ---
            qubit.reset(node.parameters.reset_type, node.parameters.simulate, log_callable=node.log)
            align()
            qubit.resonator.measure("readout", qua_vars=(Ig, Qg))
            save(Ig, Ig_st)
            save(Qg, Qg_st)
            qubit.resonator.wait(qubit.resonator.depletion_time // 4)

            # --- Excited state |e⟩ ---
            qubit.reset(node.parameters.reset_type, node.parameters.simulate, log_callable=node.log)
            align()
            qubit.xy.update_frequency(ge_if_0)
            qubit.xy.play("x180")
            align(qubit.xy.name, qubit.resonator.name)
            qubit.resonator.measure("readout", qua_vars=(Ie, Qe))
            save(Ie, Ie_st)
            save(Qe, Qe_st)
            qubit.resonator.wait(qubit.resonator.depletion_time // 4)

            # --- Second-excited state |f⟩ ---
            qubit.reset(node.parameters.reset_type, node.parameters.simulate, log_callable=node.log)
            align()
            qubit.xy.update_frequency(ge_if_0)
            qubit.xy.play("x180")
            qubit.xy.update_frequency(ef_if_0)
            qubit.xy.play("EF_x180")
            align(qubit.xy.name, qubit.resonator.name)
            qubit.resonator.measure("readout", qua_vars=(If_, Qf))
            save(If_, If_st)
            save(Qf, Qf_st)
            qubit.resonator.wait(qubit.resonator.depletion_time // 4)

        with stream_processing():
            n_st.save("n")
            Ig_st.save_all("Ig")
            Qg_st.save_all("Qg")
            Ie_st.save_all("Ie")
            Qe_st.save_all("Qe")
            If_st.save_all("If")
            Qf_st.save_all("Qf")


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

    node.log("Executing g/e/f single-shot readout calibration …")
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(node.namespace["qua_program"])
        results = fetching_tool(
            job,
            data_list=["n", "Ig", "Qg", "Ie", "Qe", "If", "Qf"],
            mode="live",
        )
        while results.is_processing():
            n, Ig, Qg, Ie, Qe, If_, Qf = results.fetch_all()
            progress_counter(n, n_avg, start_time=results.get_start_time())
        n, Ig, Qg, Ie, Qe, If_, Qf = results.fetch_all()
        node.log(job.execution_report())

    node.results["Ig"] = Ig.tolist()
    node.results["Qg"] = Qg.tolist()
    node.results["Ie"] = Ie.tolist()
    node.results["Qe"] = Qe.tolist()
    node.results["If"] = If_.tolist()
    node.results["Qf"] = Qf.tolist()


# %% {Load historical data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_data_id = node.parameters.load_data_id
    node.load_from_id(load_data_id)
    node.parameters.load_data_id = load_data_id


# %% {Analyse data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    Ig = np.asarray(node.results["Ig"])
    Qg = np.asarray(node.results["Qg"])
    Ie = np.asarray(node.results["Ie"])
    Qe = np.asarray(node.results["Qe"])
    If_ = np.asarray(node.results["If"])
    Qf = np.asarray(node.results["Qf"])

    # ----------------------------------------------------------------
    # g_ef_threshold: I threshold separating |g⟩ from {|e⟩, |f⟩}
    # Combine e and f into a single "not-g" cloud and find the I value
    # that minimises false detections between g and not-g.
    # ----------------------------------------------------------------
    I_notg = np.concatenate([Ie, If_])
    g_ef_threshold = _find_threshold_1d(Ig, I_notg)

    # ----------------------------------------------------------------
    # ge_f_threshold: Q threshold separating {|g⟩, |e⟩} from |f⟩
    # Only makes sense AFTER the I-cut has excluded the g vs ef decision.
    # We look at Q among the {e,f} sub-cloud and find the boundary.
    # ----------------------------------------------------------------
    Q_gande = np.concatenate([Qg, Qe])
    ge_f_threshold = _find_threshold_1d(Q_gande, Qf)

    node.results["g_ef_threshold"] = float(g_ef_threshold)
    node.results["ge_f_threshold"] = float(ge_f_threshold)

    # Summary stats
    g_correct = float(np.mean(Ig <= g_ef_threshold))
    ef_classified_as_g = float(np.mean(I_notg <= g_ef_threshold))
    f_correct = float(np.mean(Qf > ge_f_threshold))
    ef_wrong = float(np.mean(np.concatenate([Qg, Qe]) > ge_f_threshold))

    node.results["g_assignment_fidelity"] = g_correct
    node.results["ef_false_ground_rate"] = ef_classified_as_g
    node.results["f_assignment_fidelity"] = f_correct
    node.results["gef_false_f_rate"] = ef_wrong

    node.log(
        f"g_ef_threshold = {g_ef_threshold:.6f}  "
        f"(|g⟩ correct: {g_correct*100:.1f}%, "
        f"|e/f⟩ mis-classified as |g⟩: {ef_classified_as_g*100:.1f}%)"
    )
    node.log(
        f"ge_f_threshold  = {ge_f_threshold:.6f}  "
        f"(|f⟩ correct: {f_correct*100:.1f}%, "
        f"|g/e⟩ mis-classified as |f⟩: {ef_wrong*100:.1f}%)"
    )
    node.outcomes = {"thresholds": "successful"}


def _find_threshold_1d(blob_a: np.ndarray, blob_b: np.ndarray) -> float:
    """Return the 1D threshold that minimises false detections between blob_a and blob_b.

    Scans candidates at each unique value in the combined distribution and returns
    the threshold giving fewest total misclassifications.  blob_a is assumed to
    have the smaller mean (threshold accepts blob_a when data ≤ threshold).
    """
    combined = np.sort(np.concatenate([blob_a, blob_b]))
    best_thr = float(np.mean([np.mean(blob_a), np.mean(blob_b)]))
    best_err = len(combined) + 1
    for thr in combined:
        err = int(np.sum(blob_a > thr)) + int(np.sum(blob_b <= thr))
        if err < best_err:
            best_err = err
            best_thr = float(thr)
    return best_thr


# %% {Plot data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    Ig = np.asarray(node.results["Ig"])
    Qg = np.asarray(node.results["Qg"])
    Ie = np.asarray(node.results["Ie"])
    Qe = np.asarray(node.results["Qe"])
    If_ = np.asarray(node.results["If"])
    Qf = np.asarray(node.results["Qf"])

    g_ef_thr = node.results["g_ef_threshold"]
    ge_f_thr = node.results["ge_f_threshold"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # IQ scatter
    ax = axes[0]
    ax.scatter(Ig, Qg, s=1, alpha=0.3, label="|g⟩", color="C0")
    ax.scatter(Ie, Qe, s=1, alpha=0.3, label="|e⟩", color="C1")
    ax.scatter(If_, Qf, s=1, alpha=0.3, label="|f⟩", color="C2")
    ax.axvline(g_ef_thr, color="C0", linestyle="--", label=f"g_ef_thr={g_ef_thr:.4f}")
    ax.axhline(ge_f_thr, color="C2", linestyle="--", label=f"ge_f_thr={ge_f_thr:.4f}")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title("g/e/f IQ blobs with decision boundaries")
    ax.legend(loc="best", markerscale=5)

    # I-axis histograms (g vs ef)
    ax2 = axes[1]
    bins = np.linspace(
        min(Ig.min(), Ie.min(), If_.min()),
        max(Ig.max(), Ie.max(), If_.max()),
        100,
    )
    ax2.hist(Ig, bins=bins, alpha=0.5, label="|g⟩", color="C0", density=True)
    ax2.hist(Ie, bins=bins, alpha=0.5, label="|e⟩", color="C1", density=True)
    ax2.hist(If_, bins=bins, alpha=0.5, label="|f⟩", color="C2", density=True)
    ax2.axvline(g_ef_thr, color="k", linestyle="--", label=f"g_ef_thr={g_ef_thr:.4f}")
    ax2.set_xlabel("I")
    ax2.set_ylabel("Density")
    ax2.set_title("I-quadrature histograms (g vs ef threshold)")
    ax2.legend()

    plt.tight_layout()
    plt.show()
    node.results["figures"] = {"gef_blobs": fig}


# %% {Update state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    qubit = node.namespace.get("qubit")
    if qubit is None:
        node.log("WARNING: no qubit found — cannot update QuAM state.")
        return

    g_ef_thr = node.results["g_ef_threshold"]
    ge_f_thr = node.results["ge_f_threshold"]

    # Determine which side f is on in Q (compare f blob mean to g/e blob mean)
    Qg = np.asarray(node.results["Qg"])
    Qe = np.asarray(node.results["Qe"])
    Qf = np.asarray(node.results["Qf"])
    f_is_below = float(np.mean(Qf)) < float(np.mean(np.concatenate([Qg, Qe])))

    with node.record_state_updates():
        # Write to the same fields used by the GEF IQ blobs node so that
        # _fock_prep_qua_sfp can find them via qubit.gef_* attributes.
        qubit.gef_g_ef_threshold = g_ef_thr
        qubit.gef_ge_f_threshold = ge_f_thr
        qubit.gef_f_is_below_ge_f = f_is_below

    node.log(
        f"Updated qubit '{qubit.name}': "
        f"gef_g_ef_threshold={g_ef_thr:.6f}, "
        f"gef_ge_f_threshold={ge_f_thr:.6f}, "
        f"gef_f_is_below_ge_f={f_is_below}"
    )


# %% {Save results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
