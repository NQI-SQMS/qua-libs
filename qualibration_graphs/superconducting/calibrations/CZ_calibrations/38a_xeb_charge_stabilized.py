# %% {Imports}
"""XEB charge-stabilized CZ calibration experiment."""
from qualibrate import QualibrationNode

from quam_config import Quam
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import curve_fit

from calibration_utils.two_qubit_xeb.parameters import Parameters
from calibration_utils.two_qubit_xeb.xeb_config import XEBConfig
from calibration_utils.two_qubit_xeb.xeb import XEB, XEBResult_minimal, XEBResult
from calibration_utils.two_qubit_xeb.qua_gate import QUAGate
from qualibration_libs.parameters import get_qubit_pairs


# %% {Initialisation}
description = """
Executes a Cross-Entropy Benchmarking (XEB) experiment for CZ gate.

# Three operation modes to get measured probability
1. Experiment:
    - (set qiskit_simulate = False and numpy_simulate = False)
    - run random circuits on real qubits and OPX and acquires the experiment results
2. Numerical simulation with Qiskit Aer:
    - (set qiskit_simulate = True and numpy_simulate = False)
    - Simulates random circuit experiment
        + Measured probability is estimated from finite number of qubit measurement results, can be used to check n_shots dependence
        + Qiskit Aer noise model
            - Depolarizing error: density operator calculation with depolarizing error parameter, directly connected to XEB layer fidelity, sanity check
            - Thermal relaxation error: density operator calculation with given T1,T2, gate times, assuming white flux noise
3. Numerical simulation with NumPy-based unitary matrix multiplication:
    - (set qiskit_simulate = False and numpy_simulate = True)
    -

# 2-q unitary estimation
- (set estimate_2q_unitary = True)
- Nelder-Mean search for 2q-gate parameters (iswap/cphase/rz1/rz2 angles) from measured probability
- Minimizes average cross-entropy between measured probability and expected probability from circuit with parameterized 2-q gate

For expected probability calculation, the original Qiskit Aer-based implementation by QM is replaced with NumPy-based unitary matrix multiplication.
"""

node = QualibrationNode[Parameters, Quam](
    name="71a_xeb_charge_stabilized",
    description=description,
    parameters=Parameters(),
    machine = Quam.load()
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Set custom parameters for debugging purposes."""
    # node.parameters.qubit_pairs = ["qB2-qB4"]
    pass


# node.machine = Quam.load()

DATA_PROCESS_FOR_CLOUD_EXECUTION = False
THERMALIZATION_FACTOR = 10


def _do_setup_xeb(node: QualibrationNode[Parameters, Quam]):
    """Build xeb_config, cz_qua, xeb and store in node.namespace. Callable directly."""
    qubit_pairs = get_qubit_pairs(node)
    target_qubit_pair = qubit_pairs[0]

    cz_gate = target_qubit_pair.macros[node.parameters.cz_macro_name]
    cz_qua = QUAGate("cz", cz_gate.apply)

    xeb_config = XEBConfig(
        seqs=node.parameters.n_sequences,
        depths=np.arange(node.parameters.depth_min, node.parameters.depth_max, node.parameters.depth_step),
        n_shots=node.parameters.n_shots,
        readout_qubits=[
            target_qubit_pair.qubit_control,
            target_qubit_pair.qubit_target,
        ],
        qubits=[
            target_qubit_pair.qubit_control,
            target_qubit_pair.qubit_target,
        ],
        qubit_pairs=[target_qubit_pair],
        baseline_gate_name=node.parameters.baseline_gate,
        gate_set_choice=node.parameters.gate_set_choice_sw_or_t,
        two_qb_gate=(cz_qua if node.parameters.apply_two_qubit_gate else None),
        two_qubit_gate_idle_time_ns=node.parameters.two_qubit_gate_idle_time_ns,
        discrimination_method=node.parameters.discrimination_method,
        should_save_data=False,
        generate_new_data=True,
        disjoint_processing=False,
        reset_method="cooldown",
        reset_kwargs={
            "cooldown_time": THERMALIZATION_FACTOR * 100000,
            "max_tries": 3,
            "pi_pulse": "x180",
        },
        readout_pulse_name="readout",
        control_readout_mode=node.parameters.control_readout_mode,
        target_readout_mode=node.parameters.target_readout_mode,
    )

    xeb = XEB(
        xeb_config,
        machine=node.machine,
        cloud=DATA_PROCESS_FOR_CLOUD_EXECUTION,
        reset_type=node.parameters.reset_type,
    )

    node.namespace["qubit_pairs"] = qubit_pairs
    node.namespace["target_qubit_pair"] = target_qubit_pair
    node.namespace["xeb_config"] = xeb_config
    node.namespace["xeb"] = xeb
    node.namespace["cz_qua"] = cz_qua


# %% {Setup_XEB}
@node.run_action(skip_if=lambda n: bool(n.parameters.analysis_only_path))
def setup_xeb(node: QualibrationNode[Parameters, Quam]):
    """Build xeb_config, cz_qua, xeb and store in node.namespace."""
    _do_setup_xeb(node)


# %% {Execute_or_Load}
@node.run_action()
def execute_or_load(node: QualibrationNode[Parameters, Quam]):
    """Either load data from analysis_only_path or execute the XEB experiment."""
    if node.parameters.analysis_only_path:
        node.log(f"--- LOADING DATA FROM: {node.parameters.analysis_only_path} ---")
        node.log("Skipping Experiment Execution.")
        result = XEBResult.from_data_qualibrate(node.parameters.analysis_only_path, machine=node.machine, prefix="data")
        node.namespace["xeb_config"] = result.xeb_config
        node.namespace["result"] = result
        node.namespace["target_qubit_pair"] = result.xeb_config.qubit_pairs[0]
    else:
        # Real experiment: XEB uses heterogeneous streams (counts, gate indices, IQ)
        # with different shapes, so XarrayDataFetcher is incompatible. Use xeb.run().
        if "xeb" not in node.namespace:
            _do_setup_xeb(node)
        xeb = node.namespace["xeb"]
        node.log("--- STARTING EXPERIMENT ---")
        job = xeb.run(simulate=False)
        result = job.result()
        node.log("Received result.")
        node.namespace["result"] = result


# %% {Plot_data}
@node.run_action()
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data."""
    result = node.namespace["result"]
    xeb_config = node.namespace["xeb_config"]
    target_qubit_pair = node.namespace["target_qubit_pair"]
    machine = node.machine

    # --- Main plots ---
    node.log("--- Processing Data ---")
    figs1 = result.plot_state_heatmap()
    fig2 = result.plot_records()
    figs3 = result.plot_fidelities(fit_linear=False, fit_log_entropy=True)
    figs4 = result.plot_fidelities(fit_linear=True, fit_log_entropy=False)
    figs_leakage_hm = result.plot_leakage_heatmap()

    node.log(result.singularities)
    pct = len(result.singularities) / (xeb_config.seqs * len(xeb_config.depths)) * 100
    node.log(f"Singularities: {pct:.3f}%")
    try:
        if figs1:
            node.results["figure1"] = figs1[0]
        node.results["figure2"] = fig2
        if figs3:
            node.results["figure3"] = figs3[0]
        if figs4:
            node.results["figure4"] = figs4[0]
        if figs_leakage_hm:
            node.results["figure_leakage_heatmap1"] = figs_leakage_hm[0]
    except Exception as e:
        node.log(f"Error plotting: {e}")

    node.results["data"] = result.data
    node.results["xeb_config"] = result.xeb_config.as_dict()

    # --- Leakage Analysis (Fitted) ---
    node.log("--- Analyzing Leakage via Histogram Fitting ---")
    try:
        q_coupler_obj = machine.qubits["q3"] if "q3" in machine.qubits else None
        pop_results = result.calc_populations_by_fitting(coupler_override=q_coupler_obj)
        depths = xeb_config.depths

        def plot_pop_ax(ax, title, pops_array, leakage_states=None):
            if leakage_states is None:
                leakage_states = [2]
            if pops_array is None or not pops_array:
                ax.text(0.5, 0.5, "No Data", ha="center")
                return
            state_colors = {1: "tab:orange", 2: "tab:red"}
            state_labels = {1: r"$P_{|e\rangle}$", 2: r"$P_{|f\rangle}$"}
            plotted_any = False
            for state_idx in leakage_states:
                if pops_array.shape[1] > state_idx:
                    p_leak = pops_array[:, state_idx]
                    color = state_colors.get(state_idx, "k")
                    label = f"Leakage ({state_labels.get(state_idx, f'State {state_idx}')})"
                    ax.plot(depths, p_leak, "o-", color=color, label=label)
                    plotted_any = True
            if not plotted_any:
                ax.text(0.5, 0.5, "Requested State\nNot in Data", ha="center")
            else:
                ax.set_title(title)
                ax.set_xlabel("Cycle Depth")
                ax.set_ylabel("Population")
                ax.set_ylim(-0.02, 0.2)
                ax.grid(True)
                ax.legend()

        fig_leak_fit, axs = plt.subplots(1, 2, figsize=(10, 5))
        plot_pop_ax(axs[0], "Control Qubit Leakage", pop_results.get("control"), leakage_states=[2])
        plot_pop_ax(axs[1], "Target Qubit Leakage", pop_results.get("target"), leakage_states=[2])
        fig_leak_fit.suptitle("Leakage vs Depth (Extracted via Histogram Amplitude Fitting)")
        plt.tight_layout()
        plt.show()
        node.results["figure_leakage_fitted"] = fig_leak_fit
    except Exception as e:
        node.log(f"Error plotting fitted leakage: {e}")

    # --- Speckle Purity Analysis ---
    def purity_exponential_decay(x, a, r, b):
        return a * (r**x) + b

    node.log("--- Analyzing Speckle Purity (Normalized/Robust) ---")
    node.log("Calculating Purity...")
    result.calculate_normalized_purity()
    fig_pur = result.plot_fidelity_and_purity(log_yscale=False)
    node.results["figure_purity_speckle"] = fig_pur
    node.results["purity_avg"] = result.average_state_purity
    node.results["purity_err_per_cycle"] = result.purity_error_per_cycle
    node.log(f"Purity Error/Cycle (Normalized): {1 - result.purity_error_per_cycle:.2e}")

    fig_comp, ax = plt.subplots(figsize=(10, 7))
    xs = xeb_config.depths

    def fit_and_plot(ax, x_data, y_data_purity, color, label_prefix, marker):
        try:
            p0 = [np.max(y_data_purity) - np.min(y_data_purity), 0.95, np.min(y_data_purity)]
            bounds = ([0, 0, 0], [1.0, 1.0, 1.0])
            popt, _ = curve_fit(purity_exponential_decay, x_data, y_data_purity, p0=p0, bounds=bounds, maxfev=5000)
            y_fit = purity_exponential_decay(x_data, *popt)
            ax.plot(x_data, np.sqrt(y_data_purity), marker, color=color, alpha=0.6, label=f"{label_prefix} Data")
            ax.plot(x_data, np.sqrt(y_fit), "-", color=color, lw=2, label=f"{label_prefix} Fit (r={popt[1]:.4f})")
            return popt
        except Exception as e:
            node.log(f"Fit failed for {label_prefix}: {e}")
            ax.plot(x_data, np.sqrt(y_data_purity), marker, color=color, alpha=0.6, label=f"{label_prefix} Data")
            return None

    fit_params = fit_and_plot(ax, xs, result.average_state_purity, "tab:blue", "Data", "o")
    if fit_params is not None:
        node.results["purity_fit_params"] = fit_params
    ax.set_xlabel("Depth")
    ax.set_ylabel(r"$\sqrt{\mathrm{Purity}}$")
    ax.set_title("Speckle Purity Decay (Normalized Method)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    node.results["figure_purity_comparison"] = fig_comp

    node.log("--- Analyzing Speckle Purity (Arute et al. 2019) ---")
    node.log("Calculating Purity...")
    result.calculate_purity()
    fig_pur = result.plot_fidelity_and_purity(log_yscale=False)
    node.results["figure_purity_speckle"] = fig_pur
    node.results["purity_avg"] = result.average_state_purity
    node.results["purity_err_per_cycle"] = result.purity_error_per_cycle
    node.log(f"Purity Error/Cycle: {1 - result.purity_error_per_cycle:.2e}")

    # --- 2Q Unitary Estimation ---
    if node.parameters.estimate_2q_unitary:
        node.log("--- Running 2Q Unitary Estimation ---")
        result_mini = XEBResult_minimal(xeb_config=xeb_config)
        result_mini.gate_indices = result.data["gate_indices"]
        result_mini.measured_probs = result.measured_probs

        result_mini.calc_expected_probs(
            theta_iswap=0,
            phi_cphase=np.pi,
            phi_rz1=0,
            phi_rz2=0,
            insert_2q_gate=node.parameters.apply_two_qubit_gate,
        )
        (
            result_mini.log_XEB_fidelity,
            result_mini.log_XEB_fidelity_seq_avg,
            result_mini.log_XEB_fidelity_seq_std,
            result_mini.a_log,
            result_mini.log_XEB_layer_fidelity,
        ) = result_mini.calculate_log_XEB_fidelity()
        (
            result_mini.linear_XEB_fidelity,
            result_mini.linear_XEB_fidelity_std,
            result_mini.a_lin,
            result_mini.linear_XEB_layer_fidelity,
            _,
        ) = result_mini.calculate_linear_XEB_fidelity(en_plot=False)

        fig6 = result_mini.estimate_2q_unitary(en_plot=True, method="Nelder-Mead")
        fig7 = result_mini.plot_fidelity_vs_depth_opt(data_to_plot="linear", compare=True)
        fig8 = result_mini.plot_fidelity_vs_depth_opt(data_to_plot="log", compare=True)

        node.results["figure_unitary_estimation_convergence"] = fig6
        node.results["figure_fidelity_opt_linear"] = fig7
        node.results["figure_fidelity_opt_log"] = fig8
        node.results["estimated_2q_params"] = {
            "theta_iswap": result_mini.theta_iswap_opt,
            "phi_cphase": result_mini.phi_cphase_opt,
            "phi_rz1": result_mini.phi_rz1_opt,
            "phi_rz2": result_mini.phi_rz2_opt,
        }
        node.log("Estimation Complete.")
        node.log(f"Optimized Parameters: iSWAP={result_mini.theta_iswap_opt:.3f}, CPhase={result_mini.phi_cphase_opt:.3f}")

    # --- IQ Density & Fixed Blobs Visualization ---
    def plot_iq_density_with_blobs(res_obj, label="Data"):
        if res_obj is None:
            return
        node.log(f"--- Plotting Global Density & Blobs [{label}] ---")

        def draw_std_circle(ax, center, sigma, color, style="-", alpha=1.0, label=None):
            if sigma is None or sigma <= 0:
                return
            circ = Circle(
                xy=center,
                radius=sigma,
                edgecolor=color,
                facecolor="none",
                lw=2,
                linestyle=style,
                alpha=alpha,
                label=label,
            )
            ax.add_patch(circ)

        conf = res_obj.xeb_config
        pair = conf.qubit_pairs[0]
        channels = [
            {
                "name": "Control",
                "I": res_obj.data["I_c_all"],
                "Q": res_obj.data["Q_c_all"],
                "qubit": pair.qubit_control,
                "mode": conf.dim_c,
                "used_gmm": res_obj.data.get("gmm_params_c"),
            },
            {
                "name": "Target",
                "I": res_obj.data["I_t_all"],
                "Q": res_obj.data["Q_t_all"],
                "qubit": pair.qubit_target,
                "mode": conf.dim_t,
                "used_gmm": res_obj.data.get("gmm_params_t"),
            },
        ]
        n_plots = len(channels)
        fig, axs = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        if n_plots == 1:
            axs = [axs]
        colors = ["red", "blue", "lime", "orange"]

        for idx, ch in enumerate(channels):
            ax = axs[idx]
            i_plot = ch["I"].flatten()
            q_plot = ch["Q"].flatten()
            i_plot = i_plot[np.isfinite(i_plot)]
            q_plot = q_plot[np.isfinite(q_plot)]
            if len(i_plot) > 0:
                ax.hist2d(i_plot, q_plot, bins=100, cmap="Greys", norm=matplotlib.colors.LogNorm(), density=True)
            qubit = ch["qubit"]
            if qubit and hasattr(qubit, "resonator") and hasattr(qubit.resonator, "gef_centers"):
                centers = np.array(qubit.resonator.gef_centers)
                n_states = min(len(centers), ch["mode"])
                for s_i in range(n_states):
                    color = colors[s_i % len(colors)]
                    center = centers[s_i]
                    sigma = 0.0
                    if hasattr(qubit, "extras"):
                        sigma = qubit.extras.get(f"std_dev_{s_i}", 0.0)
                    ax.plot(
                        center[0],
                        center[1],
                        "x",
                        color=color,
                        markersize=10,
                        markeredgewidth=3,
                        label=f"Ref |{s_i}>" if s_i == 0 else None,
                    )
                    if sigma > 0:
                        draw_std_circle(ax, center, sigma, color, style="-", alpha=1.0)
            gmm = ch["used_gmm"]
            if gmm:
                means = gmm["means"]
                for s_i in range(len(means)):
                    mu = means[s_i]
                    ax.plot(
                        mu[0],
                        mu[1],
                        "+",
                        color="k",
                        markersize=15,
                        markeredgewidth=1,
                        label="Used/Fit" if s_i == 0 else None,
                    )
            ax.set_title(f"{ch['name']} Readout")
            ax.set_xlabel("I [a.u.]")
            ax.set_ylabel("Q [a.u.]")
            ax.set_aspect("equal")
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize="small")
        fig.suptitle(f"IQ Density & Reference Blobs [{label}]")
        plt.tight_layout()
        plt.show()
        return fig

    if node.parameters.discrimination_method == "gaussian":
        fig_iq = plot_iq_density_with_blobs(result, "Data")
        if fig_iq:
            node.results["figure_iq_blobs"] = fig_iq

    # --- Raw IQ Data Analysis ---
    node.log("--- Analyzing Raw IQ Data ---")
    plot_s_index = 0
    plot_d_index = 5
    try:
        n_seqs = xeb_config.seqs
        n_depths = len(xeb_config.depths)
        n_shots = xeb_config.n_shots
        readout_op_name = xeb_config.readout_pulse_name

        I_c_all = result.data["I_c_all"].reshape((n_seqs, n_depths, n_shots))
        Q_c_all = result.data["Q_c_all"].reshape((n_seqs, n_depths, n_shots))
        I_t_all = result.data["I_t_all"].reshape((n_seqs, n_depths, n_shots))
        Q_t_all = result.data["Q_t_all"].reshape((n_seqs, n_depths, n_shots))

        i_c = I_c_all[plot_s_index, plot_d_index]
        q_c = Q_c_all[plot_s_index, plot_d_index]
        i_t = I_t_all[plot_s_index, plot_d_index]
        q_t = Q_t_all[plot_s_index, plot_d_index]

        fig_iq_all, axs = plt.subplots(1, 2, figsize=(10, 5))
        qubit_c = target_qubit_pair.qubit_control
        qubit_t = target_qubit_pair.qubit_target

        mode_c = node.parameters.control_readout_mode
        op_c = qubit_c.resonator.operations[readout_op_name]
        try:
            angle_c = op_c.rotation_angle
        except Exception:
            node.log(f"Warning: Could not find rotation_angle for {qubit_c.name}. Assuming 0.")
            angle_c = 0.0
        i_c_rot, q_c_rot = i_c, q_c

        axs[0].scatter(i_c_rot, q_c_rot, alpha=0.5, s=5)
        axs[0].set_title(f"Control Qubit ({qubit_c.name}) - Mode {mode_c}")
        axs[0].set_xlabel("I' (Rotated) [a.u.]")
        axs[0].set_ylabel("Q' (Rotated) [a.u.]")
        axs[0].axis("equal")
        if mode_c == 2:
            try:
                thresh_c = op_c.threshold
                axs[0].axvline(thresh_c, color="r", ls="--", label=f"Threshold ({thresh_c:.2f})")
                axs[0].legend()
            except Exception as e:
                node.log(f"Could not plot threshold for Control: {e}")
        elif mode_c == 3:
            try:
                centers_c = qubit_c.resonator.gef_centers
                axs[0].scatter(centers_c[0][0], centers_c[0][1], c="r", marker="X", s=100, label="G Center")
                axs[0].scatter(centers_c[1][0], centers_c[1][1], c="g", marker="X", s=100, label="E Center")
                axs[0].scatter(centers_c[2][0], centers_c[2][1], c="b", marker="X", s=100, label="F Center")
                axs[0].legend()
            except Exception as e:
                node.log(f"Could not plot centers for Control: {e}")

        mode_t = node.parameters.target_readout_mode
        op_t = qubit_t.resonator.operations[readout_op_name]
        try:
            angle_t = op_t.rotation_angle
        except Exception:
            node.log(f"Warning: Could not find rotation_angle for {qubit_t.name}. Assuming 0.")
            angle_t = 0.0
        cos_t = np.cos(angle_t)
        sin_t = np.sin(angle_t)
        i_t_rot = i_t * cos_t - q_t * sin_t
        q_t_rot = i_t * sin_t + q_t * cos_t

        axs[1].scatter(i_t_rot, q_t_rot, alpha=0.5, s=5, color="C1")
        axs[1].set_title(f"Target Qubit ({qubit_t.name}) - Mode {mode_t}")
        axs[1].set_xlabel("I' (Rotated) [a.u.]")
        axs[1].axis("equal")
        if mode_t == 2:
            try:
                thresh_t = op_t.threshold
                axs[1].axvline(thresh_t, color="r", ls="--", label=f"Threshold ({thresh_t:.2f})")
                axs[1].legend()
            except Exception as e:
                node.log(f"Could not plot threshold for Target: {e}")
        elif mode_t == 3:
            try:
                centers_t = qubit_t.resonator.gef_centers
                axs[1].scatter(centers_t[0][0], centers_t[0][1], c="r", marker="X", s=100, label="G Center")
                axs[1].scatter(centers_t[1][0], centers_t[1][1], c="g", marker="X", s=100, label="E Center")
                axs[1].scatter(centers_t[2][0], centers_t[2][1], c="b", marker="X", s=100, label="F Center")
                axs[1].legend()
            except Exception as e:
                node.log(f"Could not plot centers for Target: {e}")

        plot_depth_val = xeb_config.depths[plot_d_index]
        fig_iq_all.suptitle(
            f"Rotated IQ Data for Sequence {plot_s_index}, Depth {plot_depth_val} (Index {plot_d_index})"
        )
        plt.tight_layout()
        plt.show()
        node.results["figure_raw_iq_snapshot"] = fig_iq_all
    except Exception as e:
        node.log(f"Could not plot raw IQ data. Error: {e}")
        node.log("This is expected if 'enable_iq_snapshot' was False, or if data is from numpy/qiskit.")

    # --- Overall State Population Plot ---
    # if not node.parameters.numpy_simulate and not node.parameters.qiskit_simulate:
    node.log("--- Analyzing Overall State Populations (from State Streams) ---")
    try:
        dim_c = xeb_config.dim_c
        dim_t = xeb_config.dim_t
        dim_k = xeb_config.dim_k
        total_dim = xeb_config.total_dim

        counts_c = {0: 0, 1: 0, 2: 0}
        counts_t = {0: 0, 1: 0, 2: 0}
        counts_k = {0: 0, 1: 0, 2: 0}
        total_shots = 0

        for i in range(total_dim):
            stream_name = f"s{i}"
            if stream_name in result.counts:
                stream_total_counts = np.sum(result.counts[stream_name])
            else:
                continue
            c, t, k = result._decode_state_index(i, dim_c, dim_t)
            counts_c[c] += stream_total_counts
            counts_t[t] += stream_total_counts
            counts_k[k] += stream_total_counts
            total_shots += stream_total_counts

        if total_shots == 0:
            raise Exception("Total shots found in state streams is 0.")

        qubit_names = []
        g_probs = []
        e_probs = []
        f_probs = []

        qubit_c = target_qubit_pair.qubit_control
        qubit_t = target_qubit_pair.qubit_target

        qubit_names.append(f"{qubit_c.name}\n(Mode {dim_c})")
        g_probs.append(counts_c.get(0, 0) / total_shots)
        e_probs.append(counts_c.get(1, 0) / total_shots)
        f_probs.append(counts_c.get(2, 0) / total_shots)

        qubit_names.append(f"{qubit_t.name}\n(Mode {dim_t})")
        g_probs.append(counts_t.get(0, 0) / total_shots)
        e_probs.append(counts_t.get(1, 0) / total_shots)
        f_probs.append(counts_t.get(2, 0) / total_shots)

        fig_populations_stream, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(qubit_names))
        width = 0.25
        bar_g = ax.bar(x - width, g_probs, width, label="|G> State (or State 0)")
        bar_e = ax.bar(x, e_probs, width, label="|E> State (or State 1)")
        bar_f = ax.bar(x + width, f_probs, width, label="|F> State (or State 2)")
        ax.set_ylabel("Probability")
        ax.set_title("Overall State Populations (from Classified State Streams)")
        ax.set_xticks(x)
        ax.set_xticklabels(qubit_names)
        ax.legend()
        ax.bar_label(bar_g, fmt="{:,.1%}")
        ax.bar_label(bar_e, fmt="{:,.1%}")
        ax.bar_label(bar_f, fmt="{:,.1%}")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.1)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
        node.results["figure_overall_populations_stream"] = fig_populations_stream
    except Exception as e:
        node.log(f"Could not plot overall state populations from streams. Error: {e}")
        node.log("This is expected if data is from numpy/qiskit.")


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the calibration results."""
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine.connect().close_all_qms()
    node.save()
    node.log("Results saved")


# %%
