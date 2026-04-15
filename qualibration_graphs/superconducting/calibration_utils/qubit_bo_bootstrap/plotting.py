"""
plotting.py — Visualization for the BO qubit bootstrap node (03e).

Three figure-level plotting functions:

    plot_bo_trajectory(history, qubits)
        BO trajectory: cost vs evaluation index, coloured by phase (LHS / BO).
        Also shows the running best cost.

    plot_convergence(history, qubits, convergence_tolerance_mhz)
        Convergence curve: GP-predicted optimum frequency vs evaluation index.
        Draws the convergence tolerance band as a horizontal shaded region.

    plot_rabi_with_fit(history, qubits, best_idx)
        Time-domain Rabi trace + sinusoidal fit for the best evaluation found.

All functions accept the ``history`` dict produced by ``run_bo_loop`` in
``03e_qubit_bo_bootstrap.py`` and a list of ``AnyTransmon`` qubits.
They return Matplotlib ``Figure`` objects that can be passed to
``node.save_figure(fig, name)`` for artifact storage.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

# Phase labels for the BO trajectory plot
_PHASE_COLORS = {
    "lhs_broad": "#4878cf",  # steel blue
    "lhs_zoom": "#6acc65",   # soft green
    "bo": "#d65f5f",         # muted red
}
_PHASE_LABELS = {
    "lhs_broad": "Phase 1a — Broad LHS",
    "lhs_zoom": "Phase 1b — Zoom LHS",
    "bo": "Phase 2 — BO (EI/UCB)",
}


def plot_bo_trajectory(
    history: Dict[str, List[Dict[str, Any]]],
    qubit_names: List[str],
) -> Figure:
    """
    Plot the BO cost trajectory for each qubit.

    For each qubit, one subplot shows:
    - Scatter: cost at each evaluation, coloured by phase (LHS broad / LHS zoom / BO).
    - Line: running best cost (monotonically non-increasing).
    - Vertical dashed lines separating the three phases.

    Parameters
    ──────────
    history : dict mapping qubit name → list of evaluation dicts.
        Each dict must contain:
            'phase'   : str  ('lhs_broad', 'lhs_zoom', 'bo')
            'cost'    : float
            'omega_d_mhz' : float  (drive frequency in MHz)
            'v_d'         : float  (OPX IF amplitude in V)
            (optional) 'omega_r_mhz', 'v_r' for 4D runs
    qubit_names : list of qubit name strings.

    Returns
    ──────
    Figure with one subplot per qubit (2 rows × ceil(n/2) columns).
    """
    n = len(qubit_names)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, q_name in enumerate(qubit_names):
        ax = axes_flat[idx]
        evals = history.get(q_name, [])
        if not evals:
            ax.set_title(f"{q_name}: no data")
            continue

        costs = np.array([e["cost"] for e in evals], dtype=float)
        phases = [e["phase"] for e in evals]
        x = np.arange(1, len(costs) + 1)

        # Running best
        running_best = np.minimum.accumulate(costs)

        # Phase boundaries
        phase_changes = [0]
        for i in range(1, len(phases)):
            if phases[i] != phases[i - 1]:
                phase_changes.append(i)
        phase_changes.append(len(phases))

        # Scatter per phase
        seen_phases: set = set()
        for i, (e, c) in enumerate(zip(evals, costs)):
            ph = e["phase"]
            col = _PHASE_COLORS.get(ph, "grey")
            label = _PHASE_LABELS.get(ph, ph) if ph not in seen_phases else None
            ax.scatter(x[i], c, color=col, s=40, zorder=3, label=label)
            seen_phases.add(ph)

        # Running best line
        ax.plot(x, running_best, "k--", linewidth=1.5, label="Running best", zorder=2)

        # Vertical separators between phases
        for bnd in phase_changes[1:-1]:
            ax.axvline(bnd + 0.5, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)

        ax.set_xlabel("Evaluation index", fontsize=9)
        ax.set_ylabel("Cost (lower = better)", fontsize=9)
        ax.set_title(f"BO Trajectory — {q_name}", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    # Hide unused subplots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Bayesian Optimisation — Cost Trajectory", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


def plot_convergence(
    history: Dict[str, List[Dict[str, Any]]],
    qubit_names: List[str],
    convergence_tolerance_mhz: float = 1.0,
) -> Figure:
    """
    Plot the GP-predicted optimum frequency vs evaluation index.

    Shows how the BO's best guess for the qubit frequency evolves.  When the
    curve flattens within ±convergence_tolerance_mhz, the BO has converged.

    Parameters
    ──────────
    history : same format as in ``plot_bo_trajectory``.
        Each evaluation dict must additionally contain:
            'predicted_freq_mhz' : float  (GP-predicted optimum, drive frequency)
    qubit_names : list of qubit name strings.
    convergence_tolerance_mhz : float
        Half-width of the shaded convergence band around the final predicted frequency.

    Returns
    ──────
    Figure with one subplot per qubit.
    """
    n = len(qubit_names)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, q_name in enumerate(qubit_names):
        ax = axes_flat[idx]
        evals = history.get(q_name, [])

        # Only BO phase entries have a predicted frequency (LHS evals don't)
        bo_evals = [e for e in evals if e.get("predicted_freq_mhz") is not None]
        if not bo_evals:
            ax.set_title(f"{q_name}: no BO data")
            continue

        bo_indices = [evals.index(e) + 1 for e in bo_evals]
        pred_freqs = np.array([e["predicted_freq_mhz"] for e in bo_evals], dtype=float)
        final_freq = pred_freqs[-1]

        ax.plot(bo_indices, pred_freqs, "o-", color="#d65f5f", linewidth=1.5, markersize=5)

        # Convergence tolerance band
        ax.axhspan(
            final_freq - convergence_tolerance_mhz,
            final_freq + convergence_tolerance_mhz,
            color="green",
            alpha=0.15,
            label=f"±{convergence_tolerance_mhz:.1f} MHz tolerance",
        )
        ax.axhline(final_freq, color="green", linestyle="--", linewidth=1.0)

        ax.set_xlabel("Evaluation index", fontsize=9)
        ax.set_ylabel("Predicted qubit freq. [MHz]", fontsize=9)
        ax.set_title(f"Convergence — {q_name}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("BO Convergence — Predicted Optimum Frequency", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


def plot_rabi_with_fit(
    history: Dict[str, List[Dict[str, Any]]],
    qubit_names: List[str],
    best_eval_idx: Optional[Dict[str, int]] = None,
) -> Figure:
    """
    Plot the time-domain Rabi trace + sinusoidal fit for the best evaluation.

    Parameters
    ──────────
    history : same format as above.
        Each evaluation dict must additionally contain:
            't_ns'    : list[float] or np.ndarray — duration array in nanoseconds
            'I_arr'   : list[float] or np.ndarray — averaged I quadrature (V)
            'I_fit'   : list[float] or np.ndarray — fitted sinusoid values
            'cost'    : float
            'rabi_freq_mhz' : float
            'amplitude'     : float
            'snr'           : float
    qubit_names : list of qubit name strings.
    best_eval_idx : optional dict mapping qubit name → index of best evaluation.
        If None, uses the evaluation with the lowest cost for each qubit.

    Returns
    ──────
    Figure with one subplot per qubit.
    """
    n = len(qubit_names)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, q_name in enumerate(qubit_names):
        ax = axes_flat[idx]
        evals = history.get(q_name, [])
        if not evals:
            ax.set_title(f"{q_name}: no data")
            continue

        # Select the evaluation to display
        if best_eval_idx and q_name in best_eval_idx:
            best_i = best_eval_idx[q_name]
        else:
            costs = [e.get("cost", np.inf) for e in evals]
            best_i = int(np.argmin(costs))

        e = evals[best_i]
        t_ns = np.asarray(e.get("t_ns", []))
        I_arr = np.asarray(e.get("I_arr", []))
        I_fit = np.asarray(e.get("I_fit", []))

        if len(t_ns) == 0 or len(I_arr) == 0:
            ax.set_title(f"{q_name}: missing trace data")
            continue

        # Scale to mV for readability
        scale = 1e3
        ax.plot(t_ns, I_arr * scale, ".", markersize=4, color="#4878cf", label="data")
        if len(I_fit) == len(t_ns):
            ax.plot(t_ns, I_fit * scale, "-", color="#d65f5f", linewidth=1.5, label="fit")

        # Annotate with fit quality
        rabi_freq = e.get("rabi_freq_mhz", float("nan"))
        amplitude = e.get("amplitude", float("nan"))
        snr = e.get("snr", float("nan"))
        cost = e.get("cost", float("nan"))
        omega_d_mhz = e.get("omega_d_mhz", float("nan"))
        title_lines = [
            f"{q_name}  (eval #{best_i + 1}, {e.get('phase', '?')})",
            f"$\\Omega_R$={rabi_freq:.1f} MHz  A={amplitude:.3f}  SNR={snr:.1f}  C={cost:.2f}",
            f"$\\omega_d$={omega_d_mhz:.1f} MHz  $V_d$={e.get('v_d', float('nan')):.4f} V",
        ]
        ax.set_title("\n".join(title_lines), fontsize=8)
        ax.set_xlabel("Pulse duration [ns]", fontsize=9)
        ax.set_ylabel("I [mV]", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Best Rabi Trace + Fit", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


def plot_cost_landscape_1d(
    history: Dict[str, List[Dict[str, Any]]],
    qubit_names: List[str],
) -> Figure:
    """
    Scatter plot of cost vs drive frequency for all evaluated points.

    A 1D projection of the cost landscape — useful for debugging whether the
    BO converged to the correct minimum vs a spurious mode.

    Parameters
    ──────────
    history : same format as above (requires 'omega_d_mhz' and 'cost' keys).
    qubit_names : list of qubit name strings.

    Returns
    ──────
    Figure with one subplot per qubit.
    """
    n = len(qubit_names)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, q_name in enumerate(qubit_names):
        ax = axes_flat[idx]
        evals = history.get(q_name, [])
        if not evals:
            ax.set_title(f"{q_name}: no data")
            continue

        freqs = np.array([e.get("omega_d_mhz", np.nan) for e in evals])
        costs = np.array([e.get("cost", np.nan) for e in evals])
        phases = [e.get("phase", "?") for e in evals]
        colors = [_PHASE_COLORS.get(ph, "grey") for ph in phases]

        ax.scatter(freqs, costs, c=colors, s=30, zorder=3)

        # Mark best point
        valid = np.isfinite(costs)
        if valid.any():
            best_i = int(np.argmin(costs[valid]))
            best_freq = freqs[valid][best_i]
            best_cost = costs[valid][best_i]
            ax.scatter(
                [best_freq], [best_cost],
                marker="*", s=200, color="gold", edgecolors="black",
                linewidths=0.5, zorder=5, label=f"Best: {best_freq:.1f} MHz",
            )

        # Legend patches
        patches = [
            mpatches.Patch(color=col, label=_PHASE_LABELS.get(ph, ph))
            for ph, col in _PHASE_COLORS.items()
        ]
        ax.legend(handles=patches, fontsize=7)
        ax.set_xlabel("Drive frequency [MHz]", fontsize=9)
        ax.set_ylabel("Cost", fontsize=9)
        ax.set_title(f"Cost landscape (1D projection) — {q_name}", fontsize=10)
        ax.grid(alpha=0.3)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("BO Cost Landscape — Drive Frequency Projection", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig
