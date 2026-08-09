# SPDX-License-Identifier: EUPL-1.2
# Copyright (C) 2026 Q.M Technologies Ltd. / Soon Teh
# Copyright (C) 2026 Q.M Technologies Ltd. / Hiroyuki Inoue
# Copyright (C) 2026 RIKEN / András Gunyhó
# Licensed under the EUPL v1.2.
# See: https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12

# %%
#!%load_ext autoreload
#!%autoreload 2

from dataclasses import asdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from qualibrate import QualibrationLibrary, QualibrationNode
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.plotting import grid_iter
from uncertainties import ufloat

from calibration_utils.two_qubit_randomized_benchmarking import (
    Parameters as ParametersRB,
)
from calibration_utils.two_qubit_randomized_benchmarking import (
    fit_raw_data,
    log_fitted_results,
    process_raw_dataset,
)
from calibration_utils.two_qubit_randomized_benchmarking.plot_tools import (
    QubitPairGrid,
    grid_pair_names,
)
from calibration_utils.common_utils.fitting_tools import power_law
from calibration_utils.common_utils.plotting_tools import patch_fig_info
from quam_config import Quam

library = QualibrationLibrary.get_active_library()
_RB60C_LIBRARY_KEY = "60c_cz_two_qubit_randomized_benchmarking_input_stream"

_SHARED_60C_PARAM_NAMES = (
    "qubit_pairs",
    "multiplexed",
    "num_shots",
    "num_random_sequences",
    "max_circuit_depth",
    "num_intervals",
    "interval",
    "seed",
    "exclude_CNOT_from_clifford",
    "operation",
    "use_state_discrimination",
    "use_strict_timing",
    "reset_type",
    "simulate",
    "timeout",
)


class Parameters(ParametersRB):
    """60d (CZ): run 60c_cz reference + interleaved, then compare via Qualibrate run ids."""

    operation: str = "cz_bipolar"
    """CZ macro to benchmark (qp.macros[operation]); forwarded to the child 60c_cz node."""

    reference_data_id: Optional[int] = None
    """Override: Qualibrate id for reference RB. If unset, uses snapshot_idx - 2 after execute."""
    interleaved_data_id: Optional[int] = None
    """Override: Qualibrate id for interleaved RB. If unset, uses snapshot_idx - 1 after execute."""


description = """
Runs 60c twice via QualibrationLibrary (reference, then interleaved), loads both by run id, and overlays comparison plots.
"""

node = QualibrationNode[Parameters, Quam](
    name="60d_cz_two_qubit_randomized_benchmarking_master",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)
# node.machine = Quam.load()


def _kwargs_for_60c(node_60d: QualibrationNode[Parameters, Quam]) -> dict:
    p = node_60d.parameters
    return {name: getattr(p, name) for name in _SHARED_60C_PARAM_NAMES if hasattr(p, name)}


def _make_60c_node(
    node_60d: QualibrationNode[Parameters, Quam], *, interleaved: bool, run_name: str
) -> QualibrationNode:
    """Build a 60c node from the library (same pattern as calibration graphs)."""
    rb_node = library.nodes[_RB60C_LIBRARY_KEY].copy(
        name=run_name,
        interleaved_CNOT=interleaved,
        load_data_id=None,
        **_kwargs_for_60c(node_60d),
    )
    rb_node.machine = node_60d.machine
    return rb_node


def _run_60c_via_library(node_60d: QualibrationNode[Parameters, Quam], *, interleaved: bool, run_name: str) -> int:
    """Run a full 60c campaign via node.run() and return its Qualibrate snapshot id."""
    rb_node = _make_60c_node(node_60d, interleaved=interleaved, run_name=run_name)
    node_60d.log(f"Running 60c ({run_name}, interleaved_CNOT={interleaved})...")
    rb_node.run()
    run_id = rb_node.snapshot_idx
    if run_id is None:
        raise RuntimeError(f"60c run '{run_name}' finished without a snapshot_idx")
    node_60d.log(f"60c '{run_name}' saved as #{run_id}")
    return run_id


def _resolve_run_ids(node_60d: QualibrationNode[Parameters, Quam]) -> tuple[int, int]:
    """Resolve reference/interleaved Qualibrate ids (parameters, namespace, or snapshot_idx fallback)."""
    if node_60d.parameters.reference_data_id is not None and node_60d.parameters.interleaved_data_id is not None:
        return (
            node_60d.parameters.reference_data_id,
            node_60d.parameters.interleaved_data_id,
        )

    ref = node_60d.namespace.get("reference_data_id")
    int_id = node_60d.namespace.get("interleaved_data_id")
    if ref is not None and int_id is not None:
        return ref, int_id

    current = node_60d.snapshot_idx
    if current is None:
        raise RuntimeError(
            "Could not resolve run ids — set reference_data_id/interleaved_data_id or run execute first."
        )
    reference_id = current - 2
    interleaved_id = current - 1
    node_60d.log(
        f"Resolved run ids from snapshot_idx={current}: reference=#{reference_id}, interleaved=#{interleaved_id}"
    )
    return reference_id, interleaved_id


def _ds_raw_from_loaded(loaded: QualibrationNode) -> object:
    for key in ("ds_raw", "ds_proc"):
        if key in loaded.results:
            return loaded.results[key]
    raise KeyError(f"Loaded run has no ds_raw or ds_proc (keys: {list(loaded.results)})")


def _load_runs_into_node(node_60d: QualibrationNode[Parameters, Quam]) -> None:
    reference_id, interleaved_id = _resolve_run_ids(node_60d)
    ref_loaded = QualibrationNode.load_from_id(reference_id)
    int_loaded = QualibrationNode.load_from_id(interleaved_id)
    if ref_loaded is None or int_loaded is None:
        raise RuntimeError(f"Could not load runs #{reference_id} and #{interleaved_id}")

    node_60d.results["ds_raw_reference"] = _ds_raw_from_loaded(ref_loaded)
    node_60d.results["ds_raw_interleaved"] = _ds_raw_from_loaded(int_loaded)
    node_60d.namespace["reference_data_id"] = reference_id
    node_60d.namespace["interleaved_data_id"] = interleaved_id

    # Pin qubit_pairs to the pairs actually present in the loaded reference dataset. The loaded
    # run's parameters.qubit_pairs may be None (it fell back to "all pairs" at acquisition time),
    # in which case get_qubit_pairs() would return every machine pair -- more than the data holds.
    # Downstream code keys results by the data's pairs (fit_results[qp.name], ds.sel(qubit_pair=...)),
    # so a namespace wider than the data raises IndexError/KeyError. Derive the pairs from the data.
    data_pairs = [str(p) for p in node_60d.results["ds_raw_reference"].qubit_pair.values]
    if data_pairs:
        node_60d.parameters.qubit_pairs = data_pairs
    elif ref_loaded.parameters.qubit_pairs is not None:
        node_60d.parameters.qubit_pairs = ref_loaded.parameters.qubit_pairs
    node_60d.namespace["qubit_pairs"] = get_qubit_pairs(node_60d)


def _plot_overlay_dual(node: QualibrationNode[Parameters, Quam]):
    colors = {"00": "C0", "01": "C1", "10": "C2", "11": "C3"}
    grid_names, qubit_pair_names = grid_pair_names(node.namespace["qubit_pairs"])

    grid_raw = QubitPairGrid(grid_names, qubit_pair_names)
    ds_ref_mean = node.results["ds_proc_reference"].mean("nb_of_sequences")
    ds_int_mean = node.results["ds_proc_interleaved"].mean("nb_of_sequences")
    for ax, qp_dict in grid_iter(grid_raw):
        qp_name = qp_dict["qubit"]
        d_ref = ds_ref_mean.sel(qubit_pair=qp_name)
        d_int = ds_int_mean.sel(qubit_pair=qp_name)
        x = d_ref.depths.values
        for ms in d_ref.measured_state.values:
            ax.plot(
                x,
                d_ref.state.sel(measured_state=ms).values,
                color=colors[ms],
                linestyle="-",
                alpha=0.85,
            )
            ax.plot(
                x,
                d_int.state.sel(measured_state=ms).values,
                color=colors[ms],
                linestyle="--",
                alpha=0.85,
            )
        ax.legend(
            handles=[
                Line2D([0], [0], color="k", linestyle="-", label="Reference"),
                Line2D([0], [0], color="k", linestyle="--", label="Interleaved"),
                *[
                    Line2D(
                        [0],
                        [0],
                        color=colors[ms],
                        linestyle="-",
                        label=rf"$|{ms}\rangle$",
                    )
                    for ms in colors
                ],
            ],
            loc="upper right",
            ncol=2,
            fontsize=6,
        )
        ax.axhline(0.25, color="gray", linewidth=3, alpha=0.2)
        ax.set_title(f"qubit_pair = {qp_name}")
        ax.set_ylim(-0.05, 1.05)
    for grid_row in grid_raw.axes:
        grid_row[0].set_ylabel(r"State probability")
    for last_row_ax in grid_raw.all_axes[-1]:
        last_row_ax.set_xlabel("Clifford depth")
    fig_raw = grid_raw.fig
    ref_id = node.namespace.get("reference_data_id", "?")
    int_id = node.namespace.get("interleaved_data_id", "?")
    fig_raw.suptitle(f"2Q RB raw (#{ref_id} vs #{int_id})", fontsize=16)
    fig_raw.set_size_inches(10, 8)
    fig_raw.tight_layout()

    grid_fit = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qp_dict in grid_iter(grid_fit):
        qp_name = qp_dict["qubit"]
        fit_ref = node.results["ds_fit_reference"].sel(qubit_pair=qp_name)
        fit_int = node.results["ds_fit_interleaved"].sel(qubit_pair=qp_name)

        ax.errorbar(
            fit_ref.depths,
            fit_ref.data_mean,
            yerr=fit_ref.data_sem,
            fmt="o",
            ms=3,
            color="C0",
            alpha=0.7,
        )
        ax.errorbar(
            fit_int.depths,
            fit_int.data_mean,
            yerr=fit_int.data_sem,
            fmt="o",
            ms=3,
            color="C3",
            alpha=0.7,
        )

        for fit_ds, color, linestyle in [(fit_ref, "C0", "-"), (fit_int, "C3", "--")]:
            if "fit_data" in fit_ds:
                max_depth = np.log(0.5) / np.log(fit_ds.fit_data.sel(param="p").values)
                max_depth = max(max_depth, fit_ds.depths.max().item() * 1.05)
                fit_depths = np.linspace(0, max_depth, 100)
                fitted = power_law(
                    fit_depths,
                    fit_ds.fit_data.sel(param="p").values,
                    fit_ds.fit_data.sel(param="A").values,
                    fit_ds.fit_data.sel(param="B").values,
                )
                ax.plot(fit_depths, fitted, color=color, linestyle=linestyle, linewidth=2)

        epc_ref = ufloat(fit_ref.error_per_clifford.values, fit_ref.error_per_clifford_sem.values)
        epc_int = ufloat(fit_int.error_per_clifford.values, fit_int.error_per_clifford_sem.values)
        epg_int = ufloat(fit_int.error_per_gate.values, fit_int.error_per_gate_sem.values)
        summary = (
            f"Ref EPC = {epc_ref:.3e}\n"
            f"Int EPC = {epc_int:.3e}\n"
            f"Int EPG = {epg_int:.3e} [{fit_int.epg_eval_method.values}]"
        )

        ax.axhline(0.25, color="gray", linewidth=3, alpha=0.2)
        ax.set_title(qp_name)
        ax.set_xlabel("Clifford depth")
        ax.set_ylabel(r"P($|00\rangle$)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(
            handles=[
                Line2D([0], [0], color="C0", linestyle="-", label="Reference"),
                Line2D([0], [0], color="C3", linestyle="--", label="Interleaved"),
                Line2D([0], [0], color="none", linestyle="", label=summary),
            ],
            loc="upper right",
            fontsize=6,
        )

    fig_fit = grid_fit.fig
    fig_fit.suptitle(f"2Q RB fit (#{ref_id} vs #{int_id})", fontsize=16)
    fig_fit.set_size_inches(10, 8)
    fig_fit.tight_layout()
    return fig_raw, fig_fit


# %%
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubit_pairs = ["q0-2", "q0-1"]
    # node.parameters.qubit_pairs = ["q0-2"]
    # node.parameters.interleaved_CNOT = True
    node.parameters.reference_data_id = 1573
    node.parameters.interleaved_data_id = 1574


# %%
@node.run_action(
    skip_if=node.parameters.simulate
    or node.parameters.load_data_id is not None
    or (node.parameters.reference_data_id is not None and node.parameters.interleaved_data_id is not None)
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Run 60c reference then interleaved via library.nodes[...].copy().run()."""
    node.namespace["reference_data_id"] = _run_60c_via_library(node, interleaved=False, run_name="2Q_RB_reference")
    node.namespace["interleaved_data_id"] = _run_60c_via_library(node, interleaved=True, run_name="2Q_RB_interleaved")
    node.log("Both 60c campaigns saved.")


# %%
@node.run_action(skip_if=node.parameters.simulate)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load the two 60c runs by id (from snapshot_idx or manual overrides)."""
    if node.parameters.load_data_id is not None:
        loaded = QualibrationNode.load_from_id(node.parameters.load_data_id)
        if loaded is None:
            raise RuntimeError(f"Could not load run #{node.parameters.load_data_id}")
        node.results["ds_raw_reference"] = _ds_raw_from_loaded(loaded)
        node.namespace["qubit_pairs"] = get_qubit_pairs(node)
        return

    _load_runs_into_node(node)


# %%
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    if "ds_raw_interleaved" not in node.results:
        raise RuntimeError("Missing interleaved data — run execute + load_data first.")

    if "qubit_pairs" not in node.namespace:
        node.namespace["qubit_pairs"] = get_qubit_pairs(node)

    node.parameters.interleaved_CNOT = False
    node.results["ds_proc_reference"] = process_raw_dataset(node.results["ds_raw_reference"], node)
    node.results["ds_fit_reference"], fit_ref = fit_raw_data(node.results["ds_proc_reference"], node)
    node.results["fit_results_reference"] = {k: asdict(v) for k, v in fit_ref.items()}
    log_fitted_results(node.results["fit_results_reference"], log_callable=node.log)

    for qp in node.namespace["qubit_pairs"]:
        ref = node.results["fit_results_reference"][qp.name]
        qp.extras["2QRB_p"] = ref["p"]
        qp.extras["2QRB_p_sem"] = ref["p_sem"]

    node.parameters.interleaved_CNOT = True
    node.results["ds_proc_interleaved"] = process_raw_dataset(node.results["ds_raw_interleaved"], node)
    node.results["ds_fit_interleaved"], fit_int = fit_raw_data(node.results["ds_proc_interleaved"], node)
    node.results["fit_results_interleaved"] = {k: asdict(v) for k, v in fit_int.items()}
    log_fitted_results(node.results["fit_results_interleaved"], log_callable=node.log)

    node.outcomes = {
        qp.name: (
            "successful"
            if node.results["fit_results_reference"][qp.name]["success"]
            and node.results["fit_results_interleaved"][qp.name]["success"]
            else "failed"
        )
        for qp in node.namespace["qubit_pairs"]
    }


# %%
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    fig_raw, fig_fit = _plot_overlay_dual(node)
    node.results["figures"] = {
        "2QRB_compare": fig_fit,
        "all_state_raw_compare": fig_raw,
    }
    patch_fig_info(node)
    plt.show()


# %%
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()


# %%
