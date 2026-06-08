"""BONodeController — wraps a QualibrationNode with GP-BO retry logic and BIC diagnosis.

Replaces the hand-coded FSM condition functions from bringup_graphs.py:
    should_retry_resonator_discovery
    should_repeat_punch_out
    should_repeat_spec_vs_power
    should_repeat_rabi_amplitude
    should_restart_qubit_calibration
and the old calibration_utils/bayesian_optimizer/bo_node_controller.py.

Acts as the ``on`` argument to ``graph.loop()`` in QUAlibrate.

Parameter injection:
    BO suggestions are injected directly into ``node.parameters`` attributes
    before returning True (retry). No QUAM temp_calibration involvement.

Uncertainty propagation:
    After a successful run, GaussianEstimate is stored in-memory in
    ``_estimates[qubit]``. Downstream controllers call ``get_estimate(qubit)``
    to incorporate upstream uncertainty into their GP acquisition bounds.

Reference:
    Kelly, J. et al. (2018). "Physical qubit calibration on a directed acyclic graph."
        arXiv:1803.03226 — defines the check/cal pattern that this controller extends
        with probabilistic decisions.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

from calib_framework.core.bic import BICDiagnoser
from calib_framework.core.estimates import GaussianEstimate
from calib_framework.core.node_result import NodeResult
from calib_framework.bo.optimizer import GPBayesianOptimizer, ParameterBound
from calib_framework.logging.session_logger import SessionLogger

if TYPE_CHECKING:
    pass  # QualibrationNode type hint only — no qualibrate import at module level

logger = logging.getLogger(__name__)


class BONodeController:
    """
    Wraps a QualibrationNode with GP-BO retry logic and BIC diagnosis.

    Replaces all hand-coded FSM condition functions and error code enums
    from the previous calibration system.

    Acts as the condition function for ``graph.loop()``:
        graph.loop(node, on=controller.should_repeat, max_iterations=max_iter)

    On each iteration:
    1. Reads ``node.results`` to extract raw fit data (x/y arrays).
    2. Runs BICDiagnoser to diagnose fit quality (replaces error code FSM).
    3. Computes BO cost from BICResult.
    4. Registers (params_used, cost) with GPBayesianOptimizer.
    5. Writes NodeResult to SessionLogger.
    6. Returns False (done) on success, or injects next BO suggestion into
       ``node.parameters`` and returns True (retry).

    Parameter injection (``param_map``):
    - ``None`` (default): tries bound names directly as ``node.parameters`` attributes.
    - ``dict[str, str]``: maps BO bound name → ``node.parameters`` attribute name.
    - ``Callable[[node, qubit, suggestion_dict], None]``: called for complex cases
      (e.g. converting a single "power_span_dbm" bound into ``min_power_dbm`` /
      ``max_power_dbm`` attributes).

    Upstream estimates (``upstream_controllers``):
    - Dict mapping node_id → upstream BONodeController.
    - ``get_estimate(qubit)`` is called on each to retrieve GaussianEstimate objects
      for GP acquisition bound tightening.

    Args:
        node_key: Short identifier matching the node script name.
        node_type: Key into BICDiagnoser.MODEL_REGISTRY.
        bounds: ParameterBound list defining the BO search space.
        bo_state_dir: Directory for GP observation JSON files and session logs.
        session_id: UUID string for the current session (shared across all nodes).
        logger_inst: SessionLogger instance (shared across all nodes in a graph).
        param_map: Parameter injection mapping (see class docstring).
        upstream_controllers: Dict of upstream BONodeController objects for
            estimate propagation (see class docstring).
        max_iterations: Maximum BO iterations before giving up.
        success_delta_bic: ΔBIC threshold for declaring success (default 6.0 = "moderate").
        x_axis_key: Key in ``node.results["ds_raw"][qubit]`` for the sweep axis.
            Common values: "detuning", "time", "amplitude", "frequency".
        y_axis_key: Key in ``node.results["ds_raw"][qubit]`` for the signal.
            Common values: "state", "I", "amplitude".
    """

    def __init__(
        self,
        node_key: str,
        node_type: str,
        bounds: list[ParameterBound],
        bo_state_dir: str | Path,
        session_id: str,
        logger_inst: SessionLogger,
        param_map: dict[str, str] | Callable | None = None,
        upstream_controllers: dict[str, "BONodeController"] | None = None,
        max_iterations: int = 8,
        success_delta_bic: float = 6.0,
        x_axis_key: str = "detuning",
        y_axis_key: str = "state",
    ) -> None:
        self.node_key = node_key
        self.node_type = node_type
        self.bounds = bounds
        self.bo_state_dir = Path(bo_state_dir)
        self.session_id = session_id
        self.log = logger_inst
        self.param_map = param_map
        self.upstream_controllers: dict[str, BONodeController] = upstream_controllers or {}
        self.max_iterations = max_iterations
        self.success_delta_bic = success_delta_bic
        self.x_axis_key = x_axis_key
        self.y_axis_key = y_axis_key

        self._diagnoser = BICDiagnoser(node_type)
        # Per-qubit optimisers — created lazily on first call.
        self._optimizers: dict[str, GPBayesianOptimizer] = {}
        # Per-qubit retry counters.
        self._retry_counts: dict[str, int] = {}
        # Per-qubit start times for wall-clock timing.
        self._start_times: dict[str, float] = {}
        # In-memory per-qubit GaussianEstimates (written on success, read by downstream nodes).
        self._estimates: dict[str, GaussianEstimate] = {}
        # BO suggestion applied before the most recent run (used as params_used).
        self._last_suggestion: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_estimate(self, qubit: str) -> GaussianEstimate | None:
        """
        Return the GaussianEstimate produced by this node's last successful run.

        Called by downstream BONodeControllers to tighten their GP acquisition bounds.
        Returns None if this node has not yet succeeded for the given qubit.
        """
        return self._estimates.get(qubit)

    # ------------------------------------------------------------------
    # QUAlibrate condition function interface
    # ------------------------------------------------------------------

    def should_repeat(self, node: Any, target: str) -> bool:
        """
        QUAlibrate condition function signature.

        Called after each node iteration by ``graph.loop()``.
        Returns True if this target (qubit) should iterate again.

        On True: injects the next BO suggestion directly into ``node.parameters``
        so the subsequent node run uses updated sweep parameters.

        Args:
            node: QualibrationNode (or subgraph) instance.
            target: Qubit name string (e.g. "q1").

        Returns:
            True → repeat; False → done for this target.
        """
        t_end = time.monotonic()
        t_start = self._start_times.pop(target, t_end)
        wall_clock = t_end - t_start

        retry_count = self._retry_counts.get(target, 0)
        optimizer = self._get_or_create_optimizer(target)

        # 1. Extract x/y data for BIC diagnosis
        try:
            x, y = self._extract_xy_data(node, target)
        except Exception as e:
            logger.warning("[%s] Could not extract x/y data for %s: %s", self.node_key, target, e)
            x, y = np.array([0.0, 1.0]), np.array([0.5, 0.5])

        # 2. BIC diagnosis
        bic_result = self._diagnoser.diagnose(x, y)

        # 3. BO cost
        bo_cost = self._diagnoser.to_bo_cost(bic_result)

        # 4. Parameters used this iteration: prefer cached suggestion, fall back to node.parameters
        params_used = self._last_suggestion.get(target) or self._get_node_params(node)

        # 5. Register observation with the optimiser
        if params_used:
            optimizer.register(params_used, bo_cost)
        else:
            logger.warning(
                "[%s] No params_used for %s — observation not registered with GP.",
                self.node_key, target,
            )

        # 6. Build NodeResult
        raw_fit = {}
        try:
            raw_fit = dict(node.results.get("fit_results", {}).get(target, {}))
        except Exception:
            pass

        upstream_estimates = self._get_upstream_estimates(target)
        outcome = NodeResult.outcome_from_bic(bic_result)

        output_estimate: GaussianEstimate | None = None
        if outcome == "successful":
            output_estimate = self._build_output_estimate(target, optimizer)

        node_result = NodeResult(
            node_id=self.node_key,
            qubit=target,
            session_id=self.session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            parameters_used=params_used or {},
            raw_fit=raw_fit,
            bic_result=bic_result,
            bo_cost=bo_cost,
            upstream_estimates=upstream_estimates,
            output_estimate=output_estimate,
            outcome=outcome,
            retry_count=retry_count,
            n_shots=int(raw_fit.get("n_shots", 0)),
            wall_clock_seconds=wall_clock,
        )

        # 7. Log to SessionLogger
        try:
            self.log.log_node_result(node_result)
        except Exception as e:
            logger.warning("[%s] SessionLogger write failed: %s", self.node_key, e)

        # 8. On success: store estimate in-memory and stop iterating
        if outcome == "successful" and output_estimate is not None:
            self._estimates[target] = output_estimate
            logger.info(
                "[%s] %s: SUCCESS (ΔBIC=%.1f). Estimate stored in-memory.",
                self.node_key, target, bic_result.delta_bic,
            )
            return False

        # 9. Decide whether to retry
        retry_count += 1
        self._retry_counts[target] = retry_count

        if retry_count >= self.max_iterations:
            logger.warning(
                "[%s] %s: max iterations (%d) reached. Stopping.",
                self.node_key, target, self.max_iterations,
            )
            return False

        # 10. Suggest next parameters, inject into node.parameters, cache for next params_used
        try:
            suggestion = optimizer.suggest(upstream_estimates=upstream_estimates)
            self._last_suggestion[target] = suggestion
            self._apply_suggestion_to_node(node, target, suggestion)
            logger.info(
                "[%s] %s: retry %d/%d. Next suggestion: %s",
                self.node_key, target, retry_count, self.max_iterations, suggestion,
            )
        except Exception as e:
            logger.warning("[%s] BO suggest/apply failed: %s", self.node_key, e)

        # Record start time for the next iteration's wall-clock.
        self._start_times[target] = time.monotonic()
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_suggestion_to_node(
        self, node: Any, qubit: str, suggestion: dict[str, float]
    ) -> None:
        """
        Inject a BO suggestion into node.parameters.

        Dispatches based on self.param_map type:
        - None: tries each bound name directly as a node.parameters attribute.
        - dict[str, str]: maps bo_name → node.parameters attribute name.
        - Callable: called as param_map(node, qubit, suggestion) for full control.
        """
        if self.param_map is None:
            try:
                params = getattr(node, "parameters", None)
                if params is not None:
                    for name, val in suggestion.items():
                        if hasattr(params, name):
                            setattr(params, name, val)
            except Exception as e:
                logger.warning("[%s] Default param injection failed: %s", self.node_key, e)

        elif callable(self.param_map):
            try:
                self.param_map(node, qubit, suggestion)
            except Exception as e:
                logger.warning("[%s] param_map callable failed: %s", self.node_key, e)

        else:
            # dict[str, str]: bo_name → node_param_attr
            try:
                params = getattr(node, "parameters", None)
                if params is not None:
                    for bo_name, attr_name in self.param_map.items():
                        if bo_name in suggestion:
                            setattr(params, attr_name, suggestion[bo_name])
            except Exception as e:
                logger.warning("[%s] param_map dict injection failed: %s", self.node_key, e)

    def _get_upstream_estimates(self, qubit: str) -> dict[str, GaussianEstimate]:
        """
        Collect GaussianEstimates from upstream controller objects.

        Called before GP acquisition to optionally tighten search bounds around
        upstream calibrated values.
        """
        estimates: dict[str, GaussianEstimate] = {}
        for node_id, ctrl in self.upstream_controllers.items():
            est = ctrl.get_estimate(qubit)
            if est is not None:
                estimates[node_id] = est
        return estimates

    def _get_node_params(self, node: Any) -> dict[str, float] | None:
        """
        Read current parameter values from node.parameters.

        Used as params_used on the first iteration before any BO suggestion
        has been made.
        """
        try:
            params = getattr(node, "parameters", None)
            if params is None:
                return None
            result = {}
            for b in self.bounds:
                val = getattr(params, b.name, None)
                if val is not None:
                    result[b.name] = float(val)
            return result or None
        except Exception:
            return None

    def _get_or_create_optimizer(self, qubit: str) -> GPBayesianOptimizer:
        """Get or lazily create a GPBayesianOptimizer for the given qubit."""
        if qubit not in self._optimizers:
            self._optimizers[qubit] = GPBayesianOptimizer(
                node_key=self.node_key,
                qubit=qubit,
                bounds=self.bounds,
                bo_state_dir=self.bo_state_dir,
            )
        return self._optimizers[qubit]

    def _extract_xy_data(self, node: Any, target: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract sweep x-axis and measured y-axis from node.results.

        Priority order:
        1. ``node.results["ds_raw"][target]`` — xarray Dataset from existing nodes.
           Looks for ``x_axis_key`` (coord or var) and ``y_axis_key`` (var).
        2. ``node.results["fit_results"][target]`` — fallback: reconstruct from
           fit metadata if raw data is unavailable.

        Raises ValueError if neither source provides usable data.
        """
        results = getattr(node, "results", {}) or {}

        # Strategy 1: xarray Dataset
        ds_raw = results.get("ds_raw", {})
        if isinstance(ds_raw, dict):
            ds = ds_raw.get(target)
        else:
            try:
                ds = ds_raw[target]
            except (KeyError, TypeError):
                ds = None

        if ds is not None:
            try:
                import xarray as xr
                if isinstance(ds, xr.Dataset):
                    if self.x_axis_key in ds.coords:
                        x = ds.coords[self.x_axis_key].values.ravel()
                    elif self.x_axis_key in ds:
                        x = ds[self.x_axis_key].values.ravel()
                    else:
                        x = list(ds.coords.values())[0].values.ravel() if ds.coords else np.linspace(0, 1, ds.dims.get(list(ds.dims.keys())[0], 10))
                    if self.y_axis_key in ds:
                        y = ds[self.y_axis_key].values.ravel()
                    else:
                        for key in ("state", "I", "amplitude", "signal", "R"):
                            if key in ds:
                                y = ds[key].values.ravel()
                                break
                        else:
                            y = list(ds.data_vars.values())[0].values.ravel()
                    n = min(len(x), len(y))
                    return np.asarray(x[:n], dtype=float), np.asarray(y[:n], dtype=float)
            except ImportError:
                pass
            except Exception as e:
                logger.debug("ds_raw extraction failed for %s: %s", target, e)

        # Strategy 2: reconstruct from fit_results
        fit = results.get("fit_results", {}).get(target, {})
        if fit:
            logger.debug(
                "[%s] %s: No ds_raw found; using fit_results stub for BIC (limited).",
                self.node_key, target,
            )
            return np.array([0.0, 1.0]), np.array([float(fit.get("success", 0.5))] * 2)

        raise ValueError(
            f"No usable x/y data found for qubit '{target}' in node.results. "
            "Ensure the node populates node.results['ds_raw'] or node.results['fit_results']."
        )

    def _build_output_estimate(
        self,
        qubit: str,
        optimizer: GPBayesianOptimizer,
    ) -> GaussianEstimate | None:
        """Build a GaussianEstimate from the GP posterior at the best observed point."""
        best = optimizer.best_params
        if best is None:
            return None
        first_param_name = self.bounds[0].name if self.bounds else "value"
        mean_val = best.get(first_param_name, 0.0)
        std_val = optimizer.posterior_std_at_best

        return GaussianEstimate(
            mean=mean_val,
            std=std_val,
            source_node=self.node_key,
            session_id=self.session_id,
            n_observations=optimizer.n_observations,
        )

    # ------------------------------------------------------------------
    # Management helpers
    # ------------------------------------------------------------------

    def reset(self, qubit: str) -> None:
        """Clear in-memory BO state for a specific qubit (disk files preserved)."""
        self._optimizers.pop(qubit, None)
        self._retry_counts.pop(qubit, None)
        self._start_times.pop(qubit, None)
        self._estimates.pop(qubit, None)
        self._last_suggestion.pop(qubit, None)

    def reset_all(self) -> None:
        """Clear in-memory BO state for all qubits."""
        self._optimizers.clear()
        self._retry_counts.clear()
        self._start_times.clear()
        self._estimates.clear()
        self._last_suggestion.clear()
