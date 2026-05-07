"""
BONodeController — qualibrate loop condition backed by a GP Bayesian optimiser.

Usage in a bringup graph::

    controller = BONodeController(
        node_key="03c",
        param_names=["frequency_span_mhz", "power_shift_dbm"],
        param_bounds={"frequency_span_mhz": (50.0, 800.0), "power_shift_dbm": (-30.0, 30.0)},
        cost_fn=my_cost_fn,
        get_current_params=my_fallback_reader,
        persist_dir=Path("bo_state"),
    )
    graph.loop(spec_vs_power, on=controller.should_retry, max_iterations=10)

The loop condition contract:
  - Called by qualibrate AFTER each node iteration with (node, qubit_name).
  - Returns True to retry (BO has written new params to temp_calibration).
  - Returns False to stop (converged or cost <= converged_cost).
  - On each call: reads params used, registers with GP, saves to disk, decides.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from calibration_utils.bayesian_optimizer.bo_optimizer import BOOptimizer

logger = logging.getLogger(__name__)


class BONodeController:
    """
    Bayesian optimizer controller used as a qualibrate loop condition.

    Parameters
    ----------
    node_key : str
        Short key used in ``temp_calibration[qubit].bo_suggested`` dict and
        in persistence filenames.  Use the node number prefix, e.g. ``"03c"``.
    param_names : list[str]
        Parameter names in the order the BO works with them (must match
        the keys written to ``bo_suggested`` by the node's ``create_qua_program``).
    param_bounds : dict
        ``{param_name: (low, high)}`` in physical units.
    cost_fn : callable
        ``(fit_result_dict) -> float``.  Lower is better; 0.0 = perfect.
    get_current_params : callable
        Fallback: ``(node, qubit_name) -> np.ndarray`` — reads physical params
        from the live node/machine state when ``bo_suggested`` is not yet set.
        In practice this is only called if the node did NOT set ``bo_suggested``
        in its ``create_qua_program`` (safety net).
    converged_cost : float
        Stop looping when observed cost drops to or below this value.
    persist_dir : Path or None
        Directory for ``{node_key}_{qubit}.json`` observation files.
        Loaded on first use to warm-start the GP.  ``None`` disables persistence.
    acq : str
        Acquisition function: ``"EI"`` (default) or ``"UCB"``.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        node_key: str,
        param_names: List[str],
        param_bounds: Dict[str, Tuple[float, float]],
        cost_fn: Callable[[dict], float],
        get_current_params: Callable,
        converged_cost: float = 0.05,
        persist_dir: Optional[Path] = None,
        acq: str = "EI",
        random_state: int = 42,
    ):
        self.node_key = node_key
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.cost_fn = cost_fn
        self.get_current_params = get_current_params
        self.converged_cost = converged_cost
        self.persist_dir = Path(persist_dir) if persist_dir is not None else None
        self.acq = acq
        self.random_state = random_state

        self._bounds_list: List[Tuple[float, float]] = [
            param_bounds[name] for name in param_names
        ]
        self._optimizers: Dict[str, BOOptimizer] = {}

    # ── Optimizer management ───────────────────────────────────────────────────

    def _get_or_create_bo(self, qubit: str) -> BOOptimizer:
        if qubit not in self._optimizers:
            bo = BOOptimizer(
                bounds=self._bounds_list,
                acq=self.acq,
                random_state=self.random_state,
            )
            if self.persist_dir is not None:
                self._load(qubit, bo)
            self._optimizers[qubit] = bo
        return self._optimizers[qubit]

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self, qubit: str) -> None:
        if self.persist_dir is None:
            return
        bo = self._optimizers.get(qubit)
        if bo is None or bo.n_observations == 0:
            return
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        X_phys, y = bo.all_observations
        data = {
            "node_key": self.node_key,
            "param_names": self.param_names,
            "bounds": {name: list(self.param_bounds[name]) for name in self.param_names},
            "observations": [
                [list(map(float, x)), float(c)] for x, c in zip(X_phys, y)
            ],
        }
        path = self.persist_dir / f"{self.node_key}_{qubit}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(
            f"BO [{self.node_key}/{qubit}]: saved {bo.n_observations} observations → {path}"
        )

    def _load(self, qubit: str, bo: BOOptimizer) -> None:
        if self.persist_dir is None:
            return
        path = self.persist_dir / f"{self.node_key}_{qubit}.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for obs in data.get("observations", []):
                x_phys, cost = obs
                bo.register(np.array(x_phys, dtype=float), float(cost))
            logger.info(
                f"BO [{self.node_key}/{qubit}]: warm-started with "
                f"{bo.n_observations} observations from {path}"
            )
        except Exception as exc:
            logger.warning(
                f"BO [{self.node_key}/{qubit}]: could not load state from {path}: {exc}"
            )

    # ── Serialisation helpers ──────────────────────────────────────────────────

    def _dict_to_array(self, params: dict) -> np.ndarray:
        return np.array([float(params[name]) for name in self.param_names])

    def _array_to_dict(self, x: np.ndarray) -> dict:
        return {name: float(x[i]) for i, name in enumerate(self.param_names)}

    # ── Qualibrate loop condition ──────────────────────────────────────────────

    def should_retry(self, node, qubit: str) -> bool:
        """
        Qualibrate loop condition function.

        Called after each node iteration.  Registers the observation, checks
        convergence, and (if retrying) writes the next BO-suggested parameters
        into ``temp_calibration[qubit].bo_suggested[node_key]``.

        The node is responsible for reading ``bo_suggested[node_key]`` at the
        start of its ``create_qua_program`` and applying the suggested params.
        """
        fit_result = (node.results or {}).get("fit_results", {}).get(qubit)
        if fit_result is None:
            logger.warning(
                f"BO [{self.node_key}/{qubit}]: no fit_results — stopping loop."
            )
            return False

        cost = float(self.cost_fn(fit_result))

        # ── Determine which parameters were used in the just-completed run ────
        temp_data = _get_temp_data(node.machine, qubit)
        bo_params_used = (temp_data.bo_suggested or {}).get(self.node_key)

        if bo_params_used is not None:
            x_phys = self._dict_to_array(bo_params_used)
        else:
            # Fallback: read from live node/machine state
            try:
                x_phys = np.asarray(self.get_current_params(node, qubit), dtype=float)
            except Exception as exc:
                logger.warning(
                    f"BO [{self.node_key}/{qubit}]: get_current_params failed ({exc}); "
                    "using midpoint of bounds as placeholder."
                )
                x_phys = np.array(
                    [(lo + hi) / 2 for lo, hi in self._bounds_list], dtype=float
                )

        bo = self._get_or_create_bo(qubit)
        bo.register(x_phys, cost)
        self._save(qubit)

        logger.info(
            f"BO [{self.node_key}/{qubit}]: cost={cost:.4f}  "
            f"best={bo.best_cost:.4f}  n_obs={bo.n_observations}"
        )

        if cost <= self.converged_cost:
            logger.info(
                f"BO [{self.node_key}/{qubit}]: converged (cost={cost:.4f} "
                f"<= {self.converged_cost})."
            )
            self._clear_suggestion(temp_data)
            return False

        # ── Suggest next parameters and write to temp_calibration ─────────────
        x_next = bo.suggest()
        params_next = self._array_to_dict(x_next)

        if temp_data.bo_suggested is None:
            object.__setattr__(temp_data, "bo_suggested", {})
        temp_data.bo_suggested[self.node_key] = params_next

        try:
            node.machine.save()
        except Exception as exc:
            logger.warning(
                f"BO [{self.node_key}/{qubit}]: machine.save() failed: {exc}"
            )

        logger.info(
            f"BO [{self.node_key}/{qubit}]: next suggestion → {params_next}"
        )
        return True

    def _clear_suggestion(self, temp_data) -> None:
        if temp_data.bo_suggested and self.node_key in temp_data.bo_suggested:
            del temp_data.bo_suggested[self.node_key]

    # ── Manual control ─────────────────────────────────────────────────────────

    def reset(self, qubit: str) -> None:
        """Clear BO state for a qubit (call when device has drifted significantly)."""
        self._optimizers.pop(qubit, None)
        if self.persist_dir is not None:
            path = self.persist_dir / f"{self.node_key}_{qubit}.json"
            if path.exists():
                path.unlink()
        logger.info(f"BO [{self.node_key}/{qubit}]: state cleared.")

    def reset_all(self) -> None:
        """Clear BO state for all qubits."""
        for qubit in list(self._optimizers.keys()):
            self.reset(qubit)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_temp_data(machine, qubit_name: str):
    """Return TemporaryCalibrationData for *qubit_name*, creating it if absent."""
    from quam_config.my_quam import TemporaryCalibrationData

    if machine.temp_calibration is None:
        machine.temp_calibration = {}
    if qubit_name not in machine.temp_calibration:
        machine.temp_calibration[qubit_name] = TemporaryCalibrationData()
    temp = machine.temp_calibration[qubit_name]
    # Backward-compatibility: add bo_suggested if the loaded state.json lacks it
    if not hasattr(temp, "bo_suggested"):
        object.__setattr__(temp, "bo_suggested", None)
    return temp
