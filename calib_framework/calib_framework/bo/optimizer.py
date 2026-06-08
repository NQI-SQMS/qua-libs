"""GPBayesianOptimizer — GP-BO engine with Matérn 5/2 kernel and EI acquisition.

Replaces calibration_utils/bayesian_optimizer/bo_optimizer.py.

Backward-compatible observation file format (JSON):
    {"node_key": str, "param_names": [str], "bounds": {name: [lo, hi]},
     "observations": [[[x1, x2, ...], cost], ...]}
This matches the exact schema written by the old bo_node_controller.py so that
existing bo_state/*.json files warm-start correctly.

References:
    Snoek, J., Larochelle, H., & Adams, R.P. (2012). "Practical Bayesian
        Optimization of Machine Learning Algorithms." NeurIPS. arXiv:1206.2944
    Rasmussen, C.E. & Williams, C.K.I. (2006). "Gaussian Processes for Machine
        Learning." MIT Press. ISBN 026218253X
    Matérn, B. (1960). "Spatial Variation." Springer. (Matérn 5/2 kernel)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

if TYPE_CHECKING:
    from calib_framework.core.estimates import GaussianEstimate

logger = logging.getLogger(__name__)


@dataclass
class ParameterBound:
    """
    Search bound for a single BO parameter.

    Attributes:
        name: Parameter name (used as dict key in suggest() / register()).
        low: Lower bound in physical units.
        high: Upper bound in physical units.
        log_scale: If True, optimisation is performed in log-space.
            Useful for amplitude parameters that span orders of magnitude.
    """

    name: str
    low: float
    high: float
    log_scale: bool = False

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(
                f"ParameterBound '{self.name}': low ({self.low}) must be < high ({self.high})"
            )
        if self.log_scale and self.low <= 0:
            raise ValueError(
                f"ParameterBound '{self.name}': log_scale=True requires low > 0, got {self.low}"
            )


class GPBayesianOptimizer:
    """
    GP-BO with Matérn 5/2 kernel and Expected Improvement acquisition.

    Warm-starts from disk: observations are persisted to
    ``{bo_state_dir}/{node_key}_{qubit}.json`` across sessions using
    the same JSON schema as the old bo_node_controller.py for backward
    compatibility.

    Differences from the old bo_optimizer.py:
    - Accepts ``GaussianEstimate`` priors from upstream nodes to tighten
      search bounds via ``GaussianEstimate.search_range()``.
    - Cost signal comes from ``BICDiagnoser.to_bo_cost()`` rather than a
      manually specified scalar.
    - Tracks convergence separately per (node_key, qubit) pair.
    - Exposes ``posterior_std_at_best`` for GaussianEstimate propagation.

    Args:
        node_key: Short identifier, e.g. "04_qubit_spectroscopy_vs_power".
        qubit: Qubit name, e.g. "q1".
        bounds: List of ParameterBound objects defining the search space.
        bo_state_dir: Directory for persistent observation JSON files.
        n_initial_random: Number of Latin-Hypercube samples before GP fits.
        noise_level: Initial WhiteKernel noise level for GP.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        node_key: str,
        qubit: str,
        bounds: list[ParameterBound],
        bo_state_dir: str | Path,
        n_initial_random: int = 3,
        noise_level: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.node_key = node_key
        self.qubit = qubit
        self.bounds = bounds
        self.bo_state_dir = Path(bo_state_dir)
        self.n_initial_random = n_initial_random
        self.noise_level = noise_level
        self.random_state = random_state

        self._X_phys: list[list[float]] = []  # shape (n, d)
        self._y: list[float] = []  # shape (n,)
        self._best_idx: int = -1
        self._gp: GaussianProcessRegressor | None = None
        self._rng = np.random.default_rng(random_state)

        self._file_path = self.bo_state_dir / f"{node_key}_{qubit}.json"

        # Attempt to warm-start from disk.
        self.load()

    # ------------------------------------------------------------------
    # Internal: normalisation / denormalisation
    # ------------------------------------------------------------------

    def _to_unit(self, x_phys: np.ndarray) -> np.ndarray:
        """Map physical parameters to [0, 1]^D (log-space for log_scale bounds)."""
        x = np.asarray(x_phys, dtype=float).copy()
        for i, b in enumerate(self.bounds):
            lo, hi = b.low, b.high
            if b.log_scale:
                lo, hi = math.log(lo), math.log(hi)
                x[..., i] = math.log(float(x[..., i])) if x.ndim == 1 else np.log(x[..., i])
            span = hi - lo
            x[..., i] = (x[..., i] - lo) / (span + 1e-30)
        return np.clip(x, 0.0, 1.0)

    def _to_phys(self, x_unit: np.ndarray) -> np.ndarray:
        """Map [0, 1]^D back to physical parameter space."""
        x = np.asarray(x_unit, dtype=float).copy()
        for i, b in enumerate(self.bounds):
            lo, hi = b.low, b.high
            if b.log_scale:
                lo, hi = math.log(lo), math.log(hi)
                x[..., i] = np.exp(lo + x[..., i] * (hi - lo))
            else:
                x[..., i] = lo + x[..., i] * (hi - lo)
        return x

    # ------------------------------------------------------------------
    # Internal: initial sampling
    # ------------------------------------------------------------------

    def _lhs_sample(self, n: int) -> np.ndarray:
        """
        Latin Hypercube Sampling in unit [0, 1]^D.

        Each dimension is divided into n equal intervals; one sample is
        drawn from each interval per dimension, then columns are shuffled.
        Provides better global coverage than pure random for small n.
        """
        d = len(self.bounds)
        result = np.zeros((n, d))
        for j in range(d):
            perm = self._rng.permutation(n)
            cuts = np.linspace(0, 1, n + 1)
            result[:, j] = cuts[perm] + self._rng.uniform(0, 1 / n, size=n)
        return np.clip(result, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Internal: GP construction and acquisition
    # ------------------------------------------------------------------

    def _build_gp(self) -> GaussianProcessRegressor:
        """Build a fresh GP with Matérn 5/2 kernel (same as old bo_optimizer.py)."""
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * Matern(
                length_scale=np.ones(len(self.bounds)),
                length_scale_bounds=(1e-3, 10.0),
                nu=2.5,
            )
            + WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-4, 1.0))
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self.random_state,
        )

    def _fit_gp(self) -> None:
        """Fit (or refit) the GP to all current observations."""
        if len(self._X_phys) < 2:
            self._gp = None
            return
        X_unit = np.array([self._to_unit(np.array(x)) for x in self._X_phys])
        y = np.array(self._y)
        gp = self._build_gp()
        gp.fit(X_unit, y)
        self._gp = gp

    def _ei(self, X_unit: np.ndarray) -> np.ndarray:
        """
        Expected Improvement (vectorised, cost-minimisation convention).

        EI(x) = σ(x) · [z·Φ(z) + φ(z)]
        where z = (y_best − μ(x)) / σ(x)
        and y_best = min(observed costs).
        """
        if self._gp is None or len(self._y) == 0:
            return np.zeros(len(X_unit))
        mu, sigma = self._gp.predict(X_unit, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        y_best = float(min(self._y))
        z = (y_best - mu) / sigma
        ei = sigma * (z * stats.norm.cdf(z) + stats.norm.pdf(z))
        return np.clip(ei, 0.0, None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(
        self,
        upstream_estimates: dict[str, "GaussianEstimate"] | None = None,
        tighten_param_map: dict[str, str] | None = None,
    ) -> dict[str, float]:
        """
        Suggest the next parameters to try via EI acquisition.

        If ``upstream_estimates`` are provided together with ``tighten_param_map``
        (mapping param_name → estimate_key), the search bounds for those
        parameters are tightened around the upstream mean using
        ``GaussianEstimate.search_range(base_range)``.

        Returns:
            Dict mapping parameter name → suggested value in physical units.
        """
        # Optionally tighten bounds using upstream estimates
        effective_bounds = list(self.bounds)
        if upstream_estimates and tighten_param_map:
            effective_bounds = []
            for b in self.bounds:
                est_key = tighten_param_map.get(b.name)
                if est_key and est_key in upstream_estimates:
                    est = upstream_estimates[est_key]
                    base_range = b.high - b.low
                    half_range = est.search_range(base_range) / 2.0
                    new_lo = max(b.low, est.mean - half_range)
                    new_hi = min(b.high, est.mean + half_range)
                    if new_lo < new_hi:
                        effective_bounds.append(
                            ParameterBound(b.name, new_lo, new_hi, b.log_scale)
                        )
                        continue
                effective_bounds.append(b)

        n_obs = len(self._X_phys)

        # Random exploration: Latin Hypercube sampling until n_initial_random observations
        if n_obs < self.n_initial_random:
            sample_unit = self._lhs_sample(1)[0]
            # Apply effective bounds by sampling within them
            x_unit = np.zeros(len(self.bounds))
            for i, (b, eb) in enumerate(zip(self.bounds, effective_bounds)):
                lo_unit = max(0.0, (eb.low - b.low) / (b.high - b.low + 1e-30))
                hi_unit = min(1.0, (eb.high - b.low) / (b.high - b.low + 1e-30))
                x_unit[i] = lo_unit + sample_unit[i] * (hi_unit - lo_unit)
            x_phys = self._to_phys(x_unit)
            return {b.name: float(x_phys[i]) for i, b in enumerate(self.bounds)}

        # GP-EI maximisation: fit GP and maximise Expected Improvement
        if self._gp is None:
            self._fit_gp()

        # Build unit-space bounds for the effective parameter bounds
        bounds_unit = []
        for i, (b, eb) in enumerate(zip(self.bounds, effective_bounds)):
            span = b.high - b.low + 1e-30
            lo_u = max(0.0, (eb.low - b.low) / span)
            hi_u = min(1.0, (eb.high - b.low) / span)
            bounds_unit.append((lo_u, hi_u))

        def neg_ei(x_unit_flat: np.ndarray) -> float:
            return -float(self._ei(x_unit_flat[np.newaxis])[0])

        result = opt.differential_evolution(
            neg_ei,
            bounds=bounds_unit,
            maxiter=300,
            tol=1e-5,
            popsize=15,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=self.random_state,
            workers=1,
        )
        x_phys = self._to_phys(result.x)
        return {b.name: float(x_phys[i]) for i, b in enumerate(self.bounds)}

    def register(self, params: dict[str, float], cost: float) -> None:
        """
        Register an observation and refit the GP.

        Args:
            params: Dict mapping parameter name → value in physical units.
            cost: BO cost in [0, 1] (from BICDiagnoser.to_bo_cost()).
        """
        x = [float(params[b.name]) for b in self.bounds]
        self._X_phys.append(x)
        self._y.append(float(cost))
        # Update best index
        if self._best_idx < 0 or cost < self._y[self._best_idx]:
            self._best_idx = len(self._y) - 1
        self._fit_gp()
        self.save()

    # ------------------------------------------------------------------
    # Persistence (backward-compatible with old bo_node_controller.py)
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save observations to disk in the backward-compatible JSON format."""
        self.bo_state_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "node_key": self.node_key,
            "param_names": [b.name for b in self.bounds],
            "bounds": {b.name: [b.low, b.high] for b in self.bounds},
            "observations": [
                [list(map(float, x)), float(c)]
                for x, c in zip(self._X_phys, self._y)
            ],
        }
        tmp = self._file_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self._file_path)

    def load(self) -> None:
        """Load observations from disk if the file exists (warm-start)."""
        if not self._file_path.exists():
            return
        try:
            data = json.loads(self._file_path.read_text(encoding="utf-8"))
            obs = data.get("observations", [])
            for entry in obs:
                # Support both [[x1, x2, ...], cost] and {"x": [...], "cost": c}
                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                    x_list, cost = entry
                elif isinstance(entry, dict):
                    x_list = entry.get("x", entry.get("X", []))
                    cost = entry.get("cost", entry.get("y", entry.get("Y", None)))
                else:
                    continue
                if x_list is not None and cost is not None:
                    self._X_phys.append([float(v) for v in x_list])
                    self._y.append(float(cost))
            if self._y:
                self._best_idx = int(np.argmin(self._y))
            logger.info(
                "GPBayesianOptimizer warm-start: loaded %d observations from %s",
                len(self._y),
                self._file_path,
            )
        except Exception as e:
            logger.warning("Could not load BO state from %s: %s", self._file_path, e)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_observations(self) -> int:
        """Number of registered observations."""
        return len(self._y)

    @property
    def best_cost(self) -> float:
        """Lowest observed cost. Returns 1.0 if no observations."""
        if not self._y:
            return 1.0
        return float(min(self._y))

    @property
    def best_params(self) -> dict[str, float] | None:
        """Parameters at the best observed cost. None if no observations."""
        if self._best_idx < 0:
            return None
        x = self._X_phys[self._best_idx]
        return {b.name: float(x[i]) for i, b in enumerate(self.bounds)}

    @property
    def posterior_std_at_best(self) -> float:
        """
        GP posterior standard deviation at the current best observed point.

        Used by BONodeController to populate GaussianEstimate.std for
        downstream nodes. Returns a large value (1.0) when no GP is fitted.
        """
        if self._gp is None or self._best_idx < 0:
            return 1.0
        x_best_unit = self._to_unit(np.array(self._X_phys[self._best_idx]))
        _, std = self._gp.predict(x_best_unit[np.newaxis], return_std=True)
        return float(std[0])
