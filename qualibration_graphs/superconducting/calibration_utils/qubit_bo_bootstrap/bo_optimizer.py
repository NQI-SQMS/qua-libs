"""
bo_optimizer.py — Gaussian-process Bayesian optimiser for the qubit bootstrap node.

Ported directly from ``automatic_calibration_idea/bo_optimizer.py`` (the offline
simulator, already validated on 10/10 clean scenarios and 3/5 noisy scenarios).
Only change: removed the simulator-specific import guard.

Design goals
────────────
- No heavy ML dependencies: uses sklearn GP + scipy only.
- Clean interface: register(x, cost) / suggest() / predict_optimum().
- Normalized internally: all GP work in [0,1]^D; physical units at the boundary.
- Pluggable acquisition: EI (default) or UCB.
- Global optimization of acquisition via differential_evolution — avoids local
  acquisition maxima that would reintroduce the local-minimum problem.

Why Matérn(ν=2.5)?
────────────────────
The Wolff cost landscape is smooth near the minimum (cost decreases monotonically
toward resonance) but has ridges and cliffs far from resonance (aliasing, no-oscillation
regions). Matérn 5/2 is once-differentiable — captures this without the infinite
smoothness assumption of RBF, which would incorrectly interpolate across ridges.

Upgrade path to BoTorch
────────────────────────
If sklearn GP becomes a bottleneck (N > 200 observations, slow posterior updates),
the BOOptimizer interface is clean enough to swap in BoTorch:
    register(x, cost) → train_X, train_Y tensors → SingleTaskGP
    suggest()         → qExpectedImprovement → optimize_acqf()
The caller (03e_qubit_bo_bootstrap.py) is unchanged.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

logger = logging.getLogger(__name__)


class BOOptimizer:
    """
    Lightweight GP-BO optimiser for minimizing an expensive black-box function.

    Parameters
    ──────────
    bounds : list of (lo, hi) tuples in physical units, one per parameter.
        Example 2D: [(3900e6, 4300e6), (0.001, 0.45)]  →  (freq_Hz, V_d)
        Example 4D: add (r_freq_lo, r_freq_hi), (r_amp_lo, r_amp_hi) for readout.
    acq : 'EI' (Expected Improvement, default) or 'UCB' (Upper Confidence Bound)
    kappa : UCB exploration weight κ. Only used when acq='UCB'.
        κ = 2.576 ≈ 99th percentile of standard normal.
    noise_level : initial GP WhiteKernel noise level (tuned during fitting).
    random_state : seed for reproducibility.
    """

    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        acq: str = "EI",
        kappa: float = 2.576,
        noise_level: float = 1e-3,
        random_state: int = 42,
    ):
        self.bounds = np.array(bounds, dtype=float)  # shape (D, 2)
        self.D = len(bounds)
        self.acq = acq
        self.kappa = kappa
        self.rng = np.random.RandomState(random_state)

        # GP: Constant amplitude × Matérn(5/2) + White noise
        # - ConstantKernel: overall signal amplitude (tuned by GP hyperparameter opt.)
        # - Matern(5/2): once-differentiable, ARD length scales per dimension
        # - WhiteKernel: observation noise (covers both quantum shot noise and
        #                systematic fit errors from the Rabi curve fitting step)
        kernel = (
            ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3))
            * Matern(
                length_scale=np.ones(self.D),
                length_scale_bounds=[(1e-2, 10.0)] * self.D,
                nu=2.5,
            )
            + WhiteKernel(
                noise_level=noise_level,
                noise_level_bounds=(1e-6, 1.0),
            )
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,        # numerical jitter for Cholesky stability
            normalize_y=True,  # important: cost values span a wide range
            n_restarts_optimizer=5,
        )

        self._X_norm: List[np.ndarray] = []  # normalized observations, shape (n, D)
        self._y: List[float] = []            # cost observations

        self._best_x_phys: Optional[np.ndarray] = None
        self._best_y: float = np.inf

        self._gp_fitted: bool = False

    # ── Normalization ─────────────────────────────────────────────────────────

    def _to_unit(self, x_phys: np.ndarray) -> np.ndarray:
        """Map physical coordinates to [0,1]^D (GP operates in unit cube)."""
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return (x_phys - lo) / (hi - lo)

    def _to_phys(self, x_norm: np.ndarray) -> np.ndarray:
        """Map [0,1]^D back to physical coordinates."""
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return x_norm * (hi - lo) + lo

    # ── Latin hypercube sampling ───────────────────────────────────────────────

    def lhs_samples(self, n: int) -> np.ndarray:
        """
        Generate *n* Latin Hypercube Samples in physical units, shape (n, D).

        LHS ensures each dimension is uniformly covered: the unit cube is divided
        into *n* equal intervals per dimension, and exactly one sample is placed
        randomly within each interval.  This gives far better global coverage than
        pure random sampling (no clustering, no empty regions).

        Used for Phase 1a (broad scan) and Phase 1b (zoom scan around the Phase-1a
        best point) to seed the GP before the BO acquisition loop starts.
        """
        samples_norm = np.zeros((n, self.D))
        for d in range(self.D):
            cuts = np.linspace(0.0, 1.0, n + 1)
            u = self.rng.uniform(cuts[:n], cuts[1: n + 1])
            samples_norm[:, d] = u[self.rng.permutation(n)]
        return self._to_phys(samples_norm)

    # ── Acquisition functions ──────────────────────────────────────────────────

    def _ei(self, X_norm: np.ndarray) -> np.ndarray:
        """
        Expected Improvement (vectorised).

        EI(x) = σ(x) · [z·Φ(z) + φ(z)]
        where z = (y_best − μ(x)) / σ(x), Φ = CDF, φ = PDF of N(0,1).

        Key property: even if μ(x) >> y_best (high mean → bad region), a large
        σ(x) contributes positively via φ(z), forcing exploration of uncertain
        regions.  This is what prevents convergence to local minima — unlike
        gradient descent which only follows the slope.
        """
        if not self._gp_fitted:
            return np.zeros(len(X_norm))
        mu, sigma = self.gp.predict(X_norm, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        y_best = min(self._y)
        z = (y_best - mu) / sigma
        ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
        return np.maximum(ei, 0.0)

    def _lcb(self, X_norm: np.ndarray) -> np.ndarray:
        """
        Lower Confidence Bound (negated for the maximization interface).

        LCB(x) = μ(x) − κ·σ(x)   (lower = better for minimization)
        We negate so that ``maximize(−LCB) = minimize(LCB)``.
        """
        if not self._gp_fitted:
            return np.zeros(len(X_norm))
        mu, sigma = self.gp.predict(X_norm, return_std=True)
        return -(mu - self.kappa * sigma)  # negated: higher = better for maximizer

    def _acq_value(self, X_norm: np.ndarray) -> np.ndarray:
        """Dispatch to the selected acquisition function."""
        if self.acq == "EI":
            return self._ei(X_norm)
        elif self.acq == "UCB":
            return self._lcb(X_norm)
        else:
            raise ValueError(
                f"Unknown acquisition function: {self.acq!r}. Use 'EI' or 'UCB'."
            )

    # ── Core interface ────────────────────────────────────────────────────────

    def register(self, x_phys: np.ndarray, cost: float) -> None:
        """
        Record an observed (x, cost) pair and refit the GP posterior.

        Parameters
        ──────────
        x_phys : parameter vector in physical units, shape (D,).
            2D: [omega_d_hz, v_d_V]
            4D: [omega_d_hz, v_d_V, omega_r_hz, v_r_V]
        cost   : scalar cost function value (lower = better).
            Positive values returned by ``compute_cost()`` in analysis.py.
        """
        x_phys = np.asarray(x_phys, dtype=float)
        x_norm = self._to_unit(x_phys)

        self._X_norm.append(x_norm.copy())
        self._y.append(float(cost))

        if cost < self._best_y:
            self._best_y = float(cost)
            self._best_x_phys = x_phys.copy()

        # Refit GP.  Requires ≥2 observations for meaningful hyperparameter tuning.
        if len(self._y) >= 2:
            X_arr = np.array(self._X_norm)
            y_arr = np.array(self._y)
            try:
                self.gp.fit(X_arr, y_arr)
                self._gp_fitted = True
                logger.debug(
                    "GP refitted on %d points. Kernel: %s",
                    len(self._y),
                    self.gp.kernel_,
                )
            except Exception as exc:
                logger.warning("GP fit failed (%s). Using previous model.", exc)

    def suggest(self) -> np.ndarray:
        """
        Return the next parameter vector to evaluate, in physical units.

        Uses ``scipy.optimize.differential_evolution`` to maximize the acquisition
        function over [0,1]^D.  Differential evolution is a global optimizer —
        it avoids local maxima of the acquisition surface, so the BO does not
        prematurely focus on a single region of the search space.

        Falls back to a random point if fewer than 3 observations have been
        registered (GP posterior is unreliable below this threshold).
        """
        if len(self._y) < 3 or not self._gp_fitted:
            logger.info(
                "Only %d observations — sampling randomly (need >=3 for reliable GP).",
                len(self._y),
            )
            return self._to_phys(self.rng.uniform(size=self.D))

        def neg_acq(x_norm_flat: np.ndarray) -> float:
            """Negative acquisition value (differential_evolution minimizes)."""
            val = self._acq_value(x_norm_flat.reshape(1, -1))
            # float(np.asarray(val).ravel()[0]) avoids numpy 2.x deprecation
            # where float() on a 1-D array raises a DeprecationWarning.
            return -float(np.asarray(val).ravel()[0])

        result = differential_evolution(
            neg_acq,
            bounds=[(0.0, 1.0)] * self.D,
            maxiter=300,
            tol=1e-5,
            seed=int(self.rng.randint(0, 2**31)),
            workers=1,       # single thread: avoids pickling issues with sklearn GP
            popsize=15,
            mutation=(0.5, 1.0),
            recombination=0.7,
        )

        x_phys = self._to_phys(result.x)
        logger.debug(
            "Acquisition maximised: x_phys=%s, acq_val=%.4f",
            x_phys,
            -result.fun,
        )
        return x_phys

    def predict_optimum(self) -> Tuple[np.ndarray, float]:
        """
        Return the GP posterior-predicted global optimum in physical units.

        This is the *predicted* minimum of the GP mean function, not the observed
        best point.  It is used for convergence checking: when this location
        stabilises between consecutive BO iterations (moves less than
        ``convergence_tolerance_mhz``), the BO has converged and we stop.

        Returns
        ──────
        x_opt_phys : np.ndarray, shape (D,)
            Predicted optimum location in physical units.
        mu_opt : float
            Predicted cost at that location (GP mean).
        """
        if not self._gp_fitted or len(self._y) < 3:
            # No reliable GP yet — return the observed best as a fallback.
            x_fallback = (
                self._best_x_phys
                if self._best_x_phys is not None
                else self._to_phys(np.full(self.D, 0.5))
            )
            return x_fallback, self._best_y

        def mu_fn(x_norm_flat: np.ndarray) -> float:
            return float(self.gp.predict(x_norm_flat.reshape(1, -1))[0])

        result = differential_evolution(
            mu_fn,
            bounds=[(0.0, 1.0)] * self.D,
            maxiter=300,
            tol=1e-5,
            seed=int(self.rng.randint(0, 2**31)),
            workers=1,
        )

        x_opt_phys = self._to_phys(result.x)
        mu_opt = float(self.gp.predict(result.x.reshape(1, -1))[0])
        return x_opt_phys, mu_opt

    def gp_uncertainty_at(self, x_phys: np.ndarray) -> float:
        """Return GP posterior standard deviation at a physical-unit point."""
        if not self._gp_fitted:
            return float("inf")
        x_norm = self._to_unit(np.asarray(x_phys, dtype=float))
        _, sigma = self.gp.predict(x_norm.reshape(1, -1), return_std=True)
        return float(sigma[0])

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def best_x(self) -> Optional[np.ndarray]:
        """Observed best parameter vector in physical units (or None if empty)."""
        return self._best_x_phys

    @property
    def best_cost(self) -> float:
        """Observed best cost value (lowest seen so far)."""
        return self._best_y

    @property
    def n_observations(self) -> int:
        """Total number of (x, cost) pairs registered so far."""
        return len(self._y)

    @property
    def all_observations(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (X_phys, y) arrays of all registered observations.

        Useful for plotting the BO trajectory and computing cost landscapes.
        """
        if not self._X_norm:
            return np.empty((0, self.D)), np.array([])
        X_phys = np.array([self._to_phys(x) for x in self._X_norm])
        y = np.array(self._y)
        return X_phys, y
