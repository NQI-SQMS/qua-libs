"""BIC model selection for calibration fit diagnosis.

Replaces hand-coded error code enums (PowerRabiErrorCode, QubitSpectroscopyErrorCode,
ResonatorSpectroscopyErrorCode, etc.) with statistically principled fit diagnosis
based on the Bayesian Information Criterion.

References:
    Schwarz, G. (1978). "Estimating the Dimension of a Model."
        Annals of Statistics, 6(2), 461–464. doi:10.1214/aos/1176344136
    Kass, R.E. & Raftery, A.E. (1995). "Bayes Factors."
        JASA 90(430), 773–795. doi:10.2307/2291091
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import scipy.optimize as opt
import scipy.stats as stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BIC evidence thresholds (Kass & Raftery 1995, Table 1)
# ΔBIC = BIC(null) - BIC(signal)  — positive means evidence for signal.
# ---------------------------------------------------------------------------
_EVIDENCE_THRESHOLDS = [
    (10.0, "strong"),
    (6.0, "moderate"),
    (2.0, "weak"),
    (float("-inf"), "none"),
]


def _evidence_strength(delta_bic: float) -> str:
    for threshold, label in _EVIDENCE_THRESHOLDS:
        if delta_bic >= threshold:
            return label
    return "none"


# ---------------------------------------------------------------------------
# BICResult
# ---------------------------------------------------------------------------


@dataclass
class BICResult:
    """
    Output of BICDiagnoser.diagnose().

    Attributes:
        winning_model: Name of the model with the lowest BIC score.
        delta_bic: BIC(null) - BIC(winning). Positive = evidence for signal.
        evidence_strength: 'none' | 'weak' | 'moderate' | 'strong'
            (thresholds: <2, 2–6, 6–10, >10 per Kass & Raftery 1995).
        model_bics: Dict mapping model name → BIC score.
        n_data: Number of data points.
        diagnosis: Human-readable summary string.
    """

    winning_model: str
    delta_bic: float
    evidence_strength: str
    model_bics: dict[str, float]
    n_data: int
    diagnosis: str

    def to_dict(self) -> dict:
        return {
            "winning_model": self.winning_model,
            "delta_bic": self.delta_bic,
            "evidence_strength": self.evidence_strength,
            "model_bics": self.model_bics,
            "n_data": self.n_data,
            "diagnosis": self.diagnosis,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BICResult":
        return cls(
            winning_model=d["winning_model"],
            delta_bic=float(d["delta_bic"]),
            evidence_strength=d["evidence_strength"],
            model_bics=d["model_bics"],
            n_data=int(d["n_data"]),
            diagnosis=d["diagnosis"],
        )


# ---------------------------------------------------------------------------
# BICModel protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BICModel(Protocol):
    """Interface for a fittable model used in BIC comparison."""

    name: str
    n_params: int

    def fit(self, x: np.ndarray, y: np.ndarray) -> dict:
        """Fit the model to data; return parameter dict."""
        ...

    def log_likelihood(self, x: np.ndarray, y: np.ndarray, params: dict) -> float:
        """Return log-likelihood under Gaussian noise (log-space, not raw)."""
        ...


# ---------------------------------------------------------------------------
# Concrete model implementations
# ---------------------------------------------------------------------------


def _gaussian_log_likelihood(residuals: np.ndarray) -> float:
    """
    Log-likelihood under i.i.d. Gaussian noise with MLE variance.

    LL = -n/2 * ln(2π σ²) - RSS/(2σ²)
    where σ² = RSS/n (MLE estimate).

    All computation stays in log-space to prevent numerical underflow.
    """
    n = len(residuals)
    rss = float(np.sum(residuals**2))
    if rss <= 0.0 or n <= 1:
        return 0.0
    sigma2 = rss / n
    ll = -n / 2.0 * math.log(2.0 * math.pi * sigma2) - rss / (2.0 * sigma2)
    return ll


class ConstantModel:
    """
    Null hypothesis: the data is constant plus Gaussian noise. k=1.

    Used as the reference model in all BIC comparisons.
    """

    name = "ConstantModel"
    n_params = 1

    def fit(self, x: np.ndarray, y: np.ndarray) -> dict:
        return {"offset": float(np.mean(y))}

    def log_likelihood(self, x: np.ndarray, y: np.ndarray, params: dict) -> float:
        residuals = y - params["offset"]
        return _gaussian_log_likelihood(residuals)


class LorentzianModel:
    """
    Lorentzian (dispersive) dip/peak for resonator/qubit spectroscopy. k=4.

    Model: y = offset + depth / (1 + ((x - center) / (width/2))^2)

    Parameters:
        center: Peak/dip centre frequency.
        width:  Full-width at half-maximum (FWHM).
        depth:  Signed amplitude (negative for dip, positive for peak).
        offset: Baseline offset.
    """

    name = "LorentzianModel"
    n_params = 4

    @staticmethod
    def _model(x: np.ndarray, center: float, width: float, depth: float, offset: float) -> np.ndarray:
        return offset + depth / (1.0 + ((x - center) / (width / 2.0 + 1e-30)) ** 2)

    def fit(self, x: np.ndarray, y: np.ndarray) -> dict:
        center0 = float(x[np.argmin(y)] if np.ptp(y) > 0 else np.mean(x))
        width0 = float(np.ptp(x) / 4.0)
        depth0 = float(np.min(y) - np.mean(y))
        offset0 = float(np.mean(y))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = opt.curve_fit(
                    self._model,
                    x,
                    y,
                    p0=[center0, width0, depth0, offset0],
                    maxfev=5000,
                )
            return {"center": popt[0], "width": popt[1], "depth": popt[2], "offset": popt[3]}
        except (RuntimeError, opt.OptimizeWarning, ValueError):
            return {"center": center0, "width": width0, "depth": depth0, "offset": offset0}

    def log_likelihood(self, x: np.ndarray, y: np.ndarray, params: dict) -> float:
        y_pred = self._model(x, params["center"], params["width"], params["depth"], params["offset"])
        return _gaussian_log_likelihood(y - y_pred)


class DampedCosineModel:
    """
    Damped cosine (Rabi oscillation). k=5.

    Model: y = amplitude * exp(-x / decay) * cos(2π * frequency * x + phase) + offset

    Parameters:
        amplitude: Oscillation amplitude.
        frequency: Oscillation frequency (cycles per unit x).
        decay:     Exponential decay constant (in x units).
        phase:     Phase offset (radians).
        offset:    DC offset.
    """

    name = "DampedCosineModel"
    n_params = 5

    @staticmethod
    def _model(
        x: np.ndarray,
        amplitude: float,
        frequency: float,
        decay: float,
        phase: float,
        offset: float,
    ) -> np.ndarray:
        decay_safe = abs(decay) + 1e-30
        return amplitude * np.exp(-x / decay_safe) * np.cos(2.0 * np.pi * frequency * x + phase) + offset

    def fit(self, x: np.ndarray, y: np.ndarray) -> dict:
        amp0 = float(np.ptp(y) / 2.0)
        # Rough frequency estimate via FFT
        if len(x) >= 4:
            dx = float(np.mean(np.diff(x)))
            freqs = np.fft.rfftfreq(len(x), d=dx)
            fft_mag = np.abs(np.fft.rfft(y - np.mean(y)))
            freq0 = float(freqs[np.argmax(fft_mag[1:]) + 1]) if len(freqs) > 1 else 1.0
        else:
            freq0 = 1.0
        decay0 = float(np.ptp(x))
        phase0 = 0.0
        offset0 = float(np.mean(y))
        p0 = [amp0, freq0, decay0, phase0, offset0]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = opt.curve_fit(
                    self._model,
                    x,
                    y,
                    p0=p0,
                    maxfev=10000,
                    bounds=(
                        [-np.inf, 1e-12, 1e-12, -2 * np.pi, -np.inf],
                        [np.inf, np.inf, np.inf, 2 * np.pi, np.inf],
                    ),
                )
            return {
                "amplitude": popt[0],
                "frequency": popt[1],
                "decay": popt[2],
                "phase": popt[3],
                "offset": popt[4],
            }
        except (RuntimeError, opt.OptimizeWarning, ValueError):
            return {
                "amplitude": amp0,
                "frequency": freq0,
                "decay": decay0,
                "phase": phase0,
                "offset": offset0,
            }

    def log_likelihood(self, x: np.ndarray, y: np.ndarray, params: dict) -> float:
        y_pred = self._model(
            x,
            params["amplitude"],
            params["frequency"],
            params["decay"],
            params["phase"],
            params["offset"],
        )
        return _gaussian_log_likelihood(y - y_pred)


class ExponentialDecayModel:
    """
    Exponential decay (T1 relaxation). k=3.

    Model: y = amplitude * exp(-x * decay_rate) + offset

    Parameters:
        amplitude:  Initial amplitude.
        decay_rate: Decay rate (1/T1 in units of 1/x).
        offset:     Baseline offset (should be ~0 for population).
    """

    name = "ExponentialDecayModel"
    n_params = 3

    @staticmethod
    def _model(x: np.ndarray, amplitude: float, decay_rate: float, offset: float) -> np.ndarray:
        decay_rate_safe = abs(decay_rate) + 1e-30
        return amplitude * np.exp(-x * decay_rate_safe) + offset

    def fit(self, x: np.ndarray, y: np.ndarray) -> dict:
        amp0 = float(np.ptp(y))
        # Rough decay rate: assume signal halves at midpoint of x range
        x_range = float(np.ptp(x)) + 1e-30
        rate0 = math.log(2.0) / (x_range / 2.0)
        offset0 = float(np.min(y))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = opt.curve_fit(
                    self._model,
                    x,
                    y,
                    p0=[amp0, rate0, offset0],
                    maxfev=5000,
                    bounds=([0, 1e-30, -np.inf], [np.inf, np.inf, np.inf]),
                )
            return {"amplitude": popt[0], "decay_rate": popt[1], "offset": popt[2]}
        except (RuntimeError, opt.OptimizeWarning, ValueError):
            return {"amplitude": amp0, "decay_rate": rate0, "offset": offset0}

    def log_likelihood(self, x: np.ndarray, y: np.ndarray, params: dict) -> float:
        y_pred = self._model(x, params["amplitude"], params["decay_rate"], params["offset"])
        return _gaussian_log_likelihood(y - y_pred)


# ---------------------------------------------------------------------------
# BICDiagnoser
# ---------------------------------------------------------------------------


class BICDiagnoser:
    """
    Runs BIC model selection for a given calibration node type.

    Replaces hand-coded error code enums (PowerRabiErrorCode,
    QubitSpectroscopyErrorCode, etc.) with statistically principled
    fit diagnosis based on BIC model comparison.

    BIC = -2 · LL + k · ln(n)
    The model with the *lowest* BIC is preferred.
    ΔBIC = BIC(null) - BIC(winner) — positive means evidence for signal.

    Usage:
        diagnoser = BICDiagnoser(node_type="power_rabi")
        result = diagnoser.diagnose(x_data, y_data)
        cost = diagnoser.to_bo_cost(result)   # 0=perfect, 1=complete failure

    References:
        Schwarz (1978); Kass & Raftery (1995).
    """

    MODEL_REGISTRY: dict[str, list] = {
        "resonator_spectroscopy": [ConstantModel(), LorentzianModel()],
        "resonator_punch_out": [ConstantModel(), LorentzianModel()],
        "chi_shift": [ConstantModel(), LorentzianModel()],
        "qubit_spectroscopy_vs_power": [ConstantModel(), LorentzianModel()],
        "power_rabi": [ConstantModel(), DampedCosineModel()],
        "time_rabi": [ConstantModel(), DampedCosineModel()],
        "t1": [ConstantModel(), ExponentialDecayModel()],
    }

    def __init__(self, node_type: str) -> None:
        if node_type not in self.MODEL_REGISTRY:
            available = list(self.MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown node_type '{node_type}'. Available: {available}"
            )
        self.node_type = node_type
        self.models = self.MODEL_REGISTRY[node_type]

    def _bic(self, model: BICModel, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute BIC for a model on data (x, y).

        BIC = -2 · log_likelihood + k · ln(n)

        All computation in log-space — never uses raw likelihoods.
        """
        n = len(y)
        params = model.fit(x, y)
        ll = model.log_likelihood(x, y, params)
        return -2.0 * ll + model.n_params * math.log(n)

    def diagnose(self, x: np.ndarray, y: np.ndarray) -> BICResult:
        """
        Run BIC model selection on data (x, y).

        Args:
            x: Sweep axis (frequency, time, amplitude, etc.).
            y: Measured signal (I/Q magnitude, state probability, etc.).

        Returns:
            BICResult with winning model, ΔBIC, evidence strength, and diagnosis.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(x) < 3:
            # Not enough data for meaningful model selection
            return BICResult(
                winning_model=self.models[0].name,
                delta_bic=0.0,
                evidence_strength="none",
                model_bics={m.name: float("nan") for m in self.models},
                n_data=len(x),
                diagnosis="insufficient_data",
            )

        bics: dict[str, float] = {}
        for model in self.models:
            try:
                bics[model.name] = self._bic(model, x, y)
            except Exception as e:
                logger.warning("BIC computation failed for %s: %s", model.name, e)
                bics[model.name] = float("inf")

        # Null model is always the first in the registry
        null_name = self.models[0].name
        null_bic = bics[null_name]

        # Winner = model with lowest BIC
        winner_name = min(bics, key=lambda k: bics[k])
        winner_bic = bics[winner_name]

        delta_bic = null_bic - winner_bic  # positive = evidence against null

        strength = _evidence_strength(delta_bic)

        # Human-readable diagnosis
        if winner_name == null_name:
            diagnosis = "no_signal_detected"
        elif strength == "none":
            diagnosis = "marginal_signal"
        elif strength == "weak":
            diagnosis = "weak_signal"
        elif strength == "moderate":
            diagnosis = "signal_found"
        else:
            diagnosis = "strong_signal_found"

        return BICResult(
            winning_model=winner_name,
            delta_bic=delta_bic,
            evidence_strength=strength,
            model_bics=bics,
            n_data=len(x),
            diagnosis=diagnosis,
        )

    def to_bo_cost(self, result: BICResult) -> float:
        """
        Convert BICResult to BO cost scalar in [0, 1].

        Mapping (smooth sigmoid on ΔBIC):
            cost = 0: strong signal found (ΔBIC ≥ 10)
            cost = 1: no signal or noise only (ΔBIC ≤ 0)

        Uses a logistic sigmoid shifted so that:
            ΔBIC = 0  → cost ≈ 1.0
            ΔBIC = 5  → cost ≈ 0.5
            ΔBIC = 10 → cost ≈ 0.0

        Never returns exactly 0 or 1 so the GP kernel always sees variability.
        """
        delta = result.delta_bic
        # Sigmoid: σ(z) = 1 / (1 + exp(z))
        # We want: cost ↓ as delta ↑, cost=0.5 at delta=5
        z = (delta - 5.0) / 2.0  # scale so ±4 covers 0.02–0.98
        cost = 1.0 / (1.0 + math.exp(z))
        # Clamp to (0.01, 0.99) to avoid degenerate GP observations
        return max(0.01, min(0.99, cost))
