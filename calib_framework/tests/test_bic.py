"""Unit tests for BICDiagnoser — no hardware required.

Tests verify that BIC model selection correctly identifies signals in synthetic
data and produces monotone cost functions as required by GP-BO.

All data generated with fixed random seeds for reproducibility.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from calib_framework.core.bic import (
    BICDiagnoser,
    BICResult,
    ConstantModel,
    DampedCosineModel,
    ExponentialDecayModel,
    LorentzianModel,
    _evidence_strength,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_lorentzian(n: int = 100, snr: float = 10.0, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) with a clear Lorentzian dip."""
    rng = _rng(seed)
    x = np.linspace(-50e6, 50e6, n)  # ±50 MHz sweep
    center = 0.0
    width = 5e6  # 5 MHz FWHM
    depth = -0.5  # negative = dip
    offset = 1.0
    y_clean = offset + depth / (1.0 + ((x - center) / (width / 2.0)) ** 2)
    noise_std = abs(depth) / snr
    y = y_clean + rng.normal(0, noise_std, size=n)
    return x, y


def make_rabi(n: int = 100, snr: float = 10.0, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) with a clean Rabi oscillation."""
    rng = _rng(seed)
    x = np.linspace(0, 2 * np.pi, n)
    amplitude = 0.5
    frequency = 1.0 / (2 * np.pi)  # 1 period in [0, 2π]
    decay = 1e6  # very slow decay (nearly undamped)
    offset = 0.0
    y_clean = amplitude * np.exp(-x / decay) * np.cos(2 * np.pi * frequency * x) + offset
    noise_std = amplitude / snr
    y = y_clean + rng.normal(0, noise_std, size=n)
    return x, y


def make_noise(n: int = 80, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) with pure Gaussian noise."""
    rng = _rng(seed)
    x = np.linspace(0, 1, n)
    y = rng.normal(0.5, 0.1, size=n)
    return x, y


# ---------------------------------------------------------------------------
# Test 1: ConstantModel wins on pure noise
# ---------------------------------------------------------------------------


def test_constant_wins_on_noise():
    """ConstantModel should win when data is pure Gaussian noise."""
    x, y = make_noise()
    diagnoser = BICDiagnoser("resonator_spectroscopy")
    result = diagnoser.diagnose(x, y)

    assert result.winning_model == ConstantModel.name, (
        f"Expected ConstantModel to win on noise, got {result.winning_model}. "
        f"ΔBIC={result.delta_bic:.2f}"
    )
    assert result.evidence_strength in {"none", "weak"}, (
        f"Expected no/weak evidence on noise, got '{result.evidence_strength}'"
    )


# ---------------------------------------------------------------------------
# Test 2: LorentzianModel wins on clear peak
# ---------------------------------------------------------------------------


def test_lorentzian_wins_on_clear_peak():
    """LorentzianModel must win with strong evidence (ΔBIC > 10) on a clear peak."""
    x, y = make_lorentzian(snr=20.0)
    diagnoser = BICDiagnoser("resonator_spectroscopy")
    result = diagnoser.diagnose(x, y)

    assert result.winning_model == LorentzianModel.name, (
        f"Expected LorentzianModel to win, got {result.winning_model}. "
        f"ΔBIC={result.delta_bic:.2f}"
    )
    assert result.delta_bic > 10.0, (
        f"Expected ΔBIC > 10 for clear Lorentzian, got {result.delta_bic:.2f}"
    )
    assert result.evidence_strength == "strong", (
        f"Expected 'strong' evidence, got '{result.evidence_strength}'"
    )


# ---------------------------------------------------------------------------
# Test 3: DampedCosineModel wins on Rabi oscillation
# ---------------------------------------------------------------------------


def test_damped_cosine_wins_on_rabi():
    """DampedCosineModel must win on a synthetic Rabi oscillation."""
    x, y = make_rabi(snr=15.0)
    diagnoser = BICDiagnoser("power_rabi")
    result = diagnoser.diagnose(x, y)

    assert result.winning_model == DampedCosineModel.name, (
        f"Expected DampedCosineModel to win, got {result.winning_model}. "
        f"ΔBIC={result.delta_bic:.2f}"
    )
    assert result.delta_bic > 6.0, (
        f"Expected ΔBIC > 6 for clear Rabi, got {result.delta_bic:.2f}"
    )
    assert result.evidence_strength in {"moderate", "strong"}, (
        f"Expected moderate/strong evidence, got '{result.evidence_strength}'"
    )


# ---------------------------------------------------------------------------
# Test 4: Low SNR → weak or no evidence
# ---------------------------------------------------------------------------


def test_low_snr_weak_evidence():
    """Very noisy Lorentzian should yield 'weak' or 'none' evidence strength."""
    x, y = make_lorentzian(snr=1.0)  # SNR=1: completely noise-dominated
    diagnoser = BICDiagnoser("resonator_spectroscopy")
    result = diagnoser.diagnose(x, y)

    assert result.evidence_strength in {"none", "weak"}, (
        f"Expected none/weak evidence at SNR=1, got '{result.evidence_strength}'. "
        f"ΔBIC={result.delta_bic:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 5: to_bo_cost is monotonically decreasing in ΔBIC
# ---------------------------------------------------------------------------


def test_cost_monotone():
    """
    to_bo_cost must be monotonically decreasing in ΔBIC.

    A higher ΔBIC (more evidence for signal) should map to a lower BO cost
    (better calibration outcome).
    """
    diagnoser = BICDiagnoser("power_rabi")

    delta_bics = np.linspace(-5.0, 20.0, 50)
    x_dummy = np.linspace(0, 1, 30)
    y_dummy = np.zeros(30)

    costs = []
    for delta in delta_bics:
        result = BICResult(
            winning_model="DampedCosineModel",
            delta_bic=float(delta),
            evidence_strength=_evidence_strength(float(delta)),
            model_bics={},
            n_data=30,
            diagnosis="test",
        )
        costs.append(diagnoser.to_bo_cost(result))

    # Verify monotonically decreasing (allow for floating-point ties)
    for i in range(len(costs) - 1):
        assert costs[i] >= costs[i + 1] - 1e-9, (
            f"to_bo_cost not monotone: cost[{i}]={costs[i]:.4f} > cost[{i+1}]={costs[i+1]:.4f} "
            f"at ΔBIC={delta_bics[i]:.2f}→{delta_bics[i+1]:.2f}"
        )

    # Boundary checks
    assert costs[0] > 0.8, f"Cost at ΔBIC=-5 should be near 1.0, got {costs[0]:.3f}"
    assert costs[-1] < 0.2, f"Cost at ΔBIC=20 should be near 0.0, got {costs[-1]:.3f}"


# ---------------------------------------------------------------------------
# Additional: BICResult serialisation round-trip
# ---------------------------------------------------------------------------


def test_bic_result_serialisation():
    """BICResult should serialise and deserialise correctly."""
    x, y = make_lorentzian()
    diagnoser = BICDiagnoser("resonator_spectroscopy")
    result = diagnoser.diagnose(x, y)

    d = result.to_dict()
    result2 = BICResult.from_dict(d)

    assert result2.winning_model == result.winning_model
    assert abs(result2.delta_bic - result.delta_bic) < 1e-9
    assert result2.evidence_strength == result.evidence_strength
    assert result2.n_data == result.n_data


# ---------------------------------------------------------------------------
# Additional: evidence strength thresholds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "delta_bic, expected",
    [
        (12.0, "strong"),
        (8.0, "moderate"),
        (4.0, "weak"),
        (1.0, "none"),
        (-5.0, "none"),
    ],
)
def test_evidence_thresholds(delta_bic: float, expected: str):
    """Evidence strength thresholds match Kass & Raftery (1995) Table 1."""
    assert _evidence_strength(delta_bic) == expected


# ---------------------------------------------------------------------------
# Additional: insufficient data fallback
# ---------------------------------------------------------------------------


def test_insufficient_data():
    """diagnose() should return 'none' evidence with fewer than 3 data points."""
    diagnoser = BICDiagnoser("power_rabi")
    result = diagnoser.diagnose(np.array([0.0, 1.0]), np.array([0.5, 0.6]))
    assert result.evidence_strength == "none"
    assert result.diagnosis == "insufficient_data"
