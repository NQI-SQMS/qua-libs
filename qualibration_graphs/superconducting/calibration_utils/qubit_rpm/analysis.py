"""
Analysis utilities for the Rabi Population Measurement (RPM) experiment.

The RPM measures the qubit thermal population by comparing two EF Rabi sweeps:

  'g' sweep: ge_π → ef(a) → ef_π (back-swap) → ge_π → readout
             Starts from |g⟩ deterministically.
             P_g(a) = cos²(π·a/2) — oscillates 1→0→1, minimum at a=1.

  'e' sweep: ef(a) → ge_π → readout
             Starts from the thermal state (P_th in |e⟩, 1-P_th in |g⟩).
             P_e(a) = (1-P_th) + P_th·sin²(π·a/2)

Both sweeps are fitted independently with a free 4-parameter cosine.
The amplitudes A_g, A_e give:
    P_th = A_e / (A_e + A_g)

Errors on A_g and A_e come from the lmfit covariance and are propagated
through to P_th and T_eff.

Success is gated on the scale-normalised chi² of the 'g' sweep:
    chi2_g = SS_res / ((N-4)·A_g²) ≤ CHI2_THRESHOLD

References: internal RPM notebook (2026).
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
from lmfit import Model, Parameter

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


# ── Scale-normalised chi² threshold (same convention as power_rabi node) ──────
# chi2 = SS_res / ((N - 4) * A²).  Values > CHI2_THRESHOLD mean residuals
# dominate the signal — the oscillation was not resolved.
CHI2_THRESHOLD = 2.0

# When P_th is below this value the T_eff uncertainty formula diverges;
# we display the uncertainty as "> MAX" instead.
_P_TH_MIN_FOR_ERR = 1e-3   # 0.1 %


@dataclass
class FitParameters:
    """Fit results for a single qubit's RPM experiment."""

    # ── Amplitudes and their 1-σ uncertainties ──────────────────────────────
    amp_g: float
    """Rabi oscillation amplitude for the 'g' sweep."""
    amp_g_err: float = float("nan")
    """1-σ uncertainty on amp_g from the lmfit covariance."""
    amp_e: float = float("nan")
    """Rabi oscillation amplitude for the 'e' sweep."""
    amp_e_err: float = float("nan")
    """1-σ uncertainty on amp_e from the lmfit covariance."""

    # ── Derived quantities ──────────────────────────────────────────────────
    thermal_population: float = float("nan")
    """Estimated thermal population P_th = A_e / (A_e + A_g)."""
    thermal_population_err: float = float("nan")
    """1-σ uncertainty on P_th from error propagation."""
    effective_temperature_mk: float = float("nan")
    """Effective qubit temperature [mK] derived from P_th and qubit frequency."""
    effective_temperature_mk_err: float = float("nan")
    """1-σ uncertainty on T_eff [mK].  NaN when P_th is too small to be reliable."""

    # ── Fit quality ─────────────────────────────────────────────────────────
    chi2_g: float = float("nan")
    """Scale-normalised chi² for the 'g' sweep.  chi2_g > CHI2_THRESHOLD → FAIL."""
    chi2_e: float = float("nan")
    """Scale-normalised chi² for the 'e' sweep (informational; large when P_th≈0)."""

    # ── Curve-reconstruction parameters (for plotting) ─────────────────────
    freq_g: float = float("nan")
    phi_g: float = float("nan")
    offset_g: float = float("nan")
    freq_e: float = float("nan")
    phi_e: float = float("nan")
    offset_e: float = float("nan")

    success: bool = False


# ── Internal fit helpers ───────────────────────────────────────────────────────

def _cosine(x, A, f, phi, offset):
    return A * np.cos(2 * np.pi * f * x + phi) + offset


def _fft_seed(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (A0, f0, phi0, offset0) initial guesses from FFT."""
    y_sub = y - np.mean(y)
    N = len(x)
    T = float(x[1] - x[0])
    yf = np.fft.rfft(y_sub)
    xf = np.fft.rfftfreq(N, T)
    mags = np.abs(yf[1:])
    freqs = xf[1:]
    if len(freqs) == 0 or np.max(mags) == 0:
        return 0.0, 0.5, 0.0, float(np.mean(y))
    idx = int(np.argmax(mags))
    return (
        float(2 * mags[idx] / N),
        float(freqs[idx]),
        float(np.angle(yf[1:][idx])),
        float(np.mean(y)),
    )


def _fit_cosine_free(x: np.ndarray, y: np.ndarray):
    """Fit y = A·cos(2π·f·x + φ) + offset with all 4 parameters free.

    Returns an lmfit ModelResult, or None on failure.
    """
    if len(x) < 5:
        return None
    a0, f0, phi0, off0 = _fft_seed(x, y)
    freqs_all = np.fft.rfftfreq(len(x), x[1] - x[0])
    freq_min = float(freqs_all[1]) * 0.3
    freq_max = float(freqs_all[-1]) * 3.0
    model = Model(_cosine, independent_vars=["x"])
    try:
        result = model.fit(
            y, x=x,
            A=Parameter("A", value=a0, min=0),
            f=Parameter("f", value=f0, min=freq_min, max=freq_max),
            phi=Parameter("phi", value=phi0),
            offset=Parameter("offset", value=off0),
        )
        return result
    except Exception:
        return None


def _stderr(result, name: str) -> float:
    """Return the 1-σ uncertainty of a parameter, or NaN if unavailable."""
    if result is None or result.params[name].stderr is None:
        return float("nan")
    return abs(float(result.params[name].stderr))


def _chi2_norm(x: np.ndarray, y: np.ndarray, A: float, f: float, phi: float,
               offset: float, n_free: int = 4) -> float:
    """chi2 = SS_res / ((N - n_free) * A²).  Returns inf if A ≤ 0."""
    if A <= 0 or len(x) <= n_free:
        return float("inf")
    residuals = y - _cosine(x, A, f, phi, offset)
    SS_res = float(np.sum(residuals ** 2))
    return SS_res / ((len(x) - n_free) * A ** 2)


# ── Temperature conversion ─────────────────────────────────────────────────────

def _effective_temperature_mk(freq_hz: float, p_th: float) -> float:
    """Convert thermal population to effective temperature in mK."""
    if p_th <= 0 or p_th >= 1:
        return float("nan")
    return (47.99 * freq_hz / 1e9) / np.log(1.0 / p_th - 1.0)


def _effective_temperature_mk_err(freq_hz: float, p_th: float, p_th_err: float) -> float:
    """1-σ uncertainty on T_eff [mK] via error propagation.

    From T = C / ln((1-P)/P):
        dT/dP = T² / (C · P · (1-P))
    Returns NaN when P_th < _P_TH_MIN_FOR_ERR (formula diverges).
    """
    if p_th < _P_TH_MIN_FOR_ERR or p_th >= 1 or not np.isfinite(p_th_err):
        return float("nan")
    t_eff = _effective_temperature_mk(freq_hz, p_th)
    if not np.isfinite(t_eff) or t_eff <= 0:
        return float("nan")
    C = 47.99 * freq_hz / 1e9
    return t_eff ** 2 * p_th_err / (C * p_th * (1.0 - p_th))


# ── Public API ────────────────────────────────────────────────────────────────

def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        p = res.get("thermal_population", float("nan"))
        p_err = res.get("thermal_population_err", float("nan"))
        t = res.get("effective_temperature_mk", float("nan"))
        t_err = res.get("effective_temperature_mk_err", float("nan"))
        chi2_g = res.get("chi2_g", float("nan"))
        chi2_e = res.get("chi2_e", float("nan"))

        p_str = (
            f"({100*p:.3f} ± {100*p_err:.3f}) %" if np.isfinite(p_err)
            else f"{100*p:.3f} %"
        )
        t_str = (
            f"({t:.1f} ± {t_err:.1f}) mK" if np.isfinite(t_err)
            else (f"< {t:.1f} mK (P_th too small for reliable err)" if np.isfinite(t) else "N/A")
        )
        log_callable(
            f"RPM results for qubit {q}: {status}\n"
            f"\tA_g = {res['amp_g']:.4f} ± {res.get('amp_g_err', float('nan')):.4f}\n"
            f"\tA_e = {res['amp_e']:.4f} ± {res.get('amp_e_err', float('nan')):.4f}\n"
            f"\tchi2_g = {chi2_g:.3f}  (threshold {CHI2_THRESHOLD})\n"
            f"\tchi2_e = {chi2_e:.3f}  (informational)\n"
            f"\tP_th = {p_str}\n"
            f"\tT_eff = {t_str}"
        )


def process_raw_datasets(
    ds_g: xr.Dataset,
    ds_e: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Convert I/Q to Volts for both datasets if needed."""
    if not node.parameters.use_state_discrimination:
        ds_g = convert_IQ_to_V(ds_g, node.namespace["qubits"])
        ds_e = convert_IQ_to_V(ds_e, node.namespace["qubits"])
    return ds_g, ds_e


def fit_raw_data(
    ds_g: xr.Dataset,
    ds_e: xr.Dataset,
    node: QualibrationNode,
) -> Dict[str, FitParameters]:
    """Fit both sweeps independently and compute thermal population for each qubit.

    Both the 'g' and 'e' sweeps are fitted with a free 4-parameter cosine.
    Success is gated on the scale-normalised chi² of the 'g' sweep
    (chi2_g > CHI2_THRESHOLD → oscillation not resolved → FAIL).
    """
    qubits = node.namespace["qubits"]
    signal = "state" if node.parameters.use_state_discrimination else "I"
    x = ds_g.amp_factor.values.astype(float)

    fit_results = {}
    for qubit in qubits:
        q = qubit.name
        y_g = getattr(ds_g.sel(qubit=q), signal).values.astype(float)
        y_e = getattr(ds_e.sel(qubit=q), signal).values.astype(float)

        # ── 'g' sweep: free fit ────────────────────────────────────────────
        res_g = _fit_cosine_free(x, y_g)
        if res_g is None or not res_g.success:
            fit_results[q] = FitParameters(amp_g=float("nan"), success=False)
            continue

        A_g   = abs(float(res_g.values["A"]))
        A_g_err = _stderr(res_g, "A")
        f_g   = float(res_g.values["f"])
        phi_g = float(res_g.values["phi"])
        off_g = float(res_g.values["offset"])
        c2_g  = _chi2_norm(x, y_g, A_g, f_g, phi_g, off_g)

        # ── 'e' sweep: free fit ────────────────────────────────────────────
        res_e = _fit_cosine_free(x, y_e)
        if res_e is None:
            fit_results[q] = FitParameters(
                amp_g=A_g, amp_g_err=A_g_err,
                freq_g=f_g, phi_g=phi_g, offset_g=off_g,
                chi2_g=c2_g,
                success=False,
            )
            continue

        A_e   = abs(float(res_e.values["A"]))
        A_e_err = _stderr(res_e, "A")
        f_e   = float(res_e.values["f"])
        phi_e = float(res_e.values["phi"])
        off_e = float(res_e.values["offset"])
        c2_e  = _chi2_norm(x, y_e, A_e, f_e, phi_e, off_e)

        # ── Thermal population with error propagation ──────────────────────
        den = A_e + A_g
        if den > 0:
            p_th = A_e / den
            if np.isfinite(A_e_err) and np.isfinite(A_g_err):
                # dP/dAe = Ag/den², dP/dAg = -Ae/den²
                p_th_err = np.sqrt(
                    (A_g / den ** 2 * A_e_err) ** 2 +
                    (A_e / den ** 2 * A_g_err) ** 2
                )
            else:
                p_th_err = float("nan")
        else:
            p_th = float("nan")
            p_th_err = float("nan")

        t_eff     = _effective_temperature_mk(qubit.f_01, p_th)
        t_eff_err = _effective_temperature_mk_err(qubit.f_01, p_th, p_th_err)

        # ── Success criterion: g-oscillation resolved ──────────────────────
        success = (
            c2_g <= CHI2_THRESHOLD
            and res_e.success
            and A_g > 0
            and A_e >= 0
            and np.isfinite(p_th)
            and 0.0 <= p_th < 1.0
        )

        fit_results[q] = FitParameters(
            amp_g=A_g, amp_g_err=A_g_err,
            amp_e=A_e, amp_e_err=A_e_err,
            thermal_population=p_th,
            thermal_population_err=p_th_err,
            effective_temperature_mk=t_eff,
            effective_temperature_mk_err=t_eff_err,
            chi2_g=c2_g,
            chi2_e=c2_e,
            freq_g=f_g, phi_g=phi_g, offset_g=off_g,
            freq_e=f_e, phi_e=phi_e, offset_e=off_e,
            success=success,
        )

    return fit_results
