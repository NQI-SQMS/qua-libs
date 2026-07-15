"""Analysis routines for the parity-time calibration (node 28).

Physics
-------
In the dispersive regime, a qubit coupled to a cavity mode with shift χ_eff
accumulates phase  n · χ_eff · τ  during a wait of τ nanoseconds when the
cavity contains n photons.  The Wigner-tomography parity measurement requires
this phase to equal nπ, giving:

    τ_parity = π / χ_eff = 1 / (2 · f_χ)

where f_χ = χ_eff / (2π) is the dispersive oscillation frequency.

Protocol
--------
With Fock |1⟩ in the cavity the qubit sees a detuning of exactly χ_eff from
its bare GE frequency, so P(e) follows a damped cosine at f_χ.  The analysis
uses FFT to find the initial frequency guess, then fits a damped cosine:

    P(e)(τ) = A · exp(−τ/T₂) · cos(2π · f · τ + φ) + offset

to extract f_χ precisely.  The FFT result is used as fallback when the fit
does not converge.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibration_libs.data import convert_IQ_to_V
try:
    from qualibration_libs.data import apply_confusion_correction_to_dataset
except ImportError:
    def apply_confusion_correction_to_dataset(ds, node):
        raise NotImplementedError("apply_confusion_correction_to_dataset not available in this qualibration_libs version")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FFT-based frequency extraction (used as initial guess for the fit)
# ---------------------------------------------------------------------------

def _fft_extract_frequency(
    tau_ns: np.ndarray,
    signal: np.ndarray,
) -> Tuple[float, float, bool]:
    """Extract dominant oscillation frequency via FFT peak.

    Returns
    -------
    freq_hz, amplitude, success
    """
    if len(tau_ns) < 8 or not np.any(np.isfinite(signal)):
        return np.nan, np.nan, False

    sig = np.where(np.isfinite(signal), signal, 0.0)
    sig_centered = sig - np.mean(sig)

    dt_ns = float(tau_ns[1] - tau_ns[0]) if len(tau_ns) > 1 else 16.0
    fft_mag = np.abs(np.fft.rfft(sig_centered))
    freqs_hz = np.fft.rfftfreq(len(sig_centered), d=dt_ns * 1e-9)

    peak_idx = int(np.argmax(fft_mag[1:])) + 1  # skip DC
    freq_hz = float(abs(freqs_hz[peak_idx]))
    amplitude = float(fft_mag[peak_idx]) * 2.0 / len(sig)

    if freq_hz <= 0:
        return np.nan, np.nan, False

    return freq_hz, amplitude, True


# ---------------------------------------------------------------------------
# Damped cosine fit
# ---------------------------------------------------------------------------

def _damped_cosine(t_s: np.ndarray, A: float, T2: float, f: float, phi: float, offset: float) -> np.ndarray:
    return A * np.exp(-t_s / T2) * np.cos(2 * np.pi * f * t_s + phi) + offset


def _fit_damped_cosine(
    tau_ns: np.ndarray,
    signal: np.ndarray,
    freq_guess_hz: float,
) -> Tuple[float, float, float, np.ndarray, bool]:
    """Fit a damped cosine to extract f_chi precisely.

    Returns
    -------
    freq_hz, freq_err_hz, T2_s, fit_curve, success
    """
    tau_s = tau_ns * 1e-9
    offset_guess = float(np.mean(signal))
    amp_guess = float((np.max(signal) - np.min(signal)) / 2)
    T2_guess = float(tau_s[-1] / 2)

    p0 = [amp_guess, T2_guess, freq_guess_hz, 0.0, offset_guess]
    bounds = (
        [0,          tau_s[1],  freq_guess_hz * 0.5, -np.pi, 0.0],
        [1,          tau_s[-1] * 5, freq_guess_hz * 2.0,  np.pi, 1.0],
    )

    try:
        popt, pcov = curve_fit(
            _damped_cosine, tau_s, signal,
            p0=p0, bounds=bounds,
            maxfev=10000,
        )
        perr = np.sqrt(np.diag(pcov))
        freq_hz = float(popt[2])
        freq_err_hz = float(perr[2])
        T2_s = float(popt[1])
        fit_curve = _damped_cosine(tau_s, *popt)
        # Sanity: fitted frequency should be within 30% of FFT guess
        if abs(freq_hz - freq_guess_hz) / freq_guess_hz > 0.3:
            return np.nan, np.nan, np.nan, np.array([]), False
        return freq_hz, freq_err_hz, T2_s, fit_curve, True
    except Exception:
        return np.nan, np.nan, np.nan, np.array([]), False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParityTimeFit:
    """Results for one qubit from the parity-time measurement."""

    success: bool
    """True when a clear frequency was extracted."""

    parity_time_s: float
    """Experimentally calibrated parity time τ_parity [seconds].
    τ_parity = 1 / (2 · f_χ) = π / χ_eff."""

    chi_eff_hz: float
    """Per-photon qubit frequency shift [Hz] from damped cosine fit (FFT fallback).
    Use -chi_eff_hz to update pair.chi (full shift, negative convention)."""

    chi_eff_err_hz: float = np.nan
    """1-sigma uncertainty on chi_eff_hz from the fit covariance [Hz].
    nan when FFT fallback was used."""

    T2_s: float = np.nan
    """Fitted T2 decay constant [seconds]. nan when FFT fallback was used."""

    amplitude: float = np.nan
    """Fitted oscillation amplitude."""

    fit_curve: np.ndarray = None
    """Fitted P(e) curve on the same tau grid, for plotting. None when not available."""

    used_fft_fallback: bool = False
    """True when the damped cosine fit failed and FFT peak was used instead."""

    message: str = ""
    """Human-readable status message."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_raw_dataset(dataset: xr.Dataset, node) -> xr.Dataset:
    """Convert IQ to volts (when not using state discrimination) or apply confusion correction."""
    from qualibration_libs.parameters import get_qubits
    qubits = list(get_qubits(node))
    if not node.parameters.use_state_discrimination:
        dataset = convert_IQ_to_V(dataset, qubits=qubits)
    else:
        dataset = apply_confusion_correction_to_dataset(dataset, node)
    return dataset


def fit_raw_data(
    dataset: xr.Dataset,
    node,
) -> Tuple[xr.Dataset, Dict[str, ParityTimeFit]]:
    """Fit P_e(τ) for each qubit and extract χ_eff and τ_parity.

    Parameters
    ----------
    dataset : xr.Dataset
        Raw dataset with dimension ``delay`` [ns].
    node    : QualibrationNode
        Active node (provides parameters and namespace).

    Returns
    -------
    dataset      : xr.Dataset   Input dataset (unchanged).
    fit_results  : dict[str, ParityTimeFit]
    """
    from qualibration_libs.parameters import get_qubits

    qubits = get_qubits(node)
    tau_ns = np.asarray(dataset.coords["delay"].values, dtype=float)
    n_tau = len(tau_ns)

    use_state = node.parameters.use_state_discrimination

    fit_results: Dict[str, ParityTimeFit] = {}

    for i, qubit in enumerate(qubits):
        # ── Extract signal ────────────────────────────────────────────────────
        candidates = (
            ["state", f"state{i + 1}"] if use_state else ["I", f"I{i + 1}"]
        )
        raw = None
        for key in candidates:
            if key in dataset:
                try:
                    da = dataset[key]
                    raw = (
                        da.sel(qubit=qubit.name).values
                        if "qubit" in da.dims
                        else da.values
                    )
                    break
                except Exception:
                    pass

        if raw is None:
            logger.warning("No signal found for qubit %s — skipping.", qubit.name)
            fit_results[qubit.name] = ParityTimeFit(
                success=False,
                parity_time_s=np.nan,
                chi_eff_hz=np.nan,
                amplitude=np.nan,
                fft_freqs_hz=np.array([]),
                fft_mag=np.array([]),
                message="No signal data found in dataset.",
            )
            continue

        sig = np.asarray(raw, dtype=float).ravel()[:n_tau]

        # ── Step 1: FFT to get initial frequency guess ────────────────────────
        fft_freq_hz, fft_amp, fft_ok = _fft_extract_frequency(tau_ns, sig)

        if not fft_ok:
            fit_results[qubit.name] = ParityTimeFit(
                success=False,
                parity_time_s=np.nan,
                chi_eff_hz=np.nan,
                amplitude=np.nan,
                message="FFT peak not found.",
            )
            continue

        # ── Step 2: damped cosine fit using FFT freq as guess ─────────────────
        fit_freq_hz, fit_err_hz, T2_s, fit_curve, fit_ok = _fit_damped_cosine(
            tau_ns, sig, fft_freq_hz
        )

        if fit_ok:
            freq_hz = fit_freq_hz
            used_fft = False
            message = (
                f"Damped cosine fit: f_chi = {freq_hz / 1e3:.2f} ± {fit_err_hz / 1e3:.2f} kHz  "
                f"T2 = {T2_s * 1e6:.1f} µs  "
                f"→  τ_parity = {1e9 / (2 * freq_hz):.0f} ns"
            )
        else:
            freq_hz = fft_freq_hz
            fit_err_hz = np.nan
            T2_s = np.nan
            fit_curve = None
            used_fft = True
            message = (
                f"Fit failed — FFT fallback: f_chi = {freq_hz / 1e3:.1f} kHz  "
                f"→  τ_parity = {1e9 / (2 * freq_hz):.0f} ns"
            )
            logger.warning("Damped cosine fit failed for %s, using FFT peak.", qubit.name)

        parity_time_s = 1.0 / (2.0 * freq_hz)
        fit_results[qubit.name] = ParityTimeFit(
            success=True,
            parity_time_s=parity_time_s,
            chi_eff_hz=freq_hz,
            chi_eff_err_hz=fit_err_hz,
            T2_s=T2_s,
            amplitude=fft_amp,
            fit_curve=fit_curve,
            used_fft_fallback=used_fft,
            message=message,
        )

    return dataset, fit_results


def log_fitted_results(
    fit_results: Dict[str, ParityTimeFit],
    log_callable=None,
) -> None:
    """Log parity time and χ_eff to console / node log."""
    if log_callable is None:
        log_callable = logger.info
    for qname, res in fit_results.items():
        if not res.success:
            log_callable(
                f"{qname}: parity-time fit FAILED — {res.message}"
            )
            continue
        method = "FFT" if res.used_fft_fallback else "fit"
        err_str = f" ± {res.chi_eff_err_hz / 1e3:.2f} kHz" if not np.isnan(res.chi_eff_err_hz) else ""
        T2_str = f"  T2 = {res.T2_s * 1e6:.1f} µs" if not np.isnan(res.T2_s) else ""
        log_callable(
            f"{qname} [{method}]: χ_eff/(2π) = {res.chi_eff_hz / 1e3:.2f}{err_str} kHz"
            f"{T2_str}  |  τ_parity = {res.parity_time_s * 1e9:.0f} ns"
        )
