"""Analysis routines for the Ramsey Stark-shift chi calibration (node 25).

Physics recap
-------------
In the dispersive regime, the qubit frequency is shifted by

    Δω_q = 2χ n̄_ss

when the coupled storage cavity is driven to a steady-state population of
n̄_ss photons.  For a CW drive at amplitude A (dimensionless QUA scale),
n̄_ss ∝ A².  By fitting the Ramsey oscillation frequency f_R(A) for several
drive amplitudes and forming Δf(A) = f_R(A) - f_R(0), we extract:

    Δf = (2χ · dn̄_ss/dA²) · A²   =>   slope = 2χ · (dn̄_ss/dA²)

If displacement_k is calibrated (n̄_pulsed = displacement_k · A²), we use it
as a proxy for (dn̄_ss/dA²) to obtain χ in absolute units.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

_TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Damped-cosine model
# ---------------------------------------------------------------------------

def _damped_cosine(
    t: np.ndarray,
    amplitude: float,
    T2: float,
    frequency: float,
    phase: float,
    offset: float,
) -> np.ndarray:
    """Exponentially damped cosine: A·exp(-t/T2)·cos(2π f t + φ) + c."""
    return amplitude * np.exp(-t / T2) * np.cos(_TWO_PI * frequency * t + phase) + offset


def _fit_ramsey_oscillation(
    tau_ns: np.ndarray,
    signal: np.ndarray,
) -> Tuple[float, float, bool]:
    """Fit a damped cosine to a single Ramsey trace.

    Returns
    -------
    frequency_hz : float   Fitted oscillation frequency [Hz]; NaN on failure.
    T2_ns        : float   Fitted decay time [ns]; NaN on failure.
    success      : bool
    """
    if len(tau_ns) < 5 or not np.any(np.isfinite(signal)):
        return np.nan, np.nan, False

    sig = np.where(np.isfinite(signal), signal, 0.0)

    # Initial guess via FFT
    dt_ns = float(tau_ns[1] - tau_ns[0]) if len(tau_ns) > 1 else 16.0
    fft_mag = np.abs(np.fft.rfft(sig - np.mean(sig)))
    freqs_hz = np.fft.rfftfreq(len(sig), d=dt_ns * 1e-9)
    # Skip DC (index 0)
    peak_idx = int(np.argmax(fft_mag[1:])) + 1
    f0 = float(abs(freqs_hz[peak_idx])) if peak_idx > 0 else 1e5

    A0 = (np.max(sig) - np.min(sig)) / 2.0
    T2_0 = float(tau_ns[-1]) / 2.0
    c0 = float(np.mean(sig))

    try:
        popt, _ = curve_fit(
            _damped_cosine,
            tau_ns,
            sig,
            p0=[A0, T2_0, f0, 0.0, c0],
            bounds=(
                [0.0,    10.0, 0.0,   -np.pi, -np.inf],
                [np.inf, np.inf, 1e9,  np.pi,  np.inf],
            ),
            maxfev=8000,
        )
        freq_hz = float(popt[2])
        T2_ns = float(popt[1])
        return freq_hz, T2_ns, True
    except Exception as exc:
        logger.debug("Ramsey fit failed: %s", exc)
        return np.nan, np.nan, False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RamseyStarkFit:
    """Fitted results for one qubit from the Ramsey Stark experiment."""

    amplitudes: List[float]
    """Cavity CW drive amplitudes swept."""
    ramsey_freq_hz: List[float]
    """Absolute Ramsey oscillation frequency for each amplitude [Hz]."""
    ramsey_T2_ns: List[float]
    """Fitted T2* for each amplitude [ns]."""
    delta_freq_hz: List[float]
    """Frequency shift Δf(A) = f_R(A) - f_R(0) for each amplitude [Hz]."""
    chi_slope_hz_per_amp2: float
    """Slope of the Δf vs A² linear fit [Hz / (amp scale)²].
    Equals 2χ · (dn̄_ss/dA²) in the dispersive model."""
    chi_hz: Optional[float]
    """Dispersive shift χ [Hz] extracted via displacement_k.
    None when displacement_k is not calibrated in the machine state."""
    success: bool
    """True when the Ramsey fits and the χ slope fit all converged."""
    fit_residual_hz: float = 0.0
    """RMS residual of the Δf vs A² linear fit [Hz]."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_raw_dataset(dataset, node):
    """Return the dataset unchanged (no additional processing needed)."""
    return dataset


def fit_raw_data(dataset, node) -> Tuple[object, Dict[str, RamseyStarkFit]]:
    """Fit Ramsey oscillations for every amplitude and extract χ.

    Parameters
    ----------
    dataset : xarray.Dataset
        Raw dataset with dims ``(qubit, drive_amplitude, delay)``.
    node    : QualibrationNode
        Active node (used for parameters and namespace).

    Returns
    -------
    dataset      : xarray.Dataset  (unchanged)
    fit_results  : dict[qubit_name -> RamseyStarkFit]
    """
    from qualibration_libs.parameters import get_qubits

    qubits = get_qubits(node)
    amplitudes = np.asarray(dataset.coords["drive_amplitude"].values, dtype=float)
    tau_ns = np.asarray(dataset.coords["delay"].values, dtype=float)

    # Retrieve displacement_k from machine state (best-effort)
    cavity_mode = node.namespace.get("cavity_mode")
    displacement_k: Optional[float] = None
    if cavity_mode is not None:
        dk = getattr(cavity_mode, "displacement_k", None)
        if dk is None:
            mode_name = node.parameters.mode_name
            pairs = getattr(node.machine, "cavity_transmon_pairs", {})
            for pk, pair in pairs.items():
                if pk.endswith(f"_{mode_name}"):
                    dk = getattr(pair, "displacement_k", None)
                    break
        if dk is not None and float(dk) > 0:
            displacement_k = float(dk)

    fit_results: Dict[str, RamseyStarkFit] = {}

    for i, qubit in enumerate(qubits):
        # ── Select signal array (n_amp × n_tau) ──────────────────────────────
        use_state = node.parameters.use_state_discrimination
        # XarrayDataFetcher groups state1/state2/... → "state" (with qubit dim)
        # and I1/I2/... → "I" (with qubit dim). Fall back to indexed keys for
        # datasets loaded from older runs.
        candidates = (
            ["state", f"state{i + 1}"]
            if use_state
            else ["I", f"I{i + 1}"]
        )
        raw = None
        for key in candidates:
            if key in dataset:
                try:
                    da = dataset[key]
                    if "qubit" in da.dims:
                        raw = da.sel(qubit=qubit.name).values
                    else:
                        raw = da.values
                    break
                except Exception:
                    pass
        if raw is None:
            logger.warning("No signal found for qubit %s", qubit.name)
            fit_results[qubit.name] = RamseyStarkFit(
                amplitudes=amplitudes.tolist(),
                ramsey_freq_hz=[np.nan] * len(amplitudes),
                ramsey_T2_ns=[np.nan] * len(amplitudes),
                delta_freq_hz=[np.nan] * len(amplitudes),
                chi_slope_hz_per_amp2=np.nan,
                chi_hz=None,
                success=False,
            )
            continue

        # raw shape: (n_amp, n_tau)
        freqs_hz: List[float] = []
        T2s_ns: List[float] = []
        for j in range(len(amplitudes)):
            sig = raw[j, :]
            f_hz, t2, ok = _fit_ramsey_oscillation(tau_ns, sig)
            freqs_hz.append(f_hz)
            T2s_ns.append(t2)

        freqs = np.array(freqs_hz)

        # ── Frequency shifts relative to A=0 ─────────────────────────────────
        f_ref = freqs[0] if np.isfinite(freqs[0]) else np.nanmedian(freqs)
        delta_f = freqs - f_ref  # Δf(A) = f_R(A) - f_R(0)

        # ── Linear fit: Δf = slope × A² ──────────────────────────────────────
        amp_sq = amplitudes ** 2
        valid = np.isfinite(delta_f)
        chi_slope = np.nan
        chi_hz: Optional[float] = None
        residual = np.nan
        success = False

        if np.sum(valid) >= 2:
            try:
                # Force-through-origin least squares: Δf = slope × A²
                X = amp_sq[valid].reshape(-1, 1)
                y = delta_f[valid]
                coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                chi_slope = float(coef[0])
                residual = float(np.sqrt(np.mean((y - chi_slope * X.ravel()) ** 2)))
                success = True

                if displacement_k is not None:
                    # χ = slope / (2 × displacement_k)
                    # where displacement_k ~ dn̄/dA² (approx, pulsed calibration)
                    chi_hz = chi_slope / (2.0 * displacement_k)
            except Exception as exc:
                logger.warning("Chi slope fit failed for %s: %s", qubit.name, exc)

        fit_results[qubit.name] = RamseyStarkFit(
            amplitudes=amplitudes.tolist(),
            ramsey_freq_hz=freqs_hz,
            ramsey_T2_ns=T2s_ns,
            delta_freq_hz=delta_f.tolist(),
            chi_slope_hz_per_amp2=chi_slope,
            chi_hz=chi_hz,
            success=success,
            fit_residual_hz=residual,
        )

    return dataset, fit_results


def log_fitted_results(
    fit_results: Dict[str, RamseyStarkFit],
    log_callable=None,
) -> None:
    """Log χ and slope to console / node log."""
    if log_callable is None:
        log_callable = logger.info
    for qname, res in fit_results.items():
        if not res.success:
            log_callable(f"{qname}: Ramsey Stark fit failed")
            continue
        slope_mhz = res.chi_slope_hz_per_amp2 / 1e6
        log_callable(
            f"{qname}: Δf/A² slope = {slope_mhz:.3f} MHz/A²  "
            f"(fit RMS = {res.fit_residual_hz / 1e3:.1f} kHz)"
        )
        if res.chi_hz is not None:
            log_callable(
                f"{qname}: χ = {res.chi_hz / 1e6:.3f} MHz  "
                f"(via displacement_k)"
            )
        else:
            log_callable(
                f"{qname}: χ not extracted — displacement_k not yet calibrated. "
                f"Raw slope = {slope_mhz:.3f} MHz/A²"
            )
