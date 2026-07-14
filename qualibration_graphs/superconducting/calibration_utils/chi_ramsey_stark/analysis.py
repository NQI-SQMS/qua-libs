"""Analysis routines for the Fock |1> qubit Ramsey chi calibration (node 25).

Physics recap
-------------
With the cavity in Fock |1>, the dispersive interaction shifts the qubit
frequency by chi (the per-photon dispersive shift):

    f_qubit(n=1) = f_qubit(n=0) + chi

Driving the qubit at its bare ge frequency and applying an artificial detuning
delta, the Ramsey oscillation frequency is:

    f_osc = delta + chi

so chi is extracted directly as:

    chi = f_osc - delta

chi is negative for typical transmon-cavity systems (qubit shifts down with
more photons).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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
    """Exponentially damped cosine: A*exp(-t/T2)*cos(2pi*f*t + phi) + c."""
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
class FockChiFit:
    """Fitted results for one qubit from the Fock |1> Ramsey chi experiment."""

    ramsey_freq_hz: float
    """Fitted Ramsey oscillation frequency [Hz].
    Equals artificial_detuning_hz + chi in the dispersive model."""

    ramsey_T2_ns: float
    """Fitted T2* of the qubit Ramsey in the presence of Fock |1> [ns]."""

    chi_hz: float
    """Dispersive shift per photon [Hz] = ramsey_freq_hz - artificial_detuning_hz.
    Negative for typical transmon-cavity systems."""

    success: bool
    """True when the Ramsey fit converged."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_raw_dataset(dataset, node):
    """Return the dataset unchanged (no additional processing needed)."""
    return dataset


def fit_raw_data(dataset, node) -> Tuple[object, Dict[str, FockChiFit]]:
    """Fit the Ramsey oscillation for every qubit and extract chi.

    Parameters
    ----------
    dataset : xarray.Dataset
        Raw dataset with dims ``(qubit, idle_time)``.
    node    : QualibrationNode
        Active node (used for parameters and namespace).

    Returns
    -------
    dataset      : xarray.Dataset  (unchanged)
    fit_results  : dict[qubit_name -> FockChiFit]
    """
    from qualibration_libs.parameters import get_qubits

    qubits = get_qubits(node)
    tau_ns = np.asarray(dataset.coords["idle_time"].values, dtype=float)
    art_det_hz = float(node.parameters.artificial_detuning_hz)

    fit_results: Dict[str, FockChiFit] = {}

    for i, qubit in enumerate(qubits):
        # Select signal array (n_tau,)
        use_state = node.parameters.use_state_discrimination
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
            fit_results[qubit.name] = FockChiFit(
                ramsey_freq_hz=np.nan,
                ramsey_T2_ns=np.nan,
                chi_hz=np.nan,
                success=False,
            )
            continue

        f_hz, t2_ns, success = _fit_ramsey_oscillation(tau_ns, raw)
        chi_hz = (f_hz - art_det_hz) if success else np.nan

        fit_results[qubit.name] = FockChiFit(
            ramsey_freq_hz=f_hz,
            ramsey_T2_ns=t2_ns,
            chi_hz=chi_hz,
            success=success,
        )

    return dataset, fit_results


def log_fitted_results(
    fit_results: Dict[str, FockChiFit],
    log_callable=None,
) -> None:
    """Log chi and T2* to console / node log."""
    if log_callable is None:
        log_callable = logger.info
    for qname, res in fit_results.items():
        if not res.success:
            log_callable(f"{qname}: Fock |1> Ramsey fit failed")
            continue
        log_callable(
            f"{qname}: f_osc = {res.ramsey_freq_hz / 1e6:.4f} MHz  "
            f"T2* = {res.ramsey_T2_ns / 1e3:.2f} us  "
            f"chi = {res.chi_hz / 1e6:.4f} MHz"
        )
