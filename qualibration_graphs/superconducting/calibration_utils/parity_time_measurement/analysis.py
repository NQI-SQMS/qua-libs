"""Analysis routines for the parity-time calibration (node 30).

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
The cavity is displaced to a coherent state with n̄ ≈ 1 photon and a Ramsey
sequence  selective_y90 → wait(τ) → selective_y90  is swept over a range of delays τ.
For a coherent-state cavity, the signal is a Poisson-weighted sum of harmonics
at n·f_χ (n = 0, 1, 2, ...) rather than a simple damped cosine, so a
damped-cosine fit is not appropriate.  Instead, χ_eff is read directly from
the FFT peak of the mean-subtracted signal.

Note: this calibration differs from node 25 (chi_ramsey_stark), which uses a
CW cavity drive to measure the steady-state dispersive shift.  Here we use a
pulsed displacement, the same hardware path as the actual Wigner experiment,
so AC Stark shifts from the probe pulse are automatically included.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from qualibration_libs.data import convert_IQ_to_V, apply_confusion_correction_to_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FFT-based frequency extraction
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
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParityTimeFit:
    """Results for one qubit from the parity-time measurement."""

    success: bool
    """True when a clear FFT peak was found."""

    parity_time_s: float
    """Experimentally calibrated parity time τ_parity [seconds].
    τ_parity = 1 / (2 · f_χ) = π / χ_eff."""

    chi_eff_hz: float
    """Per-photon qubit frequency shift [Hz] extracted from the FFT peak.
    This equals the PNRS peak spacing (n=0→1) and is 2× the Hamiltonian
    coupling pair.chi (i.e. chi_eff_hz = 2 · pair.chi · 2π in angular units).
    Use chi_eff_hz / 2 to update pair.chi."""

    amplitude: float
    """FFT-estimated oscillation amplitude."""

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

        # ── FFT-based extraction ──────────────────────────────────────────────
        freq_hz, amp, ok = _fft_extract_frequency(tau_ns, sig)

        if ok:
            parity_time_s = 1.0 / (2.0 * freq_hz)
            message = (
                f"PNRS spacing/(2π) = {freq_hz / 1e3:.1f} kHz  "
                f"→  pair.chi = {freq_hz / 2e3:.1f} kHz  "
                f"→  τ_parity = {parity_time_s * 1e9:.0f} ns"
            )
            fit_results[qubit.name] = ParityTimeFit(
                success=True,
                parity_time_s=parity_time_s,
                chi_eff_hz=freq_hz,
                amplitude=amp,
                message=message,
            )
        else:
            fit_results[qubit.name] = ParityTimeFit(
                success=False,
                parity_time_s=np.nan,
                chi_eff_hz=np.nan,
                amplitude=np.nan,
                message="FFT peak not found.",
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
        log_callable(
            f"{qname}: PNRS spacing/(2π) = {res.chi_eff_hz / 1e3:.2f} kHz  |  "
            f"pair.chi/(2π) = {res.chi_eff_hz / 2e3:.2f} kHz  |  "
            f"τ_parity = {res.parity_time_s * 1e9:.0f} ns"
        )
