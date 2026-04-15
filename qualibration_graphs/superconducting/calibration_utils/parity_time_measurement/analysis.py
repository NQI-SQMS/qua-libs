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
sequence  y90 → wait(τ) → y90  is swept over a range of delays τ.  The
measured P(e) oscillates primarily at f_χ (dominated by the n=1 component of
the Poisson photon distribution).  A damped-cosine fit extracts f_χ and
therefore τ_parity.

Note: this calibration differs from node 25 (chi_ramsey_stark), which uses a
CW cavity drive to measure the steady-state dispersive shift.  Here we use a
pulsed displacement, the same hardware path as the actual Wigner experiment,
so AC Stark shifts from the probe pulse are automatically included.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
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
    """A · exp(−t/T2) · cos(2π f t + φ) + C  [t in ns, f in Hz]."""
    return amplitude * np.exp(-t / T2) * np.cos(_TWO_PI * frequency * t + phase) + offset


def _fit_single_trace(
    tau_ns: np.ndarray,
    signal: np.ndarray,
) -> Tuple[float, float, float, float, float, bool]:
    """Fit a single P_e(τ) trace to a damped cosine.

    Returns
    -------
    amplitude, T2_ns, frequency_hz, phase_rad, offset, success
    """
    if len(tau_ns) < 8 or not np.any(np.isfinite(signal)):
        return np.nan, np.nan, np.nan, 0.0, np.nan, False

    sig = np.where(np.isfinite(signal), signal, 0.0)

    # Initial frequency guess from FFT of the mean-subtracted signal
    dt_ns = float(tau_ns[1] - tau_ns[0]) if len(tau_ns) > 1 else 16.0
    fft_mag = np.abs(np.fft.rfft(sig - np.mean(sig)))
    freqs_hz = np.fft.rfftfreq(len(sig), d=dt_ns * 1e-9)
    peak_idx = int(np.argmax(fft_mag[1:])) + 1  # skip DC
    f0 = float(abs(freqs_hz[peak_idx])) if peak_idx > 0 else 1e5

    A0   = (float(np.max(sig)) - float(np.min(sig))) / 2.0
    T2_0 = float(tau_ns[-1]) / 2.0
    c0   = float(np.mean(sig))

    try:
        popt, _ = curve_fit(
            _damped_cosine,
            tau_ns,
            sig,
            p0=[A0, T2_0, f0, 0.0, c0],
            bounds=(
                [0.0,    4.0,   0.0,   -np.pi, 0.0   ],
                [1.0,    np.inf, 1e9,   np.pi,  1.0   ],
            ),
            maxfev=10_000,
        )
        return (float(popt[0]), float(popt[1]), float(popt[2]),
                float(popt[3]), float(popt[4]), True)
    except Exception as exc:
        logger.debug("Parity-time fit failed: %s", exc)
        return np.nan, np.nan, np.nan, 0.0, np.nan, False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParityTimeFit:
    """Fitted results for one qubit from the parity-time measurement."""

    success: bool
    """True when the damped-cosine fit converged."""

    parity_time_s: float
    """Experimentally calibrated parity time τ_parity [seconds].
    τ_parity = 1 / (2 · f_χ) = π / χ_eff."""

    chi_eff_hz: float
    """Effective dispersive shift χ_eff / (2π) [Hz] extracted from the fit.
    May differ from the bare χ measured by PNRS peak spacing."""

    T2_star_ns: float
    """Fitted Ramsey decay time T2* [ns]."""

    amplitude: float
    """Fitted oscillation amplitude (dimensionless)."""

    phase_rad: float
    """Fitted initial phase [rad]."""

    offset: float
    """Fitted DC offset of the Ramsey trace."""

    message: str = ""
    """Human-readable status message."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_raw_dataset(dataset: xr.Dataset, node) -> xr.Dataset:
    """Return dataset unchanged — no additional pre-processing required."""
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
    ds_fit       : xr.Dataset   Dataset with fitted curve added as ``P_e_fit``.
    fit_results  : dict[str, ParityTimeFit]
    """
    from qualibration_libs.parameters import get_qubits

    qubits = get_qubits(node)
    tau_ns = np.asarray(dataset.coords["delay"].values, dtype=float)
    n_tau = len(tau_ns)

    use_state = node.parameters.use_state_discrimination
    ds_fit = dataset.copy(deep=True)

    fit_results: Dict[str, ParityTimeFit] = {}
    fitted_curves = []

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
            logger.warning("No signal found for qubit %s — skipping fit.", qubit.name)
            fit_results[qubit.name] = ParityTimeFit(
                success=False,
                parity_time_s=np.nan,
                chi_eff_hz=np.nan,
                T2_star_ns=np.nan,
                amplitude=np.nan,
                phase_rad=np.nan,
                offset=np.nan,
                message="No signal data found in dataset.",
            )
            fitted_curves.append(np.full(n_tau, np.nan))
            continue

        sig = np.asarray(raw, dtype=float).ravel()[:n_tau]

        # ── Fit ───────────────────────────────────────────────────────────────
        amp, T2, freq_hz, phase, offset, ok = _fit_single_trace(tau_ns, sig)

        if ok and freq_hz > 0:
            parity_time_s = 1.0 / (2.0 * freq_hz)
            curve = _damped_cosine(tau_ns, amp, T2, freq_hz, phase, offset)
            message = (
                f"χ_eff/(2π) = {freq_hz / 1e3:.1f} kHz  "
                f"→  τ_parity = {parity_time_s * 1e9:.0f} ns"
            )
            fit_results[qubit.name] = ParityTimeFit(
                success=True,
                parity_time_s=parity_time_s,
                chi_eff_hz=freq_hz,
                T2_star_ns=T2,
                amplitude=amp,
                phase_rad=phase,
                offset=offset,
                message=message,
            )
        else:
            fit_results[qubit.name] = ParityTimeFit(
                success=False,
                parity_time_s=np.nan,
                chi_eff_hz=np.nan,
                T2_star_ns=np.nan,
                amplitude=np.nan,
                phase_rad=np.nan,
                offset=np.nan,
                message="Damped-cosine fit did not converge.",
            )
            curve = np.full(n_tau, np.nan)

        fitted_curves.append(curve)

    # Attach fitted curves to dataset
    qubit_names = [q.name for q in qubits]
    ds_fit["P_e_fit"] = xr.DataArray(
        np.array(fitted_curves),
        dims=["qubit", "delay"],
        coords={"qubit": qubit_names, "delay": tau_ns},
        attrs={"long_name": "Fitted P(|e⟩)", "units": ""},
    )

    return ds_fit, fit_results


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
            f"{qname}: χ_eff/(2π) = {res.chi_eff_hz / 1e3:.2f} kHz  |  "
            f"τ_parity = {res.parity_time_s * 1e9:.0f} ns  |  "
            f"T2* = {res.T2_star_ns / 1e3:.2f} µs"
        )
