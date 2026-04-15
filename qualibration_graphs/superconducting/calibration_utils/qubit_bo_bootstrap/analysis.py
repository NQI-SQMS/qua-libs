"""
analysis.py — Rabi trace fitting and Wolff cost function for the BO bootstrap node.

Two public functions:

    fit_rabi_trace(I_arr, t_ns)
        Fit a sinusoidal oscillation to a time-domain Rabi trace.
        Multi-start strategy: Lomb-Scargle seed + curve_fit, then falls back
        to a grid of starting frequencies if the first attempt fails.
        Returns a ``BoFitResult`` dataclass.

    compute_cost(fit, target_rabi_freq_mhz, w_rabi, w_amp, w_snr)
        Evaluate the Wolff cost function on a fit result.
        Returns a scalar cost value (lower = better).

Wolff cost function
────────────────────
Introduced in Wolff et al. APS 2026 (Pfafflab / UIUC):

    C = w_rabi × |Ω_R − Ω_T| / Ω_T  −  w_amp × log(A)  −  w_snr × log(SNR)

where:
    Ω_R = generalised Rabi frequency from the fit (includes detuning contribution)
    Ω_T = target Rabi frequency (user-set hyperparameter, default 20 MHz)
    A   = normalised oscillation amplitude ∈ (0, 1]
          A → 1 at resonance (Ω_d ≫ detuning), A → 0 far off resonance
    SNR = fit_amplitude / fit_noise_floor

The three terms simultaneously push toward:
    1. Drive on resonance  (Rabi term)
    2. Correct drive amplitude (amplitude term)
    3. Good readout conditions (SNR term)

All three terms are needed: the Rabi term alone cannot distinguish between
"on resonance at low drive" and "off resonance at high drive" (both give
Ω_R ≈ Ω_T). The amplitude term breaks this degeneracy.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import lombscargle

logger = logging.getLogger(__name__)

# ── Physical constants ─────────────────────────────────────────────────────────
_MIN_AMPLITUDE = 1e-9   # prevent log(0) in cost function
_MIN_SNR = 1e-9         # prevent log(0) in cost function


# ── Fit result dataclass ──────────────────────────────────────────────────────

@dataclass
class BoFitResult:
    """
    Results from fitting a single time-domain Rabi trace for the BO node.

    All frequency values are in MHz for readability and cost-function computation.
    """

    rabi_freq_mhz: float
    """Generalised Rabi frequency Ω_R in MHz, extracted from the sinusoidal fit.
    Ω_R = 1 / (2 × pi_half_period).  At resonance Ω_R ≈ Ω_d (drive Rabi frequency).
    Off resonance Ω_R = sqrt(Ω_d² + Δ²) where Δ = ω_d − ω_q."""

    amplitude: float
    """Normalised oscillation amplitude A ∈ (0, 1].
    Defined as A = 2·fit_a / signal_range, where fit_a is the half-amplitude of the
    sinusoid and signal_range is the peak-to-peak range of the raw I trace.
    A → 1 at resonance (maximum contrast), A → 0 far off resonance."""

    snr: float
    """Signal-to-noise ratio: fit_a / rms_residual.
    Low SNR indicates either very low drive amplitude or poor readout conditions."""

    fit_a: float
    """Half-amplitude of the fitted sinusoid in physical IQ units (Volts or arbitrary)."""

    fit_f_mhz: float
    """Fitted oscillation frequency in MHz (same as rabi_freq_mhz for a simple 1D fit)."""

    fit_phi: float
    """Fitted phase offset in radians."""

    fit_offset: float
    """Fitted DC offset in the same units as the raw data."""

    success: bool
    """True if the fit converged and produced a physically reasonable result.
    False if curve_fit failed or the fit amplitude is below the noise floor."""

    residual_rms: float
    """RMS of the fit residuals in physical units."""


# ── Oscillation model ─────────────────────────────────────────────────────────

def _sinusoid(t: np.ndarray, a: float, f: float, phi: float, offset: float) -> np.ndarray:
    """
    Sinusoidal model: y(t) = a·cos(2π·f·t + phi) + offset

    Parameters
    ──────────
    t      : time array in nanoseconds
    a      : half-amplitude (Volts or arbitrary IQ units)
    f      : frequency in MHz (=GHz·1e-3 · 1e3 = cycles per ns)
    phi    : phase offset in radians
    offset : DC offset
    """
    return a * np.cos(2.0 * np.pi * f * t + phi) + offset


# ── Robust fitting ────────────────────────────────────────────────────────────

def fit_rabi_trace(
    I_arr: np.ndarray,
    t_ns: np.ndarray,
    target_rabi_freq_mhz: float = 20.0,
    n_restarts: int = 5,
) -> BoFitResult:
    """
    Fit a sinusoidal oscillation to a time-domain Rabi trace.

    Strategy
    ────────
    1. Estimate the dominant frequency from the Lomb-Scargle periodogram
       (robust to uneven sampling and handles very slow oscillations).
    2. Run scipy.optimize.curve_fit from the Lomb-Scargle seed.
    3. If the first attempt fails or gives an unreasonable result, run
       *n_restarts* additional attempts from a grid of starting frequencies
       spanning [0.5, 2×target_rabi_freq] MHz.
    4. Keep the fit with the lowest residual RMS across all attempts.

    Parameters
    ──────────
    I_arr : np.ndarray, shape (n_pts,)
        Averaged I quadrature values (Volts), as returned by the QUA job.
        Should already be averaged over num_shots (the QUA stream_processing
        buffer().average() step produces this).
    t_ns : np.ndarray, shape (n_pts,)
        Drive pulse duration in nanoseconds (same length as I_arr).
        Must match the durations_ns array used in the QUA program.
    target_rabi_freq_mhz : float
        Target Rabi frequency in MHz.  Used to define the frequency search range
        for the fallback grid restarts (see n_restarts).
    n_restarts : int
        Number of curve_fit restarts with different starting frequencies.
        More restarts increase robustness at the cost of fitting time (~ms each).

    Returns
    ──────
    BoFitResult : fitted parameters + derived quantities (amplitude, SNR, success).
    """
    t = np.asarray(t_ns, dtype=float)
    y = np.asarray(I_arr, dtype=float)

    if len(t) < 5:
        logger.warning("Too few data points (%d) to fit Rabi trace.", len(t))
        return _failed_fit()

    signal_range = float(np.ptp(y))  # peak-to-peak range
    if signal_range < 1e-12:
        logger.warning("Signal range ~0 — no oscillation visible.")
        return _failed_fit()

    # ── Lomb-Scargle frequency seed ───────────────────────────────────────────
    # Evaluate at frequencies from 0.5 MHz to the Nyquist limit.
    dt_ns = float(np.median(np.diff(t)))
    f_nyquist_mhz = 0.5 / dt_ns * 1e3  # dt in ns → f in MHz
    f_min_mhz = max(0.1, 0.5 / (t[-1] - t[0]) * 1e3)
    freqs_mhz = np.linspace(f_min_mhz, min(f_nyquist_mhz, 4 * target_rabi_freq_mhz), 500)
    angular_freqs = 2.0 * np.pi * freqs_mhz * 1e-3  # rad / ns

    try:
        pgram = lombscargle(t, y - np.mean(y), angular_freqs, normalize=True)
        f_ls_mhz = float(freqs_mhz[np.argmax(pgram)])
    except Exception:
        f_ls_mhz = target_rabi_freq_mhz  # fallback if Lomb-Scargle fails

    # Starting amplitude and offset
    a0 = signal_range / 2.0
    offset0 = float(np.mean(y))

    # Grid of starting frequencies for restarts
    f_grid_mhz = np.linspace(
        max(0.5, 0.5 * target_rabi_freq_mhz),
        2.5 * target_rabi_freq_mhz,
        n_restarts,
    )
    start_freqs = np.concatenate([[f_ls_mhz], f_grid_mhz])

    # Bounds: amplitude > 0, frequency > 0, offset free
    bounds_lo = [0.0, 0.0, -np.pi, -np.inf]
    bounds_hi = [np.inf, f_nyquist_mhz, np.pi, np.inf]

    best_result: Optional[BoFitResult] = None
    best_rms = np.inf

    for f_start in start_freqs:
        try:
            popt, _ = curve_fit(
                _sinusoid,
                t,
                y,
                p0=[a0, f_start, 0.0, offset0],
                bounds=(bounds_lo, bounds_hi),
                maxfev=5000,
            )
            a_fit, f_fit_mhz, phi_fit, offset_fit = popt
            y_fit = _sinusoid(t, *popt)
            residuals = y - y_fit
            rms = float(np.sqrt(np.mean(residuals**2)))

            if rms < best_rms and a_fit > 0 and f_fit_mhz > 0:
                best_rms = rms
                best_result = _build_result(
                    a_fit, f_fit_mhz, phi_fit, offset_fit, rms, signal_range
                )
        except Exception as exc:
            logger.debug("curve_fit failed for f_start=%.2f MHz: %s", f_start, exc)
            continue

    if best_result is None:
        logger.info("All curve_fit attempts failed — returning failed fit.")
        return _failed_fit()

    # Mark as failed if amplitude is below a heuristic noise threshold
    if best_result.amplitude < 0.05:
        logger.info(
            "Fit amplitude too low (A=%.3f) — no meaningful oscillation.",
            best_result.amplitude,
        )
        best_result = BoFitResult(
            **{**best_result.__dict__, "success": False}  # type: ignore[arg-type]
        )

    return best_result


def _build_result(
    a_fit: float,
    f_fit_mhz: float,
    phi_fit: float,
    offset_fit: float,
    residual_rms: float,
    signal_range: float,
) -> BoFitResult:
    """Construct a BoFitResult from curve_fit output."""
    # Normalise amplitude: A ∈ (0, 1], peaks at 1 on resonance
    amplitude = float(2.0 * a_fit / max(signal_range, _MIN_AMPLITUDE))
    amplitude = min(amplitude, 1.0)  # clamp: shouldn't exceed 1 for a well-fit trace

    # SNR: fit half-amplitude divided by residual noise floor
    snr = float(a_fit / max(residual_rms, _MIN_SNR))

    return BoFitResult(
        rabi_freq_mhz=f_fit_mhz,
        amplitude=amplitude,
        snr=snr,
        fit_a=a_fit,
        fit_f_mhz=f_fit_mhz,
        fit_phi=phi_fit,
        fit_offset=offset_fit,
        success=True,
        residual_rms=residual_rms,
    )


def _failed_fit() -> BoFitResult:
    """Return a sentinel BoFitResult for a failed fit (success=False)."""
    return BoFitResult(
        rabi_freq_mhz=0.0,
        amplitude=0.0,
        snr=0.0,
        fit_a=0.0,
        fit_f_mhz=0.0,
        fit_phi=0.0,
        fit_offset=0.0,
        success=False,
        residual_rms=float("inf"),
    )


# ── Wolff cost function ───────────────────────────────────────────────────────

def compute_cost(
    fit: BoFitResult,
    target_rabi_freq_mhz: float = 20.0,
    w_rabi: float = 10.0,
    w_amp: float = 3.0,
    w_snr: float = 1.0,
) -> float:
    """
    Evaluate the Wolff cost function on a ``BoFitResult``.

    Cost function (lower = better):

        C = w_rabi × |Ω_R − Ω_T| / Ω_T  −  w_amp × log(A)  −  w_snr × log(SNR)

    where:
        Ω_R  = fit.rabi_freq_mhz (generalised Rabi frequency from the curve fit)
        Ω_T  = target_rabi_freq_mhz (user-set; default 20 MHz)
        A    = fit.amplitude ∈ (0, 1] (normalised oscillation amplitude)
        SNR  = fit.snr (signal-to-noise ratio)

    When the fit fails (fit.success is False), a large penalty cost is returned
    (1000.0) to tell the GP that this region of the search space is uninformative.
    This prevents the BO from wasting evaluations on clearly bad points.

    Parameters
    ──────────
    fit : BoFitResult
        Output from ``fit_rabi_trace()``.
    target_rabi_freq_mhz : float
        Target Rabi frequency Ω_T in MHz.
    w_rabi : float
        Weight of the Rabi-frequency penalty term. Default 10.0.
        Dominates the cost landscape → drives the optimizer toward resonance first.
    w_amp : float
        Weight of the amplitude penalty term. Default 3.0.
        −w_amp·log(A) penalises low oscillation amplitude (off resonance).
    w_snr : float
        Weight of the SNR penalty term. Default 1.0.
        −w_snr·log(SNR) penalises low drive amplitude and poor readout conditions.

    Returns
    ──────
    float : scalar cost (lower = better).
    """
    if not fit.success:
        # Failed fit: maximum penalty — tells GP this is a dead region.
        return 1000.0

    rabi_term = w_rabi * abs(fit.rabi_freq_mhz - target_rabi_freq_mhz) / max(target_rabi_freq_mhz, 1e-9)
    amp_term = -w_amp * np.log(max(fit.amplitude, _MIN_AMPLITUDE))
    snr_term = -w_snr * np.log(max(fit.snr, _MIN_SNR))

    cost = float(rabi_term + amp_term + snr_term)
    logger.debug(
        "Cost: %.3f  (rabi=%.3f, amp=%.3f, snr=%.3f)  "
        "Omega_R=%.2f MHz  A=%.3f  SNR=%.2f",
        cost, rabi_term, amp_term, snr_term,
        fit.rabi_freq_mhz, fit.amplitude, fit.snr,
    )
    return cost
