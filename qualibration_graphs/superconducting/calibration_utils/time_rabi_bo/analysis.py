"""
Time-Rabi fitting and Wolff cost function.

Physics recap
─────────────
For a driven two-level system with drive frequency ω_d and qubit frequency ω_q:

    Ω_R = sqrt(Ω_d² + Δ²)        Δ = ω_d − ω_q  (detuning)
    A   ∝ Ω_d / Ω_R              oscillation amplitude (peaks at Δ=0, i.e. resonance)
    Ω_d ∝ V_d                    drive Rabi frequency ∝ drive voltage

Time-Rabi signal (I quadrature, square pulse of duration t):
    I(t) = A · cos(Ω_R · t + φ) + offset

Cost function (Wolff et al., APS 2026):
    C = w_Ω · |Ω_R − Ω_T| / Ω_T  −  w_A · log(A)  −  w_SNR · log(SNR)

    - Term 1 penalises Ω_R ≠ Ω_T. At resonance (Δ=0) with correct V_d,
      Ω_R = Ω_d = Ω_T → zero. Off-resonance or wrong amplitude → large.
    - Term 2 rewards high oscillation amplitude A (peaks on resonance).
    - Term 3 rewards high readout SNR.

The global minimum of C is at (ω_d = ω_q, V_d = V_π) simultaneously.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import curve_fit
from typing import Optional, Tuple, List


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class TimeRabiFitResult:
    """Outcome of fitting one time-Rabi trace."""
    rabi_freq_mhz: float
    """Fitted Rabi frequency Ω_R in MHz."""

    amplitude: float
    """Normalised oscillation amplitude A ∈ (0, 1].
    Computed as A_fit / signal_range to be dimensionless."""

    snr: float
    """Signal-to-noise ratio: A_fit / residual_RMS. > 1 means oscillation visible."""

    cost: float
    """Wolff cost function value C. Lower is better. NaN until compute_cost() called."""

    fit_success: bool
    """True if chi2 ≤ chi2_threshold, SNR ≥ min_snr, and Ω_R is finite and positive."""

    raw_trace: np.ndarray
    """Raw ADC signal array used for fitting (selected quadrature)."""

    chi2: float = np.nan
    """Reduced chi²: SS_res / ((N-4) × A²). Same definition as power_rabi node.
    chi2 ≤ 2 means the oscillation amplitude exceeds the noise floor (SNR ≥ 0.71).
    chi2 > 2 means residuals dominate — the fitter is fitting noise, not a Rabi signal."""

    omega_d_hz: float = 0.0
    """Drive frequency used for this measurement."""

    power_dbm: float = 0.0
    """Drive power in dBm used for this measurement."""

    v_d: float = 0.0
    """Drive amplitude in V (kept for backward compatibility)."""

    raw_I: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Raw I-quadrature ADC signal (before quadrature selection)."""

    raw_Q: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Raw Q-quadrature ADC signal (before quadrature selection)."""


# ── Fit model ──────────────────────────────────────────────────────────────────

def _rabi_model(t_us: np.ndarray, A: float, omega_mhz: float, phi: float, offset: float) -> np.ndarray:
    """
    Sinusoidal model for time-Rabi traces.

    Parameters work in µs and MHz (natural units for transmon experiments):
        t_us     : time axis in microseconds
        A        : oscillation amplitude
        omega_mhz: Rabi frequency in MHz (× 2π converts to rad/µs)
        phi      : phase offset
        offset   : DC offset

    Returns I(t) = A · cos(2π · omega_mhz · t_us + phi) + offset
    """
    return A * np.cos(2 * np.pi * omega_mhz * t_us + phi) + offset


# ── Main fitting function ──────────────────────────────────────────────────────

def fit_time_rabi(
    t_ns: np.ndarray,
    signal: np.ndarray,
    target_rabi_freq_mhz: float,
    chi2_threshold: float = 5.0,
    min_snr: float = 1.5,
) -> TimeRabiFitResult:
    """
    Fit a time-Rabi trace and extract Ω_R, A, SNR.

    Parameters
    ──────────
    t_ns                : time axis in nanoseconds (integer multiples of 4)
    signal              : averaged I-quadrature ADC signal, shape (len(t_ns),)
    target_rabi_freq_mhz: Ω_T in MHz (used for initial fit guess and bounds)
    chi2_threshold      : reduced chi² above which fit is deemed failed
    min_snr             : minimum SNR below which fit is deemed failed

    Returns
    ───────
    TimeRabiFitResult with rabi_freq_mhz, amplitude, snr, fit_success, raw_trace.
    cost is NaN — call compute_cost() separately.
    """
    t_us = t_ns * 1e-3  # ns → µs

    # ── Initial guesses ────────────────────────────────────────────────────────
    signal_range = signal.max() - signal.min()
    A0      = max(signal_range / 2, 1e-6)
    offset0 = signal.mean()
    omega0  = target_rabi_freq_mhz
    phi0    = 0.0

    p0 = [A0, omega0, phi0, offset0]

    # Bounds: amplitude positive, freq in [0.05, 20] × Ω_T, phase ∈ [−π, π]
    lb = [0.0,          omega0 * 0.05, -np.pi, -np.inf]
    ub = [signal_range * 5, omega0 * 20.0,  np.pi,  np.inf]

    try:
        popt, pcov = curve_fit(
            _rabi_model, t_us, signal,
            p0=p0, bounds=(lb, ub),
            maxfev=5000,
        )
        A_fit, omega_fit, phi_fit, offset_fit = popt

        # ── Derived quantities ─────────────────────────────────────────────
        rabi_freq_mhz = float(omega_fit)

        # Normalised amplitude: A_fit divided by signal range → dimensionless, ∈ (0,1]
        amplitude = float(max(A_fit / max(signal_range, 1e-9), 1e-6))

        # SNR: fitted amplitude / residual RMS
        fitted = _rabi_model(t_us, *popt)
        residuals = signal - fitted
        noise_rms = float(max(residuals.std(), 1e-9))
        snr = float(max(A_fit / noise_rms, 1e-6))

        # Reduced chi²: SS_res / ((N − 4) · A²)
        N = len(signal)
        n_dof = max(N - 4, 1)
        chi2_dof = float(np.sum(residuals ** 2) / (n_dof * max(A_fit ** 2, 1e-12)))

        fit_success = (
            chi2_dof < chi2_threshold
            and snr >= min_snr
            and np.isfinite(rabi_freq_mhz)
            and rabi_freq_mhz > 0
        )

        return TimeRabiFitResult(
            rabi_freq_mhz=rabi_freq_mhz,
            amplitude=amplitude,
            snr=snr,
            cost=np.nan,
            chi2=chi2_dof,
            fit_success=fit_success,
            raw_trace=signal.copy(),
        )

    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        # Fit completely failed — return sentinel values that will score as high cost
        return TimeRabiFitResult(
            rabi_freq_mhz=target_rabi_freq_mhz * 20.0,  # large → large cost term 1
            amplitude=1e-6,                               # small → large cost term 2
            snr=1e-6,                                     # small → large cost term 3
            cost=np.nan,
            chi2=np.inf,
            fit_success=False,
            raw_trace=signal.copy(),
        )


# ── Cost function ──────────────────────────────────────────────────────────────

def compute_cost(
    fit: TimeRabiFitResult,
    target_rabi_freq_mhz: float,
    w_rabi: float = 10.0,
    w_amp:  float = 3.0,
    w_snr:  float = 1.0,
    w_chi2: float = 2.0,
    failed_cost_sentinel: float = 1e6,
) -> float:
    """
    Evaluate the Wolff cost function with a soft chi² penalty.

    Extended formula (vs. original Wolff):
        C = w_Ω·|Ω_R−Ω_T|/Ω_T − w_A·log(A) − w_SNR·log(SNR) + w_χ²·max(0, log(χ²/2))

    The chi² penalty term is 0 for clean fits (χ² ≤ 2) and grows logarithmically
    for noisy ones.  This keeps costs finite and informative for ALL convergent fits,
    giving the GP real gradient information instead of a binary 1e6 sentinel.

    Only fits where scipy.optimize.curve_fit diverged entirely (chi2=inf) return
    failed_cost_sentinel — those carry zero useful information.

    Lower C → better calibration. Global minimum at Ω_R=Ω_T, A=1, SNR→∞, χ²→0.

    Parameters
    ──────────
    fit                  : TimeRabiFitResult from fit_time_rabi()
    target_rabi_freq_mhz : Ω_T in MHz
    w_rabi, w_amp, w_snr : Wolff cost weights
    w_chi2               : weight on the soft chi² penalty (0 disables it)
    failed_cost_sentinel : value returned only for completely diverged fits (chi2=inf)
    """
    # Only use sentinel for RuntimeError/diverged fits — chi2 is inf
    if not np.isfinite(fit.chi2):
        return failed_cost_sentinel

    term_rabi = w_rabi * abs(fit.rabi_freq_mhz - target_rabi_freq_mhz) / target_rabi_freq_mhz
    term_amp  = -w_amp  * np.log(max(fit.amplitude, 1e-9))
    term_snr  = -w_snr  * np.log(max(fit.snr,       1e-9))
    # Soft chi2 penalty: 0 when chi2 ≤ 2, log-grows above.
    # chi2=5 → +1.8, chi2=50 → +6.4, chi2=150 → +9.0 — keeps costs in [0, ~30]
    # even for noisy traces, so the GP landscape has meaningful gradients everywhere.
    term_chi2 = w_chi2 * max(0.0, np.log(max(fit.chi2, 1e-9) / 2.0))

    return float(term_rabi + term_amp + term_snr + term_chi2)


# ── Utility: quick Ramsey T2* estimate ────────────────────────────────────────

def fit_ramsey_t2star(
    t_ns: np.ndarray,
    signal: np.ndarray,
    detuning_mhz: float,
) -> Tuple[Optional[float], bool]:
    """
    Fit a Ramsey trace to extract T2* (ns).

    Used by the post-convergence sanity check in the bootstrap node.

    Model: A · exp(−t / T2*) · cos(2π · detuning_mhz · t_us + φ) + offset

    Returns (T2star_ns, fit_success). T2star_ns is None if fit failed.
    """
    t_us = t_ns * 1e-3

    def _ramsey_model(t, A, T2star_us, phi, offset):
        return A * np.exp(-t / max(T2star_us, 0.001)) * np.cos(2 * np.pi * detuning_mhz * t + phi) + offset

    A0       = (signal.max() - signal.min()) / 2
    T2star0  = float(t_us.max()) / 3
    p0 = [A0, T2star0, 0.0, signal.mean()]
    lb = [0.0,  0.001, -np.pi, -np.inf]
    ub = [np.inf, t_us.max() * 100, np.pi, np.inf]

    try:
        popt, _ = curve_fit(_ramsey_model, t_us, signal, p0=p0, bounds=(lb, ub), maxfev=3000)
        T2star_us = float(popt[1])
        T2star_ns = T2star_us * 1e3
        return T2star_ns, True
    except Exception:
        return None, False


# ── Fit curve helper ───────────────────────────────────────────────────────────

def rabi_fit_curve(t_ns: np.ndarray, fit: TimeRabiFitResult) -> Optional[np.ndarray]:
    """
    Evaluate the fitted sinusoidal model on t_ns (nanoseconds).

    Returns the fitted curve array if the fit succeeded, otherwise None.
    Uses the fitted Rabi frequency and the raw_trace signal to re-derive
    amplitude/phase/offset via a quick re-fit (avoids storing all fit params).
    """
    if not fit.fit_success or len(fit.raw_trace) == 0:
        return None
    t_us = t_ns * 1e-3
    signal = fit.raw_trace
    signal_range = float(signal.max() - signal.min())
    A0 = max(signal_range / 2, 1e-6)
    p0 = [A0, fit.rabi_freq_mhz, 0.0, float(signal.mean())]
    lb = [0.0, fit.rabi_freq_mhz * 0.5, -np.pi, -np.inf]
    ub = [signal_range * 5, fit.rabi_freq_mhz * 2.0, np.pi, np.inf]
    try:
        popt, _ = curve_fit(_rabi_model, t_us, signal, p0=p0, bounds=(lb, ub), maxfev=2000)
        return _rabi_model(t_us, *popt)
    except Exception:
        return None
