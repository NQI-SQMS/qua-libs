"""
Resonator circle-fit pipeline for the 02a calibration node.

Implements the multi-step algorithm from CircleFit/circuit.py
(Probst et al., Rev. Sci. Instrum. 86, 024703 (2015), arXiv:1410.3365)
but operates directly on numpy arrays — no DataModule dependency.

Pipeline (mirrors circuit.__init__):
  1. Estimate and remove cable delay
  2. Subtract linear amplitude background
  3. Lorentzian pre-fit → initial Ql, fr
  4. Frequency-dependent weights
  5. Weighted algebraic circle fit
  6. Detect off-resonance angle theta0
  7. Normalise circle by off-resonance point (a, alpha)
  8. Estimate impedance-mismatch angle phi0
  9. Final nonlinear model fit (complex or magnitude)
 10. Derive Qc_real, Qi; return CircleFitParameters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from . import fit_toolbox as ft

log = logging.getLogger(__name__)


@dataclass
class CircleFitParameters:
    """Results of a resonator circle fit for a single qubit/resonator."""

    Q_loaded: float
    """Loaded quality factor Q_L = 1 / (1/Q_i + 1/Q_e)."""
    Q_internal: float
    """Internal quality factor Q_i (loss channel)."""
    Q_external: float
    """External/coupling quality factor Q_e."""
    kappa_Hz: float
    """Total linewidth κ/2π [Hz] = f0_Hz / Q_loaded."""
    f0_Hz: float
    """Resonance frequency [Hz]."""
    phi0: float
    """Impedance mismatch angle φ₀ [rad]."""
    delay_ns: float
    """Estimated cable delay [ns]."""
    success: bool
    """True when all Q-factor fields are finite and positive."""


def run_circle_fit(
    freq_Hz: np.ndarray,
    S21: np.ndarray,
    geometry: str = "transmission",
    delay_ns: float | None = None,
    comb_slopes: bool = True,
    final_mag: bool = False,
    weight_width: float = 1.0,
    fit_range: float = 0.10,
) -> CircleFitParameters:
    """Run the Probst circle-fit pipeline on raw IQ data.

    Parameters
    ----------
    freq_Hz : array — probe frequencies [Hz]
    S21 : complex array — demodulated signal (I + jQ), any overall scale
    geometry : 'transmission' (hanger S21) or 'reflection' (one-port S11)
    delay_ns : manual cable delay [ns]; None → auto-estimated from phase slope
    comb_slopes : use weighted full-range phase fit (True) or endpoint average (False)
    final_mag : fit only |S21| in the final step (more robust under heavy phase noise)
    weight_width : linewidth multiplier for the FWHM-based weight window
    fit_range : fraction of endpoints used for delay and background estimation

    Returns
    -------
    CircleFitParameters with all Q fields set to NaN and success=False on failure.
    """
    _nan = CircleFitParameters(
        Q_loaded=float("nan"), Q_internal=float("nan"), Q_external=float("nan"),
        kappa_Hz=float("nan"), f0_Hz=float("nan"), phi0=float("nan"),
        delay_ns=float("nan"), success=False,
    )

    freq_Hz = np.asarray(freq_Hz, dtype=float)
    S21 = np.asarray(S21, dtype=complex)

    if len(freq_Hz) < 10:
        log.warning("circle_fit: too few points (%d), skipping", len(freq_Hz))
        return _nan

    # ── 1. Cable delay ────────────────────────────────────────────────────────
    freq_GHz = freq_Hz * 1e-9
    try:
        if delay_ns is None:
            phase_deg = np.degrees(np.unwrap(np.angle(S21)))
            delay_ns, _ = ft.get_delay_arrays(freq_GHz, phase_deg,
                                              comb_slopes=comb_slopes,
                                              f_range=fit_range)
        S21_corr = S21 * np.exp(2j * np.pi * freq_GHz * delay_ns)
    except Exception as exc:
        log.warning("circle_fit: delay estimation failed: %s", exc)
        return _nan

    # ── 2. Background subtraction ─────────────────────────────────────────────
    try:
        S21_norm, _bg_pars = ft.subtract_linear_bg(freq_GHz, S21_corr,
                                                   fit_range=fit_range)
    except Exception as exc:
        log.warning("circle_fit: background subtraction failed: %s", exc)
        return _nan

    # ── 3. Lorentzian pre-fit → initial Ql, fr ────────────────────────────────
    try:
        lor_p, _ = ft.fit_lorentzian(freq_GHz, S21_norm)
        _, _, Ql_est, fr_est = lor_p
        Ql_est = abs(Ql_est)
        fr_est = abs(fr_est)
    except Exception as exc:
        log.warning("circle_fit: Lorentzian pre-fit failed: %s", exc)
        return _nan

    # ── 4. Weights ────────────────────────────────────────────────────────────
    weights = ft.get_weights(freq_GHz, Ql_est, fr_est, weight_width)

    # ── 5. Weighted algebraic circle fit ──────────────────────────────────────
    try:
        xc, yc, r0 = ft.fit_circle_weights(freq_GHz, S21_norm,
                                            fr_est, Ql_est, weights)
        zc = complex(xc, yc)
    except Exception as exc:
        log.warning("circle_fit: circle fit failed: %s", exc)
        return _nan

    # ── 6. Off-resonance angle theta0 ─────────────────────────────────────────
    try:
        theta0, _ = ft.fit_theta0(freq_GHz, S21_norm, Ql_est, fr_est, zc)
    except Exception as exc:
        log.warning("circle_fit: theta0 fit failed: %s", exc)
        return _nan

    # ── 7. Normalise circle by off-resonance point ────────────────────────────
    offrespoint = zc + r0 * np.exp(1j * theta0)
    a = abs(offrespoint)
    alpha = np.angle(offrespoint)

    if geometry == "reflection":
        circle_norm = S21_norm * np.exp(1j * (np.pi - alpha)) / a
        zc_norm = zc * np.exp(1j * (np.pi - alpha)) / a
    else:  # transmission / notch
        circle_norm = S21_norm * np.exp(-1j * alpha) / a
        zc_norm = zc * np.exp(-1j * alpha) / a

    r_norm = r0 / a

    # ── 8. Phi0 (impedance mismatch) ──────────────────────────────────────────
    if geometry == "reflection":
        phi0 = np.arcsin(np.clip(zc_norm.imag / r_norm, -1.0, 1.0))
        absQc_est = Ql_est / r_norm          # factor-2 difference vs notch
    else:
        phi0 = -np.arcsin(np.clip(zc_norm.imag / r_norm, -1.0, 1.0))
        absQc_est = Ql_est / (2.0 * r_norm)

    # ── 9. Final nonlinear model fit ──────────────────────────────────────────
    try:
        if geometry == "reflection":
            if final_mag:
                fit_p, _ = ft.fit_mag_refl(freq_GHz, circle_norm,
                                            Ql_est, absQc_est, fr_est, phi0,
                                            weights)
                Ql_fit, absQc_fit, fr_fit = fit_p[:3]
                phi0_fit = phi0
            else:
                fit_p, _ = ft.fit_model_refl(freq_GHz, circle_norm,
                                              Ql_est, absQc_est, fr_est, phi0,
                                              weights)
                Ql_fit, absQc_fit, fr_fit, phi0_fit = np.abs(fit_p)
        else:
            if final_mag:
                fit_p, _ = ft.fit_mag_notch(freq_GHz, circle_norm,
                                             Ql_est, absQc_est, fr_est, phi0,
                                             weights)
                Ql_fit, absQc_fit, fr_fit = fit_p[:3]
                phi0_fit = phi0
            else:
                fit_p, _ = ft.fit_model_notch(freq_GHz, circle_norm,
                                               Ql_est, absQc_est, fr_est, phi0,
                                               weights)
                Ql_fit, absQc_fit, fr_fit, phi0_fit = np.abs(fit_p)
    except Exception as exc:
        log.warning("circle_fit: final model fit failed: %s", exc)
        return _nan

    # ── 10. Quality factors ───────────────────────────────────────────────────
    if geometry == "reflection":
        Qc_real = absQc_fit   # no cos(phi0) factor for reflection
    else:
        Qc_real = Ql_fit / (2.0 * r_norm * np.cos(phi0_fit))

    try:
        Qi = 1.0 / (1.0 / Ql_fit - 1.0 / Qc_real)
    except ZeroDivisionError:
        Qi = float("nan")

    f0_Hz = fr_fit * 1e9   # GHz → Hz

    all_ok = all(np.isfinite(v) and v > 0
                 for v in (Ql_fit, Qi, Qc_real, f0_Hz))

    return CircleFitParameters(
        Q_loaded=float(Ql_fit),
        Q_internal=float(Qi),
        Q_external=float(Qc_real),
        kappa_Hz=float(f0_Hz / Ql_fit) if Ql_fit > 0 else float("nan"),
        f0_Hz=float(f0_Hz),
        phi0=float(phi0_fit),
        delay_ns=float(delay_ns),
        success=bool(all_ok),
    )


def plot_circle_fit(
    freq_Hz: np.ndarray,
    S21: np.ndarray,
    result: CircleFitParameters,
    title: str = "",
) -> Figure:
    """4-panel diagnostic plot for a circle-fit result.

    Panels: raw IQ spiral | delay-corrected IQ + fitted circle |
            |S21| vs frequency | phase vs frequency.
    """
    freq_GHz = freq_Hz * 1e-9
    S21 = np.asarray(S21, dtype=complex)

    if np.isfinite(result.delay_ns):
        S21_corr = S21 * np.exp(2j * np.pi * freq_GHz * result.delay_ns)
    else:
        S21_corr = S21.copy()

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    # ── A: raw IQ ─────────────────────────────────────────────────────────────
    ax = axes[0]
    sc = ax.scatter(S21.real, S21.imag, c=freq_GHz, cmap="plasma",
                    s=12, zorder=5, edgecolors="none")
    ax.plot(S21.real, S21.imag, "-", color="grey", lw=0.5, alpha=0.4)
    ax.set_xlabel("I"); ax.set_ylabel("Q")
    ax.set_title("Raw IQ", fontsize=10)
    ax.set_aspect("equal", "datalim"); ax.grid(True, alpha=0.3)

    # ── B: delay-corrected IQ + circle ────────────────────────────────────────
    ax = axes[1]
    ax.scatter(S21_corr.real, S21_corr.imag, c=freq_GHz, cmap="plasma",
               s=12, zorder=5, edgecolors="none")
    ax.plot(S21_corr.real, S21_corr.imag, "-", color="grey", lw=0.5, alpha=0.4)
    ax.set_xlabel("I"); ax.set_ylabel("Q")
    ax.set_title(f"Delay-corrected IQ  (τ={result.delay_ns:.2f} ns)", fontsize=10)
    ax.set_aspect("equal", "datalim"); ax.grid(True, alpha=0.3)

    # ── C: magnitude ──────────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(freq_GHz, 20 * np.log10(np.abs(S21)), ".", ms=3, color="steelblue",
            label="data")
    if np.isfinite(result.f0_Hz):
        ax.axvline(result.f0_Hz * 1e-9, color="r", lw=1.0, ls="--",
                   label=f"f₀={result.f0_Hz*1e-9:.6f} GHz")
    ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("|S21| (dB)")
    ax.set_title("Magnitude", fontsize=10)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ── D: phase ──────────────────────────────────────────────────────────────
    ax = axes[3]
    ax.plot(freq_GHz, np.degrees(np.unwrap(np.angle(S21_corr))), "-",
            color="darkorange", lw=1.2)
    ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("Phase (deg)")
    ax.set_title("Delay-corrected phase", fontsize=10)
    ax.grid(True, alpha=0.3)

    suptitle = title or "Resonator circle fit"
    if result.success:
        suptitle += (
            f"   f₀={result.f0_Hz*1e-9:.6f} GHz"
            f"  Q_L={result.Q_loaded:.0f}"
            f"  Q_i={result.Q_internal:.0f}"
            f"  Q_e={result.Q_external:.0f}"
            f"  κ/2π={result.kappa_Hz*1e-3:.1f} kHz"
        )
    else:
        suptitle += "  [fit failed]"
    fig.suptitle(suptitle, fontsize=10)

    plt.tight_layout()
    return fig
