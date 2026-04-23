"""Analysis for displacement calibration via photon-number splitting (node 28).

Experiment:
  For each displacement amplitude A, a qubit spectroscopy scan is performed.
  The resulting spectrum contains peaks at frequencies ω_q - 2χ·n (n=0,1,2,...),
  with amplitudes proportional to the Poisson photon-number distribution P(n).

Procedure:
  1. For each amplitude A: auto-fit increasing number of Gaussians until chi² < threshold.
  2. Compute n̄ = Σ n·P(n) from fitted distribution.
  3. Fit n̄(A) = k·A² to obtain calibration constant k.
  4. A₁ph = 1/√k is the displacement amplitude that deposits exactly 1 photon on average.
  5. chi_hz = peak_spacing / 2 = χ/(2π) [Hz] is the Hamiltonian coupling constant,
     averaged across all amplitudes where peaks are visible.

Convention: chi = χ/(2π) [Hz] in H/ħ = χ a†a σz.  Peak spacing = 2*chi.

State updates:
  - cavity_mode.cavity_mode_drive.operations["displacement"].amplitude = amp_for_one_photon
  - CavityTransmonPair.chi = chi_hz
  - CavityTransmonPair.displacement_k = k
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
try:
    from qualibration_libs.data import apply_confusion_correction_to_dataset
except ImportError:
    def apply_confusion_correction_to_dataset(ds, node):
        raise NotImplementedError("apply_confusion_correction_to_dataset not available in this qualibration_libs version")


@dataclass
class FitParameters:
    """Fit results for a single qubit's displacement calibration."""
    k: float
    """Calibration constant: n̄ = k · A²."""
    amp_for_one_photon: float
    """Displacement amplitude_scale that deposits exactly 1 photon on average: 1/√k."""
    chi_hz: float
    """Mean peak spacing 2χ [Hz], averaged over amplitudes with visible peaks."""
    nbar_vs_amp: List[Tuple[float, float]]
    """Raw calibration pairs [(A, n̄), ...] before fitting."""
    num_peaks_used: int
    """Number of Gaussian peaks used in the best fit (auto-detected)."""
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        log_callable(
            f"[28] {q}: {status} | A₁ph = {res['amp_for_one_photon']:.4f} "
            f"| k = {res['k']:.3f} | chi = {res['chi_hz'] * 1e-3:.3f} kHz "
            f"| peaks detected = {res['num_peaks_used']}"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    else:
        ds = apply_confusion_correction_to_dataset(ds, node)
    return ds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _multi_gaussian(x, *params):
    """Sum of N Gaussians.  params = [a1, x1, sigma1, a2, x2, sigma2, ..., offset]."""
    n = (len(params) - 1) // 3
    offset = params[-1]
    result = np.full_like(x, offset, dtype=float)
    for i in range(n):
        a, x0, sigma = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        result += a * np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    return result


def _fit_spectrum_n(x: np.ndarray, y: np.ndarray, n_peaks: int) -> Optional[np.ndarray]:
    """Attempt to fit exactly n_peaks Gaussians to a 1-D spectrum.

    Returns popt or None on failure.
    popt layout: [a0, x0, s0, a1, x1, s1, ..., offset]
    """
    try:
        y_smooth = np.convolve(y, np.ones(5) / 5, mode="same")
        peak_indices = sorted(
            np.argsort(y_smooth)[-n_peaks:].tolist(), key=lambda i: x[i]
        )
        offset0 = float(np.min(y))
        sigma0 = float((x[-1] - x[0]) / (n_peaks * 4))
        p0 = []
        for idx in peak_indices:
            p0 += [float(y_smooth[idx] - offset0), float(x[idx]), sigma0]
        p0.append(offset0)

        lower = ([0, x.min(), 1] * n_peaks) + [float("-inf")]
        upper = ([float("inf"), x.max(), (x[-1] - x[0])] * n_peaks) + [float("inf")]

        popt, _ = curve_fit(_multi_gaussian, x, y, p0=p0, bounds=(lower, upper), maxfev=20000)
        return popt
    except Exception:
        return None


def _fit_spectrum_auto(
    x: np.ndarray,
    y: np.ndarray,
    max_peaks: int,
    chi2_threshold: float,
) -> Tuple[Optional[np.ndarray], int]:
    """Try increasing numbers of Gaussian peaks until reduced chi² < chi2_threshold.

    Follows the same convention as power_rabi:
        chi2 = SS_res / ((N - P) * a²)
    where P = 3*n + 1 free parameters and a = peak-to-peak of the fitted curve.
    chi2 ≤ 2 means residual RMS ≤ a/√2  (signal clearly resolved above noise).

    Returns (best_popt, n_peaks_used).  best_popt is None if every attempt failed.
    """
    N = len(x)
    best_popt: Optional[np.ndarray] = None
    best_n: int = 0
    best_chi2: float = np.inf

    for n in range(1, max_peaks + 1):
        popt = _fit_spectrum_n(x, y, n)
        if popt is None:
            continue
        y_fit = _multi_gaussian(x, *popt)
        a = float(np.max(y_fit) - np.min(y_fit))
        if a <= 0:
            continue
        P = 3 * n + 1  # free parameters
        dof = max(N - P, 1)
        SS_res = float(np.sum((y - y_fit) ** 2))
        chi2 = SS_res / (dof * a ** 2)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_popt = popt
            best_n = n
        if chi2 < chi2_threshold:
            break

    return best_popt, best_n


def _extract_nbar_and_chi(popt: np.ndarray, n_peaks: int) -> Tuple[float, float]:
    """From Gaussian fit parameters compute n̄ and chi.

    Peak areas ∝ amplitude × sigma (Gaussian area = a·σ·√(2π), but we only need
    relative weights, so area_n = a_n * |sigma_n|).
    n̄ = Σ n · P(n),  P(n) = area_n / Σ area_i
    chi = mean spacing between adjacent peak centres.
    """
    areas = np.array([abs(popt[3 * i]) * abs(popt[3 * i + 2]) for i in range(n_peaks)])
    areas = np.maximum(areas, 0.0)
    total = areas.sum()
    if total <= 0:
        return float("nan"), float("nan")
    probs = areas / total
    nbar = float(np.dot(np.arange(n_peaks), probs))

    positions = np.array([popt[3 * i + 1] for i in range(n_peaks)])
    positions_sorted = np.sort(positions)
    if len(positions_sorted) >= 2:
        spacings = np.diff(positions_sorted)
        chi_hz = float(np.mean(spacings)) / 2.0  # peak_spacing/2 = χ/(2π)
    else:
        chi_hz = float("nan")

    return nbar, chi_hz


def _nbar_model(A, k):
    return k * A ** 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    signal_name = "state" if node.parameters.use_state_discrimination else "I"
    max_peaks = node.parameters.max_peaks
    chi2_threshold = node.parameters.chi2_threshold
    fit_results = {}

    for q in ds.qubit.values:
        ds_q = ds.sel(qubit=q)
        power_dbm = ds_q.power.values
        # Reconstruct linear amplitude prefactors from dBm values (max power → scale=1.0)
        amps = 10 ** ((power_dbm - power_dbm.max()) / 20)
        x_hz = ds_q.detuning.values

        nbar_list: List[Tuple[float, float]] = []
        chi_list: List[float] = []
        peaks_per_amp: List[int] = []

        # Per-amplitude Gaussian fitting with automatic peak detection
        for amp_idx, A in enumerate(amps):
            y = getattr(ds_q.isel(power=amp_idx), signal_name).values
            popt, n_detected = _fit_spectrum_auto(x_hz, y, max_peaks, chi2_threshold)
            if popt is None or n_detected == 0:
                continue
            nbar, chi_hz = _extract_nbar_and_chi(popt, n_detected)
            if np.isfinite(nbar):
                nbar_list.append((float(A), float(nbar)))
                peaks_per_amp.append(n_detected)
            if np.isfinite(chi_hz) and chi_hz > 0:
                chi_list.append(chi_hz)

        mean_chi = float(np.mean(chi_list)) if chi_list else float("nan")
        # Report the median number of peaks detected across amplitudes
        n_peaks_used = int(np.median(peaks_per_amp)) if peaks_per_amp else 0

        # Fit n̄(A) = k·A²
        try:
            valid = [(A, nb) for A, nb in nbar_list if np.isfinite(nb) and A > 0]
            if len(valid) < 3:
                raise ValueError("too few valid points")
            A_arr = np.array([v[0] for v in valid])
            nbar_arr = np.array([v[1] for v in valid])
            k0 = float(np.mean(nbar_arr / (A_arr ** 2 + 1e-12)))
            popt_k, _ = curve_fit(_nbar_model, A_arr, nbar_arr, p0=[k0], maxfev=5000)
            k_fit = float(popt_k[0])
            amp_one = float(1.0 / np.sqrt(k_fit)) if k_fit > 0 else float("nan")
            success = bool(np.isfinite(amp_one) and k_fit > 0)
            fit_results[q] = FitParameters(
                k=k_fit,
                amp_for_one_photon=amp_one,
                chi_hz=mean_chi,
                nbar_vs_amp=nbar_list,
                num_peaks_used=n_peaks_used,
                success=success,
            )
        except Exception:
            fit_results[q] = FitParameters(
                k=float("nan"),
                amp_for_one_photon=float("nan"),
                chi_hz=mean_chi,
                nbar_vs_amp=nbar_list,
                num_peaks_used=n_peaks_used,
                success=False,
            )

    return ds, fit_results
