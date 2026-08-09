"""Analysis for photon number resolved spectroscopy (node 29).

Sweeps qubit ge frequency with selective_x180 to resolve photon-number-split peaks.
Auto-detects number of peaks by fitting increasing Gaussians with shared sigma
until chi² < threshold.  Peak spacing = |chi| (dispersive shift).
Normalised peak amplitudes give P(n) directly.

Convention: chi [Hz] is the full per-photon qubit frequency shift, stored with its
physical sign.  For typical transmon-cavity systems chi < 0 (more photons lower the
qubit frequency) and |chi| equals the PNRS peak spacing (n=0→n=1 frequency gap).
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from calibration_utils.shared import apply_confusion_matrix_correction


def apply_confusion_correction_to_dataset(ds, node):
    return apply_confusion_matrix_correction(ds, node.namespace["qubits"])


@dataclass
class FitParameters:
    """Fit results for a single qubit's photon number resolved spectroscopy measurement."""
    chi_hz: float
    """Full per-photon qubit frequency shift [Hz], stored negative for typical
    transmon-cavity systems.  |chi_hz| equals the PNRS peak spacing (n=0→n=1)."""
    peak_positions_hz: List[float]
    """Detuning positions of photon-number peaks [Hz], ordered n=0, n=1, n=2, ...
    (most positive detuning first, i.e. index 0 ≈ 0 detuning = vacuum peak)."""
    peak_amplitudes: List[float]
    """Raw fitted amplitude of each Gaussian peak, ordered n=0, n=1, n=2, ..."""
    peak_probabilities: List[float]
    """Normalised photon-number probabilities P(n) = A_n / sum(A_i),
    ordered n=0, n=1, n=2, ... so index matches photon number."""
    num_peaks_used: int
    """Number of Gaussian peaks detected by the auto-fitting routine."""
    success: bool
    sigma_hz: float = float("nan")
    """Shared Gaussian sigma (1-σ half-width) [Hz]."""
    offset: float = float("nan")
    """Fitted baseline P(e) floor."""
    nbar: float = float("nan")
    """Mean photon number n̄ extracted by fitting P(n) to a Poisson distribution."""


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        chi_khz = res["chi_hz"] * 1e-3 if np.isfinite(res["chi_hz"]) else float("nan")
        peaks_khz = [p * 1e-3 for p in res["peak_positions_hz"]]
        probs = [f"{p:.3f}" for p in res["peak_probabilities"]]
        nbar = res.get("nbar", float("nan"))
        nbar_str = f"{nbar:.3f}" if np.isfinite(nbar) else "nan"
        log_callable(
            f"[24] {q}: {status} | chi = {chi_khz:.3f} kHz "
            f"| peaks = {res['num_peaks_used']} "
            f"| n\u0304 = {nbar_str} "
            f"| positions (kHz) = {[f'{p:.3f}' for p in peaks_khz]} "
            f"| P(n) = {probs}"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    else:
        ds = apply_confusion_correction_to_dataset(ds, node)
    return ds


def _gaussian_sum_shared_sigma(x, sigma, offset, *amp_pos_pairs):
    """Sum of N Gaussians all sharing the same sigma.

    params layout (after sigma, offset):  amp_0, pos_0, amp_1, pos_1, ...
    Params layout (after sigma, offset): amp_0, pos_0, amp_1, pos_1, ...
    """
    result = np.full_like(x, float(offset), dtype=float)
    for i in range(len(amp_pos_pairs) // 2):
        a = amp_pos_pairs[2 * i]
        x0 = amp_pos_pairs[2 * i + 1]
        result += a * np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    return result


def _fit_spectrum_n(x: np.ndarray, y: np.ndarray, n_peaks: int) -> Optional[np.ndarray]:
    """Attempt to fit exactly n_peaks Gaussians with a shared sigma.

    Returns popt = [sigma, offset, amp_0, pos_0, amp_1, pos_1, ...] or None.
    """
    from scipy.signal import find_peaks as _find_peaks
    try:
        y_smooth = np.convolve(y, np.ones(5) / 5, mode="same")
        n_pts = len(x)
        # Enforce spatial separation: peaks must be at least span/(n_peaks*4) apart.
        min_dist = max(1, n_pts // (n_peaks * 4))
        found, _ = _find_peaks(y_smooth, distance=min_dist)
        if len(found) >= n_peaks:
            # Take the n_peaks tallest among spatially-separated candidates
            peak_indices = sorted(
                sorted(found.tolist(), key=lambda i: y_smooth[i])[-n_peaks:],
                key=lambda i: x[i],
            )
        else:
            # Fallback: top-N from the full array (original behaviour)
            peak_indices = sorted(
                np.argsort(y_smooth)[-n_peaks:].tolist(),
                key=lambda i: x[i],
            )
        x_span = float(x[-1] - x[0])
        sigma0 = x_span / (n_peaks * 4)
        offset0 = float(np.min(y))

        p0 = [sigma0, offset0]
        lower = [x_span / (n_peaks * 20), float("-inf")]
        upper = [x_span, float("inf")]
        for idx in peak_indices:
            amp = float(y_smooth[idx] - offset0)
            p0 += [amp, float(x[idx])]
            lower += [0.0, float(x.min())]
            upper += [float("inf"), float(x.max())]

        popt, _ = curve_fit(
            _gaussian_sum_shared_sigma,
            x, y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000,
        )
        return popt
    except Exception:
        return None


def _fit_spectrum_auto(
    x: np.ndarray,
    y: np.ndarray,
    max_peaks: int,
    chi2_threshold: float,
) -> Tuple[Optional[np.ndarray], int]:
    """Try increasing numbers of shared-sigma Gaussians until reduced chi² < threshold.

    Convention (same as power_rabi):
        chi2 = SS_res / ((N - P) * a²)
    where P = 2 + 2*n free parameters and a = peak-to-peak of the fitted curve.

    Returns (best_popt, n_peaks_used).  best_popt is None if every attempt failed.
    """
    N = len(x)
    best_popt: Optional[np.ndarray] = None
    best_n: int = 0
    best_chi2: float = np.inf

    # Minimum acceptable peak amplitude: 3% of the data peak-to-peak range.
    # Peaks below this threshold are indistinguishable from noise.
    data_range = float(np.max(y) - np.min(y))
    min_amp = 0.03 * data_range

    for n in range(1, max_peaks + 1):
        popt = _fit_spectrum_n(x, y, n)
        if popt is None:
            continue

        # Reject if any fitted peak amplitude is below the noise floor
        amps = [float(popt[2 + 2 * i]) for i in range(n)]
        if any(a < min_amp for a in amps):
            continue

        y_fit = _gaussian_sum_shared_sigma(x, *popt)
        a = float(np.max(y_fit) - np.min(y_fit))
        if a <= 0:
            continue
        P = 2 + 2 * n  # sigma, offset, + (amp, pos) × n
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


def _fit_poisson_nbar(probs: List[float]) -> float:
    """Fit a list of P(n) values to a Poisson distribution, return best-fit n̄.

    Minimises sum of squared deviations between probs and Poisson(n, nbar).
    Returns nan on failure.
    """
    from scipy.special import factorial
    from scipy.optimize import minimize_scalar

    n_vals = np.arange(len(probs))
    p_obs = np.array(probs, dtype=float)

    def residual(nbar):
        if nbar <= 0:
            return np.inf
        p_fit = np.exp(-nbar) * nbar ** n_vals / factorial(n_vals)
        return float(np.sum((p_obs - p_fit) ** 2))

    try:
        result = minimize_scalar(residual, bounds=(1e-6, 20.0), method="bounded")
        return float(result.x) if result.success else float("nan")
    except Exception:
        return float("nan")


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    signal_name = "state" if node.parameters.use_state_discrimination else "I"
    max_peaks = node.parameters.max_peaks
    chi2_threshold = node.parameters.chi2_threshold
    fit_results = {}

    for q in ds.qubit.values:
        ds_q = ds.sel(qubit=q)
        x = ds_q.detuning.values  # Hz
        y = getattr(ds_q, signal_name).values

        popt, n_detected = _fit_spectrum_auto(x, y, max_peaks, chi2_threshold)

        if popt is not None and n_detected > 0:
            # popt = [sigma, offset, amp_0, pos_0, amp_1, pos_1, ...]
            pos_unsorted = [float(popt[3 + 2 * i]) for i in range(n_detected)]
            amps_unsorted = [float(popt[2 + 2 * i]) for i in range(n_detected)]

            # Sort ascending (most negative first) to compute chi from spacings
            sort_idx_asc = np.argsort(pos_unsorted)
            positions_asc = [pos_unsorted[j] for j in sort_idx_asc]
            amps_asc = [amps_unsorted[j] for j in sort_idx_asc]

            if len(positions_asc) >= 2:
                spacings = [positions_asc[i + 1] - positions_asc[i]
                            for i in range(len(positions_asc) - 1)]
                chi_hz = -float(np.mean(spacings))  # full per-photon shift, negative
            else:
                chi_hz = float("nan")

            # Reverse so index 0 = n=0 (vacuum, detuning ≈ 0, most positive position)
            positions = positions_asc[::-1]
            amps_sorted = amps_asc[::-1]

            # Normalise to P(n); index 0 = P(n=0)
            total_amp = sum(max(a, 0.0) for a in amps_sorted)
            if total_amp > 0:
                probs = [max(a, 0.0) / total_amp for a in amps_sorted]
            else:
                probs = [1.0 / n_detected] * n_detected

            # Fit P(n) to Poisson to extract mean photon number n̄
            nbar = _fit_poisson_nbar(probs)

            success = bool(np.isfinite(chi_hz) and chi_hz < 0)
            fit_results[str(q)] = FitParameters(
                chi_hz=chi_hz,
                peak_positions_hz=positions,
                peak_amplitudes=amps_sorted,
                peak_probabilities=probs,
                num_peaks_used=n_detected,
                success=success,
                sigma_hz=float(popt[0]),
                offset=float(popt[1]),
                nbar=nbar,
            )
        else:
            fit_results[str(q)] = FitParameters(
                chi_hz=float("nan"),
                peak_positions_hz=[],
                peak_amplitudes=[],
                peak_probabilities=[],
                num_peaks_used=0,
                success=False,
            )

    return ds, fit_results
