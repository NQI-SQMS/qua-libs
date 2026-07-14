"""Analysis for the displacement Wigner calibration node (32).

Experiment:
  For each displacement amplitude scale a, the cavity is displaced by a·D_unit,
  then a parity Ramsey is performed on the qubit:

      x90 → wait(t_parity) → x90 → measure

  where t_parity = 1/(4·chi_hz) = 1/(2·peak_spacing) so that the qubit
  accumulates a phase of π per photon — measuring the photon-number parity ⟨(-1)^n⟩.
  chi_hz = χ/(2π) [Hz] is the Hamiltonian coupling constant (H/ħ = χ a†a σz);
  the per-photon qubit shift is 2·chi_hz, so τ_parity = 1/(2·2·chi_hz) = 1/(4·chi_hz).

  For a cavity in a coherent state |α = a·α_unit⟩ the measured signal is:

      P_e(a) = offset + amplitude · exp(-2·(a / sigma)²)

  which is a 1D slice of the Wigner function W(α) ∝ exp(-2|α|²).
  The fitted sigma is the amplitude_scale corresponding to one zero-point
  fluctuation of the cavity field, i.e. the unit displacement scale.

State update:
  - cavity_mode.cavity_mode_drive.operations["displacement"].amplitude
    is multiplied by sigma so that amplitude_scale=1 → 1 photon.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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
    """Fit results for a single qubit's Wigner calibration."""
    sigma: float
    """Unit displacement amplitude_scale (= A₁ph): scale for 1 photon."""
    amplitude: float
    """Fitted Gaussian amplitude (ideally ≈ 0.5 for perfect parity)."""
    offset: float
    """Fitted baseline (ideally ≈ 0.5)."""
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        log_callable(
            f"[32] {q}: {status} | sigma (A₁ph) = {res['sigma']:.4f} "
            f"| amplitude = {res['amplitude']:.3f} | offset = {res['offset']:.3f}"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    else:
        ds = apply_confusion_correction_to_dataset(ds, node)
    return ds


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def wigner_gaussian(a: np.ndarray, amplitude: float, sigma: float, offset: float) -> np.ndarray:
    """1D Wigner slice model: Gaussian in displacement amplitude scale.

    Parameters
    ----------
    a : array-like
        Displacement amplitude scales (dimensionless, can be negative).
    amplitude : float
        Peak amplitude above baseline (≈ 0.5 for ideal vacuum + perfect parity).
    sigma : float
        Width parameter = unit displacement scale (A₁ph).
    offset : float
        Baseline probability (≈ 0.5).
    """
    return amplitude * np.exp(-2.0 * (a / sigma) ** 2) + offset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_chi_hz(node: QualibrationNode) -> Optional[float]:
    """Return chi_hz = χ/(2π) [Hz] from parameters or CavityTransmonPair."""
    if node.parameters.chi_hz is not None:
        val = float(node.parameters.chi_hz)
        if np.isfinite(val) and val > 0:
            return val

    mode_name = node.parameters.mode_name
    pairs = getattr(node.machine, "cavity_transmon_pairs", None)
    if pairs:
        for q in node.namespace.get("qubits", []):
            key = f"{q.name}_{mode_name}"
            pair = pairs.get(key)
            if pair is not None and getattr(pair, "chi", None) is not None:
                chi = float(pair.chi)
                if np.isfinite(chi) and chi > 0:
                    return chi

    return None


def resolve_parity_time_ns(node: QualibrationNode) -> int:
    """Return the parity time in nanoseconds (rounded to nearest 4 ns).

    Priority: node.parameters.parity_time_ns → computed from chi_hz.
    Raises ValueError if neither is available.
    """
    if node.parameters.parity_time_ns is not None:
        t = int(node.parameters.parity_time_ns)
        t = max(t // 4, 4) * 4  # round to clock cycle
        return t

    chi_hz = _get_chi_hz(node)
    if chi_hz is None:
        raise ValueError(
            "Cannot determine parity_time_ns: set node.parameters.parity_time_ns "
            "or node.parameters.chi_hz (or run node 28 first to populate chi on "
            "CavityTransmonPair)."
        )
    # τ = 1/(2·peak_spacing) = 1/(2·2·chi_hz) = 1/(4·chi_hz)
    t_ns = 1e9 / (4.0 * chi_hz)
    t_ns_rounded = max(int(round(t_ns / 4)) * 4, 16)
    logging.getLogger(__name__).info(
        f"parity_time_ns computed from chi_hz={chi_hz*1e-3:.1f} kHz → {t_ns_rounded} ns"
    )
    return t_ns_rounded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit a Gaussian to the 1D Wigner slice P_e(a) for each qubit.

    Returns the original dataset (unchanged) and a dict of FitParameters per qubit.
    """
    signal_name = "state" if node.parameters.use_state_discrimination else "I"
    fit_results: Dict[str, FitParameters] = {}

    for q in ds.qubit.values:
        ds_q = ds.sel(qubit=q)
        a_arr = ds_q.amp.values.astype(float)
        signal = ds_q[signal_name].values.astype(float)

        # Initial guesses
        offset0 = float(np.mean(signal[np.abs(a_arr) > 0.8 * np.max(np.abs(a_arr))]))
        amplitude0 = float(np.max(signal)) - offset0
        # Width: half-max point
        half_max = offset0 + amplitude0 / 2.0
        above = a_arr[signal > half_max]
        sigma0 = float(np.max(np.abs(above))) if len(above) > 0 else 0.5

        try:
            popt, _ = curve_fit(
                wigner_gaussian,
                a_arr,
                signal,
                p0=[amplitude0, sigma0, offset0],
                bounds=([0.0, 1e-4, 0.0], [1.0, 5.0, 1.0]),
                maxfev=10000,
            )
            amplitude_fit, sigma_fit, offset_fit = popt
            success = bool(
                np.isfinite(sigma_fit)
                and sigma_fit > 0
                and amplitude_fit > 0.05  # at least some contrast
            )
            fit_results[str(q)] = FitParameters(
                sigma=float(sigma_fit),
                amplitude=float(amplitude_fit),
                offset=float(offset_fit),
                success=success,
            )
        except Exception:
            fit_results[str(q)] = FitParameters(
                sigma=float("nan"),
                amplitude=float("nan"),
                offset=float("nan"),
                success=False,
            )

    return ds, fit_results
