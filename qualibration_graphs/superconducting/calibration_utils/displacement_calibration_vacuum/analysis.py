"""Analysis for the displacement vacuum-population calibration node (35).

Experiment:
  For each displacement amplitude scale a, the cavity is displaced to a coherent
  state |α = a · A_unit⟩, then a π-pulse is applied on the qubit (selective_x180
  or x180) and the qubit state is measured.

  The selective_x180 is spectrally tuned to the n=0 qubit transition, so it flips
  the qubit only when the cavity is in the vacuum state |0⟩.  The signal is:

      P_e(a) = amplitude · exp(-(a / sigma)²) + offset

  where sigma = A_1ph is the amplitude_scale that produces exactly 1 photon on
  average.  This follows from P(n=0) = exp(-n̄) = exp(-(a/A_1ph)²) for a coherent
  state.

State update:
  - cavity_mode.cavity_mode_drive.operations["displacement"].amplitude
    is multiplied by sigma so that amplitude_scale=1 → 1 photon.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Tuple

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
    """Fit results for a single qubit."""
    sigma: float
    """Unit displacement amplitude_scale (A_1ph): the scale for exactly 1 photon."""
    amplitude: float
    """Fitted peak amplitude (ideally ~0.5 for perfect selective π-flip contrast)."""
    offset: float
    """Fitted baseline (ideally ~0 for perfect state discrimination)."""
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        log_callable(
            f"[35] {q}: {status} | A_1ph (sigma) = {res['sigma']:.4f} "
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

def vacuum_population(a: np.ndarray, amplitude: float, sigma: float, offset: float) -> np.ndarray:
    """Vacuum state population model: Gaussian decay in displacement amplitude.

    P_e(a) = amplitude · exp(-(a / sigma)²) + offset

    Parameters
    ----------
    a : array-like
        Displacement amplitude scale values.
    amplitude : float
        Peak probability contrast (≈ 0.5 for ideal selective π-flip).
    sigma : float
        Unit displacement scale A_1ph (displacement that gives n̄ = 1).
    offset : float
        Baseline probability at large amplitudes (ideally 0).
    """
    return amplitude * np.exp(-(a / sigma) ** 2) + offset


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit the vacuum-population Gaussian to P_e(a) for each qubit.

    Returns the dataset (unchanged) and a dict of FitParameters per qubit.
    """
    signal_name = "state" if node.parameters.use_state_discrimination else "I"
    fit_results: Dict[str, FitParameters] = {}

    for q in ds.qubit.values:
        ds_q = ds.sel(qubit=q)
        a_arr = ds_q.amp.values.astype(float)
        signal = ds_q[signal_name].values.astype(float)

        # Initial guesses
        offset0 = float(np.mean(signal[a_arr > 0.8 * a_arr.max()]))
        amplitude0 = float(signal[a_arr == a_arr.min()].mean()) - offset0
        # sigma: amplitude drops to 1/e of peak at a = sigma
        target = offset0 + amplitude0 / np.e
        above = a_arr[signal > target]
        sigma0 = float(above.max()) if len(above) > 0 else float(a_arr.max()) * 0.5

        try:
            a_max = float(a_arr.max())
            popt, _ = curve_fit(
                vacuum_population,
                a_arr,
                signal,
                p0=[amplitude0, sigma0, offset0],
                bounds=([0.0, 1e-4, -0.5], [1.5, a_max * 2, 1.0]),
                maxfev=10000,
            )
            amplitude_fit, sigma_fit, offset_fit = popt
            success = bool(
                np.isfinite(sigma_fit)
                and sigma_fit > 0
                and amplitude_fit > 0.02  # require at least ~2% contrast
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
