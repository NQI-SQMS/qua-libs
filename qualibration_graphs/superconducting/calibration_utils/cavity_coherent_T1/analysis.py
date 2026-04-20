"""Analysis for the cavity coherent T1 node (33).

Experiment:
  The cavity is prepared in a coherent state |alpha> by playing the displacement
  pulse at scale `displacement_scale`.  After a variable wait time t_eff the
  qubit selective pi-pulse is applied.  The selective pulse only flips the qubit
  when the cavity is in vacuum (n=0), so the measured qubit excited-state
  probability is proportional to P(n=0):

      P(n=0 | coherent state with nbar) = exp(-nbar)

  Since nbar decays exponentially:

      nbar(t) = nbar_0 * exp(-t / T1)

  the measured signal is:

      P_e(t) = A * exp(-nbar_0 * exp(-t / T1)) + offset

  This is a "double exponential" (Gumbel-like) decay.  Fitting it extracts
  the cavity T1.

Note: the x-axis (t) is the *total* physical wait time in nanoseconds,
i.e.  delay_repeats * clock_cycles * 4 ns.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V, apply_confusion_correction_to_dataset


@dataclass
class CoherentT1Fit:
    """Fit results for a single qubit / cavity mode."""
    T1_ns: float
    """Fitted cavity T1 [ns]."""
    T1_error_ns: float
    """1-sigma uncertainty on T1 [ns]."""
    nbar0: float
    """Fitted initial mean photon number."""
    amplitude: float
    """Fitted signal amplitude."""
    offset: float
    """Fitted signal baseline."""
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        T1_us = res["T1_ns"] * 1e-3
        T1_err_us = res["T1_error_ns"] * 1e-3
        log_callable(
            f"[33] {q}: {status} | T1 = {T1_us:.1f} ± {T1_err_us:.1f} µs "
            f"| nbar0 = {res['nbar0']:.2f}"
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

def coherent_T1_model(t_ns: np.ndarray, A: float, nbar0: float, T1_ns: float, offset: float) -> np.ndarray:
    """Double-exponential decay for coherent-state T1 measurement.

    P_e(t) = A * exp(-nbar0 * exp(-t / T1)) + offset

    Parameters
    ----------
    t_ns : array-like
        Physical wait time [ns] (= delay_repeats * clock_cycles * 4).
    A : float
        Signal amplitude.
    nbar0 : float
        Initial mean photon number.
    T1_ns : float
        Cavity T1 [ns].
    offset : float
        Signal baseline.
    """
    return A * np.exp(-nbar0 * np.exp(-np.asarray(t_ns) / T1_ns)) + offset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, CoherentT1Fit]]:
    """Fit the coherent T1 double-exponential model to the dataset."""
    signal_name = "state" if node.parameters.use_state_discrimination else "I"
    nbar0_guess = node.parameters.displacement_scale ** 2
    fit_results: Dict[str, CoherentT1Fit] = {}

    for q in ds.qubit.values:
        ds_q = ds.sel(qubit=q)
        t_ns = ds_q.idle_time.values.astype(float)  # physical time in ns
        signal = ds_q[signal_name].values.astype(float)

        # Initial guesses
        offset0 = float(np.min(signal))
        A0 = float(np.max(signal)) - offset0
        T1_0 = float(t_ns[len(t_ns) // 2])  # midpoint as initial guess

        try:
            popt, pcov = curve_fit(
                coherent_T1_model,
                t_ns,
                signal,
                p0=[A0, nbar0_guess, T1_0, offset0],
                bounds=(
                    [0.0, 0.01, t_ns[1], -1.0],
                    [2.0, 100.0, t_ns[-1] * 10, 1.0],
                ),
                maxfev=20000,
            )
            A_fit, nbar_fit, T1_fit, offset_fit = popt
            perr = np.sqrt(np.diag(pcov))
            T1_err = float(perr[2])
            success = bool(
                np.isfinite(T1_fit) and T1_fit > 0
                and T1_fit < t_ns[-1] * 5
                and T1_err / T1_fit < 1.0
            )
            fit_results[str(q)] = CoherentT1Fit(
                T1_ns=float(T1_fit),
                T1_error_ns=T1_err,
                nbar0=float(nbar_fit),
                amplitude=float(A_fit),
                offset=float(offset_fit),
                success=success,
            )
        except Exception:
            fit_results[str(q)] = CoherentT1Fit(
                T1_ns=float("nan"),
                T1_error_ns=float("nan"),
                nbar0=float("nan"),
                amplitude=float("nan"),
                offset=float("nan"),
                success=False,
            )

    return ds, fit_results
