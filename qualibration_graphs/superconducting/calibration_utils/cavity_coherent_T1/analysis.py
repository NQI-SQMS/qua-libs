"""Analysis for the cavity coherent T1 node (33).

Experiment:
  The cavity is prepared in a coherent state |alpha> by playing the displacement
  pulse at amplitude_scale = displacement_alpha / displacement_alpha_max.
  After a variable wait time t_eff the qubit selective pi-pulse is applied.
  The selective pulse only flips the qubit when the cavity is in vacuum (n=0),
  so the measured qubit excited-state probability is proportional to P(n=0):

      P(n=0 | coherent state with nbar) = exp(-nbar)

  Since nbar decays exponentially:

      nbar(t) = nbar_0 * exp(-t / T1)    with nbar_0 = |alpha_0|^2

  the measured signal is:

      P_e(t) = A * exp(-nbar_0 * exp(-t / T1)) + offset

  This is a "double exponential" (Gumbel-like) decay.  Fitting it extracts
  the cavity T1 and the initial photon number nbar_0 = |alpha_0|^2.

  After fitting, the dataset is augmented with the inferred photon-number
  decay nbar_data(t) obtained by inverting the Gumbel model:

      nbar_data(t) = -ln((P_e(t) - offset_fit) / A_fit)

  This allows plotting |alpha(t)|^2 = nbar(t) directly as a simple exponential.

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
from qualibration_libs.data import convert_IQ_to_V
try:
    from qualibration_libs.data import apply_confusion_correction_to_dataset
except ImportError:
    def apply_confusion_correction_to_dataset(ds, node):
        raise NotImplementedError("apply_confusion_correction_to_dataset not available in this qualibration_libs version")


@dataclass
class CoherentT1Fit:
    """Fit results for a single qubit / cavity mode."""
    T1_ns: float
    """Fitted cavity T1 [ns]."""
    T1_error_ns: float
    """1-sigma uncertainty on T1 [ns]."""
    nbar0: float
    """Fitted initial mean photon number |alpha_0|^2."""
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
            f"| nbar0 = |alpha_0|^2 = {res['nbar0']:.2f}"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert raw IQ streams to physical units, subtract the cross-Kerr baseline,
    and optionally derive a vacuum-population probability for state discrimination.

    Two modes are supported, selected by node.parameters.subtract_baseline:

    subtract_baseline = True  (dual-sequence, cross-Kerr correction)
    ----------------------------------------------------------------
    The QUA program acquires two independent sub-sequences per time point:
      * Signal   (I, Q)   — cavity displaced + wait t + selective qubit π-pulse.
      * Baseline (Ib, Qb) — cavity displaced + wait t + NO qubit π-pulse.

    Step 1 — Convert to Volts
        Apply (×2^12 / readout_length) to I, Q, Ib, Qb.

    Step 2 — Baseline subtraction  (IQ domain, BEFORE any threshold)
        I_corr(t) = I(t) − Ib(t)  ≈  P_vacuum(t) · (I_e − I_g)
        Q_corr(t) = Q(t) − Qb(t)
        Removes the time-dependent cross-Kerr resonator offset (which changes as
        the cavity photon number decays during the T1 sweep).

    Step 3 — State discrimination  (optional, applied to I_corr)
        If use_state_discrimination is True, I_corr is normalised per-qubit to
        [0, 1] using the tail values at the longest wait times as the reference
        (≈ I_e − I_g at t→∞ where P_vacuum → 1), giving a continuous P_vacuum
        estimate.  A hard binary threshold is NOT used here because thresholding
        and averaging do not commute.

    subtract_baseline = False  (original single-sequence)
    -------------------------------------------------------
    When use_state_discrimination is True, the binary state was computed per-shot
    in QUA and is already present as 'state' in the dataset.
    When use_state_discrimination is False, I is converted to Volts and used
    directly.
    """
    qubits = node.namespace["qubits"]

    # -----------------------------------------------------------------------
    # PATH A — dual-sequence with cross-Kerr baseline subtraction
    # -----------------------------------------------------------------------
    if node.parameters.subtract_baseline:
        # Step 1: Convert all four quadrature streams to Volts.
        ds = convert_IQ_to_V(ds, qubits, IQ_list=["I", "Q", "Ib", "Qb"])

        # Step 2: Baseline subtraction in the IQ plane.
        # Ib(t) ≈ cross-Kerr shift(t) + I_g  (qubit in |g⟩, no π-pulse).
        # I(t)  = cross-Kerr shift(t) + P_vac(t)·I_e + (1−P_vac(t))·I_g.
        # I_corr = I − Ib  ≈  P_vac(t) · (I_e − I_g).
        ds = ds.assign(
            I=ds["I"] - ds["Ib"],   # cross-Kerr-corrected in-phase quadrature
            Q=ds["Q"] - ds["Qb"],   # cross-Kerr-corrected quadrature (informational)
        )

        # Step 3: Optional state discrimination on the baseline-corrected I.
        if node.parameters.use_state_discrimination:
            state_arrays = []
            t_values = ds["idle_time"].values

            for q in qubits:
                I_q = ds["I"].sel(qubit=q.name)

                # Estimate the peak (≈ I_e − I_g) from the points at the longest
                # wait times, where the cavity has decayed to vacuum (P_vacuum → 1).
                # Average the top 10% (≥ 1 point) to reduce noise on the estimate.
                n_peak = max(1, len(t_values) // 10)
                peak_idx = np.argsort(t_values)[-n_peak:]
                peak = float(I_q.values[peak_idx].mean())

                if abs(peak) < 1e-9:
                    state_q = xr.DataArray(
                        I_q.values.copy(), coords=I_q.coords, dims=I_q.dims,
                    )
                else:
                    state_q = xr.DataArray(
                        np.clip(I_q.values / peak, 0.0, 1.0),
                        coords=I_q.coords,
                        dims=I_q.dims,
                    )

                state_arrays.append(state_q)

            state = xr.concat(state_arrays, dim="qubit").assign_coords(qubit=ds.qubit)
            ds = ds.assign(state=state)
            ds = apply_confusion_correction_to_dataset(ds, node)

    # -----------------------------------------------------------------------
    # PATH B — original single-sequence (no baseline subtraction)
    # -----------------------------------------------------------------------
    else:
        if node.parameters.use_state_discrimination:
            ds = apply_confusion_correction_to_dataset(ds, node)
        else:
            ds = convert_IQ_to_V(ds, qubits)

    return ds


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def coherent_T1_model(t_ns: np.ndarray, A: float, nbar0: float, T1_ns: float, offset: float) -> np.ndarray:
    """Double-exponential (Gumbel) decay for coherent-state T1 measurement.

    P_e(t) = A * exp(-nbar0 * exp(-t / T1)) + offset

    Parameters
    ----------
    t_ns : array-like
        Physical wait time [ns] (= delay_repeats * clock_cycles * 4).
    A : float
        Signal amplitude.
    nbar0 : float
        Initial mean photon number |alpha_0|^2.
    T1_ns : float
        Cavity T1 [ns].
    offset : float
        Signal baseline.
    """
    return A * np.exp(-nbar0 * np.exp(-np.asarray(t_ns) / T1_ns)) + offset


def invert_gumbel_to_nbar(signal: np.ndarray, A: float, offset: float) -> np.ndarray:
    """Invert the Gumbel model to recover nbar(t) = |alpha(t)|^2.

    nbar(t) = -ln((P_e(t) - offset) / A)

    Points where (P_e - offset) / A <= 0 are masked to NaN.
    """
    ratio = (signal - offset) / A
    valid = ratio > 0
    nbar = np.full_like(signal, np.nan, dtype=float)
    nbar[valid] = -np.log(ratio[valid])
    return nbar


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, CoherentT1Fit]]:
    """Fit the coherent T1 double-exponential model to the dataset.

    After fitting, augments the dataset with a 'nbar_data' variable containing
    the inferred |alpha(t)|^2 = nbar(t) obtained by inverting the Gumbel model
    using the fitted A and offset for each qubit.
    """
    signal_name = "state" if node.parameters.use_state_discrimination else "I"
    nbar0_guess = node.parameters.displacement_alpha ** 2
    fit_results: Dict[str, CoherentT1Fit] = {}

    nbar_data_list = []

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
            # Invert the Gumbel model to get nbar(t) = |alpha(t)|^2
            nbar_data_list.append(invert_gumbel_to_nbar(signal, A_fit, offset_fit))
        except Exception:
            fit_results[str(q)] = CoherentT1Fit(
                T1_ns=float("nan"),
                T1_error_ns=float("nan"),
                nbar0=float("nan"),
                amplitude=float("nan"),
                offset=float("nan"),
                success=False,
            )
            nbar_data_list.append(np.full(len(t_ns), np.nan))

    # Add nbar_data variable to the dataset (shape: qubit × idle_time)
    ds = ds.assign(
        nbar_data=xr.DataArray(
            np.array(nbar_data_list),
            dims=["qubit", "idle_time"],
            attrs={"long_name": "|α(t)|²  (photon number)", "units": "photons"},
        )
    )

    return ds, fit_results
