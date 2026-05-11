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
    """Convert raw IQ streams to physical units, subtract the cross-Kerr baseline,
    and optionally derive a vacuum-population probability for state discrimination.

    Two modes are supported, selected by node.parameters.subtract_baseline:

    subtract_baseline = True  (dual-sequence, cross-Kerr correction)
    ----------------------------------------------------------------
    The QUA program acquires two independent sub-sequences per amplitude point:
      * Signal   (I, Q)   — cavity displaced + qubit π-pulse.
      * Baseline (Ib, Qb) — cavity displaced + NO qubit π-pulse.

    Step 1 — Convert to Volts
        Apply (×2^12 / readout_length) to I, Q, Ib, Qb so they share the same scale.

    Step 2 — Baseline subtraction  (IQ domain, BEFORE any threshold)
        I_corr(a) = I(a) − Ib(a)  ≈  P_vacuum(a) · (I_e − I_g)
        Q_corr(a) = Q(a) − Qb(a)
        Removes the amplitude-dependent cross-Kerr resonator offset.

    Step 3 — State discrimination  (optional, applied to I_corr)
        If use_state_discrimination is True, I_corr is normalised per-qubit to
        [0, 1] using the peak value at the smallest |amplitude| points as the
        reference (≈ I_e − I_g), giving a continuous P_vacuum estimate.
        A hard binary threshold is NOT used here because thresholding and
        averaging do not commute: applying I_corr > thr to the *averaged* I_corr
        (one float per point) would yield a step function, not a smooth Gaussian.
        The resulting 'state' variable is passed to the confusion-matrix correction.

    subtract_baseline = False  (original single-sequence)
    -------------------------------------------------------
    The QUA program runs only the signal sequence.  When use_state_discrimination
    is True, the binary state was computed per-shot in QUA (average of 0/1 outcomes
    = proper probability) and is already present as 'state' in the dataset.
    When use_state_discrimination is False, I is converted to Volts and used
    directly.  This path exactly reproduces the original node behaviour.
    """
    qubits = node.namespace["qubits"]

    # -----------------------------------------------------------------------
    # PATH A — dual-sequence with cross-Kerr baseline subtraction
    # -----------------------------------------------------------------------
    if node.parameters.subtract_baseline:
        # Step 1: Convert all four quadrature streams to Volts.
        # The same readout-length normalisation applies to both signal (I, Q) and
        # baseline (Ib, Qb) because they use the same readout operation.
        ds = convert_IQ_to_V(ds, qubits, IQ_list=["I", "Q", "Ib", "Qb"])

        # Step 2: Baseline subtraction in the IQ plane.
        # Ib(a) ≈ cross-Kerr shift(a) + I_g  (qubit in |g⟩, no π-pulse).
        # I(a)  = cross-Kerr shift(a) + P_vac(a)·I_e + (1−P_vac(a))·I_g.
        # I_corr = I − Ib  ≈  P_vac(a) · (I_e − I_g).
        # ds["Ib"] and ds["Qb"] are retained for diagnostics but not used further.
        ds = ds.assign(
            I=ds["I"] - ds["Ib"],   # cross-Kerr-corrected in-phase quadrature
            Q=ds["Q"] - ds["Qb"],   # cross-Kerr-corrected quadrature (informational)
        )

        # Step 3: Optional state discrimination on the baseline-corrected I.
        if node.parameters.use_state_discrimination:
            state_arrays = []
            amp_values = ds["amp"].values

            for q in qubits:
                I_q = ds["I"].sel(qubit=q.name)

                # Estimate the peak (≈ I_e − I_g) from the points closest to a = 0
                # (smallest |amplitude|), where P_vacuum ≈ 1 and the qubit is most
                # likely to be flipped.  Average the bottom 10 % (≥ 1 point) to
                # reduce noise on the estimate.
                n_peak = max(1, len(amp_values) // 10)
                peak_idx = np.argsort(np.abs(amp_values))[:n_peak]
                peak = float(I_q.values[peak_idx].mean())

                if abs(peak) < 1e-9:
                    # Degenerate / no-signal case: keep raw I_corr so the fit still
                    # has a finite signal to work with.
                    state_q = xr.DataArray(
                        I_q.values.copy(), coords=I_q.coords, dims=I_q.dims,
                    )
                else:
                    # Normalise I_corr to [0, 1]  ≈  P_vacuum(a).
                    # Clip to [0, 1] to keep the probability physically meaningful
                    # in the presence of noise or imperfect baseline cancellation.
                    state_q = xr.DataArray(
                        np.clip(I_q.values / peak, 0.0, 1.0),
                        coords=I_q.coords,
                        dims=I_q.dims,
                    )

                state_arrays.append(state_q)

            # Stack per-qubit arrays back along the qubit dimension.
            state = xr.concat(state_arrays, dim="qubit").assign_coords(qubit=ds.qubit)
            ds = ds.assign(state=state)

            # Apply confusion-matrix correction.
            # P_true = (P_meas − ge) / (ee − ge) is linear in P_meas and applies
            # equally well to the continuous probability estimate above.
            ds = apply_confusion_correction_to_dataset(ds, node)

    # -----------------------------------------------------------------------
    # PATH B — original single-sequence (no baseline subtraction)
    # -----------------------------------------------------------------------
    else:
        if node.parameters.use_state_discrimination:
            # 'state' was computed per-shot in QUA (average of 0/1 binary outcomes)
            # and is already present in the dataset as a proper probability.
            # Apply confusion-matrix correction directly.
            ds = apply_confusion_correction_to_dataset(ds, node)
        else:
            # No state discrimination: convert raw I/Q to Volts for fitting.
            ds = convert_IQ_to_V(ds, qubits)

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
    # Choose the fit signal:
    #   'state' — normalised P_vacuum ∈ [0, 1], produced by process_raw_dataset
    #             when use_state_discrimination=True (confusion-corrected).
    #   'I'     — baseline-corrected I quadrature in Volts (continuous signal).
    # In both cases the variable follows the Gaussian model
    #   f(a) = amplitude · exp(-(a/sigma)²) + offset
    # and yields the same sigma; only the amplitude scale differs.
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
