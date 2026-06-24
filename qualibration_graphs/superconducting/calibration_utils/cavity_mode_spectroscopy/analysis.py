"""
Analysis utilities for the cavity mode spectroscopy measurement (node 27).

The sequence sweeps the cavity drive frequency and probes photon occupation via
a narrow-bandwidth qubit pulse (selective_x180). When the cavity drive is on
resonance with the cavity mode, photons are deposited; the dispersive coupling
shifts the qubit frequency by χ per photon, so the selective probe pulse goes
off-resonant and the qubit excitation probability drops.

The result is a DIP in qubit state vs. cavity drive detuning centred on the
bare cavity resonance. Fitting a Lorentzian dip gives the resonance frequency.

State update:
    cavity_mode.cavity_mode_drive.RF_frequency  →  cavity resonance frequency [Hz]
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
    """Fit results for a single qubit's cavity mode spectroscopy."""
    frequency_hz: float
    """Fitted cavity resonance frequency [Hz] (absolute RF frequency)."""
    fwhm_hz: float
    """Fitted FWHM of the dip [Hz]."""
    center_detuning_hz: float
    """Detuning offset from the current RF_frequency [Hz]."""
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        log_callable(
            f"Results for qubit {q}: {status}\n"
            f"\tCavity resonance: {1e-9 * res['frequency_hz']:.6f} GHz | "
            f"FWHM: {1e-6 * res['fwhm_hz']:.3f} MHz | "
            f"Detuning offset: {1e-6 * res['center_detuning_hz']:.3f} MHz"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert raw IQ streams to physical units and, when subtract_baseline=True,
    remove the cavity-probe-drive-induced IQ offset before any thresholding.

    subtract_baseline = True  (dual-sequence, drive-leakage correction)
    ----------------------------------------------------------------
    The QUA program acquires two independent sub-sequences per detuning point:
      * Signal   (I, Q)   - cavity probe drive + qubit probe pulse.
      * Baseline (Ib, Qb) - cavity probe drive + NO qubit probe pulse.

    I_corr(f) = I(f) - Ib(f), Q_corr(f) = Q(f) - Qb(f), in Volts, before any
    threshold. Removes the drive-induced offset that otherwise produces
    spurious peaks at higher probe power (mirrors
    displacement_calibration_vacuum's baseline subtraction).

    If use_state_discrimination is True, a continuous excitation-probability
    estimate is derived from I_corr by normalising against the points with
    the largest |detuning| (expected off-resonance, where the qubit should be
    fully excited by the probe pulse) -- the mirror image of
    displacement_calibration_vacuum's "points closest to a=0" reference.

    subtract_baseline = False  (original single-sequence)
    -------------------------------------------------------
    Unchanged: per-shot QUA state discrimination, or raw I in Volts.
    """
    qubits = node.namespace["qubits"]

    if node.parameters.subtract_baseline:
        ds = convert_IQ_to_V(ds, qubits, IQ_list=["I", "Q", "Ib", "Qb"])
        ds = ds.assign(
            I=ds["I"] - ds["Ib"],
            Q=ds["Q"] - ds["Qb"],
        )

        if node.parameters.use_state_discrimination:
            state_arrays = []
            detuning_values = ds["detuning"].values

            for q in qubits:
                I_q = ds["I"].sel(qubit=q.name)

                # Reference "peak" (fully excited, no dispersive shift) from
                # the tail of the detuning sweep -- the edges are expected to
                # be off-resonance.
                n_peak = max(1, len(detuning_values) // 10)
                peak_idx = np.argsort(np.abs(detuning_values))[-n_peak:]
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
    else:
        if not node.parameters.use_state_discrimination:
            ds = convert_IQ_to_V(ds, qubits)
        else:
            ds = apply_confusion_correction_to_dataset(ds, node)

    cavity_mode = _get_cavity_mode(node)
    rf_freq = cavity_mode.cavity_mode_drive.RF_frequency
    full_freq = np.array(
        [ds.detuning.values + rf_freq for _ in node.namespace["qubits"]]
    )
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def _lorentzian_dip(x, amplitude, center, hwhm, offset):
    """Lorentzian dip: y = offset - amplitude * hwhm² / (hwhm² + (x - center)²)"""
    return offset - amplitude * hwhm ** 2 / (hwhm ** 2 + (x - center) ** 2)


def _fit_lorentzian_dip(x: np.ndarray, y: np.ndarray):
    """Fit a Lorentzian dip; returns (center, fwhm, success)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    offset0 = float(np.max(y))
    i_min = int(np.argmin(y))
    center0 = float(x[i_min])
    amplitude0 = max(float(offset0 - np.min(y)), 1e-12)
    half_depth = offset0 - 0.5 * amplitude0
    try:
        left_idx = np.where(y[:i_min + 1] > half_depth)[0]
        right_idx = np.where(y[i_min:] > half_depth)[0]
        left = float(x[left_idx[-1]]) if len(left_idx) else float(x[0])
        right = float(x[i_min + right_idx[0]]) if len(right_idx) else float(x[-1])
        hwhm0 = max(0.5 * abs(right - left), float(abs(x[1] - x[0])))
    except (IndexError, ValueError):
        hwhm0 = float((x[-1] - x[0]) / 8)

    try:
        popt, _ = curve_fit(
            _lorentzian_dip, x, y,
            p0=[amplitude0, center0, hwhm0, offset0],
            bounds=([0, -np.inf, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
            maxfev=5000,
        )
        amplitude, center, hwhm, offset = popt
        fwhm = 2.0 * abs(hwhm)
        return float(center), float(fwhm), True
    except Exception:
        return center0, 2.0 * hwhm0, False


def _get_cavity_mode(node: QualibrationNode):
    mode_name = node.parameters.mode_name
    for cav in node.machine.cavities.values():
        mode = getattr(cav, mode_name, None)
        if mode is not None:
            return mode
    raise KeyError(f"Cavity mode '{mode_name}' not found in machine.cavities")


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit the cavity resonance dip for each qubit."""
    cavity_mode = _get_cavity_mode(node)
    rf_freq = cavity_mode.cavity_mode_drive.RF_frequency
    span_hz = node.parameters.frequency_span_in_mhz * 1e6
    min_dip = node.parameters.min_dip_fraction

    signal_name = "state" if node.parameters.use_state_discrimination else "I"

    fit_results = {}
    for q in ds.qubit.values:
        ds_q = ds.sel(qubit=q)
        x = ds_q.detuning.values.astype(float)
        y = getattr(ds_q, signal_name).values.astype(float)

        # Validate dip depth
        dip_depth = float(np.max(y) - np.min(y))
        signal_range = max(float(np.max(y)), 1e-12)
        if dip_depth < min_dip * signal_range:
            fit_results[q] = FitParameters(
                frequency_hz=rf_freq,
                fwhm_hz=float("nan"),
                center_detuning_hz=0.0,
                success=False,
            )
            continue

        center_det, fwhm, ok = _fit_lorentzian_dip(x, y)

        in_range = abs(center_det) < span_hz
        success = ok and np.isfinite(center_det) and in_range

        fit_results[q] = FitParameters(
            frequency_hz=rf_freq + center_det if success else rf_freq,
            fwhm_hz=fwhm,
            center_detuning_hz=center_det if success else 0.0,
            success=success,
        )

    return ds, fit_results
