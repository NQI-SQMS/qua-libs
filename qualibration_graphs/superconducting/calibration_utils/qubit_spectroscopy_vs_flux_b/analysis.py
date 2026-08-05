"""Analysis utilities for qubit spectroscopy versus flux calibration."""

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import xarray as xr
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from qualibrate import QualibrationNode
from qualibration_libs.analysis import peaks_dips
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V


@dataclass
class FitParameters:
    """Stores the relevant node-specific fitted parameters used to update the state at the end of the node.

    ``dv_phi0`` / ``phi0_current`` / ``m_pH`` are retained (always NaN here) only
    so the saved schema stays identical to node 09; this node fits a local
    quadratic (parabola) on the arch and has no sinusoid period to derive them from.
    """

    success: bool
    qubit_frequency: float
    frequency_shift: float
    idle_offset: float
    dv_phi0: float = float("nan")
    phi0_current: float = float("nan")
    m_pH: float = float("nan")
    upper_sweet_spot_flux: float = float("nan")
    upper_sweet_spot_frequency: float = float("nan")
    lower_sweet_spot_flux: float = float("nan")
    lower_sweet_spot_frequency: float = float("nan")


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Logs the node-specific fitted results for all qubits from the fit results."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_idle_offset = f"\tidle offset: {fit_results[q]['idle_offset'] * 1e3:.0f} mV | "
        s_freq = f"Qubit frequency: {1e-9 * fit_results[q]['qubit_frequency']:.3f} GHz | "
        s_shift = f"(shift of {1e-6 * fit_results[q]['frequency_shift']:.0f} MHz)\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        s_sweet = ""
        if not np.isnan(fit_results[q].get("upper_sweet_spot_frequency", float("nan"))):
            s_sweet += (
                f"\tupper sweet spot: {fit_results[q]['upper_sweet_spot_flux'] * 1e3:.1f} mV / "
                f"{1e-9 * fit_results[q]['upper_sweet_spot_frequency']:.4f} GHz\n"
            )
        if not np.isnan(fit_results[q].get("lower_sweet_spot_frequency", float("nan"))):
            s_sweet += (
                f"\tlower sweet spot: {fit_results[q]['lower_sweet_spot_flux'] * 1e3:.1f} mV / "
                f"{1e-9 * fit_results[q]['lower_sweet_spot_frequency']:.4f} GHz\n"
            )
        log_callable(s_qubit + s_idle_offset + s_freq + s_shift + s_sweet)


def _resolve_target_flux(qubit, user_target):
    """Resolve target flux: user override > stored offset > 0.0 fallback."""
    if user_target is not None:
        return float(user_target)
    flux_point = getattr(getattr(qubit, "z", None), "flux_point", None)
    val = None
    if flux_point == "joint":
        val = getattr(qubit.z, "joint_offset", None)
    elif flux_point == "independent":
        val = getattr(qubit.z, "independent_offset", None)
    try:
        return float(val) if val is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _isolate_primary_arch(q_peaks, target_flux):
    """Isolate a single arch around target_flux from peak_position vs flux.

    Strategy:
    1. Drop NaN, median-filter peak-position vs flux.
    2. Find local maxima with scipy.signal.find_peaks; pick the one whose
       flux is closest to target_flux.
    3. Find local minima of the same filtered curve; the closest minima on
       the left and right of the chosen maximum bound the arch window.

    Returns ``(arch_flux, arch_freq_filt, arch_freq_raw, info)`` or ``None``
    if not enough data. ``arch_freq_filt`` is the median-filtered curve used
    for robust peak/min detection and parabola fitting; ``arch_freq_raw``
    contains the original measured peak positions for accurate sweet-spot
    reporting.
    """
    valid = q_peaks.dropna(dim="flux_bias")
    n_valid = valid.sizes["flux_bias"]
    if n_valid < 5:
        return None

    flux_vals = valid.flux_bias.values
    freq_vals = valid.values.squeeze()
    if freq_vals.ndim == 0:
        return None

    kernel = max(3, n_valid // 10) | 1
    filtered = median_filter(freq_vals, size=kernel)

    max_idx, _ = find_peaks(filtered)
    if len(max_idx) == 0:
        max_idx = np.array([int(np.argmax(filtered))])

    distances = np.abs(flux_vals[max_idx] - target_flux)
    idx_center = int(max_idx[int(np.argmin(distances))])

    min_idx, _ = find_peaks(-filtered)
    left_candidates = min_idx[min_idx < idx_center]
    right_candidates = min_idx[min_idx > idx_center]
    idx_left = int(left_candidates.max()) if len(left_candidates) > 0 else 0
    idx_right = int(right_candidates.min()) if len(right_candidates) > 0 else n_valid - 1

    if idx_right - idx_left < 4:
        idx_left = 0
        idx_right = n_valid - 1

    arch_flux = flux_vals[idx_left : idx_right + 1]
    arch_freq_filt = filtered[idx_left : idx_right + 1]
    arch_freq_raw = freq_vals[idx_left : idx_right + 1]
    info = {
        "idx_center": idx_center,
        "idx_left": idx_left,
        "idx_right": idx_right,
        "n_valid": n_valid,
        "target_flux": float(target_flux),
    }
    return arch_flux, arch_freq_filt, arch_freq_raw, info


def _fit_parabola_per_qubit(arch_flux, arch_freq):
    """Fit f(x) = c2*x^2 + c1*x + c0 to arch points.

    Returns None if too few points or if c2 >= 0 (upward-opening parabola
    means we did not capture an arch top).
    """
    if arch_flux.size < 5:
        return None
    coefs = np.polyfit(arch_flux, arch_freq, deg=2)
    c2, c1, c0 = float(coefs[0]), float(coefs[1]), float(coefs[2])
    if not np.isfinite(c2) or c2 >= 0:
        return None
    v0 = -c1 / (2.0 * c2)
    f0 = c0 - (c1 * c1) / (4.0 * c2)
    return {"c2": c2, "c1": c1, "c0": c0, "v0": float(v0), "f0": float(f0)}


def _validate_lower_ss_with_parabola(
    arch_flux, arch_freq_filt, arch_freq_raw, parabola, tol_hz
):
    """Find lowest measured freq within arch window and accept as lower SS
    only if it lies on/near the fitted parabola.

    Order:
    1. (Already done) parabola fit on the filtered arch curve.
    2. Use the median-FILTERED curve to robustly locate the index of the
       minimum (avoids being pulled by spurious noisy points).
    3. Report the RAW measured frequency at that flux as the lower SS value
       (the filtered value would be biased toward the median of the kernel
       window and shifts the apparent minimum upward, especially near the
       arch edges).
    4. Validate by comparing the raw measured value against the parabola
       prediction at that flux: accept if the gap is within ``tol_hz``.
    """
    if arch_flux.size == 0:
        return float("nan"), float("nan")
    idx_min = int(np.argmin(arch_freq_filt))
    min_flux = float(arch_flux[idx_min])
    min_freq_raw = float(arch_freq_raw[idx_min])

    pred = parabola["c2"] * min_flux * min_flux + parabola["c1"] * min_flux + parabola["c0"]
    if abs(min_freq_raw - pred) <= tol_hz:
        return min_flux, min_freq_raw
    return float("nan"), float("nan")


def process_raw_dataset(ds, node):
    """Process raw dataset: convert I/Q to V, add amplitude/phase, RF freq coord."""
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    full_freq = np.array([ds.detuning + q.xy.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    current = ds.flux_bias / node.parameters.input_line_impedance_in_ohm
    ds = ds.assign_coords({"current": (["flux_bias"], current.data)})
    ds.current.attrs["long_name"] = "Current"
    ds.current.attrs["units"] = "A"
    attenuation_factor = 10 ** (-node.parameters.line_attenuation_in_db / 20)
    attenuated_current = ds.current * attenuation_factor
    ds = ds.assign_coords({"attenuated_current": (["flux_bias"], attenuated_current.values)})
    ds.attenuated_current.attrs["long_name"] = "Attenuated Current"
    ds.attenuated_current.attrs["units"] = "A"
    return ds


def fit_raw_data(ds, node):
    """Fit qubit frequency vs flux with a local quadratic (parabola) on the arch."""
    peak_freq = peaks_dips(ds.IQ_abs, dim="detuning", prominence_factor=5)
    return _fit_raw_data_parabola(peak_freq, node)


def _fit_raw_data_parabola(peak_freq, node):
    """Parabola path: isolate primary arch around target_flux, fit quadratic,
    report sweet spots from vertex."""
    qubits = node.namespace["qubits"]
    user_target = getattr(node.parameters, "target_flux", None)
    half_span = node.parameters.flux_offset_span_in_v / 2
    tol_hz = float(node.parameters.frequency_span_in_mhz) * 1e6 * 0.2

    qubit_names = list(peak_freq.position.qubit.values)
    per_qubit = {}
    for q in qubit_names:
        qb = next((x for x in qubits if x.name == q), None)
        rf_freq = qb.xy.RF_frequency if qb is not None else 0.0
        target_flux = _resolve_target_flux(qb, user_target) if qb is not None else 0.0

        q_peaks = peak_freq.position.sel(qubit=[q])
        arch = _isolate_primary_arch(q_peaks, target_flux)
        parabola = None
        lower = (float("nan"), float("nan"))
        arch_flux_min = float("nan")
        arch_flux_max = float("nan")
        if arch is not None:
            arch_flux, arch_freq_filt, arch_freq_raw, _info = arch
            # Use the (more robust) filtered curve for parabola fitting.
            parabola = _fit_parabola_per_qubit(arch_flux, arch_freq_filt)
            if parabola is not None:
                lower = _validate_lower_ss_with_parabola(
                    arch_flux,
                    arch_freq_filt,
                    arch_freq_raw,
                    parabola,
                    tol_hz=tol_hz,
                )
                arch_flux_min = float(arch_flux.min())
                arch_flux_max = float(arch_flux.max())
        if parabola is None:
            parabola = {
                "c2": float("nan"),
                "c1": float("nan"),
                "c0": float("nan"),
                "v0": float("nan"),
                "f0": float("nan"),
            }

        v0 = parabola["v0"]
        f0 = parabola["f0"]
        idle_offset = v0
        qubit_frequency = float(f0 + rf_freq) if np.isfinite(f0) else float("nan")
        upper_ss_flux = v0
        upper_ss_freq = qubit_frequency

        in_range = np.isfinite(idle_offset) and abs(idle_offset) <= half_span
        success = bool(in_range and np.isfinite(qubit_frequency))

        per_qubit[q] = {
            "rf_freq": rf_freq,
            "target_flux": target_flux,
            "parabola": parabola,
            "arch_flux_min": arch_flux_min,
            "arch_flux_max": arch_flux_max,
            "idle_offset": idle_offset,
            "qubit_frequency": qubit_frequency,
            "frequency_shift": float(f0) if np.isfinite(f0) else float("nan"),
            "upper_sweet_spot_flux": float(upper_ss_flux) if np.isfinite(upper_ss_flux) else float("nan"),
            "upper_sweet_spot_frequency": float(upper_ss_freq) if np.isfinite(upper_ss_freq) else float("nan"),
            "lower_sweet_spot_flux": float(lower[0]),
            "lower_sweet_spot_frequency": (
                float(lower[1] + rf_freq) if np.isfinite(lower[1]) else float("nan")
            ),
            "success": success,
        }

    fit = _build_parabola_fit_dataset(qubit_names, per_qubit, peak_freq.position)

    fit_results_out = {
        q: FitParameters(
            success=per_qubit[q]["success"],
            qubit_frequency=per_qubit[q]["qubit_frequency"],
            frequency_shift=per_qubit[q]["frequency_shift"],
            idle_offset=per_qubit[q]["idle_offset"],
            upper_sweet_spot_flux=per_qubit[q]["upper_sweet_spot_flux"],
            upper_sweet_spot_frequency=per_qubit[q]["upper_sweet_spot_frequency"],
            lower_sweet_spot_flux=per_qubit[q]["lower_sweet_spot_flux"],
            lower_sweet_spot_frequency=per_qubit[q]["lower_sweet_spot_frequency"],
        )
        for q in qubit_names
    }
    return fit, fit_results_out


def _attach_fit_attrs(fit):
    """Attach standard long_name/units attributes to the common fit dataset coords."""
    fit.idle_offset.attrs = {"long_name": "idle flux bias", "units": "V"}
    fit.freq_shift.attrs = {"long_name": "frequency shift", "units": "Hz"}
    fit.sweet_spot_frequency.attrs = {"long_name": "sweet spot frequency", "units": "Hz"}
    fit.upper_sweet_spot_flux.attrs = {"long_name": "upper sweet spot flux bias", "units": "V"}
    fit.upper_sweet_spot_frequency.attrs = {"long_name": "upper sweet spot frequency", "units": "Hz"}
    fit.lower_sweet_spot_flux.attrs = {"long_name": "lower sweet spot flux bias", "units": "V"}
    fit.lower_sweet_spot_frequency.attrs = {"long_name": "lower sweet spot frequency", "units": "Hz"}


def _build_parabola_fit_dataset(qubit_names, per_qubit, peak_freq_position):
    """Assemble the parabola-mode fit dataset from per-qubit results.

    Centralises what would otherwise be a long, repetitive ``xr.Dataset``
    construction. ``key`` looks up a top-level field of the per-qubit dict
    while ``key, sub`` looks up a nested field.
    """
    def col(key, sub=None):
        if sub is None:
            return ("qubit", np.array([per_qubit[q][key] for q in qubit_names]))
        return ("qubit", np.array([per_qubit[q][key][sub] for q in qubit_names]))

    fit = xr.Dataset(
        coords={
            "qubit": qubit_names,
            "c2": col("parabola", "c2"),
            "c1": col("parabola", "c1"),
            "c0": col("parabola", "c0"),
            "idle_offset": col("idle_offset"),
            "freq_shift": col("frequency_shift"),
            "sweet_spot_frequency": col("qubit_frequency"),
            "arch_flux_min": col("arch_flux_min"),
            "arch_flux_max": col("arch_flux_max"),
            "target_flux": col("target_flux"),
            "upper_sweet_spot_flux": col("upper_sweet_spot_flux"),
            "upper_sweet_spot_frequency": col("upper_sweet_spot_frequency"),
            "lower_sweet_spot_flux": col("lower_sweet_spot_flux"),
            "lower_sweet_spot_frequency": col("lower_sweet_spot_frequency"),
            "success": col("success"),
        }
    )
    fit = fit.assign(peak_freq=peak_freq_position)
    _attach_fit_attrs(fit)
    fit.attrs["fit_model"] = "parabola"
    return fit
