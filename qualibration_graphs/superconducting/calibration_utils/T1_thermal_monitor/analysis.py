"""
Analysis utilities for the T1 + thermal population monitor node (node 34).

Each iteration runs a full T1 decay sweep and a two-sequence RPM measurement.
Results are accumulated and stored as a combined xarray Dataset.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from lmfit import Model, Parameter

from qualibration_libs.data import convert_IQ_to_V
from qualibration_libs.analysis import fit_decay_exp


@dataclass
class IterResult:
    """Accumulated per-iteration result for a single qubit."""

    qubit: str
    t_sec: List[float] = field(default_factory=list)
    """Wall-clock elapsed time at each iteration [s]."""
    T1_us: List[float] = field(default_factory=list)
    """Fitted T1 [µs] at each iteration."""
    T1_error_us: List[float] = field(default_factory=list)
    """T1 fit uncertainty [µs]."""
    P_th: List[float] = field(default_factory=list)
    """Thermal population at each iteration."""
    T_eff_mk: List[float] = field(default_factory=list)
    """Effective qubit temperature [mK]."""


# ── T1 fitting ────────────────────────────────────────────────────────────────

def fit_T1_dataset(ds: xr.Dataset, use_state_discrimination: bool) -> Dict[str, dict]:
    """Fit an exponential decay to a single T1 sweep dataset.

    Returns {qubit: {"T1_us", "T1_error_us", "success"}}.
    """
    signal = ds.state if use_state_discrimination else ds.I
    fit_data = fit_decay_exp(signal, "idle_time")

    results = {}
    for q in ds.qubit.values:
        fd = fit_data.sel(qubit=q)
        decay = float(fd.sel(fit_vals="decay").values)
        decay_var = float(fd.sel(fit_vals="decay_decay").values)

        if decay >= 0 or not np.isfinite(decay):
            results[q] = {"T1_us": float("nan"), "T1_error_us": float("nan"), "success": False}
            continue

        tau_ns = -1.0 / decay
        tau_err_ns = tau_ns * (np.sqrt(max(decay_var, 0.0)) / abs(decay))
        success = (tau_ns > 16) and (tau_err_ns / tau_ns < 1.0)
        results[q] = {
            "T1_us": tau_ns * 1e-3,
            "T1_error_us": tau_err_ns * 1e-3,
            "success": success,
        }
    return results


# ── RPM fitting ───────────────────────────────────────────────────────────────

def _fit_sinusoid_amplitude(x: np.ndarray, y: np.ndarray) -> Tuple[float, bool]:
    """Fit A·cos(2π·f·x + φ) + offset to y and return (|A|, success)."""
    if len(x) < 5:
        return 0.0, False

    y_sub = y - np.mean(y)
    N = len(x)
    T = float(x[1] - x[0])
    yf = np.fft.rfft(y_sub)
    xf = np.fft.rfftfreq(N, T)
    mags = np.abs(yf[1:])
    freqs = xf[1:]

    if len(freqs) == 0 or np.max(mags) == 0:
        return 0.0, False

    idx = int(np.argmax(mags))
    f0 = float(freqs[idx])
    a0 = float(2 * mags[idx] / N)
    phi0 = float(np.angle(yf[1:][idx]))
    off0 = float(np.mean(y))

    def _cosine(x, A, f, phi, offset):
        return A * np.cos(2 * np.pi * f * x + phi) + offset

    try:
        result = Model(_cosine, independent_vars=["x"]).fit(
            y, x=x,
            A=Parameter("A", value=a0, min=0),
            f=Parameter("f", value=f0, min=float(freqs[0]) * 0.3, max=float(freqs[-1]) * 3),
            phi=Parameter("phi", value=phi0),
            offset=Parameter("offset", value=off0),
        )
        return abs(float(result.values["A"])), True
    except Exception:
        return 0.0, False


def _effective_temperature_mk(freq_hz: float, p_th: float) -> float:
    """Convert thermal population to effective temperature in mK."""
    if p_th <= 0 or p_th >= 1 or not np.isfinite(p_th):
        return float("nan")
    return (47.99 * freq_hz / 1e9) / np.log(1.0 / p_th - 1.0)


def fit_rpm_datasets(
    ds_g: xr.Dataset,
    ds_e: xr.Dataset,
    use_state_discrimination: bool,
    qubits,
) -> Dict[str, dict]:
    """Fit both RPM sweeps and extract thermal population for each qubit.

    Returns {qubit: {"P_th", "T_eff_mk", "success"}}.
    """
    signal = "state" if use_state_discrimination else "I"
    x = ds_g.amp_factor.values.astype(float)
    results = {}

    for qubit in qubits:
        q = qubit.name
        y_g = getattr(ds_g.sel(qubit=q), signal).values.astype(float)
        y_e = getattr(ds_e.sel(qubit=q), signal).values.astype(float)

        A_g, ok_g = _fit_sinusoid_amplitude(x, y_g)
        A_e, ok_e = _fit_sinusoid_amplitude(x, y_e)

        success = ok_g and ok_e and (A_g + A_e) > 1e-6
        if success:
            p_th = A_e / (A_e + A_g)
            t_eff = _effective_temperature_mk(qubit.f_01, p_th)
        else:
            p_th = float("nan")
            t_eff = float("nan")

        results[q] = {"P_th": p_th, "T_eff_mk": t_eff, "success": success}
    return results


# ── Dataset building ──────────────────────────────────────────────────────────

def build_dataset(iter_results: Dict[str, IterResult]) -> xr.Dataset:
    """Assemble accumulated IterResult objects into a single xarray Dataset.

    Dimensions: (qubit, iteration).
    Variables: T1_us, T1_error_us, P_th, T_eff_mk, elapsed_s.
    """
    qubit_names = list(iter_results.keys())
    n_iter = max(len(r.t_sec) for r in iter_results.values())

    def _pad(lst):
        arr = np.full(n_iter, float("nan"))
        arr[: len(lst)] = lst
        return arr

    coords = {"qubit": qubit_names, "iteration": np.arange(n_iter)}

    return xr.Dataset(
        {
            "T1_us": xr.DataArray(
                np.stack([_pad(iter_results[q].T1_us) for q in qubit_names]),
                dims=["qubit", "iteration"],
                attrs={"long_name": "T1 relaxation time", "units": "µs"},
            ),
            "T1_error_us": xr.DataArray(
                np.stack([_pad(iter_results[q].T1_error_us) for q in qubit_names]),
                dims=["qubit", "iteration"],
                attrs={"long_name": "T1 fit uncertainty", "units": "µs"},
            ),
            "P_th": xr.DataArray(
                np.stack([_pad(iter_results[q].P_th) for q in qubit_names]),
                dims=["qubit", "iteration"],
                attrs={"long_name": "thermal population", "units": ""},
            ),
            "T_eff_mk": xr.DataArray(
                np.stack([_pad(iter_results[q].T_eff_mk) for q in qubit_names]),
                dims=["qubit", "iteration"],
                attrs={"long_name": "effective temperature", "units": "mK"},
            ),
            "elapsed_s": xr.DataArray(
                np.stack([_pad(iter_results[q].t_sec) for q in qubit_names]),
                dims=["qubit", "iteration"],
                attrs={"long_name": "elapsed time", "units": "s"},
            ),
        },
        coords=coords,
    )


def reconstruct_iter_results(ds: xr.Dataset) -> Dict[str, IterResult]:
    """Reconstruct IterResult objects from a saved dataset (for load_data)."""
    results = {}
    for q in ds.qubit.values:
        q = str(q)
        ds_q = ds.sel(qubit=q)
        # Drop NaN-padded trailing entries
        valid = np.isfinite(ds_q.elapsed_s.values)
        results[q] = IterResult(
            qubit=q,
            t_sec=ds_q.elapsed_s.values[valid].tolist(),
            T1_us=ds_q.T1_us.values[valid].tolist(),
            T1_error_us=ds_q.T1_error_us.values[valid].tolist(),
            P_th=ds_q.P_th.values[valid].tolist(),
            T_eff_mk=ds_q.T_eff_mk.values[valid].tolist(),
        )
    return results


# ── Logging ───────────────────────────────────────────────────────────────────

def log_iteration(
    iteration: int,
    n_iter: int,
    t_sec: float,
    T1_fits: Dict[str, dict],
    rpm_fits: Dict[str, dict],
    log_callable=None,
):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    t_min = t_sec / 60.0
    parts = []
    for q in T1_fits:
        t1_s = f"T1={T1_fits[q]['T1_us']:.1f}µs" if T1_fits[q]["success"] else "T1=FAIL"
        p_s = (
            f"P_th={100 * rpm_fits[q]['P_th']:.3f}%"
            if rpm_fits[q]["success"] else "P_th=FAIL"
        )
        parts.append(f"{q}: {t1_s}  {p_s}")
    log_callable(
        f"Iter {iteration + 1}/{n_iter}  t={t_min:.1f} min  |  " + "  ".join(parts)
    )


def log_summary(iter_results: Dict[str, IterResult], log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, r in iter_results.items():
        T1_v = [v for v in r.T1_us if np.isfinite(v)]
        P_v = [v for v in r.P_th if np.isfinite(v)]
        if T1_v and P_v:
            log_callable(
                f"Summary {q}:  "
                f"T1 n={len(T1_v)}  mean={np.mean(T1_v):.2f}µs  std={np.std(T1_v):.2f}µs  |  "
                f"P_th n={len(P_v)}  mean={100*np.mean(P_v):.3f}%  std={100*np.std(P_v):.3f}%"
            )
        else:
            log_callable(f"Summary {q}: insufficient successful fits.")
