"""
Analysis utilities for the Rabi Population Measurement (RPM) experiment.

The RPM measures the qubit thermal population by comparing two EF Rabi sweeps:

  'g' sweep: ge_π → ef(a) → ef_π (back-swap) → ge_π → readout
             Starts from |g⟩ deterministically. The back-swap maps |f⟩→|e⟩→|g⟩
             and leaves |e⟩→|f⟩→|f⟩, so the readout signal is:
             P_g(a) = cos²(π·a/2) — oscillates 1→0→1 with minimum at a=1.

  'e' sweep: ef(a) → ge_π → readout
             Starts from the thermal state (P_th in |e⟩, 1-P_th in |g⟩).
             The |g⟩ component maps to |e⟩ after ge_π giving state=1 always;
             the |e⟩ component undergoes EF Rabi then back-maps via ge_π:
             P_e(a) = (1-P_th) + P_th·sin²(π·a/2)

The amplitudes A_g, A_e of the sinusoidal components give:
    P_th = A_e / (A_e + A_g)

References: T. Kim et al., Taeyoon RPM notebook (2026).
"""
import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from lmfit import Model, Parameter

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


@dataclass
class FitParameters:
    """Fit results for a single qubit's RPM experiment."""
    amp_g: float
    """Rabi oscillation amplitude (sinusoidal component) for the 'g' sweep."""
    amp_e: float
    """Rabi oscillation amplitude (sinusoidal component) for the 'e' sweep."""
    thermal_population: float
    """Estimated thermal population P_th = A_e / (A_e + A_g)."""
    effective_temperature_mk: float
    """Effective qubit temperature [mK] derived from P_th and qubit frequency."""
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        log_callable(
            f"RPM results for qubit {q}: {status}\n"
            f"\tA_g={res['amp_g']:.4f}  A_e={res['amp_e']:.4f}\n"
            f"\tP_th = ({100*res['thermal_population']:.3f} ± ?) %\n"
            f"\tT_eff = {res['effective_temperature_mk']:.1f} mK"
        )


def _fit_sinusoid(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, bool]:
    """Fit y = A·cos(2π·f·x + phi) + offset.  Returns (A, f, phi, offset, success)."""
    if len(x) < 5:
        return 0.0, 0.0, 0.0, float(np.mean(y)), False

    y_sub = y - np.mean(y)
    N = len(x)
    T = float(x[1] - x[0])
    yf = np.fft.rfft(y_sub)
    xf = np.fft.rfftfreq(N, T)
    mags = np.abs(yf[1:])
    freqs = xf[1:]
    if len(freqs) == 0 or np.max(mags) == 0:
        return 0.0, 0.0, 0.0, float(np.mean(y)), False

    idx = int(np.argmax(mags))
    f0 = float(freqs[idx])
    a0 = float(2 * mags[idx] / N)
    phi0 = float(np.angle(yf[1:][idx]))
    off0 = float(np.mean(y))

    def _cosine(x, A, f, phi, offset):
        return A * np.cos(2 * np.pi * f * x + phi) + offset

    try:
        model = Model(_cosine, independent_vars=["x"])
        result = model.fit(
            y, x=x,
            A=Parameter("A", value=a0, min=0),
            f=Parameter("f", value=f0, min=float(freqs[0]) * 0.3, max=float(freqs[-1]) * 3),
            phi=Parameter("phi", value=phi0),
            offset=Parameter("offset", value=off0),
        )
        A   = abs(float(result.values["A"]))
        f   = float(result.values["f"])
        phi = float(result.values["phi"])
        off = float(result.values["offset"])
        return A, f, phi, off, True
    except Exception:
        return 0.0, 0.0, 0.0, float(np.mean(y)), False


def _effective_temperature_mk(freq_hz: float, p_th: float) -> float:
    """Convert thermal population P_th to effective temperature in mK."""
    if p_th <= 0 or p_th >= 1:
        return float("nan")
    unit_temp_mk = 47.99  # mK per GHz photon energy
    freq_ghz = freq_hz / 1e9
    return unit_temp_mk * freq_ghz / np.log(1.0 / p_th - 1.0)


def process_raw_datasets(
    ds_g: xr.Dataset,
    ds_e: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Convert I/Q to Volts for both datasets if needed."""
    if not node.parameters.use_state_discrimination:
        ds_g = convert_IQ_to_V(ds_g, node.namespace["qubits"])
        ds_e = convert_IQ_to_V(ds_e, node.namespace["qubits"])
    return ds_g, ds_e


def fit_raw_data(
    ds_g: xr.Dataset,
    ds_e: xr.Dataset,
    node: QualibrationNode,
) -> Dict[str, FitParameters]:
    """Fit both sweeps and compute thermal population for each qubit."""
    qubits = node.namespace["qubits"]
    signal = "state" if node.parameters.use_state_discrimination else "I"
    x = ds_g.amp_factor.values.astype(float)

    fit_results = {}
    for qubit in qubits:
        q = qubit.name
        y_g = getattr(ds_g.sel(qubit=q), signal).values.astype(float)
        y_e = getattr(ds_e.sel(qubit=q), signal).values.astype(float)

        A_g, _, _, _, ok_g = _fit_sinusoid(x, y_g)
        A_e, _, _, _, ok_e = _fit_sinusoid(x, y_e)

        success = ok_g and ok_e and (A_g + A_e) > 0
        if success:
            p_th = A_e / (A_e + A_g)
            t_eff = _effective_temperature_mk(qubit.f_01, p_th)
        else:
            p_th = float("nan")
            t_eff = float("nan")

        fit_results[q] = FitParameters(
            amp_g=A_g,
            amp_e=A_e,
            thermal_population=p_th,
            effective_temperature_mk=t_eff,
            success=success,
        )
    return fit_results
