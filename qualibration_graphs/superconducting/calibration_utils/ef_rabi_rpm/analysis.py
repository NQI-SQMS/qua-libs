"""
Analysis utilities for the EF Rabi RPM (f-state preparation check) experiment.

The sequence prepares |e⟩ via ge_π, sweeps the EF pulse amplitude, uses a
back-swap (ef_π → ge_π) to map f-state population to the measurable g/e basis,
and reads out with standard ge discrimination:

  ge_π → ef(a) → ef_π (back-swap) → ge_π → readout

The signal P(a) = cos²(π·a/2): starts at 1 (a=0 → back-swap leaves |f⟩),
minimum at a=1 (correct π pulse → |f⟩→|e⟩→|g⟩ after back-swaps), returns
to 1 at a=2.

The first minimum of the fitted sinusoid gives the EF π-pulse amplitude:
    new_ef_amplitude = a_min × current_ef_x180_amplitude
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
    """Fit results for a single qubit's EF Rabi RPM experiment."""
    pi_amp_factor: float
    """Amplitude scale factor at which the EF π rotation occurs (should be ≈1 if calibrated)."""
    pi_amp: float
    """Calibrated EF π-pulse amplitude [V]."""
    rabi_period: float
    """Full Rabi period in amplitude-factor units."""
    fit_A: float
    """Fitted cosine amplitude."""
    fit_f: float
    """Fitted cosine frequency [1/amp_factor_units]."""
    fit_phi: float
    """Fitted cosine phase [rad]."""
    fit_offset: float
    """Fitted cosine offset."""
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        log_callable(
            f"EF Rabi RPM results for qubit {q}: {status}\n"
            f"\tEF π-amp factor = {res['pi_amp_factor']:.3f} "
            f"(ideal = 1.000)  →  amplitude = {1e3*res['pi_amp']:.2f} mV"
        )


def _fit_cosine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, bool]:
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


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit the EF Rabi RPM oscillation and extract the π-pulse amplitude."""
    qubits = node.namespace["qubits"]
    signal = "state" if node.parameters.use_state_discrimination else "I"
    x = ds.amp_factor.values.astype(float)

    fit_results = {}
    fit_data = {}
    for qubit in qubits:
        q = qubit.name
        y = getattr(ds.sel(qubit=q), signal).values.astype(float)
        A, f, phi, off, ok = _fit_cosine(x, y)
        fit_data[q] = {"A": A, "f": f, "phi": phi, "offset": off}

        if ok and f > 0:
            # P(a) = A·cos(2π·f·a + phi) + off
            # Minimum (π-pulse) where derivative = 0 and second derivative > 0:
            # First minimum of A·cos(...): cos(2π·f·a + phi) = -1 → 2π·f·a + phi = π
            # a_min = (π - phi) / (2π·f), adjusted to be in [0, max_amp_factor]
            a_min_raw = (np.pi - phi) / (2 * np.pi * f)
            # Find the first positive minimum
            period = 1.0 / f
            a_min = a_min_raw % period
            if a_min <= 0:
                a_min += period

            current_amp = qubit.xy.operations[node.parameters.ef_operation].amplitude
            pi_amp = a_min * current_amp
            success = 0 < a_min <= node.parameters.max_amp_factor
        else:
            a_min = float("nan")
            pi_amp = float("nan")
            success = False

        fd = fit_data[q]
        fit_results[q] = FitParameters(
            pi_amp_factor=a_min,
            pi_amp=pi_amp,
            rabi_period=1.0 / f if ok and f > 0 else float("nan"),
            fit_A=fd["A"],
            fit_f=fd["f"],
            fit_phi=fd["phi"],
            fit_offset=fd["offset"],
            success=success,
        )

    return ds, fit_results
