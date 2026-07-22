"""Analysis utilities for Ramsey-based coupler flux long distortion characterization.

Maps the measured Ramsey phase at each delay time to an effective coupler flux
amplitude using a reference Ramsey amplitude sweep, then fits a sum of decaying
exponentials to the residual flux response.

Signal -> flux response pipeline mirrors the qubit Ramsey analysis (see
``qubit_flux_long_distortion_ramsey/analysis.py`` for the mermaid diagram
and the full mathematical derivation). The only physical differences are
that the long pulse is on the coupler (not the qubit) line, the readout
qubit is one of the qubit-pair members, and the IIR amplitude gets a 1.11
gain factor in ``update_state`` to compensate for the coupler-to-qubit
transduction asymmetry.

Fit model (finite-pulse, used when ``flux_settle_time_in_ns`` > 0)
[Aggarwal et al. arXiv:2503.08645 Appendix H, Eq. (H1), adapted]:

    y(t_delay) = a_dc + sum_i a_i (1 - exp(-T_pulse/tau_i)) exp(-t_delay/tau_i)

The amplitudes ``a_i`` returned by ``multi_exp_fit_global(..., t_pulse_ns=T_pulse)``
are de-attenuated step-response amplitudes, so the IIR coefficient formula
``A_i = a_i / a_dc`` (gain-normalized IIR tap; Rol et al. arXiv:1907.04818 Eq. (S22),
s(t)=g(1+A e^{-t/tau_IIR})u(t), with g <-> a_dc and A <-> a_i/a_dc) gives the
correct per-pole pre-distortion strength directly.
"""
# %%
from typing import Dict

import numpy as np
import xarray as xr
from calibration_utils.qubit_flux_long_distortion_qubitspec.analysis import (
    FluxDistortionExpFitResult,
    multi_exp_fit_global,
)
from qualibration_libs.data import convert_IQ_to_V

# %%
def _robust_unwrap_1d(phase):
    """Unwrap a 1-D phase array using linear-trend prediction.

    ``np.unwrap`` corrects jumps exceeding pi between consecutive samples,
    but fails when the *true* phase change between neighbours exceeds pi
    (common with log-spaced time axes).  This function predicts the
    expected value from linear extrapolation of the two preceding points
    and snaps to the nearest 2*pi branch of that prediction.
    """
    # Needed when the log-spaced time axis causes inter-sample phase steps larger than pi, which would cause standard np.unwrap to fold onto the wrong branch and corrupt the coupler flux amplitude mapping.
    period = 2 * np.pi
    out = np.array(phase, dtype=float)
    if len(out) < 2:
        return out
    for i in range(1, len(out)):
        predicted = out[i - 1] if i < 2 else 2 * out[i - 1] - out[i - 2]
        out[i] -= np.round((out[i] - predicted) / period) * period
    return out


def _fourier_phase(data: xr.DataArray, dim: str = "frame") -> xr.DataArray:
    """Extract oscillation phase via Fourier projection at the fundamental frequency.

    Projects the mean-subtracted signal onto cos(2*pi*frame) and sin(2*pi*frame)
    and returns arctan2 of the projections.  Unlike a nonlinear fit this is
    deterministic and immune to local-minimum convergence artifacts that cause
    spurious ~pi phase jumps.
    """
    # Replaces a nonlinear sinusoidal fit so the coupler Ramsey phase at each delay time is extracted robustly without risk of converging to a wrong local minimum.
    coord = data.coords[dim]
    cos_basis = xr.DataArray(np.cos(2 * np.pi * coord.values), dims=[dim], coords={dim: coord})
    sin_basis = xr.DataArray(np.sin(2 * np.pi * coord.values), dims=[dim], coords={dim: coord})

    centered = data - data.mean(dim=dim)
    cos_proj = (centered * cos_basis).sum(dim=dim)
    sin_proj = (centered * sin_basis).sum(dim=dim)
    return xr.apply_ufunc(np.arctan2, -sin_proj, cos_proj)


def _get_signal_and_ref_keys(ds: xr.Dataset) -> tuple[str, str | None]:
    """Return the data variable names for the signal and reference measurements."""
    # Handles both state-discrimination and raw-IQ datasets so the coupler Ramsey analysis pipeline remains agnostic to the measurement back-end.
    if "state" in ds.data_vars:
        signal_key = "state"
        ref_key = "state_ref" if "state_ref" in ds.data_vars else None
    elif "I" in ds.data_vars:
        signal_key = "I"
        ref_key = "I_ref" if "I_ref" in ds.data_vars else None
    else:
        raise ValueError("Dataset must contain 'state' or 'I' data variable")
    return signal_key, ref_key


def extract_ramsey_phase(ds: xr.Dataset) -> xr.DataArray:
    """Extract Ramsey phase from the signal measurement at each (qubit, time).

    Projects the signal onto cos/sin at the fundamental frame-rotation frequency.
    Returns the raw Fourier phase in [-pi, pi] without unwrapping along time,
    so that each delay point can be independently mapped to the reference
    calibration curve.
    """
    # Produces the per-delay-time phase values that will be inverted through the reference calibration curve to reconstruct the coupler flux amplitude at each step-response time point.
    signal_key, _ = _get_signal_and_ref_keys(ds)
    return _fourier_phase(ds[signal_key], "frame")


def extract_reference_calibration(ds: xr.Dataset) -> xr.DataArray | None:
    """Extract Ramsey phase vs coupler amplitude from the reference sweep.

    The reference Ramsey amplitude sweep provides a calibration curve mapping
    coupler flux amplitude to Ramsey phase.  This function extracts the
    Fourier phase along ``frame`` for each amplitude value and unwraps along
    the amplitude axis.

    Returns
    -------
    xr.DataArray or None
        Phase with dims ``(qubit, a)``, or ``None`` if no reference amplitude
        sweep is present in the dataset.
    """
    # Builds the coupler-flux-amplitude-to-phase lookup table; without this calibration curve the signal phase cannot be converted back to physical coupler flux units.
    _, ref_key = _get_signal_and_ref_keys(ds)
    if ref_key is None or ref_key not in ds.data_vars:
        return None
    ref_data = ds[ref_key]
    if "a" not in ref_data.dims:
        return None

    ref_phase = _fourier_phase(ref_data, "frame")
    ref_phase = xr.apply_ufunc(
        _robust_unwrap_1d,
        ref_phase,
        input_core_dims=[["a"]],
        output_core_dims=[["a"]],
        vectorize=True,
    )
    return ref_phase


def _map_phase_to_amplitude(
    sig_phases: np.ndarray,
    ref_amps: np.ndarray,
    ref_phases: np.ndarray,
) -> np.ndarray:
    """Invert the reference phase(amplitude) calibration curve.

    Each signal phase is snapped to the 2pi branch closest to the reference
    phase range before interpolation, so no prior unwrapping of the signal
    along time is required.  Values outside the calibration range are clipped
    to the boundary amplitude.
    """
    # Converts measured coupler Ramsey signal phases to effective coupler flux amplitudes point-by-point using linear interpolation on the sorted reference curve, 
    # avoiding the need for global phase unwrapping along the time axis.
    sort_idx = np.argsort(ref_phases)
    ref_phases_sorted = ref_phases[sort_idx]
    ref_amps_sorted = ref_amps[sort_idx]

    ref_center = 0.5 * (ref_phases_sorted[0] + ref_phases_sorted[-1])
    adjusted = sig_phases - np.round((sig_phases - ref_center) / (2 * np.pi)) * (2 * np.pi)

    return np.interp(adjusted, ref_phases_sorted, ref_amps_sorted)


def process_raw_dataset(ds: xr.Dataset, node) -> xr.Dataset:
    """Preprocess Ramsey raw dataset: convert IQ to volts if applicable."""
    # Normalises the raw hardware output into calibrated voltage units before Fourier phase extraction, matching the pre-processing convention used across all distortion calibration pipelines.
    if "I" in ds or "Q" in ds:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def fit_raw_data(ds: xr.Dataset, node) -> tuple[xr.Dataset, Dict[str, FluxDistortionExpFitResult]]:
    """Ramsey analysis: map signal phase to amplitude via reference calibration, fit exponentials.

    Steps
    -----
    1. Extract raw Ramsey phase at each delay time from the signal measurement.
    2. Extract phase-vs-amplitude calibration from the reference Ramsey sweep.
    3. For each delay, map the measured phase to the effective coupler amplitude
       by inverting the reference calibration (each point is independent, no
       unwrapping or late-time assumptions).
    4. Compute flux distortion = effective_amplitude - ramsey_flux_amplitude_in_v.
    5. Add ``coupler_flux_amplitude_in_v`` to obtain the total flux response.
    6. Fit a sum of decaying exponentials to the flux response.
    """
    # Main analysis entry point for the Ramsey-based coupler-flux distortion measurement; transforms raw Ramsey oscillation data into fitted exponential distortion parameters using the co-measured reference sweep as the phase-to-coupler-flux calibration.
    qubits = node.namespace["qubits"]

    signal_phase = extract_ramsey_phase(ds)
    ref_cal = extract_reference_calibration(ds)

    flux_response = xr.full_like(signal_phase, np.nan, dtype=float)
    ramsey_flux_amp = node.parameters.ramsey_flux_amplitude_in_v
    coupler_flux_amp = getattr(node.parameters, "coupler_flux_amplitude_in_v", None)

    if ref_cal is not None:
        ref_amps = ref_cal.coords["a"].values

        for q in qubits:
            ref_phases_q = ref_cal.sel(qubit=q.name).values
            sig_phases_q = signal_phase.sel(qubit=q.name).values

            eff_amp = _map_phase_to_amplitude(sig_phases_q, ref_amps, ref_phases_q)
            distortion = eff_amp - ramsey_flux_amp
            total_flux = -distortion + (coupler_flux_amp if coupler_flux_amp is not None else 0)
            flux_response.loc[{"qubit": q.name}] = total_flux
    else:
        print(
            "WARNING: No reference amplitude sweep found in dataset. "
            "Cannot map phase to flux — flux_response will be NaN."
        )

    ds = ds.copy()
    ds["signal_phase"] = signal_phase
    if ref_cal is not None:
        ds["ref_phase_cal"] = ref_cal
    ds["flux_response"] = flux_response

    fit_results: Dict[str, FluxDistortionExpFitResult] = {}
    t_pulse_ns = float(getattr(node.parameters, "flux_settle_time_in_ns", 0)) or None
    n_exponentials = int(getattr(node.parameters, "n_exponentials", 3))
    for q in qubits:
        t_data = flux_response.sel(qubit=q.name).time.values
        y_data = flux_response.sel(qubit=q.name).values
        # Mask out NaN samples and the t=0 origin so the log-time weighting and
        # exponential decomposition stay well-defined.
        mask = (~np.isnan(y_data)) & (t_data > 0)
        fit_results[q.name] = multi_exp_fit_global(
            t_data[mask],
            y_data[mask],
            n_exponentials=n_exponentials,
            t_pulse_ns=t_pulse_ns,
            verbose=True,
        )
    return ds, fit_results
