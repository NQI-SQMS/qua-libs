"""Analysis utilities for Ramsey-based qubit flux long distortion characterization.

Maps the measured Ramsey phase at each delay time to an effective qubit flux
amplitude using a reference Ramsey amplitude sweep, then fits a sum of decaying
exponentials to the residual flux response.

Signal -> flux response pipeline (mermaid):

    flowchart TD
        rawIQ["raw IQ vs frame for each t_delay"]
            --> phase["Fourier projection phi(t)=atan2(-sin_proj, cos_proj)"]
        refSweep["reference IQ vs frame for each ramsey_flux_amplitude"]
            --> refPhase["phi_ref(A_flux) (unwrapped along amplitude)"]
        phase --> invert["per-t inverse interp:\n  eff_amp(t) = phi_ref^-1(phi(t))"]
        refPhase --> invert
        invert --> distortion["residual flux:\n  delta(t) = eff_amp(t) - ramsey_flux_amp"]
        distortion --> stepResp["step-rise reformulation:\n  y(t) = qubit_flux_amp - delta(t)"]
        stepResp --> fit["multi_exp_fit_global(t_pulse_ns=flux_settle_time_in_ns)\n  -> a_dc, {(a_i, tau_i)}"]
        fit --> iir["IIR: A_i = a_i / a_dc, tau_i unchanged"]

Key equations
-------------
1. Ramsey phase accumulation during the wait window of length T_wait at
   instantaneous effective amplitude A_eff(t):

       phi(t_delay) = 2 pi * Delta_f(A_eff(t_delay)) * T_wait

2. Reference calibration (no preceding long pulse) gives phi_ref(A_flux);
   invert via 1-D interpolation to map measured phi -> A_eff.

3. Step-rise reformulation: a positive residual delta(t) (long-pulse tail
   that has not yet decayed) is recast as the equivalent step-rise response
   y(t) = qubit_flux_amp - delta(t). So at t = 0+ (immediately after the
   long pulse is turned off) y ~ 0, and at t -> infinity y -> qubit_flux_amp
   = a_dc.

4. Finite-pulse multi-exponential fit
   [Aggarwal et al. arXiv:2503.08645 Appendix H, Eq. (H1), adapted]:

       y(t_delay) = a_dc + sum_i a_i (1 - exp(-T_pulse/tau_i)) exp(-t_delay/tau_i)

   The multi_exp_fit_global(..., t_pulse_ns=T_pulse) call returns
   de-attenuated amplitudes a_i, so the IIR coefficient formula
   A_i = a_i / a_dc (gain-normalized IIR tap; Rol et al. arXiv:1907.04818 Eq. (S22),
   s(t)=g(1+A e^{-t/tau_IIR})u(t), with g <-> a_dc and A <-> a_i/a_dc) yields the
   correct per-pole pre-distortion strength directly.
"""
from typing import Dict

import numpy as np
import xarray as xr
from calibration_utils.qubit_flux_long_distortion_qubitspec.analysis import (
    FluxDistortionExpFitResult,
    multi_exp_fit_global,
)
from qualibration_libs.data import convert_IQ_to_V


def _robust_unwrap_1d(phase):
    """Unwrap a 1-D phase array using linear-trend prediction.

    ``np.unwrap`` corrects jumps exceeding pi between consecutive samples,
    but fails when the *true* phase change between neighbours exceeds pi
    (common with log-spaced time axes).  This function predicts the
    expected value from linear extrapolation of the two preceding points
    and snaps to the nearest 2*pi branch of that prediction.
    """
    # Needed when the log-spaced time axis causes inter-sample phase steps larger than pi, which would cause standard np.unwrap to fold onto the wrong branch and corrupt the flux mapping.
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
    # Replaces a nonlinear sinusoidal fit to robustly extract the Ramsey oscillation phase at each delay time point without risk of converging to wrong local minima.
    coord = data.coords[dim]
    cos_basis = xr.DataArray(np.cos(2 * np.pi * coord.values), dims=[dim], coords={dim: coord})
    sin_basis = xr.DataArray(np.sin(2 * np.pi * coord.values), dims=[dim], coords={dim: coord})

    centered = data - data.mean(dim=dim)
    cos_proj = (centered * cos_basis).sum(dim=dim)
    sin_proj = (centered * sin_basis).sum(dim=dim)
    return xr.apply_ufunc(np.arctan2, -sin_proj, cos_proj)


def _get_signal_and_ref_keys(ds: xr.Dataset) -> tuple[str, str | None]:
    """Return the data variable names for the signal and reference measurements."""
    # Handles both state-discrimination and raw-IQ datasets so the rest of the analysis pipeline remains agnostic to the measurement back-end.
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
    # Produces the per-delay-time phase values that will be inverted through the reference calibration curve to reconstruct the qubit flux amplitude at each point in the step response.
    signal_key, _ = _get_signal_and_ref_keys(ds)
    return _fourier_phase(ds[signal_key], "frame")


def extract_reference_calibration(ds: xr.Dataset) -> xr.DataArray | None:
    """Extract Ramsey phase vs qubit flux amplitude from the reference sweep.

    The reference Ramsey amplitude sweep provides a calibration curve mapping
    qubit flux amplitude to Ramsey phase.  This function extracts the
    Fourier phase along ``frame`` for each amplitude value and unwraps along
    the amplitude axis.

    Returns
    -------
    xr.DataArray or None
        Phase with dims ``(qubit, a)``, or ``None`` if no reference amplitude
        sweep is present in the dataset.
    """
    # Builds the lookup table that relates Ramsey phase to flux amplitude; without this calibration it is impossible to convert the measured signal phase back to physical flux units.
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
    # Converts measured signal phases to effective qubit flux amplitudes point-by-point, using linear interpolation on the sorted reference curve to avoid the need for global phase unwrapping along the time axis.
    sort_idx = np.argsort(ref_phases)
    ref_phases_sorted = ref_phases[sort_idx]
    ref_amps_sorted = ref_amps[sort_idx]

    ref_center = 0.5 * (ref_phases_sorted[0] + ref_phases_sorted[-1])
    adjusted = sig_phases - np.round((sig_phases - ref_center) / (2 * np.pi)) * (2 * np.pi)

    return np.interp(adjusted, ref_phases_sorted, ref_amps_sorted)


def _assess_branch_risk(sig_phases: np.ndarray, ref_phases: np.ndarray) -> dict:
    """Flag when the per-point np.round phase->flux inversion may be aliased.

    ``_map_phase_to_amplitude`` selects, for each delay independently, the 2*pi
    branch nearest the centre of the reference window.  That is exact *only*
    while the true phase trajectory stays within a single 2*pi window.  If the
    distortion-induced phase swing approaches/exceeds 2*pi the per-point choice
    can switch branches between adjacent delays and corrupt the recovered
    *shape* (and therefore the fitted IIR taps).

    We estimate the trajectory's own peak-to-peak swing by unwrapping the signal
    phase along time -- diagnostic only; the inversion itself stays per-point.
    ``ref_span`` is reported for context: a reference that itself spans > 2*pi is
    multi-valued, so a wrong *absolute* branch also becomes possible (this only
    shifts the DC level / a_dc and is harmless for the IIR taps as long as the
    swing is small, which is why the trigger is the swing, not the span).

    Returns a dict with ``level`` ('ok' | 'marginal' | 'high'), an integer
    ``code`` (0 | 1 | 2), and ``sig_swing_frac`` / ``ref_span_frac`` in units of
    2*pi.
    """
    two_pi = 2 * np.pi
    ref_span = float(np.ptp(ref_phases)) if np.size(ref_phases) else 0.0
    sig_swing = float(np.ptp(_robust_unwrap_1d(sig_phases))) if np.size(sig_phases) else 0.0
    sig_frac = sig_swing / two_pi
    ref_frac = ref_span / two_pi
    if sig_frac >= 1.0:          # swing >= 2*pi -> branch selection definitely aliases the shape
        level, code = "high", 2
    elif sig_frac >= 0.5:        # swing >= pi -> within 2x of the aliasing limit
        level, code = "marginal", 1
    else:
        level, code = "ok", 0
    return {"level": level, "code": code, "sig_swing_frac": sig_frac, "ref_span_frac": ref_frac}


def annotate_branch_risk(fig, ds: xr.Dataset) -> bool:
    """Draw a warning box on ``fig`` if any qubit was flagged with branch-aliasing
    risk by :func:`fit_raw_data`.

    Terminal warnings are easy to miss, so the same information is stamped onto
    the saved flux-response figures.  Safe to call unconditionally: it is a no-op
    (returns ``False``) when the dataset carries no risk metadata or every qubit
    is ``ok``.
    """
    if "branch_risk_code" not in ds:
        return False
    qnames = [str(n) for n in ds["branch_risk_code"].coords["qubit"].values]
    lines = []
    for nm in qnames:
        code = int(ds["branch_risk_code"].sel(qubit=nm).values)
        if code < 1:
            continue
        sw = float(ds["branch_sig_swing"].sel(qubit=nm).values)
        rs = float(ds["branch_ref_span"].sel(qubit=nm).values)
        tag = "HIGH" if code >= 2 else "marginal"
        lines.append(f"{nm}: {tag} — phase swing {sw:.2f}×2π, ref span {rs:.2f}×2π")
    if not lines:
        return False
    msg = (
        "⚠ BRANCH-ALIASING RISK: per-point phase→flux inversion (np.round) may be unreliable\n"
        + "\n".join(lines)
        + "\n(true phase approaches/exceeds one 2π window; see _map_phase_to_amplitude)"
    )
    fig.text(
        0.5, 0.01, msg, ha="center", va="bottom", fontsize=8, color="white",
        bbox=dict(boxstyle="round", facecolor="crimson", alpha=0.9),
    )
    try:
        fig.subplots_adjust(bottom=0.22)
    except Exception:
        pass
    return True


def process_raw_dataset(ds: xr.Dataset, node) -> xr.Dataset:
    """Preprocess Ramsey raw dataset: convert IQ to volts if applicable."""
    # Ensures the raw IQ data is in calibrated voltage units before phase extraction, mirroring the pre-processing step used in the spectroscopy-based pipeline.
    if "I" in ds or "Q" in ds:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def fit_raw_data(ds: xr.Dataset, node) -> tuple[xr.Dataset, Dict[str, FluxDistortionExpFitResult]]:
    """Ramsey analysis: map signal phase to amplitude via reference calibration, fit exponentials.

    Steps
    -----
    1. Extract raw Ramsey phase at each delay time from the signal measurement.
    2. Extract phase-vs-amplitude calibration from the reference Ramsey sweep.
    3. For each delay, map the measured phase to the effective qubit flux amplitude
       by inverting the reference calibration (each point is independent, no
       unwrapping or late-time assumptions).
    4. Compute flux distortion = effective_amplitude - ramsey_flux_amplitude_in_v.
    5. Add ``qubit_flux_amplitude_in_v`` to obtain the total flux response.
    6. Fit a sum of decaying exponentials to the flux response.
    """
    # Main analysis entry point for the Ramsey-based qubit-flux distortion measurement; transforms raw Ramsey oscillation data into fitted exponential distortion parameters using the co-measured reference sweep as the phase-to-flux calibration.
    qubits = node.namespace["qubits"]

    signal_phase = extract_ramsey_phase(ds)
    ref_cal = extract_reference_calibration(ds)

    flux_response = xr.full_like(signal_phase, np.nan, dtype=float)
    ramsey_flux_amp = node.parameters.ramsey_flux_amplitude_in_v
    qubit_flux_amp = getattr(node.parameters, "qubit_flux_amplitude_in_v", None)

    branch_risk: Dict[str, dict] = {}
    if ref_cal is not None:
        ref_amps = ref_cal.coords["a"].values

        for q in qubits:
            ref_phases_q = ref_cal.sel(qubit=q.name).values
            sig_phases_q = signal_phase.sel(qubit=q.name).values

            eff_amp = _map_phase_to_amplitude(sig_phases_q, ref_amps, ref_phases_q)
            distortion = eff_amp - ramsey_flux_amp
            total_flux = -distortion + (qubit_flux_amp if qubit_flux_amp is not None else 0)
            flux_response.loc[{"qubit": q.name}] = total_flux

            # Diagnostic: is the per-point np.round branch selection trustworthy here?
            risk = _assess_branch_risk(sig_phases_q, ref_phases_q)
            branch_risk[q.name] = risk
            if risk["level"] != "ok":
                print(
                    f"WARNING [{q.name}]: phase->flux branch-aliasing risk = {risk['level'].upper()}. "
                    f"signal phase swing = {risk['sig_swing_frac']:.2f} x 2pi, "
                    f"reference span = {risk['ref_span_frac']:.2f} x 2pi. "
                    "_map_phase_to_amplitude selects a 2pi branch per point (np.round); this is exact "
                    "only while the phase stays within one 2pi window, so the fitted distortion shape "
                    "may be aliased -- see the warning drawn on the flux-response figures."
                )
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
    if ref_cal is not None and branch_risk:
        # Persist the branch-aliasing diagnostic so the plotting layer can warn on
        # the figure (see annotate_branch_risk); terminal warnings get missed.
        qnames = [q.name for q in qubits]
        ds["branch_risk_code"] = xr.DataArray(
            [branch_risk[n]["code"] for n in qnames], dims=["qubit"], coords={"qubit": qnames}
        )
        ds["branch_sig_swing"] = xr.DataArray(
            [branch_risk[n]["sig_swing_frac"] for n in qnames], dims=["qubit"], coords={"qubit": qnames},
            attrs={"long_name": "signal phase peak-to-peak swing", "units": "2*pi"},
        )
        ds["branch_ref_span"] = xr.DataArray(
            [branch_risk[n]["ref_span_frac"] for n in qnames], dims=["qubit"], coords={"qubit": qnames},
            attrs={"long_name": "reference phase span", "units": "2*pi"},
        )

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
