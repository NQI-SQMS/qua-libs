"""
Analysis utilities for the GEF dispersive shift measurement.

Three resonator spectra are acquired:
  - state 0 (|g⟩): qubit thermalized, no drive
  - state 1 (|e⟩): qubit driven to |e⟩ via x180_ge
  - state 2 (|f⟩): qubit driven to |f⟩ via x180_ge + x180_ef

The dispersive shifts:
  chi_ge = f_resonator(|e⟩) - f_resonator(|g⟩)
  chi_ef = f_resonator(|f⟩) - f_resonator(|e⟩)

The optimal readout frequency is set to f_resonator(|e⟩), which maximises
contrast between |g⟩ and |e⟩ while giving good sensitivity for the |f⟩ state.

All spectra are fitted with Lorentzian dip models.
"""
import logging
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from qualibration_libs.analysis.models import lorentzian_dip


@dataclass
class FitParametersGEF:
    """Fit results for the GEF dispersive shift measurement."""
    f_g: float
    """Resonator frequency when qubit is in |g⟩ [Hz]."""
    f_e: float
    """Resonator frequency when qubit is in |e⟩ [Hz]."""
    f_f: float
    """Resonator frequency when qubit is in |f⟩ [Hz]."""
    chi_ge_hz: float
    """GE dispersive shift chi_ge = f_e - f_g [Hz]."""
    chi_ef_hz: float
    """EF dispersive shift chi_ef = f_f - f_e [Hz]."""
    f_optimal: float
    """Optimal readout frequency = f_resonator(|e⟩) [Hz]."""
    success: bool
    # Lorentzian fit parameters (default nan = fit not attempted / failed)
    kappa_g_hz: float = float("nan")
    """Half-linewidth (HWHM) of |g⟩ resonator dip [Hz]."""
    kappa_e_hz: float = float("nan")
    """Half-linewidth (HWHM) of |e⟩ resonator dip [Hz]."""
    kappa_f_hz: float = float("nan")
    """Half-linewidth (HWHM) of |f⟩ resonator dip [Hz]."""
    amplitude_g: float = float("nan")
    amplitude_e: float = float("nan")
    amplitude_f: float = float("nan")
    offset_g: float = float("nan")
    offset_e: float = float("nan")
    offset_f: float = float("nan")
    center_g_hz: float = float("nan")
    """Fitted dip center detuning for |g⟩ [Hz]."""
    center_e_hz: float = float("nan")
    """Fitted dip center detuning for |e⟩ [Hz]."""
    center_f_hz: float = float("nan")
    """Fitted dip center detuning for |f⟩ [Hz]."""
    # Phase fit: φ(ω) = φ₀ − arctan(2·(ω − ωr) / κ)
    phi0_g: float = float("nan")
    phi0_e: float = float("nan")
    phi0_f: float = float("nan")
    kappa_phase_g_hz: float = float("nan")
    kappa_phase_e_hz: float = float("nan")
    kappa_phase_f_hz: float = float("nan")
    center_phase_g_hz: float = float("nan")
    center_phase_e_hz: float = float("nan")
    center_phase_f_hz: float = float("nan")
    chi2_g: float = float("nan")
    """Reduced chi-squared of Lorentzian amplitude fit for |g⟩: SS_res / ((N-4)*amp²)."""
    chi2_e: float = float("nan")
    """Reduced chi-squared of Lorentzian amplitude fit for |e⟩: SS_res / ((N-4)*amp²)."""
    chi2_f: float = float("nan")
    """Reduced chi-squared of Lorentzian amplitude fit for |f⟩: SS_res / ((N-4)*amp²)."""


def _compute_lorentzian_chi2(
    x: np.ndarray, y: np.ndarray, center: float, kappa: float, amplitude: float, offset: float
) -> float:
    """Reduced chi-squared for a Lorentzian dip fit: SS_res / ((N-4)*amp²).

    Returns inf when amplitude <= 0 or N <= 4 (fit cannot be evaluated).
    """
    N = len(x)
    if N <= 4 or amplitude <= 0:
        return float("inf")
    y_fit = lorentzian_dip(x, amplitude, center, kappa, offset)
    SS_res = float(np.sum((y - y_fit) ** 2))
    return SS_res / ((N - 4) * amplitude ** 2)


def _phase_initial_guesses(x: np.ndarray, phase: np.ndarray):
    """Estimate center and kappa from phase data for the arctan fit.

    Uses the peak of |dφ/dx| as the center and the HWHM of that peak as kappa.
    Falls back to span/8 for kappa if the width cannot be resolved.
    Applies light smoothing before gradient estimation to suppress noise spikes.
    """
    n_smooth = max(3, len(phase) // 10)
    phase_smooth = uniform_filter1d(phase, size=n_smooth)
    dphase = np.abs(np.gradient(phase_smooth, x))
    i_max = int(np.argmax(dphase))
    center0 = float(x[i_max])
    half_max = 0.5 * dphase[i_max]
    try:
        left_idx = np.where(dphase[:i_max + 1] < half_max)[0]
        right_idx = np.where(dphase[i_max:] < half_max)[0]
        left = float(x[left_idx[-1]]) if len(left_idx) else float(x[0])
        right = float(x[i_max + right_idx[0]]) if len(right_idx) else float(x[-1])
        kappa0 = max(0.5 * abs(right - left), float(x[1] - x[0]))
    except (IndexError, ValueError):
        kappa0 = float((x[-1] - x[0]) / 8)
    return center0, kappa0


def _fit_phase(x: np.ndarray, phase: np.ndarray, center0: float, kappa0: float):
    """Fit resonator phase: φ(ω) = φ₀ − arctan(2·(ω − ωr) / κ)

    Returns (phi0, center_hz, kappa_hz, success).
    """
    x = np.asarray(x, dtype=float)
    phase = np.asarray(phase, dtype=float)
    phi0_0 = float(np.mean(phase))
    center0 = float(center0)
    kappa0 = max(float(kappa0), float(x[1] - x[0]))

    def _model(xv, phi0, center, kappa):
        return phi0 - np.arctan(2.0 * (xv - center) / kappa)

    try:
        popt, _ = curve_fit(
            _model, x, phase,
            p0=[phi0_0, center0, kappa0],
            bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]),
            maxfev=5000,
        )
        phi0, center, kappa = popt
        return float(phi0), float(center), float(abs(kappa)), True
    except Exception:
        return phi0_0, center0, kappa0, False


def _fit_lorentzian_dip(x: np.ndarray, y: np.ndarray):
    """Fit a Lorentzian dip to 1D data.

    Returns (center_hz, kappa_hz, amplitude, offset, success) where kappa is HWHM.
    Model: y = offset - amplitude * kappa^2 / (kappa^2 + (x - center)^2)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    offset0 = float(np.max(y))
    i_min = int(np.argmin(y))
    center0 = float(x[i_min])
    amplitude0 = float(offset0 - np.min(y))
    amplitude0 = max(amplitude0, 1e-12)

    half_depth = offset0 - 0.5 * amplitude0
    try:
        left_idx = np.where(y[:i_min + 1] > half_depth)[0]
        right_idx = np.where(y[i_min:] > half_depth)[0]
        left = float(x[left_idx[-1]]) if len(left_idx) else float(x[0])
        right = float(x[i_min + right_idx[0]]) if len(right_idx) else float(x[-1])
        kappa0 = max(0.5 * abs(right - left), float(x[1] - x[0]))
    except (IndexError, ValueError):
        kappa0 = float((x[-1] - x[0]) / 8)

    try:
        popt, _ = curve_fit(
            lorentzian_dip, x, y,
            p0=[amplitude0, center0, kappa0, offset0],
            bounds=([0, -np.inf, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
            maxfev=5000,
        )
        amplitude, center, kappa, offset = popt
        return float(center), float(abs(kappa)), float(amplitude), float(offset), True
    except Exception:
        return center0, kappa0, amplitude0, offset0, False


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        log_callable(
            f"Results for qubit {q}: {status}\n"
            f"\tf_g: {1e-9 * res['f_g']:.5f} GHz "
            f"(kappa_g FWHM: {2e-3 * res.get('kappa_g_hz', float('nan')):.1f} kHz)\n"
            f"\tf_e: {1e-9 * res['f_e']:.5f} GHz "
            f"(kappa_e FWHM: {2e-3 * res.get('kappa_e_hz', float('nan')):.1f} kHz)\n"
            f"\tf_f: {1e-9 * res['f_f']:.5f} GHz "
            f"(kappa_f FWHM: {2e-3 * res.get('kappa_f_hz', float('nan')):.1f} kHz)\n"
            f"\tchi_ge: {1e-3 * res['chi_ge_hz']:.1f} kHz | "
            f"chi_ef: {1e-3 * res['chi_ef_hz']:.1f} kHz | "
            f"f_opt (|e>): {1e-9 * res['f_optimal']:.5f} GHz"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Convert raw I/Q to Volts and add a full_freq coordinate."""
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    # Unwrap phase along the detuning axis so wrap discontinuities don't confuse the arctan fit
    if "phase" in ds.data_vars:
        det_axis = list(ds.phase.dims).index("detuning")
        unwrapped = np.unwrap(ds.phase.values, axis=det_axis)
        ds["phase"] = xr.DataArray(unwrapped, dims=ds.phase.dims, coords=ds.phase.coords, attrs=ds.phase.attrs)
    full_freq = np.array(
        [ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]]
    )
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParametersGEF]]:
    """Fit |g⟩, |e⟩, and |f⟩ resonator spectra with Lorentzians and extract chi.

    Expects ``ds`` to have a ``qubit_state`` dimension with values ``[0, 1, 2]``
    (0 = ground, 1 = excited, 2 = f-state).

    Returns
    -------
    ds : xr.Dataset  (unchanged — fit curves reconstructed in plotting)
    fit_results : dict[str, FitParametersGEF]
    """
    span_hz = node.parameters.frequency_span_in_mhz * 1e6
    chi2_threshold = getattr(node.parameters, "chi2_threshold", 3.0)
    fit_on_phase = getattr(node.parameters, "fit_on_phase", False)
    fit_results = {}

    for q_obj in node.namespace["qubits"]:
        q = q_obj.name
        rr_freq = q_obj.resonator.RF_frequency
        dfs = ds.detuning.values.astype(float)

        amp_g = ds.IQ_abs.sel(qubit=q, qubit_state=0).values
        amp_e = ds.IQ_abs.sel(qubit=q, qubit_state=1).values
        amp_f = ds.IQ_abs.sel(qubit=q, qubit_state=2).values

        cg, kg, ag, og, ok_g = _fit_lorentzian_dip(dfs, amp_g)
        ce, ke, ae, oe, ok_e = _fit_lorentzian_dip(dfs, amp_e)
        cf, kf, af, of_, ok_f = _fit_lorentzian_dip(dfs, amp_f)

        # Chi-squared quality check for amplitude fits
        chi2_g = _compute_lorentzian_chi2(dfs, amp_g, cg, kg, ag, og) if ok_g else float("inf")
        chi2_e = _compute_lorentzian_chi2(dfs, amp_e, ce, ke, ae, oe) if ok_e else float("inf")
        chi2_f = _compute_lorentzian_chi2(dfs, amp_f, cf, kf, af, of_) if ok_f else float("inf")
        amp_ok_g = ok_g and chi2_g <= chi2_threshold
        amp_ok_e = ok_e and chi2_e <= chi2_threshold
        amp_ok_f = ok_f and chi2_f <= chi2_threshold

        # Phase fit: φ(ω) = φ₀ − arctan(2·(ω − ωr) / κ)
        # Always run — used as primary when fit_on_phase=True, fallback otherwise
        phi0_g = phi0_e = phi0_f = float("nan")
        kappa_ph_g = kappa_ph_e = kappa_ph_f = float("nan")
        center_ph_g = center_ph_e = center_ph_f = float("nan")
        ok_ph_g = ok_ph_e = ok_ph_f = False
        if "phase" in ds.data_vars:
            ph_g = ds.phase.sel(qubit=q, qubit_state=0).values
            ph_e = ds.phase.sel(qubit=q, qubit_state=1).values
            ph_f = ds.phase.sel(qubit=q, qubit_state=2).values
            # When fit_on_phase=True, always derive initial guesses from the phase data itself;
            # the amplitude fit may have found the wrong feature (e.g. a small dip instead
            # of the main resonance peak) and would poison the phase fit starting point.
            if fit_on_phase:
                c0_g, k0_g = _phase_initial_guesses(dfs, ph_g)
                c0_e, k0_e = _phase_initial_guesses(dfs, ph_e)
                c0_f, k0_f = _phase_initial_guesses(dfs, ph_f)
            else:
                c0_g, k0_g = (cg, kg) if amp_ok_g else _phase_initial_guesses(dfs, ph_g)
                c0_e, k0_e = (ce, ke) if amp_ok_e else _phase_initial_guesses(dfs, ph_e)
                c0_f, k0_f = (cf, kf) if amp_ok_f else _phase_initial_guesses(dfs, ph_f)
            phi0_g, center_ph_g, kappa_ph_g, ok_ph_g = _fit_phase(dfs, ph_g, c0_g, k0_g)
            phi0_e, center_ph_e, kappa_ph_e, ok_ph_e = _fit_phase(dfs, ph_e, c0_e, k0_e)
            phi0_f, center_ph_f, kappa_ph_f, ok_ph_f = _fit_phase(dfs, ph_f, c0_f, k0_f)

        if fit_on_phase:
            # Phase is the primary fit — use phase centers directly
            if ok_ph_g and np.isfinite(center_ph_g):
                cg = center_ph_g
            if ok_ph_e and np.isfinite(center_ph_e):
                ce = center_ph_e
            if ok_ph_f and np.isfinite(center_ph_f):
                cf = center_ph_f
            success_g = ok_ph_g and np.isfinite(cg) and abs(cg) < span_hz
            success_e = ok_ph_e and np.isfinite(ce) and abs(ce) < span_hz
            success_f = ok_ph_f and np.isfinite(cf) and abs(cf) < span_hz
        else:
            # Amplitude is the primary fit; phase is a fallback when amplitude chi2 is poor
            if not amp_ok_g and ok_ph_g and np.isfinite(center_ph_g) and abs(center_ph_g) < span_hz:
                node.log(
                    f"[{q}] |g⟩: amplitude fit poor (chi2={chi2_g:.2f} > {chi2_threshold}); "
                    f"using phase fit center {rr_freq + center_ph_g:.6e} Hz"
                )
                cg = center_ph_g
            if not amp_ok_e and ok_ph_e and np.isfinite(center_ph_e) and abs(center_ph_e) < span_hz:
                node.log(
                    f"[{q}] |e⟩: amplitude fit poor (chi2={chi2_e:.2f} > {chi2_threshold}); "
                    f"using phase fit center {rr_freq + center_ph_e:.6e} Hz"
                )
                ce = center_ph_e
            if not amp_ok_f and ok_ph_f and np.isfinite(center_ph_f) and abs(center_ph_f) < span_hz:
                node.log(
                    f"[{q}] |f⟩: amplitude fit poor (chi2={chi2_f:.2f} > {chi2_threshold}); "
                    f"using phase fit center {rr_freq + center_ph_f:.6e} Hz"
                )
                cf = center_ph_f
            success_g = (amp_ok_g or ok_ph_g) and np.isfinite(cg) and abs(cg) < span_hz
            success_e = (amp_ok_e or ok_ph_e) and np.isfinite(ce) and abs(ce) < span_hz
            success_f = (amp_ok_f or ok_ph_f) and np.isfinite(cf) and abs(cf) < span_hz

        f_g = rr_freq + cg
        f_e = rr_freq + ce
        f_f = rr_freq + cf

        success = success_g and success_e and success_f

        chi_ge_hz = f_e - f_g if success else float("nan")
        chi_ef_hz = f_f - f_e if success else float("nan")
        # Optimal readout frequency: f_resonator(|e⟩) — maximises |e⟩ vs |g⟩ contrast
        f_optimal = f_e if success_e else rr_freq

        fit_results[q] = FitParametersGEF(
            f_g=f_g,
            f_e=f_e,
            f_f=f_f,
            chi_ge_hz=chi_ge_hz,
            chi_ef_hz=chi_ef_hz,
            f_optimal=f_optimal,
            success=success,
            kappa_g_hz=kg,
            kappa_e_hz=ke,
            kappa_f_hz=kf,
            amplitude_g=ag,
            amplitude_e=ae,
            amplitude_f=af,
            offset_g=og,
            offset_e=oe,
            offset_f=of_,
            center_g_hz=cg,
            center_e_hz=ce,
            center_f_hz=cf,
            phi0_g=phi0_g,
            phi0_e=phi0_e,
            phi0_f=phi0_f,
            kappa_phase_g_hz=kappa_ph_g,
            kappa_phase_e_hz=kappa_ph_e,
            kappa_phase_f_hz=kappa_ph_f,
            center_phase_g_hz=center_ph_g,
            center_phase_e_hz=center_ph_e,
            center_phase_f_hz=center_ph_f,
            chi2_g=chi2_g if np.isfinite(chi2_g) else float("nan"),
            chi2_e=chi2_e if np.isfinite(chi2_e) else float("nan"),
            chi2_f=chi2_f if np.isfinite(chi2_f) else float("nan"),
        )

    return ds, fit_results
