"""Analysis utilities for cryoscope experiment: flux line step response fitting."""
from dataclasses import dataclass

import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from qualibrate import QualibrationNode

from calibration_utils.qubit_flux_long_distortion_qubitspec.analysis import (
    _frequency_to_flux,
    _frequency_to_flux_deviation,
    _load_dispersion_curve,
    _load_spectroscopy_curve,
    multi_exp_fit_global,
)
from qualibration_libs.analysis import fit_oscillation, unwrap_phase
from qualibration_libs.data import convert_IQ_to_V
from scipy.signal import savgol_filter


def savgol(da, dim, range=3, order=2):
    """Apply Savitzky-Golay filter to smooth data."""

    def diff_func(x):
        return savgol_filter(x, range, order, deriv=0, delta=1)

    return xr.apply_ufunc(diff_func, da, input_core_dims=[[dim]], output_core_dims=[[dim]])


def diff_savgol(da, dim, range=3, order=2):
    """Apply Savitzky-Golay filter to compute derivative."""

    def diff_func(x):
        return savgol_filter(x / (2 * np.pi), range, order, deriv=1, delta=1)

    return xr.apply_ufunc(diff_func, da, input_core_dims=[[dim]], output_core_dims=[[dim]])


def cryoscope_frequency(ds, stable_time_indices, quad_term=-1, sg_range=3, sg_order=2):
    """Extract flux response from frequency data using cryoscope analysis.

    Mean-detuning derivative f = d(phi/2pi)/dt following Rol et al. arXiv:1907.04818,
    Eq. (3) [Delta_f_R = (phi_{tau+dtau} - phi_tau)/(2*pi*dtau); cf. Eq. (2) for the
    phase-integral relation]; the derivative is obtained with the second-order
    Savitzky-Golay filter described in the same paper.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Dataset containing unwrapped phase data with a "time" dimension.
        May also contain a "qubit" dimension for multi-qubit processing.
    stable_time_indices : tuple of (int, int)
        Time range ``(start, end)`` in ns considered as the stable (flat)
        region for normalization.  Only used when ``quad_term`` is the
        default sentinel value ``-1``.
    quad_term : float or xr.DataArray, optional
        Quadratic term ``freq_vs_flux_01_quad_term`` relating qubit frequency
        to flux.  When processing multiple qubits, pass an ``xr.DataArray``
        with a ``qubit`` dimension so that each qubit uses its own value.
        The default ``-1`` triggers self-normalization of the flux response.
    sg_range : int, optional
        Savitzky-Golay filter window length (default 3).
    sg_order : int, optional
        Savitzky-Golay polynomial order (default 2).

    Returns
    -------
    xr.Dataset
        Dataset with ``"freq"`` and ``"flux"`` variables (and input data if converted from DataArray).
    """
    if isinstance(ds, xr.DataArray):
        ds = ds.copy().to_dataset(name="phase")
        _phase = ds["phase"]
    else:
        ds = ds.copy()
        _phase = ds[list(ds.data_vars)[0]]

    freq_cryoscope = diff_savgol(_phase, "time", range=sg_range, order=sg_order)

    ds["freq"] = freq_cryoscope

    flux_cryoscope = np.sqrt(np.abs(1e9 * freq_cryoscope / quad_term)).fillna(0)

    # Self-normalization is only applied when quad_term is the default
    # sentinel value (-1).  When a real quad_term (scalar or DataArray) is
    # provided the raw flux values are kept.  The isinstance check prevents
    # an ambiguous truth-value error when quad_term is an xr.DataArray.
    if np.isscalar(quad_term) and quad_term == -1:
        flux_cryoscope = flux_cryoscope / flux_cryoscope.sel(
            time=slice(stable_time_indices[0], stable_time_indices[1])
        ).mean(dim="time")

    ds["flux"] = flux_cryoscope

    return ds


def expdecay(x, s, a, t):
    """Exponential decay defined as 1 + a * np.exp(-x / t).
    :param x: numpy array for the time vector in ns
    :param a: float for the exponential amplitude
    :param t0: time shift
    :param t: float for the exponential decay time in ns
    :return: numpy array for the exponential decay
    """
    return s * (1 + a * np.exp(-(x) / t))


def two_expdecay(x, s, a, t, a2, t2):
    """Double exponential decay defined as s * (1 + a * np.exp(-x / t) + a2 * np.exp(-x / t2)).
    :param x: numpy array for the time vector in ns
    :param s: float for the scaling factor
    :param a: float for the first exponential amplitude
    :param t: float for the first exponential decay time in ns
    :param a2: float for the second exponential amplitude
    :param t2: float for the second exponential decay time in ns
    :return: numpy array for the double exponential decay
    """
    return s * (1 + a * np.exp(-(x) / t) + a2 * np.exp(-(x) / t2))


def single_exp(da, plot=True):
    """Fit single exponential decay to data."""
    first_vals = da.sel(time=slice(0, 1)).mean().values
    final_vals = da.sel(time=slice(20, None)).mean().values
    print(first_vals, final_vals)

    fit = da.curvefit(
        "time",
        expdecay,
        p0={"a": 1 - first_vals / final_vals, "t": 50, "s": final_vals},
    ).curvefit_coefficients

    fit_vals = dict(zip(fit.to_dict()["coords"]["param"]["data"], fit.to_dict()["data"]))

    t_s = 1
    alpha = np.exp(-t_s / fit_vals["t"])
    A = fit_vals["a"]
    fir = [1 / (1 + A), -alpha / (1 + A)]
    iir = [(A + alpha) / (1 + A)]

    if plot:
        fig, ax = plt.subplots()
        ax.plot(da.time, da, label="data")
        ax.plot(da.time, expdecay(da.time, **fit_vals), label="fit")
        ax.grid("all")
        ax.legend()
        print(f"Qubit - FIR: {fir}\nIIR: {iir}")
    else:
        fig = None
        ax = None
    return fir, iir, fig, ax, (da.time, expdecay(da.time, **fit_vals))


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Convert IQ data to voltage if state discrimination is not used."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode):
    """Fit raw cryoscope data with exponential models for each qubit.

    This function supports multiple qubits.  The oscillation fitting and phase
    unwrapping are performed across all qubits at once (leveraging xarray
    vectorization).  The ``cryoscope_frequency`` step uses a per-qubit
    ``quad_term`` DataArray so that each qubit's ``freq_vs_flux_01_quad_term``
    is applied independently.  Finally, the exponential-decay optimisation is
    run per qubit and the results are collected in a dictionary keyed by qubit
    name.  Pattern follows ``qubit_flux_long_distortion_qubitspec/analysis.py``.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset containing ``I``/``Q`` or ``state`` data with dimensions
        ``(qubit, time, frame)``.
    node : QualibrationNode
        Node containing parameters, machine and namespace.

    Returns
    -------
    ds_fit : xr.Dataset
        Dataset with ``"freq"`` and ``"flux"`` variables, retaining
        the ``qubit`` dimension so that per-qubit data can be selected with
        ``ds_fit.flux.sel(qubit=name)``.
    fit_results : dict[str, FitParameters]
        Dictionary mapping each qubit name to its ``FitParameters`` dataclass
        (containing ``success``, ``components``, ``a_dc``).
    """

    # --- 1. Detect whether the dataset contains I/Q or state data -----------
    if hasattr(ds, "I"):
        data = "I"
    elif hasattr(ds, "state"):
        data = "state"
    else:
        raise ValueError("Dataset must contain either 'I' or 'state' data")

    # --- 2. Phase extraction (all qubits at once via xarray vectorization) --
    dafit = fit_oscillation(ds[data], "frame")
    daphi = unwrap_phase(dafit.sel(fit_vals="phi"), "time")
    sg_order = 2
    sg_range = 3

    # --- 3. Quad terms: use quadratic model when qubit_pairs absent (qubit nodes), else flux from dispersion
    qubits = node.namespace["qubits"]
    qubit_pairs = node.namespace.get("qubit_pairs")
    if qubit_pairs is not None:
        quad_terms = xr.DataArray(
            np.nan * np.ones(len(qubits)),
            coords={"qubit": [q.name for q in qubits]},
            dims=["qubit"],
        )
    else:
        quad_terms = xr.DataArray(
            [getattr(node.machine.qubits[q.name], "freq_vs_flux_01_quad_term", None) or np.nan for q in qubits],
            coords={"qubit": [q.name for q in qubits]},
            dims=["qubit"],
        )

    # --- 4. Compute cryoscope frequency and flux
    ds_fit = cryoscope_frequency(
        daphi,
        quad_term=quad_terms,
        stable_time_indices=(node.parameters.cryoscope_len - 20, node.parameters.cryoscope_len),
        sg_order=sg_order,
        sg_range=sg_range,
    )
    # When use_spectroscopy_data=True (qubit nodes only): overwrite flux from spectroscopy curve.
    # This converts the measured cryoscope frequency into flux via the measured freq-vs-flux
    # dispersion curve (freq-vs-flux lookup and interpolation).
    use_spec = getattr(node.parameters, "use_spectroscopy_data", False)
    spec_run_id = getattr(node.parameters, "spectroscopy_run_id", None)
    if qubit_pairs is None and use_spec and spec_run_id is not None:
        for i, q in enumerate(qubits):
            curve = _load_spectroscopy_curve(spec_run_id, q.name, q.xy.RF_frequency)
            if curve is not None:
                # Cryoscope freq convention: freq = f_drive - f_qubit (positive when qubit is below drive).
                # Subtract to get the actual qubit frequency: abs_freq = RF - cryoscope_freq.
                abs_freq_q = q.xy.RF_frequency - ds_fit["freq"].sel(qubit=q.name).values * 1e9
                idle_qubit_freq = q.xy.RF_frequency
                ds_fit["flux"].values[i, :] = _frequency_to_flux_deviation(
                    abs_freq_q,
                    curve[0],
                    curve[1],
                    idle_qubit_freq,
                )
            else:
                print(f"  WARNING: spectroscopy curve unavailable for {q.name}, using quad_term flux")

    # When qubit_pairs present (coupler nodes): overwrite flux from dispersion curve
    if qubit_pairs is not None:
        abs_freq_Hz = ds_fit["freq"] * 1e9
        # Node 25 (short distortion): use relative detuning (peak_frequency) from dispersion
        for i, q in enumerate(qubits):
            if i >= len(qubit_pairs):
                break
            coupler = qubit_pairs[i].coupler
            curve = _load_dispersion_curve(node, q, coupler, frequency_var="peak_frequency")
            if curve is not None:
                flux_bias, abs_peak = curve
                freq_vals = abs_freq_Hz.sel(qubit=q.name).values
                flux_vals = _frequency_to_flux(freq_vals, flux_bias, np.abs(abs_peak))
                ds_fit["flux"].values[i, :] = flux_vals
            else:
                ds_fit["flux"].values[i, :] = np.nan

    # --- 5. Per-qubit exponential fit ---------------------------------------
    #     For each qubit, extract its 1-D flux response and run the global
    #     multi-exponential fit (driven by a single n_exponentials parameter).
    #     Imported from the long-distortion module so both nodes share one
    #     well-tested fitter.
    fit_results = {}
    n_exp = int(node.parameters.n_exponentials)
    time_vals = np.asarray(ds_fit.time.values, dtype=float)

    for q in qubits:
        flux_vals = np.asarray(ds_fit.flux.sel(qubit=q.name).values, dtype=float)
        mask = np.isfinite(flux_vals) & (time_vals > 0)
        if mask.sum() < max(2 * n_exp + 1, 4):
            fit_results[q.name] = FitParameters(success=False, components=[], a_dc=float("nan"))
            continue
        res = multi_exp_fit_global(time_vals[mask], flux_vals[mask], n_exp, verbose=True)
        fit_results[q.name] = FitParameters(
            success=bool(res["fit_successful"]),
            components=list(res["a_tau_tuple"]),
            a_dc=float(res["a_dc"]) if np.isfinite(res["a_dc"]) else float("nan"),
        )

    return ds_fit, fit_results


def _extract_relevant_fit_parameters(ds: xr.Dataset, node: QualibrationNode):
    """Extract relevant fit parameters from the dataset and add metadata.

    .. deprecated::
        This helper is no longer called by ``fit_raw_data`` since the move to
        per-qubit fitting.  It is kept for backward compatibility with any
        external code that may reference it.  Prefer using the ``fit_results``
        dictionary returned directly by ``fit_raw_data`` instead.
    """
    # Assess whether the fit was successful or not

    # Check if ds has fit_results (normal case) or use ds directly (error case)
    if "fit_results" in ds:
        fit = ds["fit_results"]
    else:
        fit = ds

    fit_results = {}

    # Get qubit names from the node if qubit dimension doesn't exist
    if hasattr(fit, "qubit") and hasattr(fit.qubit, "values"):
        qubit_names = fit.qubit.values
    else:
        qubit_names = [q.name for q in node.namespace["qubits"]]

    for q in qubit_names:
        success = fit.attrs.get("fit_success", False)
        # Reconstruct components from stored 1D arrays if available
        if "fit_component_amps" in fit.attrs and "fit_component_taus_ns" in fit.attrs:
            amps = fit.attrs.get("fit_component_amps", [])
            taus = fit.attrs.get("fit_component_taus_ns", [])
            try:
                components = [zip(list(amps), list(taus))]
            except Exception:
                components = []
        else:
            # Backward compatibility (older in-memory attribute, not NetCDF-safe but maybe present at runtime)
            components = fit.attrs.get("fit_components", [])
        a_dc = fit.attrs.get("fit_a_dc", None)

        fit_results[q] = FitParameters(success=success, components=components, a_dc=a_dc)
    return ds, fit_results


def log_fitted_results(fit_results: dict, log_callable=print):
    """Log the fitted results for each qubit.

    Parameters
    ----------
    fit_results : dict
        Dictionary containing fit results for each qubit.
    log_callable : callable, optional
        Function to use for logging (default is print).
    """
    for qubit_name, fit_result in fit_results.items():
        log_callable(f"=== {qubit_name} ===")
        if getattr(fit_result, "success", False):
            log_callable("Overall fit: SUCCESSFUL")
        else:
            log_callable("Overall fit: FAILED")

        # New logging for FitParametersNEW structure
        if hasattr(fit_result, "components") and fit_result.components is not None:
            components = fit_result.components
            a_dc = getattr(fit_result, "a_dc", None)
            if a_dc is not None:
                log_callable(f"  DC term (a_dc): {a_dc:.6g}")
            if isinstance(components, (list, tuple)) and len(components) > 0:
                log_callable("  Exponential components (amplitude, tau [ns]):")
                for idx, comp in enumerate(components, start=1):
                    try:
                        amp, tau = comp
                        if a_dc not in (None, 0):
                            log_callable(f"    #{idx}: amp = {amp:.6g} (rel {amp / a_dc:.3f}), tau = {tau:.3f} ns")
                        else:
                            log_callable(f"    #{idx}: amp = {amp:.6g}, tau = {tau:.3f} ns")
                    except Exception:
                        log_callable(f"    #{idx}: {comp}")
            else:
                log_callable("  No exponential components fitted.")
        else:
            # Backwards compatibility: old FitParameters style
            if hasattr(fit_result, "fit1_success") or hasattr(fit_result, "fit2_success"):
                if getattr(fit_result, "fit1_success", False):
                    A = getattr(fit_result, "fit1_A", None)
                    tau = getattr(fit_result, "fit1_tau", None)
                    if A is not None and tau is not None:
                        log_callable(f"  Single exp: A = {A:.6g}, tau = {tau:.3f} ns")
                if getattr(fit_result, "fit2_success", False):
                    A1 = getattr(fit_result, "fit2_A1", None)
                    tau1 = getattr(fit_result, "fit2_tau1", None)
                    A2 = getattr(fit_result, "fit2_A2", None)
                    tau2 = getattr(fit_result, "fit2_tau2", None)
                    if None not in (A1, tau1, A2, tau2):
                        log_callable(
                            f"  Double exp: A1 = {A1:.6g}, tau1 = {tau1:.3f} ns | A2 = {A2:.6g}, tau2 = {tau2:.3f} ns"
                        )
        log_callable("")


@dataclass
class FitParameters:
    """Stores cryoscope fit parameters: exponential components and DC term."""

    # List of (amplitude, tau) tuples for each exponential component
    components: list
    # Constant (DC) term
    a_dc: float
    # Overall success flag
    success: bool = False


def fit_fir_data(ds_fit: xr.Dataset, node) -> dict:
    """Run FIR filter analysis on the cryoscope flux step response.

    Follows the notebook pipeline:
      1. Normalize flux by stable-region tail mean.
      2. Resample from 1 GS/s to 2 GS/s on ``ds_fit.time`` (same axis as the plot).
      3. Grid search over (L, lam1, lam2) to fit the best forward FIR.
      4. Invert to obtain the pre-distortion filter h_inv.
      5. Validate the corrected response at 1 GS/s.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fitted dataset containing a ``flux`` variable with dimensions
        ``(qubit, time)``.
    node : QualibrationNode
        Node object providing ``node.parameters`` (FIR grid-search settings)
        and ``node.namespace["qubits"]``.

    Returns
    -------
    dict
        Keyed by qubit name.  Each value is a dict with keys:
        ``success``, ``forward_fir``, ``inverse_fir``, ``normalized_1gs``,
        ``corrected_1gs``, ``time_1gs``, ``time_2gs``, ``normalized_2gs``,
        ``fig_fir_fit``, ``fig_fir_inverse``.
    """
    from calibration_utils.qubit_flux_short_distortion.fir_utils import (
        analyze_and_plot_inverse_fir_auto,
        estimate_noise_floor,
        resample_to_target_rate,
    )
    from scipy.signal import lfilter

    params = node.parameters
    qubits = node.namespace["qubits"]
    fir_results = {}

    for q in qubits:
        flux_raw = ds_fit.flux.sel(qubit=q.name).values
        if np.all(np.isnan(flux_raw)):
            node.log(f"  {q.name}: flux is all NaN — skipping FIR")
            fir_results[q.name] = {"success": False}
            continue

        # Normalize by stable-region tail mean
        tail_mean = float(np.nanmean(flux_raw[-10:]))
        if tail_mean == 0:
            tail_mean = 1.0
        normalized_1gs = flux_raw / tail_mean

        # Resample 1 GS/s → 2 GS/s
        time_1gs_arr = np.asarray(ds_fit.time.values, dtype=float)
        normalized_2gs, time_2gs = resample_to_target_rate(
            normalized_1gs,
            original_Ts=1,
            target_Ts=0.5,
            t_original_ns=time_1gs_arr,
        )

        # Single data-driven path: AIC selects L (<= fir_max_taps), GCV + L-curve
        # select the forward and inverse regularisation strengths. sigma and
        # lam_smooth are pinned by the cryoscope Nyquist; nothing else to tune.
        h_fir, h_inv, _best_reconstructed, fig_fir_fit, fig_inv_fir, auto_info = (
            analyze_and_plot_inverse_fir_auto(
                response=normalized_2gs,
                time=time_2gs,
                Ts=0.5,
                max_taps=params.fir_max_taps,
                M=None,
                sigma_ns=None,
                alpha=1.0,
                criterion="both",
                verbose=True,
            )
        )
        chosen_meta = {
            "auto_chosen_L": auto_info["L"],
            "auto_chosen_lam": auto_info["lam"],
            "auto_chosen_lam_smooth": auto_info["lam_smooth"],
            "auto_chosen_sigma_ns": auto_info["sigma_ns"],
            "auto_criterion_forward": auto_info["criterion_forward"],
            "auto_criterion_inverse": auto_info["criterion_inverse"],
            "auto_forward_nrms": auto_info["forward_nrms"],
        }

        # Corrected response validation at 1 GS/s
        ideal_1gs = np.ones(len(normalized_1gs))
        predistorted = lfilter(h_inv, 1, ideal_1gs)
        corrected = lfilter(h_fir, 1, predistorted)
        corrected_norm = corrected / float(np.nanmean(corrected[-10:]))

        # --- Noise-floor triangulation (sigma_A tail-std, sigma_B first-diff,
        # sigma_C fit-implied).  Run on the 1 GS/s raw grid: at 2 GS/s the
        # cubic interpolation correlates adjacent samples and biases sigma_B
        # downward (false-positive WARN).  sigma_C only available in auto mode
        # and lives in the 2 GS/s fit domain — close enough in magnitude for
        # the loose 1.5 ratio test.
        sigma_C = None
        ht = float(auto_info.get("forward_hat_trace", 0.0))
        rss = float(auto_info.get("forward_rss", np.nan))
        dof = len(normalized_2gs) - ht
        if np.isfinite(rss) and dof > 0:
            sigma_C = float(np.sqrt(rss / dof))
        noise_info = estimate_noise_floor(normalized_1gs, Ts=1.0, sigma_C=sigma_C)

        fir_results[q.name] = {
            "success": True,
            "forward_fir": h_fir.tolist(),
            "inverse_fir": h_inv.tolist(),
            "normalized_1gs": normalized_1gs.tolist(),
            "corrected_1gs": corrected_norm.tolist(),
            "time_1gs": ds_fit.time.values.tolist(),
            "time_2gs": time_2gs.tolist(),
            "normalized_2gs": normalized_2gs.tolist(),
            "mode": "auto",
            **chosen_meta,
            "noise_sigma_A_tail_std":    noise_info["sigma_A"],
            "noise_sigma_B_first_diff":  noise_info["sigma_B"],
            "noise_sigma_C_fit_implied": noise_info["sigma_C"],
            "noise_sigma_displayed":     noise_info["displayed"],
            "noise_ratio_AB":            noise_info["ratio_AB"],
            "noise_ratio_fit":           noise_info["ratio_fit"],
            "noise_ratio_max_min":       noise_info["ratio_max_min"],
            "noise_estimate_status":     noise_info["status"],
            "noise_estimate_msg":        noise_info["msg_short"],
            # matplotlib figures — excluded from node.results (not JSON-serialisable)
            "fig_fir_fit": fig_fir_fit,
            "fig_fir_inverse": fig_inv_fir,
        }
        sigma_C_str = "n/a" if noise_info["sigma_C"] is None else f"{noise_info['sigma_C']:.2e}"
        mode_tag = "auto"
        node.log(
            f"  {q.name}: FIR done ({mode_tag}) — forward {len(h_fir)} taps, inverse {len(h_inv)} taps"
        )
        node.log(
            f"  {q.name}: noise sigma_A={noise_info['sigma_A']:.2e} sigma_B={noise_info['sigma_B']:.2e} "
            f"sigma_C={sigma_C_str}  ratio_AB={noise_info['ratio_AB']:.2f}  "
            f"ratio_fit={noise_info['ratio_fit']:.2f}  -> {noise_info['msg_short']}"
        )
        if noise_info["status"] == "warn_tail":
            node.log(
                f"  {q.name}: tail may not be settled (sigma_A >> sigma_B); "
                f"try a longer cryoscope_len."
            )
        elif noise_info["status"] == "warn_fit":
            node.log(
                f"  {q.name}: tail looks clean but fit residual exceeds noise floor "
                f"(ratio_fit={noise_info['ratio_fit']:.2f}). Likely upstream IIR underfits "
                f"a long-tau component; consider raising n_exponentials or revisiting the "
                f"cryoscope amplitude / detuning."
            )

    return fir_results
