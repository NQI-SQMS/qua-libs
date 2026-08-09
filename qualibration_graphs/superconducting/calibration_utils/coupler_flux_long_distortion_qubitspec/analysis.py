"""Analysis for coupler flux long distortion (qubitspec dataload variant).

Extends the coupler-flux branch of qubit_flux_long_distortion_qubitspec with two
explicit dispersion-curve loading paths (priority order):

1. Qubit spectroscopy vs coupler flux (node 10 raw data) via ``spectroscopy_run_id``
   — highest resolution, uses ``_load_coupler_spectroscopy_curve``.
2. Ramsey vs coupler flux via ``ramsey_vs_flux_run_id``
   — via ``_load_ramseyflux_curve_from_param``.
3. Falls back to ``qubit.extras[{coupler}_dispersion_load_id]`` (original behaviour).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr

from calibration_utils.qubit_flux_long_distortion_qubitspec.analysis import (
    FluxDistortionExpFitResult,
    _frequency_to_flux,
    _load_dispersion_curve,
    extract_center_freqs_iq,
    extract_center_freqs_state,
    fit_raw_data as _pi_flux_fit_raw_data,
    multi_exp_fit_global,
    process_raw_dataset,
    _read_node_data_dict
)

def _derive_coupler_flux_from_decouple(
    detuning_hz: float,
    decouple_offset: float,
    curve: Tuple[np.ndarray, np.ndarray],
    branch: str = "auto",
) -> Optional[Tuple[float, float]]:
    """Derive absolute coupler flux for a signed detuning from the decouple_offset.

    Unlike ``_derive_flux_amp`` (which always targets *below* idle and returns
    a relative offset), this function:

    * looks up the qubit frequency at ``decouple_offset`` on the dispersion
      curve,
    * applies a **signed** detuning (positive = freq above decouple-point freq,
      negative = freq below),
    * interpolates the curve to find the absolute coupler-flux value that
      produces the target frequency, restricting the search to the requested
      ``branch`` and (within that branch) picking the crossing closest to
      ``decouple_offset``.

    Sign convention
    ---------------
    ``detuning_hz`` is **signed**: positive moves the target frequency above the
    decouple-point frequency, negative moves it below.  Whether a given sign is
    reachable depends on the dispersion curve coverage; if the target frequency
    falls outside the curve range, ``None`` is returned (callers typically fall
    back to a fixed amplitude).

    Branch selection
    ----------------
    A non-monotonic curve (e.g. an avoided crossing or parabola) may cross the
    target frequency on both sides of the decouple point.  ``branch`` controls
    which side is searched:

    * ``"left"``  — only crossings with ``flux <= decouple_offset``.
    * ``"right"`` — only crossings with ``flux >= decouple_offset``.
    * ``"auto"`` (default, or any other value) — search all crossings; if more
      than one remains, a warning is emitted and the crossing nearest the
      decouple point is used.

    Parameters
    ----------
    detuning_hz : float
        Signed detuning in Hz from the qubit frequency at the decouple point.
        Positive moves the target frequency **up**, negative moves it **down**.
    decouple_offset : float
        Coupler flux (V) at the decouple operating point (from state.json).
    curve : tuple of (flux_array, freq_array)
        Dispersion curve (qubit freq vs coupler flux) from spectroscopy or
        Ramsey data.
    branch : str, optional
        Which side of the decouple point to search for the crossing:
        ``"left"``, ``"right"``, or ``"auto"`` (default).

    Returns
    -------
    (coupler_flux_abs, freq_at_decouple) or None
        The absolute coupler flux (V) that achieves the target frequency,
        and the qubit frequency at the decouple point (Hz).
        Returns None if the target is outside the curve range or no crossing
        exists on the requested branch.
    """
    curve_flux, curve_freq = curve
    if len(curve_flux) < 2:
        return None

    decouple_idx = int(np.argmin(np.abs(curve_flux - decouple_offset)))
    decouple_flux_on_curve = float(curve_flux[decouple_idx])
    if abs(decouple_flux_on_curve - decouple_offset) > 0.01:
        import warnings
        warnings.warn(
            f"decouple_offset={decouple_offset:.4f} V is outside the dispersion "
            f"curve range [{curve_flux.min():.4f}, {curve_flux.max():.4f}] V "
            f"(nearest={decouple_flux_on_curve:.4f} V, gap="
            f"{abs(decouple_flux_on_curve - decouple_offset):.4f} V). "
            f"Reference frequency may be inaccurate."
        )
    freq_at_decouple = float(curve_freq[decouple_idx])

    target_freq = freq_at_decouple + detuning_hz

    diff = curve_freq - target_freq
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        import warnings
        warnings.warn(
            f"Target frequency {target_freq / 1e9:.6f} GHz "
            f"(decouple freq {freq_at_decouple / 1e9:.6f} GHz + detuning "
            f"{detuning_hz / 1e6:+.2f} MHz) is outside the dispersion curve "
            f"frequency range [{curve_freq.min() / 1e9:.6f}, "
            f"{curve_freq.max() / 1e9:.6f}] GHz. "
            f"Reduce |detuning_in_mhz| or acquire wider Ramsey/spectroscopy data."
        )
        return None

    # Interpolate every zero-crossing of (curve_freq - target_freq).
    crossing_fluxes = []
    for idx in sign_changes:
        f1, f2 = curve_freq[idx], curve_freq[idx + 1]
        x1, x2 = curve_flux[idx], curve_flux[idx + 1]
        frac = (target_freq - f1) / (f2 - f1) if abs(f2 - f1) > 0 else 0.0
        flux_val = x1 + frac * (x2 - x1)
        crossing_fluxes.append(flux_val)

    # Restrict to the requested branch (side of the decouple point).
    if branch == "left":
        candidate_fluxes = [f for f in crossing_fluxes if f <= decouple_offset]
    elif branch == "right":
        candidate_fluxes = [f for f in crossing_fluxes if f >= decouple_offset]
    else:
        candidate_fluxes = list(crossing_fluxes)

    if len(candidate_fluxes) == 0:
        import warnings
        warnings.warn(
            f"No crossing for target frequency {target_freq / 1e9:.6f} GHz on "
            f"branch='{branch}' relative to decouple_offset={decouple_offset:.4f} V "
            f"(all crossings at flux={['%.4f' % f for f in crossing_fluxes]} V). "
            f"Try branch='auto' or a different detuning sign."
        )
        return None

    if len(candidate_fluxes) > 1:
        import warnings
        warnings.warn(
            f"{len(candidate_fluxes)} crossings found for target frequency "
            f"{target_freq / 1e9:.6f} GHz on branch='{branch}' "
            f"(flux={['%.4f' % f for f in candidate_fluxes]} V). "
            f"Picking the one nearest decouple_offset={decouple_offset:.4f} V; "
            f"set branch='left'/'right' to disambiguate explicitly."
        )

    # Pick the candidate crossing closest to the decouple_offset.
    distances = [abs(f - decouple_offset) for f in candidate_fluxes]
    best = int(np.argmin(distances))
    return float(candidate_fluxes[best]), freq_at_decouple


def _load_ramseyflux_curve_from_param(
    run_id: Optional[int],
    qubit,
    coupler,
    node,
    frequency_var: str = "abs_peak_frequency",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load qubit-freq vs coupler-flux dispersion curve from a run specified by run_id.

    The Ramsey-vs-coupler-flux node stores coupler flux **relative to the
    coupler's decouple_offset** (i.e. 0 V in the saved data = decouple_offset).
    This function converts the flux axis to **absolute** values by adding
    ``coupler.decouple_offset`` so that downstream code can work in absolute
    coupler-flux coordinates.

    Supports the same two dataset layouts as the original:
    - Ramsey-based: ``qubit_frequency`` data var + ``coupler_flux`` dimension
    - Legacy:       ``coupler_flux`` data var + ``frequency_var`` data var

    Returns
    -------
    (flux_bias_absolute, frequency) as 1-D arrays, or None if run_id is None
    or load fails.  ``flux_bias_absolute`` is in absolute coupler-flux volts.
    """
    if run_id is None:
        return None
    print(f"Loading Ramsey-flux curve for {qubit.name} / {coupler.name} from run_id {run_id}")
    try:
        data = _read_node_data_dict(run_id)
        ds_fit = data["ds_fit"]

        decouple_offset = getattr(coupler, "decouple_offset", 0.0) or 0.0

        # --- Ramsey-based layout: qubit_frequency var + coupler_flux dim ---
        if "qubit_frequency" in ds_fit.data_vars and "coupler_flux" in ds_fit.dims:
            flux_bias_rel = ds_fit.coupler_flux.values
            if "qubit_pair" in ds_fit.dims:
                qp_names_in_ds = [str(qp) for qp in ds_fit.qubit_pair.values]
                qubit_match = None
                for pair_name, pair in node.machine.qubit_pairs.items():
                    if pair.coupler.name == coupler.name and str(pair_name) in qp_names_in_ds:
                        qubit_match = str(pair_name)
                        break
                if qubit_match is None:
                    print(f"No qubit_pair match found for {qubit.name} / {coupler.name}")
                    return None
                frequency = ds_fit["qubit_frequency"].sel(qubit_pair=qubit_match).values
            else:
                frequency = ds_fit["qubit_frequency"].values
            flux_bias_abs = flux_bias_rel + decouple_offset
            print(f"  Ramsey flux axis shifted by decouple_offset={decouple_offset:.4f} V "
                  f"-> absolute range [{flux_bias_abs[0]:.4f}, {flux_bias_abs[-1]:.4f}] V")
            return flux_bias_abs, frequency

        # --- Legacy layout: coupler_flux data var + frequency_var ---
        flux_bias_rel = ds_fit["coupler_flux"].values
        qubit_match = None
        for pair_name, pair in node.machine.qubit_pairs.items():
            if pair.coupler.name == coupler.name:
                qubit_match = str(pair_name)
                break
        if qubit_match is None or frequency_var not in ds_fit.data_vars:
            print(f"Cannot find '{frequency_var}' for {qubit.name} / {coupler.name} in run {run_id}")
            return None
        frequency = ds_fit[frequency_var].sel(qubit_pair=qubit_match).values
        flux_bias_abs = flux_bias_rel + decouple_offset
        print(f"  Ramsey flux axis shifted by decouple_offset={decouple_offset:.4f} V "
              f"-> absolute range [{flux_bias_abs[0]:.4f}, {flux_bias_abs[-1]:.4f}] V")
        return flux_bias_abs, frequency

    except Exception as e:
        print(f"Error loading dispersion curve from run {run_id}: {e}")
        return None


def _load_coupler_spectroscopy_curve(
    run_id: Optional[int],
    qubit,
    coupler,
    node,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load qubit-freq vs coupler-flux curve from a qubit-spectroscopy-vs-coupler-flux run (node 10).

    Reuses ``_extract_spectroscopy_curve`` from qubit_flux_long_distortion_qubitspec.  The raw
    dataset from node 10 (``10_qubit_spectroscopy_vs_coupler_flux``) has dims
    ``(qubit, detuning, flux_bias)`` where ``flux_bias`` is the coupler flux, but its ``qubit``
    coordinate holds **qubit-pair names** (e.g. ``"q0-1"``) rather than the measured qubit name
    (e.g. ``"q0"``); the measured qubit name lives in a separate ``measured_qubit_name`` coord.
    We therefore resolve the correct ``qubit`` coordinate label before extracting, instead of
    selecting blindly by ``qubit.name`` (which raises "not all values found in index 'qubit'").

    Returns
    -------
    (flux_bias, frequency) as 1-D arrays, or None if run_id is None or load fails.
    """
    # Resolves the dataset's actual qubit-coordinate label (pair name vs qubit name) so the
    # node-10 coupler dataset can be indexed correctly before running the DP extraction pipeline.
    if run_id is None:
        return None
    print(f"Loading spectroscopy curve for {qubit.name} / {coupler.name} from run_id {run_id}")
    try:
        from calibration_utils.qubit_flux_long_distortion_qubitspec.analysis import (
            _extract_spectroscopy_curve,
        )
        ds_raw = _read_node_data_dict(run_id)["ds_raw"]
        qvals = [str(v) for v in np.atleast_1d(ds_raw.qubit.values)]

        # Resolve the qubit coordinate label used by this dataset.
        key = None
        if qubit.name in qvals:
            # Older datasets store data keyed by the measured qubit name directly.
            key = qubit.name
        else:
            # Node-10 layout: qubit coord = pair names; map via the (unique) coupler.
            for pair_name, pair in node.machine.qubit_pairs.items():
                if pair.coupler.name == coupler.name and str(pair_name) in qvals:
                    key = str(pair_name)
                    break
            # Fallback: single match on the measured_qubit_name coordinate.
            if key is None and "measured_qubit_name" in ds_raw.coords:
                matches = [
                    qv
                    for qv, m in zip(qvals, np.atleast_1d(ds_raw.measured_qubit_name.values))
                    if str(m) == qubit.name
                ]
                if len(matches) == 1:
                    key = matches[0]

        if key is None:
            print(
                f"  WARNING: could not resolve qubit coord for {qubit.name}/{coupler.name} "
                f"in run #{run_id} (qubit coords={qvals}); skipping spectroscopy path"
            )
            return None

        curve = _extract_spectroscopy_curve(ds_raw, key, qubit.xy.RF_frequency)
        if curve is not None:
            print(
                f"  Loaded spectroscopy curve for {qubit.name} (coord '{key}') from run #{run_id}: "
                f"{len(curve[0])} pts, flux=[{curve[0][0]:.4f}, {curve[0][-1]:.4f}] V"
            )
        return curve
    except Exception as e:
        print(f"Error loading spectroscopy curve from run {run_id}: {e}")
        return None


def fit_raw_data(ds: xr.Dataset, node) -> tuple[xr.Dataset, Dict[str, FluxDistortionExpFitResult]]:
    """Compute center_freqs, flux_response, and fit exponential cascade.

    Mirrors pi_flux.fit_raw_data for the coupler-flux branch.  The only difference
    is that the dispersion curve is loaded from ``node.parameters.ramsey_vs_flux_run_id``
    (via ``_load_ramseyflux_curve_from_param``) with a fallback to
    ``qubit.extras[dispersion_load_id]`` (via the original ``_load_dispersion_curve``).

    For qubit-flux nodes (qubit_pairs absent in namespace), delegates entirely to
    pi_flux.fit_raw_data to avoid code duplication.
    """
    # Entry point for the coupler-flux variant of the distortion calibration; tries the three curve-loading strategies in priority order before running the same exponential fit pipeline as the qubit-flux branch.
    qubits = node.namespace["qubits"]
    qubit_pairs = node.namespace.get("qubit_pairs")

    if qubit_pairs is None:
        # Not a coupler-flux node — delegate to pi_flux unchanged
        return _pi_flux_fit_raw_data(ds, node)

    # ------------------------------------------------------------------
    # Shared pre-processing (mirrors pi_flux.fit_raw_data)
    # ------------------------------------------------------------------
    dfs = (
        node.namespace.get("sweep_axes", {}).get("detuning").values
        if "sweep_axes" in node.namespace
        else (ds.get("detuning").values if "detuning" in ds.dims else ds.get("freq").values)
    )
    if "detuning" not in ds.dims and "freq" in ds.dims:
        ds = ds.rename({"freq": "detuning"})

    if node.parameters.use_state_discrimination and "state" in ds.data_vars:
        center_freqs = extract_center_freqs_state(ds, dfs)
    else:
        center_freqs = extract_center_freqs_iq(ds, dfs)

    # ------------------------------------------------------------------
    # Coupler-flux branch
    # ------------------------------------------------------------------
    ds = ds.assign_coords(
        {
            "freq_full": (
                ["qubit", "detuning"],
                np.array([dfs + q.xy.RF_frequency for q in qubits]),
            ),
            "detuning": (
                ["qubit", "detuning"],
                np.array([dfs for _ in qubits]),
            ),
            "flux": (
                ["qubit", "detuning"],
                np.full((len(qubits), len(dfs)), np.nan, dtype=float),
            ),
        }
    )

    abs_freq = center_freqs + xr.DataArray(
        [q.xy.RF_frequency for q in qubits],
        coords={"qubit": [q.name for q in qubits]},
        dims=["qubit"],
    )
    flux_response = xr.full_like(center_freqs, np.nan, dtype=float)

    use_spec = getattr(node.parameters, "use_spectroscopy_data", False)
    spec_run_id = getattr(node.parameters, "spectroscopy_run_id", None)
    use_ramsey = getattr(node.parameters, "use_ramsey_vs_flux_data", True)
    ramsey_run_id = getattr(node.parameters, "ramsey_vs_flux_run_id", None)

    # Coupler flux operating point — used for branch selection.
    coupler_flux_center = node.namespace.get("coupler_flux_center")
    if coupler_flux_center is None:
        flux_sweep_min = getattr(node.parameters, "flux_sweep_min_volt", None)
        flux_sweep_max = getattr(node.parameters, "flux_sweep_max_volt", None)
        coupler_flux_center = (
            (flux_sweep_min + flux_sweep_max) / 2
            if flux_sweep_min is not None and flux_sweep_max is not None
            else getattr(node.parameters, "coupler_flux_amplitude_in_v", None)
        )

    # Retrieve per-pair decouple offsets (persisted by create_qua_program)
    decouple_offsets = node.namespace.get("decouple_offsets")
    if decouple_offsets is None:
        decouple_offsets = [qp.coupler.decouple_offset for qp in qubit_pairs]

    for i, q in enumerate(qubits):
        if i >= len(qubit_pairs):
            break
        coupler = qubit_pairs[i].coupler

        curve = None
        if use_spec and spec_run_id is not None:
            curve = _load_coupler_spectroscopy_curve(spec_run_id, q, coupler, node)
        if curve is None and use_ramsey and ramsey_run_id is not None:
            curve = _load_ramseyflux_curve_from_param(ramsey_run_id, q, coupler, node)
        if curve is None:
            curve = _load_dispersion_curve(node, q, coupler)

        # Filter curve to the branch containing the operating flux point.
        # Use the decouple_offset as the reference: the coupler_flux_center
        # is on one side of the decouple point, so keep that side of the curve.
        if coupler_flux_center is not None and curve is not None:
            dec_off = decouple_offsets[i] if i < len(decouple_offsets) else None
            ref_flux = dec_off if dec_off is not None else 0.0
            if float(coupler_flux_center) >= ref_flux:
                branch_mask = curve[0] >= ref_flux
            else:
                branch_mask = curve[0] <= ref_flux
            if np.sum(branch_mask) >= 2:
                curve = (curve[0][branch_mask], curve[1][branch_mask])

        if curve is not None and len(curve[0]) >= 2:
            flux_bias, abs_peak = curve
            abs_freq_q = abs_freq.sel(qubit=q.name).values
            flux_vals = _frequency_to_flux(abs_freq_q, flux_bias, abs_peak)
            flux_response.values[i, :] = flux_vals

    ds = ds.copy()
    ds["center_freqs"] = center_freqs
    ds["flux_response"] = flux_response

    n_exponentials = int(getattr(node.parameters, "n_exponentials", 3))
    fit_results: Dict[str, FluxDistortionExpFitResult] = {}
    for q in qubits:
        qf = flux_response.sel(qubit=q.name)
        t_data = np.asarray(qf.time.values, dtype=float)
        y_data = np.asarray(qf.values, dtype=float)
        # Mask out NaN samples and the t=0 origin so the log-time weighting and
        # exponential decomposition stay well-defined (mirrors _fit_pi_flux_all_qubits).
        mask = np.isfinite(y_data) & (t_data > 0)
        fit_results[q.name] = multi_exp_fit_global(
            t_data[mask],
            y_data[mask],
            n_exponentials=n_exponentials,
            verbose=True,
        )
    return ds, fit_results
