"""Analysis module for the JAZZ2-N SNZ amplitude / t_phi_eff scan.

The node measures P_|00> on a 3-D (amplitude_scale, t_phi_eff, N) volume
with N = 2k (paper convention; m = N + 1 = 1, 3, 5, ...). The map used
for the optimum search is the N-averaged 2-D map

    <P_|00>>_N(amp, t_phi_eff) = mean_N P_|00>(amp, t_phi_eff, N).

When N_min == N_max the average is trivially a single-N map and recovers
the original behaviour. The optimum (amp*, t_phi_eff*) is the point where
<P_|00>>_N is maximised, since P_|00> = 1 implies both perfect CZ angle
AND zero leakage (and averaging preserves both signatures).

We localise the optimum by

1. discrete argmax of the averaged grid;
2. if the discrete argmax is in the interior of the grid (not on any
   edge), fit a 2-D quadratic ``f(x, y) = a x^2 + b y^2 + c x y + d x +
   e y + f0`` to a 5x5 patch around it via least squares, and report the
   critical point of the fit (with sanity checks: Hessian must be
   negative definite, critical point must lie inside the patch);
3. otherwise we report the discrete argmax and mark success=False.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode

from calibration_utils.snz_b_over_a import decompose_t_phi_eff


@dataclass
class FitResults:
    """JAZZ2-N SNZ fit results for a single qubit pair."""

    optimal_amplitude: float
    """Absolute optimal CZ amplitude (Volts), i.e. amp_scale_optimal * stored amplitude."""
    optimal_amplitude_scale: float
    """Optimal amplitude scale factor (dimensionless; multiplied with stored amplitude)."""
    optimal_t_phi_eff: float
    """Optimal effective idle time (ns)."""
    optimal_t_phi: int
    """Integer idle samples at the optimum (from decompose_t_phi_eff)."""
    optimal_b_over_a: float
    """B/A transition-sample ratio at the optimum (from decompose_t_phi_eff)."""
    optimal_p00: float
    """Achieved P_|00> at the optimum (sub-grid via 2-D quadratic when applicable)."""
    success: bool
    """True iff the discrete argmax is in the interior of the swept grid."""
    fit_method: str = "argmax"
    """Either 'quadratic_2d' (sub-grid) or 'argmax' (interior check failed or quadratic refusal)."""


def coerce_to_even(n: int) -> int:
    """Coerce an integer to the nearest even integer >= 0."""
    if n < 0:
        return 0
    return 2 * int(round(n / 2.0))


def log_fitted_results(fit_results: Dict[str, FitResults], log_callable=None):
    """Log the JAZZ2-N SNZ fit results per qubit pair."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qp_name, fr in fit_results.items():
        status = "SUCCESS" if fr.success else "FAIL (edge/refusal)"
        msg = (
            f"Results for qubit pair {qp_name}: {status}!\n"
            f"\tOptimal amplitude  : {fr.optimal_amplitude:.6f} V "
            f"(scale {fr.optimal_amplitude_scale:.6f})\n"
            f"\tOptimal t_phi_eff  : {fr.optimal_t_phi_eff:.4f} ns\n"
            f"\t  -> t_phi         : {fr.optimal_t_phi} samples\n"
            f"\t  -> B/A           : {fr.optimal_b_over_a:.4f}\n"
            f"\tAchieved P_|00>    : {fr.optimal_p00:.4f}  (method={fr.fit_method})"
        )
        log_callable(msg)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Augment the raw dataset with an absolute-amplitude coordinate per qubit pair."""
    qubit_pairs = node.namespace["qubit_pairs"]
    operation = node.parameters.operation

    def abs_amp(qp, amp_rel):
        return amp_rel * qp.macros[operation].flux_pulse_qubit.amplitude

    ds = ds.assign_coords(
        {
            "amp_full": (
                ["qubit_pair", "amplitude"],
                np.array([abs_amp(qp, ds.amplitude.values) for qp in qubit_pairs]),
            )
        }
    )
    return ds


def _is_interior(i: int, j: int, n_i: int, n_j: int) -> bool:
    """Discrete argmax is in the interior of an (n_i, n_j) grid."""
    return 0 < i < n_i - 1 and 0 < j < n_j - 1


def _fit_2d_quadratic(
    amp_vals: np.ndarray,
    tpe_vals: np.ndarray,
    p_grid: np.ndarray,
    i_max: int,
    j_max: int,
    half_window: int = 2,
) -> Tuple[float, float, float, bool]:
    """Fit ``f(x,y) = a x^2 + b y^2 + c x y + d x + e y + f0`` to a patch
    around ``(i_max, j_max)`` and return the critical point.

    Returns
    -------
    amp_star, tpe_star, p_star, ok :
        ``ok`` is True iff the fit produced a maximum (Hessian negative
        definite) whose location lies inside the fitted patch and inside
        the (amp, t_phi_eff) sweep window. Otherwise the function falls
        back to the discrete-grid values at ``(i_max, j_max)``.
    """
    n_amp, n_tpe = p_grid.shape
    i_lo, i_hi = max(0, i_max - half_window), min(n_amp, i_max + half_window + 1)
    j_lo, j_hi = max(0, j_max - half_window), min(n_tpe, j_max + half_window + 1)

    sub_amp = amp_vals[i_lo:i_hi]
    sub_tpe = tpe_vals[j_lo:j_hi]
    sub_p = p_grid[i_lo:i_hi, j_lo:j_hi]

    # Need at least 6 finite samples for a 6-parameter fit.
    finite = np.isfinite(sub_p)
    if int(finite.sum()) < 6:
        return float(amp_vals[i_max]), float(tpe_vals[j_max]), float(p_grid[i_max, j_max]), False

    x0, y0 = float(amp_vals[i_max]), float(tpe_vals[j_max])
    X, Y = np.meshgrid(sub_amp, sub_tpe, indexing="ij")
    xc = (X - x0).ravel()
    yc = (Y - y0).ravel()
    z = sub_p.ravel()
    mask = np.isfinite(z)
    xc, yc, z = xc[mask], yc[mask], z[mask]

    M = np.column_stack([xc**2, yc**2, xc * yc, xc, yc, np.ones_like(xc)])
    try:
        coeffs, *_ = np.linalg.lstsq(M, z, rcond=None)
    except np.linalg.LinAlgError:
        return x0, y0, float(p_grid[i_max, j_max]), False

    a, b, c, d, e, f0 = (float(v) for v in coeffs)

    # Critical point: solve [[2a, c], [c, 2b]] [x*, y*] = [-d, -e].
    det = 4.0 * a * b - c * c
    if abs(det) < 1e-12:
        return x0, y0, float(p_grid[i_max, j_max]), False
    x_star = (c * e - 2.0 * b * d) / det
    y_star = (c * d - 2.0 * a * e) / det

    # Must be a maximum: Hessian = [[2a, c], [c, 2b]] negative-definite.
    if not (a < 0.0 and det > 0.0):
        return x0, y0, float(p_grid[i_max, j_max]), False

    # Critical point must lie inside the fitted patch.
    dx_max = float(max(abs(sub_amp[0] - x0), abs(sub_amp[-1] - x0)))
    dy_max = float(max(abs(sub_tpe[0] - y0), abs(sub_tpe[-1] - y0)))
    if abs(x_star) > dx_max or abs(y_star) > dy_max:
        return x0, y0, float(p_grid[i_max, j_max]), False

    amp_star = x0 + x_star
    tpe_star = y0 + y_star
    # Critical point must also lie inside the swept window.
    if not (float(amp_vals[0]) <= amp_star <= float(amp_vals[-1])):
        return x0, y0, float(p_grid[i_max, j_max]), False
    if not (float(tpe_vals[0]) <= tpe_star <= float(tpe_vals[-1])):
        return x0, y0, float(p_grid[i_max, j_max]), False

    p_star = a * x_star**2 + b * y_star**2 + c * x_star * y_star + d * x_star + e * y_star + f0
    return amp_star, tpe_star, float(p_star), True


def _fit_one_pair(
    amp_vals: np.ndarray, tpe_vals: np.ndarray, p_grid: np.ndarray
) -> Tuple[float, float, float, bool, str]:
    """Discrete argmax of ``p_grid`` (shape ``(n_amp, n_tpe)``) + 2-D quadratic refinement.

    Returns ``(amp_star, tpe_star, p_star, interior_success, method)``.
    """
    if p_grid.ndim != 2 or p_grid.shape != (len(amp_vals), len(tpe_vals)):
        return float("nan"), float("nan"), float("nan"), False, "none"
    if not np.any(np.isfinite(p_grid)):
        return float("nan"), float("nan"), float("nan"), False, "none"

    p_finite = np.where(np.isfinite(p_grid), p_grid, -np.inf)
    flat_idx = int(np.argmax(p_finite))
    i_max, j_max = np.unravel_index(flat_idx, p_grid.shape)

    interior = _is_interior(int(i_max), int(j_max), len(amp_vals), len(tpe_vals))
    if interior:
        amp_star, tpe_star, p_star, ok = _fit_2d_quadratic(amp_vals, tpe_vals, p_grid, int(i_max), int(j_max))
        method = "quadratic_2d" if ok else "argmax"
    else:
        amp_star = float(amp_vals[i_max])
        tpe_star = float(tpe_vals[j_max])
        p_star = float(p_grid[i_max, j_max])
        method = "argmax"

    return amp_star, tpe_star, p_star, bool(interior), method


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitResults]]:
    """Fit the JAZZ2-N SNZ data per qubit pair.

    Averages ``p00`` over the ``N`` axis (no-op when ``N_min == N_max``),
    runs the 2-D quadratic refinement on the resulting (amp, t_phi_eff) map,
    and augments ``ds_fit`` with the per-pair ``p00_avg`` data variable plus
    optimal-point coordinates.
    """
    qubit_pairs = node.namespace["qubit_pairs"]
    operation = node.parameters.operation

    if "p" not in ds:
        raise RuntimeError("JAZZ2-N SNZ analysis requires 'p' in the dataset (state discrimination).")

    amp_vals = ds.amplitude.values
    tpe_vals = ds.t_phi_eff.values

    # Average over the N axis if present. If absent (single-N legacy datasets), use p00 directly.
    if "N" in ds["p"].dims:
        p00_avg = ds["p"].mean(dim="N", keep_attrs=True)
    else:
        p00_avg = ds["p"]

    opt_amps_abs = []
    opt_amps_scale = []
    opt_tpes = []
    opt_t_phis = []
    opt_b_over_as = []
    opt_p00s = []
    successes = []
    methods = []
    qp_names = ds.qubit_pair.values
    fit_results: Dict[str, FitResults] = {}

    for qp_name in qp_names:
        qp = next(qp for qp in qubit_pairs if qp.name == qp_name)
        # Averaged map has dims (qubit_pair, amplitude, t_phi_eff).
        p_grid = p00_avg.sel(qubit_pair=qp_name).transpose("amplitude", "t_phi_eff").values
        amp_scale, tpe_star, p_star, ok, method = _fit_one_pair(amp_vals, tpe_vals, np.asarray(p_grid))

        stored_amp = qp.macros[operation].flux_pulse_qubit.amplitude
        amp_abs = amp_scale * stored_amp if np.isfinite(amp_scale) else np.nan
        if np.isfinite(tpe_star):
            t_phi, b_over_a = decompose_t_phi_eff(float(tpe_star))
        else:
            t_phi, b_over_a = 0, float("nan")

        opt_amps_abs.append(amp_abs)
        opt_amps_scale.append(amp_scale)
        opt_tpes.append(tpe_star)
        opt_t_phis.append(int(t_phi))
        opt_b_over_as.append(float(b_over_a))
        opt_p00s.append(p_star)
        successes.append(ok)
        methods.append(method)

        fit_results[str(qp_name)] = FitResults(
            optimal_amplitude=float(amp_abs),
            optimal_amplitude_scale=float(amp_scale),
            optimal_t_phi_eff=float(tpe_star),
            optimal_t_phi=int(t_phi),
            optimal_b_over_a=float(b_over_a),
            optimal_p00=float(p_star),
            success=bool(ok),
            fit_method=str(method),
        )

    ds_fit = ds.assign({"p00_avg": p00_avg.rename("p00_avg")})
    ds_fit = ds_fit.assign_coords(
        {
            "optimal_amplitude": ("qubit_pair", np.array(opt_amps_abs, dtype=float)),
            "optimal_amplitude_scale": ("qubit_pair", np.array(opt_amps_scale, dtype=float)),
            "optimal_t_phi_eff": ("qubit_pair", np.array(opt_tpes, dtype=float)),
            "optimal_t_phi": ("qubit_pair", np.array(opt_t_phis, dtype=int)),
            "optimal_b_over_a": ("qubit_pair", np.array(opt_b_over_as, dtype=float)),
            "optimal_p00": ("qubit_pair", np.array(opt_p00s, dtype=float)),
            "success": ("qubit_pair", np.array(successes, dtype=bool)),
            "fit_method": ("qubit_pair", np.array(methods, dtype=object)),
        }
    )
    ds_fit.optimal_amplitude.attrs = {"long_name": "optimal CZ amplitude", "units": "V"}
    ds_fit.optimal_amplitude_scale.attrs = {"long_name": "optimal CZ amplitude scale", "units": "a.u."}
    ds_fit.optimal_t_phi_eff.attrs = {"long_name": "optimal t_phi_eff", "units": "ns"}
    ds_fit.optimal_p00.attrs = {"long_name": "achieved P_|00>", "units": "a.u."}
    ds_fit.p00_avg.attrs = {"long_name": "<P_|00>>_N", "units": "a.u."}
    return ds_fit, fit_results
