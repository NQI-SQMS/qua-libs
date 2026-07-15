"""
Optimised Wigner tomography — probe-displacement generation and density-matrix
reconstruction.

Probe-displacement optimisation follows Reinhold (2019) Appendix C and G;
the Laguerre-expansion tomography matrix follows Cahill & Glauber (1969).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_genlaguerre

from qualibrate import QualibrationNode

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tomography matrix and its gradient
# ─────────────────────────────────────────────────────────────────────────────

def wigner_matrix_and_grad(disps: np.ndarray, N_ph: int):
    """Build the tomography matrix M (N_exp × N_ph²) and its gradients.

    The (α, n, m) element (row = displacement α, column = flat index n*N_ph+m):

        M[α, n, m] = e^{-2|α|²} · (-1)^m · sqrt(m!/n!) · (2α)^{n-m}
                     · L_m^{n-m}(4|α|²)    for n ≥ m

    with M[α, m, n] = conj(M[α, n, m]) for n < m.

    Returns
    -------
    M      : (N_exp, N_ph²) complex128
    dM_dr  : (N_exp, N_ph²) complex128  — ∂M/∂Re(α)
    dM_di  : (N_exp, N_ph²) complex128  — ∂M/∂Im(α)
    """
    disps = np.asarray(disps, dtype=np.complex128)
    N_exp = len(disps)
    N2 = N_ph * N_ph

    log_fact = np.zeros(N_ph, dtype=np.float64)
    for k in range(1, N_ph):
        log_fact[k] = log_fact[k - 1] + math.log(k)

    abs2 = disps.real ** 2 + disps.imag ** 2
    two_a = 2.0 * disps
    four_abs2 = 4.0 * abs2
    exp_prefactor = np.exp(-2.0 * abs2)

    M = np.zeros((N_exp, N2), dtype=np.complex128)
    dM_dr = np.zeros((N_exp, N2), dtype=np.complex128)
    dM_di = np.zeros((N_exp, N2), dtype=np.complex128)

    for n in range(N_ph):
        for m in range(n + 1):
            col = n * N_ph + m
            k = n - m
            pf = math.exp(0.5 * (log_fact[m] - log_fact[n]))
            sign_m = (-1) ** m
            two_a_pow = two_a ** k
            L = eval_genlaguerre(m, k, four_abs2)

            M[:, col] = (exp_prefactor * sign_m * pf) * two_a_pow * L

            if m > 0:
                dL_dt = -eval_genlaguerre(m - 1, k + 1, four_abs2)
            else:
                dL_dt = np.zeros(N_exp, dtype=np.float64)

            common = exp_prefactor * sign_m * pf

            dE_dr = -4.0 * disps.real * exp_prefactor
            dP_dr = (2.0 * k) * (two_a ** (k - 1)) if k > 0 else np.zeros(N_exp, dtype=np.complex128)
            dL_dr = 8.0 * disps.real * dL_dt

            dM_dr[:, col] = common * (
                dE_dr / exp_prefactor * two_a_pow * L
                + dP_dr * L
                + two_a_pow * dL_dr
            )

            dE_di = -4.0 * disps.imag * exp_prefactor
            dP_di = (2.0j * k) * (two_a ** (k - 1)) if k > 0 else np.zeros(N_exp, dtype=np.complex128)
            dL_di = 8.0 * disps.imag * dL_dt

            dM_di[:, col] = common * (
                dE_di / exp_prefactor * two_a_pow * L
                + dP_di * L
                + two_a_pow * dL_di
            )

    for n in range(N_ph):
        for m in range(n + 1, N_ph):
            col = n * N_ph + m
            col_conj = m * N_ph + n
            M[:, col] = np.conj(M[:, col_conj])
            dM_dr[:, col] = np.conj(dM_dr[:, col_conj])
            dM_di[:, col] = np.conj(dM_di[:, col_conj])

    return M, dM_dr, dM_di


# ─────────────────────────────────────────────────────────────────────────────
# Condition-number cost + gradient for optimisation
# ─────────────────────────────────────────────────────────────────────────────

def _cost_and_grad(r_disps: np.ndarray, N_ph: int, pin_first: bool = True):
    """κ(M) and its gradient w.r.t. the flat real parameter vector."""
    N = len(r_disps) // 2
    disps = r_disps[:N] + 1j * r_disps[N:]

    M, dM_dr, dM_di = wigner_matrix_and_grad(disps, N_ph)
    U, S, Vd = np.linalg.svd(M, full_matrices=False)

    if S[-1] < 1e-15:
        return 1e12, np.zeros_like(r_disps)

    kappa = S[0] / S[-1]
    u0, uN = U[:, 0], U[:, -1]
    v0, vN = Vd[0, :], Vd[-1, :]

    grad_r = np.zeros(N, dtype=np.float64)
    grad_i = np.zeros(N, dtype=np.float64)

    for k in range(N):
        dMk_dr = dM_dr[[k], :]
        dMk_di = dM_di[[k], :]

        dS0_dr = np.real(np.conj(u0[k]) * (dMk_dr @ np.conj(v0)))[0]
        dSN_dr = np.real(np.conj(uN[k]) * (dMk_dr @ np.conj(vN)))[0]
        dS0_di = np.real(np.conj(u0[k]) * (dMk_di @ np.conj(v0)))[0]
        dSN_di = np.real(np.conj(uN[k]) * (dMk_di @ np.conj(vN)))[0]

        smin = S[-1]
        grad_r[k] = (dS0_dr * smin - S[0] * dSN_dr) / smin ** 2
        grad_i[k] = (dS0_di * smin - S[0] * dSN_di) / smin ** 2

    if pin_first:
        grad_r[0] = 0.0
        grad_i[0] = 0.0

    return float(kappa), np.concatenate([grad_r, grad_i])


# ─────────────────────────────────────────────────────────────────────────────
# Displacement optimisation
# ─────────────────────────────────────────────────────────────────────────────

def optimize_displacements(
    N_ph: int,
    N_exp: Optional[int] = None,
    n_restarts: int = 3,
    ftol: float = 1e-6,
    seed: Optional[int] = None,
    max_displacement: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """Find N_exp complex displacements that minimise κ(M).

    Parameters
    ----------
    N_ph             : Fock cutoff (reconstruction truncation)
    N_exp            : number of probe points (default N_ph² + 30)
    n_restarts       : random restarts; best result is returned
    ftol             : L-BFGS-B convergence tolerance
    seed             : RNG seed for reproducibility
    max_displacement : clamp each Re/Im component to ±max_displacement
                       (photon units); set ≈ (2 - margin) * displacement_k
                       to respect the firmware ±2 amplitude limit.

    Returns
    -------
    disps : (N_exp,) complex128 — optimised displacements (index 0 = origin)
    kappa : float — achieved condition number
    """
    if N_exp is None:
        N_exp = N_ph ** 2 + 30

    if max_displacement is not None:
        lo, hi = -max_displacement, max_displacement
        bounds = [(lo, hi)] * (2 * N_exp)
        bounds[0] = (0.0, 0.0)
        bounds[N_exp] = (0.0, 0.0)
    else:
        bounds = None

    rng = np.random.default_rng(seed)
    best_kappa = np.inf
    best_x = None

    for restart in range(n_restarts):
        x0_re = rng.standard_normal(N_exp)
        x0_im = rng.standard_normal(N_exp)
        x0_re[0] = 0.0
        x0_im[0] = 0.0
        if max_displacement is not None:
            x0_re = np.clip(x0_re, lo, hi)
            x0_im = np.clip(x0_im, lo, hi)
        x0 = np.concatenate([x0_re, x0_im])

        result = minimize(
            _cost_and_grad,
            x0,
            args=(N_ph,),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"ftol": ftol, "maxiter": 5000, "maxfun": 50000},
        )

        kappa = result.fun
        logger.info(f"  restart {restart + 1}/{n_restarts}:  κ = {kappa:.4f}")
        if kappa < best_kappa:
            best_kappa = kappa
            best_x = result.x

    disps = best_x[:N_exp] + 1j * best_x[N_exp:]
    disps[0] = 0.0 + 0.0j
    return disps, float(best_kappa)


# ─────────────────────────────────────────────────────────────────────────────
# Load-or-optimise helper
# ─────────────────────────────────────────────────────────────────────────────

def load_or_optimize(
    npz_path: str | Path,
    N_ph: int,
    N_exp: Optional[int] = None,
    **kwargs,
) -> tuple[np.ndarray, float]:
    """Load pre-computed displacements or run optimisation and cache.

    Parameters
    ----------
    npz_path : path to .npz file (created if missing)
    N_ph     : Fock cutoff — must match the stored value when loading
    N_exp    : probe points (default N_ph² + 30)
    **kwargs : forwarded to optimize_displacements

    Returns
    -------
    disps : (N_exp,) complex128
    kappa : float
    """
    npz_path = Path(npz_path)
    if npz_path.exists():
        data = np.load(npz_path)
        disps = data["disps"]
        kappa = float(data["kappa"])
        stored_N_ph = int(data.get("N_ph", N_ph))
        if stored_N_ph != N_ph:
            raise ValueError(
                f"Loaded npz has N_ph={stored_N_ph} but requested N_ph={N_ph}. "
                f"Delete '{npz_path}' to rerun optimisation."
            )
        logger.info(f"Loaded {len(disps)} displacements from '{npz_path}'  (κ={kappa:.3f})")
        return disps, kappa

    if N_exp is None:
        N_exp = N_ph ** 2 + 30
    logger.info(f"Optimising {N_exp} displacements for N_ph={N_ph} …")
    disps, kappa = optimize_displacements(N_ph, N_exp=N_exp, **kwargs)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, disps=disps, kappa=kappa, N_ph=N_ph, N_exp=N_exp)
    logger.info(f"Saved to '{npz_path}'  (κ={kappa:.3f})")
    return disps, kappa


def resolve_probe_displacements(node: QualibrationNode, results_dir: Optional[Path] = None):
    """Load or optimise probe displacements according to node parameters.

    Returns (disps, kappa, npz_path_used).
    """
    N_ph = node.parameters.n_fock_cutoff
    N_exp = node.parameters.n_probe_points

    if node.parameters.probe_displacements_path is not None:
        npz_path = Path(node.parameters.probe_displacements_path)
    else:
        if results_dir is None:
            results_dir = Path(".")
        npz_path = results_dir / f"probe_displacements_N{N_ph}.npz"

    disps, kappa = load_or_optimize(npz_path, N_ph, N_exp=N_exp)
    return disps, kappa, str(npz_path)


# ─────────────────────────────────────────────────────────────────────────────
# Chi-Hz helper (mirrors displacement_wigner_calibration)
# ─────────────────────────────────────────────────────────────────────────────

def _get_chi_hz(node: QualibrationNode) -> Optional[float]:
    if node.parameters.chi_hz is not None:
        return abs(float(node.parameters.chi_hz))
    mode_name = node.parameters.mode_name
    for pair in node.machine.cavity_transmon_pairs.values():
        if pair.cavity_mode_name == mode_name:
            chi = getattr(pair, "chi", None)
            if chi is not None:
                return abs(float(chi))
    return None


def resolve_parity_time_ns(node: QualibrationNode) -> int:
    """Return parity wait time [ns] (multiple of 4 ns).

    Priority:
      1. node.parameters.parity_time_ns        (explicit override)
      2. pair.parity_time from QuAM            (written by node 28 — preferred)
      3. computed from node.parameters.chi_hz  (explicit chi override)
      4. computed from pair.chi in QuAM        (fallback)
    """
    if node.parameters.parity_time_ns is not None:
        t = int(node.parameters.parity_time_ns)
        return max(t // 4, 4) * 4

    # Check pair.parity_time written by node 28 (experimentally calibrated)
    mode_name = node.parameters.mode_name
    for pair in node.machine.cavity_transmon_pairs.values():
        if pair.cavity_mode_name == mode_name:
            pt = getattr(pair, "parity_time", None)
            if pt is not None:
                t_ns = max(int(round(float(pt) * 1e9 / 4)) * 4, 16)
                logger.info(f"parity_time_ns from QuAM pair.parity_time → {t_ns} ns")
                return t_ns
            break

    # Fallback: compute from chi
    chi_hz = _get_chi_hz(node)
    if chi_hz is None:
        raise ValueError(
            "Cannot determine parity_time_ns: run node 28 first (writes pair.parity_time), "
            "or set node.parameters.parity_time_ns / node.parameters.chi_hz explicitly."
        )
    t_ns = 1e9 / (4.0 * chi_hz)
    t_ns_rounded = max(int(round(t_ns / 4)) * 4, 16)
    logger.info(f"parity_time_ns computed from chi_hz={chi_hz*1e-3:.1f} kHz → {t_ns_rounded} ns")
    return t_ns_rounded


# ─────────────────────────────────────────────────────────────────────────────
# Density-matrix reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_state(
    disps: np.ndarray,
    parity_measurements: np.ndarray,
    N_ph: int,
) -> np.ndarray:
    """Reconstruct the density matrix from polarity-subtracted parity data.

    Parameters
    ----------
    disps               : (N_exp,) complex — probe displacement positions
    parity_measurements : (N_exp,) real   — P_e(polarity=+1) − P_e(polarity=-1) = ⟨(−1)^N⟩_displaced
    N_ph                : Fock cutoff

    Returns
    -------
    rho : (N_ph, N_ph) complex128, Hermitian, PSD, Tr=1
    """
    disps = np.asarray(disps, dtype=np.complex128)
    parity_measurements = np.asarray(parity_measurements, dtype=np.float64)

    M, _, _ = wigner_matrix_and_grad(disps, N_ph)
    rhs = parity_measurements

    trace_row = np.zeros(N_ph * N_ph, dtype=np.complex128)
    for n in range(N_ph):
        trace_row[n * N_ph + n] = 1.0

    lam = np.sqrt(len(disps))
    M_aug = np.vstack([M, lam * trace_row.reshape(1, -1)])
    rhs_aug = np.concatenate([rhs, [lam * 1.0]])

    rho_vec, _, _, _ = np.linalg.lstsq(M_aug, rhs_aug, rcond=None)
    rho = rho_vec.reshape(N_ph, N_ph)

    rho = 0.5 * (rho + rho.conj().T)

    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.real(eigvals)

    i = 0
    while i < len(eigvals) and eigvals[i] < 0:
        deficit = -eigvals[i]
        eigvals[i] = 0.0
        remaining = len(eigvals) - i - 1
        if remaining > 0:
            eigvals[i + 1:] -= deficit / remaining
        i += 1

    total = np.sum(eigvals)
    if total > 0:
        eigvals /= total

    return (eigvecs * eigvals) @ eigvecs.conj().T


def compute_fidelity(rho: np.ndarray, target) -> float:
    """Fidelity F = Tr[target · rho] (target may be a ket or density matrix)."""
    target = np.asarray(target, dtype=np.complex128)
    if target.ndim == 1:
        return float(np.real(target.conj() @ rho @ target))
    return float(np.real(np.trace(target @ rho)))
