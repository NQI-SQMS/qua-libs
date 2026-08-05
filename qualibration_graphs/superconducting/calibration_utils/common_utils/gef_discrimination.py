"""Offline g/e/f (or g/e) state discrimination from raw IQ + calibrated centers.

Mirrors the real-time ``readout_state_gef`` macro but runs on the PC, so you can use a
better metric (L2 / Gaussian) than the real-time L1 argmin. Intended to give the
``use_state_discrimination=False`` (raw IQ) path of general nodes a proper offline
classifier into g/e/f, reusing the calibrated ``qubit.resonator.gef_centers``.

UNITS (the #1 bug source): ``I``, ``Q`` and ``centers`` MUST be in the SAME units.
``qubit.resonator.gef_centers`` are stored in raw-demod units (= center_V * length / 2**12).
- raw I/Q straight from ``measure``      -> pass ``gef_centers`` directly.
- I/Q already in Volts (convert_IQ_to_V) -> convert centers with :func:`centers_to_volts`.

Metric guide (cost / quality both increase L1 < L2 < Gaussian):
- "l1"       : Manhattan distance; matches the real-time macro (use for cross-checks).
               NOT rotation-invariant.
- "l2"       : Euclidean nearest-center (rotation-invariant). Default.
- "gaussian" : maximum 2D-isotropic Gaussian likelihood; needs per-state ``sigmas``;
               accounts for differing blob spread (same form used by the XEB analysis).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

_METHODS = ("l2", "gaussian", "l1")


def centers_to_volts(gef_centers, readout_length: float, demod_factor: int = 1) -> np.ndarray:
    """Convert ``gef_centers`` (raw-demod units) to Volts, matching ``convert_IQ_to_V``.

    ``convert_IQ_to_V`` does ``V = raw * demod_factor * 2**12 / length``; since the centers
    were stored as ``center_V * length / 2**12``, the inverse is::

        center_V = gef_centers * demod_factor * 2**12 / length

    Use ``demod_factor=2`` if the data was converted with ``single_demod=True``.
    """
    return np.asarray(gef_centers, dtype=float) * demod_factor * 2**12 / readout_length


def _scores(I, Q, centers, method, sigmas):
    """Per-state score with shape (..., K). Returns (score, sense) where sense is
    'min' (distances) or 'max' (log-likelihood)."""
    I = np.asarray(I, dtype=float)
    Q = np.asarray(Q, dtype=float)
    centers = np.asarray(centers, dtype=float)
    dI = I[..., None] - centers[:, 0]  # broadcast over last axis = state index
    dQ = Q[..., None] - centers[:, 1]
    if method == "l1":
        return np.abs(dI) + np.abs(dQ), "min"
    if method == "l2":
        return dI**2 + dQ**2, "min"  # squared Euclidean (argmin == Euclidean argmin)
    if method == "gaussian":
        if sigmas is None:
            raise ValueError("method='gaussian' requires `sigmas` (per-state std dev).")
        var = np.asarray(sigmas, dtype=float)[: centers.shape[0]] ** 2 + 1e-12
        # 2D isotropic log-likelihood, dropping the constant -ln(2*pi):
        #   ln[ (1/(2*pi*var)) * exp(-d^2 / (2*var)) ]  ->  -ln(var) - d^2 / (2*var)
        loglik = -np.log(var) - (dI**2 + dQ**2) / (2 * var)
        return loglik, "max"
    raise ValueError(f"unknown method {method!r}; choose from {_METHODS}")


def classify_iq(I, Q, centers, *, method: str = "l2", sigmas=None, n_states: int | None = None) -> np.ndarray:
    """Assign each (I, Q) sample to a state index (0=g, 1=e, 2=f).

    Parameters
    ----------
    I, Q : array_like
        Measured quadratures (any broadcastable shape). Same units as ``centers``.
    centers : array_like, shape (K, 2)
        State centers ``[[I_0, Q_0], ...]``; row k is state k.
    method : {"l2", "gaussian", "l1"}
        See module docstring. Default "l2".
    sigmas : array_like, shape (K,), optional
        Per-state std dev, required for ``method="gaussian"``.
    n_states : int, optional
        Use only the first ``n_states`` centers (e.g. 2 for g/e). Defaults to all.

    Returns
    -------
    numpy.ndarray
        Integer state index per sample, with the broadcast shape of ``I``, ``Q``.
    """
    centers = np.asarray(centers, dtype=float)[: (n_states or len(centers))]
    score, sense = _scores(I, Q, centers, method, sigmas)
    reduce = np.argmin if sense == "min" else np.argmax
    return reduce(score, axis=-1).astype(int)


def gef_posteriors(I, Q, centers, sigmas, *, n_states: int | None = None) -> np.ndarray:
    """Soft ``P(state | IQ)`` with shape (..., K), for soft counts / fidelity weighting."""
    centers = np.asarray(centers, dtype=float)[: (n_states or len(centers))]
    loglik, _ = _scores(I, Q, centers, "gaussian", sigmas)
    loglik = loglik - loglik.max(axis=-1, keepdims=True)  # numerical stability
    p = np.exp(loglik)
    return p / p.sum(axis=-1, keepdims=True)


def estimate_sigmas(I_by_state, Q_by_state) -> np.ndarray:
    """Per-state isotropic sigma from prepared-state IQ clouds.

    ``sigma_k = sqrt(0.5 * (var(I_k) + var(Q_k)))``. Use when ``qubit.extras['std_dev_k']``
    is not available.

    Parameters
    ----------
    I_by_state, Q_by_state : sequence (len K) of 1D arrays
        Single-shot I / Q values for each prepared state.
    """
    return np.array(
        [np.sqrt(0.5 * (np.var(i) + np.var(q))) for i, q in zip(I_by_state, Q_by_state)]
    )


def classify_dataarray(
    I_da: xr.DataArray,
    Q_da: xr.DataArray,
    centers,
    *,
    method: str = "l2",
    sigmas=None,
    n_states: int | None = None,
) -> xr.DataArray:
    """xarray wrapper around :func:`classify_iq`; returns a ``state`` DataArray that
    shares the dims/coords of ``I_da``."""
    return xr.apply_ufunc(
        lambda i, q: classify_iq(i, q, centers, method=method, sigmas=sigmas, n_states=n_states),
        I_da,
        Q_da,
    ).rename("state")
