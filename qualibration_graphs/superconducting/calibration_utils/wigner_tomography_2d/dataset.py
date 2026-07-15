"""xarray dataset utilities for 2D Wigner tomography raw measurements."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr


def build_raw_dataset(
    probe_disps: np.ndarray,
    state_plus: np.ndarray,
    state_minus: np.ndarray,
    parity: np.ndarray,
    **attrs,
) -> xr.Dataset:
    """Build an xarray Dataset with raw Wigner tomography measurements.

    The dataset is self-contained: probe displacements are stored as data
    variables so it can be loaded and re-analysed without the .npz file.

    Parameters
    ----------
    probe_disps  : (N_probe,) complex128 — displacement points in photon units
    state_plus   : (N_probe,) — P_e for polarity +1
    state_minus  : (N_probe,) — P_e for polarity -1
    parity       : (N_probe,) — state_minus - state_plus  (∝ W(α))
    **attrs      : stored as dataset attributes (metadata)

    Returns
    -------
    xr.Dataset with dimensions ["probe_idx"]
    """
    probe_idx = np.arange(len(probe_disps))
    return xr.Dataset(
        {
            "state_plus": xr.DataArray(
                state_plus, dims=["probe_idx"],
                attrs={"long_name": "Excitation probability (polarity +1)"},
            ),
            "state_minus": xr.DataArray(
                state_minus, dims=["probe_idx"],
                attrs={"long_name": "Excitation probability (polarity -1)"},
            ),
            "parity": xr.DataArray(
                parity, dims=["probe_idx"],
                attrs={"long_name": "Wigner measurement: P_- − P_+"},
            ),
            "probe_disps_real": xr.DataArray(
                probe_disps.real, dims=["probe_idx"],
                attrs={"units": "photons", "long_name": "Re(displacement)"},
            ),
            "probe_disps_imag": xr.DataArray(
                probe_disps.imag, dims=["probe_idx"],
                attrs={"units": "photons", "long_name": "Im(displacement)"},
            ),
        },
        coords={"probe_idx": probe_idx},
        attrs=attrs,
    )


def dataset_to_arrays(
    ds: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract numpy arrays from a raw Wigner dataset.

    Returns
    -------
    probe_disps  : (N_probe,) complex128
    state_plus   : (N_probe,) float64
    state_minus  : (N_probe,) float64
    parity       : (N_probe,) float64
    """
    probe_disps = (
        ds["probe_disps_real"].values + 1j * ds["probe_disps_imag"].values
    )
    return (
        probe_disps,
        ds["state_plus"].values,
        ds["state_minus"].values,
        ds["parity"].values,
    )


def load_dataset(path: str | Path) -> xr.Dataset:
    """Load a raw Wigner tomography dataset from a NetCDF file."""
    return xr.open_dataset(path)
