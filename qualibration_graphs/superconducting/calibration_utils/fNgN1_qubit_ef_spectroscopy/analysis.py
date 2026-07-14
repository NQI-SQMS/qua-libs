"""Analysis utilities for qubit ef spectroscopy at Fock |N⟩."""
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


def fit_raw_data(ds: xr.Dataset, frequency_span_in_mhz: float, use_state_discrimination: bool):
    """Fit a Lorentzian peak to ef state-vs-detuning data at a given Fock level.

    Returns (ds_fit, fit_results) where fit_results maps qubit name → dict with
    keys: frequency_hz (detuning from sweep center), fwhm_hz, success.
    """
    signal_name = "state" if use_state_discrimination else "I"
    signal = getattr(ds, signal_name, None)
    if signal is None:
        signal = ds.I

    fit_results = {}
    fit_arrays = {}

    for q in ds.qubit.values:
        y = signal.sel(qubit=q).values.astype(float)
        x = ds.detuning.values.astype(float)

        try:
            y_smooth = savgol_filter(y, window_length=min(11, len(y) // 4 * 2 + 1), polyorder=2)
            peak_idx = int(np.argmax(y_smooth))
            f0_guess = float(x[peak_idx])
            amp_guess = float(y_smooth[peak_idx]) - float(np.mean(y))
            bkg_guess = float(np.mean(y))
            fwhm_guess = float(frequency_span_in_mhz) * 0.5e6

            def lorentzian(f, a, f0, gamma, c):
                return c + a * (gamma / 2) ** 2 / ((f - f0) ** 2 + (gamma / 2) ** 2)

            p0 = [amp_guess, f0_guess, fwhm_guess, bkg_guess]
            bounds = (
                [0, x.min(), 1e3, -np.inf],
                [1.5, x.max(), float(frequency_span_in_mhz) * 1e6, np.inf],
            )
            popt, _ = curve_fit(lorentzian, x, y, p0=p0, bounds=bounds, maxfev=5000)
            f0_fit = float(popt[1])
            fit_results[q] = {
                "frequency_hz": f0_fit,
                "fwhm_hz": float(abs(popt[2])),
                "success": True,
            }
            fit_arrays[q] = lorentzian(x, *popt)
        except Exception:
            fit_results[q] = {
                "frequency_hz": float("nan"),
                "fwhm_hz": float("nan"),
                "success": False,
            }
            fit_arrays[q] = np.full(len(x), np.nan)

    ds_fit = ds.copy()
    ds_fit = ds_fit.assign(
        fit=xr.DataArray(
            [fit_arrays[q] for q in ds.qubit.values],
            dims=["qubit", "detuning"],
            coords={"qubit": ds.qubit, "detuning": ds.detuning},
        )
    )
    return ds_fit, fit_results
