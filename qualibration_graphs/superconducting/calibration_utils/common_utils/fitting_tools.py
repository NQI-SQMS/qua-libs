import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


###################################################
# General
###################################################
def sanitize_y_err(y_err):
    if y_err is not None:
        if all(y_err) == 0:
            y_err = None
        else:
            y_err = np.clip(y_err, 1e-12, None)

    return y_err


def IQR_filter(x, y=None, same_dim=False):
    """Outlier removal using the IQR rule on ``y``.

    Parameters
    ----------
    x : array_like
        First coordinate, or if ``y`` is omitted, the values to filter (treated as ``y``).
    y : array_like or None
        Values used to compute quartiles and the inclusion mask. If None, ``x`` is used as ``y``
        and the return is only the filtered ``y`` (or same-length ``y`` with outliers as nan).
    same_dim : bool
        If False (default), return only inlier points (shorter arrays).
        If True, return arrays with the same length as the input, with outliers replaced by nan.
    """
    if y is None:
        y_arr = np.asarray(x, dtype=float)
        q75, q25 = np.percentile(y_arr, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        mask = (y_arr > lower_bound) & (y_arr < upper_bound)
        if same_dim:
            out = y_arr.copy()
            out[~mask] = np.nan
            return out
        return y_arr[mask]

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    q75, q25 = np.percentile(y_arr, [75, 25])
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    mask = (y_arr > lower_bound) & (y_arr < upper_bound)
    if same_dim:
        x_out = x_arr.copy()
        y_out = y_arr.copy()
        x_out[~mask] = np.nan
        y_out[~mask] = np.nan
        return x_out, y_out
    return x_arr[mask], y_arr[mask]


###################################################
# Power law fitting
###################################################


def power_law(x, p, A, B):
    return A * (p**x) + B


def fit_power_law(x, y, y_err=None, bounds=([0, 0, 0], [np.inf, np.inf, np.inf])):
    order = np.argsort(x)
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]

    B0 = y[-1]
    y_shifted = y - B0
    mask = y_shifted > 1e-8
    if np.count_nonzero(mask) >= 2:
        # log(y - B) = log(A) + x * log(p)
        slope, intercept = np.polyfit(x[mask], np.log(y_shifted[mask]), 1)
        p0, A0 = np.exp(slope), np.exp(intercept)
    else:
        A0 = max(y[0] - B0, 1e-12)
        dx = x[1] - x[0]
        p0 = ((y[1] - B0) / A0) ** (1.0 / dx) if dx > 0 else 0.99

    initial_params = [np.clip(p, lo, hi) for p, lo, hi in zip([p0, A0, B0], *bounds)]
    y_err = sanitize_y_err(y_err)
    popt, pcov = curve_fit(power_law, x, y, p0=initial_params, sigma=y_err, bounds=bounds, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


###################################################
# Sinusoidal fitting
###################################################
def _zero_crossings(t, y, min_separation=0.01, smooth_window=0.05, gradient_percentile=20):
    """
    Detect zero-crossing times in a signal with noise suppression and gradient filtering.

    Locates time points where signal transitions through zero by detecting sign changes
    between consecutive samples and computing crossing times via linear interpolation.
    Filters out noise-induced spurious crossings using gradient magnitude percentile,
    and enforces minimum spacing between detections.

    Parameters
    ----------
    t : array_like, shape (N,)
        Time coordinate array. Must have same length as `y`.
    y : array_like, shape (N,)
        Signal amplitude array. Crossings are detected where sign changes occur.
    min_separation : float or None, optional
        Minimum required time interval between consecutive detected crossings.
        Crossings separated by less than this threshold are filtered, keeping only
        the first occurrence. If None, defaults to 1% of data duration: 0.01×(t[-1] - t[0]).
    smooth_window : float, int, or None, optional
        Smoothing kernel specification for noise reduction before crossing detection:

        - If 0 < smooth_window < 1: treated as fraction of array length.
          Window size = max(3, int(smooth_window × len(y)))
        - If smooth_window ≥ 1: integer window size (number of points)
        - If None, 0, or False: no smoothing applied

        Uses simple moving average (boxcar) convolution. Default: 0.05 (5% of data length).

    gradient_percentile : float or None, optional
        Percentile threshold for filtering crossings by gradient magnitude. Only crossings
        with gradients above this percentile are retained, eliminating low-slope noise-induced
        fluctuations while preserving high-gradient real crossings. Range: 0-100.
        If None or False, no gradient filtering applied. Default: 20 (only keep top 80%).

    Returns
    -------
    crossing_times : ndarray, shape (M,)
        Array of interpolated time values where zero crossings occur, in ascending order.
        Empty array if no crossings detected or if signal has constant sign.

    Algorithm
    ---------
    1. Convert inputs to float arrays
    2. Apply rolling mean smoothing if requested (mode='same' to preserve length)
    3. Detect sign changes using np.signbit (handles ±0 correctly)
    4. Compute crossing times by linear interpolation: t_cross = t[i] + Δt × |y[i]|/(|y[i]| + |y[i+1]|)
    5. Calculate gradient magnitude |dy/dt| at each sign-change point
    6. Filter by gradient percentile to eliminate noise-induced spurious crossings
    7. Filter crossings closer than min_separation using greedy forward selection
    8. Return sorted array of crossing times

    Implementation Notes
    --------------------
    - Uses np.signbit instead of y<0 for robust zero handling (distinguishes +0 from -0)
    - Prevents division by zero via safe denominator: max(|y[i]| + |y[i+1]|, 1e-15)
    - Guards array indexing to avoid out-of-bounds access when idx+1 used
    - Sub-sample accuracy achieved through linear interpolation between bracketing points
    - Gradient filtering exploits that real zero crossings have high |dy/dt| while noise
      fluctuations produce low-gradient sign changes
    - Gradient computed as |Δy|/|Δt| at sign-change points

    Applications
    ------------
    - Frequency estimation from oscillating signals (period = 2 × avg crossing spacing)
    - Phase-locked loop analysis and timing recovery
    - Period extraction from noisy quasi-periodic waveforms
    - Segment identification in zero-mean oscillatory data with noise

    Examples
    --------
    Detect crossings in clean 5 Hz sine wave:

    >>> t = np.linspace(0, 1, 500)
    >>> y = np.sin(2 * np.pi * 5 * t)
    >>> crossings = _zero_crossings(t, y)
    >>> len(crossings)
    10
    >>> # Expected: 10 crossings (5 complete periods × 2 crossings per period)

    Filter high-amplitude noise with gradient threshold:

    >>> y_noisy = y + 0.3 * np.random.randn(len(y))
    >>> crossings_filtered = _zero_crossings(t, y_noisy, gradient_percentile=20)
    >>> # Retains only top 80% of gradients, removing noise-induced spurious crossings

    Enforce minimum crossing separation:

    >>> crossings_spaced = _zero_crossings(t, y_noisy, min_separation=0.05)
    >>> # No two crossings closer than 50 ms apart

    See Also
    --------
    _omega_tau_from_segments : Uses zero crossings to estimate frequency and decay
    numpy.signbit : Sign bit function used for robust zero detection
    """
    y = np.asarray(y, float)
    t = np.asarray(t, float)

    # Apply rolling mean smoothing
    # if smooth_window:
    #     if smooth_window > 1:
    #         kernel = np.ones(smooth_window) / smooth_window
    #         y = np.convolve(y, kernel, mode="same")
    #     elif smooth_window < 1:
    #         # treat as fraction of total length
    #         window_size = max(3, int(smooth_window * len(y)))
    #         kernel = np.ones(window_size) / window_size
    #         y = np.convolve(y, kernel, mode="same")

    s = np.signbit(y)
    idx = np.where(s[:-1] != s[1:])[0]
    # Guard against potential out-of-range when using idx+1
    if idx.size:
        idx = idx[idx < (len(t) - 1)]
    if idx.size == 0:
        return np.array([])

    # Linear interpolation for zero-crossing positions
    # Avoid division by zero by using safe denominator
    y_abs = np.abs(y[idx]) + np.abs(y[idx + 1])
    y_abs = np.maximum(y_abs, 1e-15)  # Prevent division by zero
    t0 = t[idx] + (t[idx + 1] - t[idx]) * (np.abs(y[idx]) / y_abs)

    # Filter by gradient magnitude to reject noise-induced spurious crossings
    # Real zero crossings have high |dy/dt|, while noise fluctuations have low gradients
    # if gradient_percentile and gradient_percentile > 0 and gradient_percentile < 100:
    #     dy = np.abs(y[idx + 1] - y[idx])
    #     dt = np.abs(t[idx + 1] - t[idx])
    #     gradient = dy / np.maximum(dt, 1e-15)

    #     # Keep only crossings with gradients above the percentile threshold
    #     grad_threshold = np.percentile(gradient, gradient_percentile)
    #     mask_grad = gradient >= grad_threshold

    #     idx = idx[mask_grad]
    #     t0 = t0[mask_grad]

    # minimum separation as a fraction of total duration
    # min_separation_t = min_separation * (t[-1] - t[0])

    # keep = [0]
    # for i in range(1, len(t0)):
    #     if t0[i] - t0[keep[-1]] > min_separation_t:
    #         keep.append(i)

    # validate the zero crossings by checking if the intervals are consistent with
    # the observed frequency
    # TODO

    return t0


def _omega_tau_from_segments(t, y, verbose: bool = False):
    """
    Estimate angular frequency and decay time constant from zero-crossing analysis.

    Analyzes oscillating signal by:
    1. Detecting zero crossings to segment the waveform
    2. Estimating angular frequency (omega) from median half-period
    3. Extracting amplitude envelope from peak values in each segment
    4. Fitting exponential decay to envelope: A*exp(-t/tau)
    5. Determining phase sign from initial signal direction

    Parameters
    ----------
    t : array_like
        Time array (1D), must be same length as y.
    y : array_like
        Signal array (1D), assumed centered around zero.

    Returns
    -------
    omega : float
        Estimated angular frequency in rad/s. Computed as π divided by the
        median half-period between zero crossings. Returns 0.0 if period
        cannot be determined reliably.
    tau : float or None
        Estimated decay time constant in same units as t. If the exponential
        fit to the amplitude envelope has negative slope (indicating decay),
        tau = -1/slope. Returns None if no decay detected (positive slope) or
        insufficient data for fitting.
    phase_sign : float
        Sign indicator for initial phase: +1.0 or -1.0. Determined from the
        sign of the first amplitude peak or initial slope direction. Used to
        resolve phase ambiguity in fit initialization.

    Notes
    -----
    Fallback Behavior:
    - If < 2 zero crossings found (less than half period):
      * omega estimated from max/min separation
      * tau set to None
      * phase_sign from initial slope

    Amplitude Envelope Extraction:
    - Segments signal between consecutive zero crossings
    - Finds maximum absolute value and its time in each segment
    - Fits log(amplitude) vs time using linear regression
    - Only returns tau if fit shows decay (negative slope)

    Robustness Features:
    - Uses 60th percentile of half-periods to avoid noise-induced fast crossings
    - Prepends virtual zero crossing if first segment is nearly complete
    - Filters out zero/negative amplitudes before log-linear fit
    - Returns None for tau if fit indicates growth instead of decay

    Examples
    --------
    >>> t = np.linspace(0, 10, 1000)
    >>> y = 0.8 * np.exp(-t/2) * np.cos(2*np.pi*t)  # 1 Hz, tau=2s
    >>> omega, tau, phase_sign = _omega_tau_from_segments(t, y)
    >>> print(f"Frequency: {omega/(2*np.pi):.2f} Hz, Decay: {tau:.2f} s")
    Frequency: 1.00 Hz, Decay: 2.00 s
    """
    zt = _zero_crossings(t, y)
    t_span = t[-1] - t[0]

    if len(zt) >= 2:
        # take the top quantile of zero-crossing separations to avoid noise-induced fast crossings
        d = np.diff(zt)
        med_halfT = np.quantile(d, 0.75)
        T = 2 * med_halfT
        omega = 2 * np.pi / T

        # validate if the T is at similar to expectation, if not assume there is just noise
        expected_zt_count = t_span / med_halfT
        if verbose:
            print(f"Expected zt count: {expected_zt_count}, actual zt count: {len(zt)}")
        if len(zt) <= expected_zt_count - 1:
            zt = [np.median(zt)]

    if len(zt) < 2:
        if verbose:
            print("Not enough zero crossings to estimate omega and tau")
        # Fallback guesses for less than half a period
        # Crude period from extremum spacing; guard against degenerate spacing
        T_fallback = 4 * t_span
        if len(zt) == 0:
            T = T_fallback
        else:
            min_max_t = np.array([t[np.argmin(y)], t[np.argmax(y)]])
            T = 4 * np.max(np.abs(min_max_t - zt))
        omega = 2 * np.pi / T

        # Phase sign from initial slope; fallback to first-sample sign if slope is tiny
        if len(y) > 1:
            dy = y[int(len(y) * 0.25)] - y[0]
            phase_sign = -np.sign(dy)
        else:
            phase_sign = np.sign(y[0])

        return omega, None, phase_sign

    # segment-wise envelope samples
    amps, times = [], []
    # use segments bounded by consecutive zero crossings
    for zt_start, zt_end in zip(zt[:-1], zt[1:]):
        msk = (t >= zt_start) & (t <= zt_end)
        if np.any(msk):
            t_seg = t[msk]
            y_seg = y[msk]
            abs_amp = abs(y_seg)
            max_abs_amp = np.argmax(abs_amp)
            amps.append(y_seg[max_abs_amp])
            times.append(t_seg[max_abs_amp])

    if len(amps) == 0:
        # No segments found, return fallback
        tau = t_span / 5.0 if len(t) > 1 else 1.0
        return omega, None, 1.0

    amps = np.asarray(amps)
    times = np.asarray(times)

    # Determine the sign of the phase from the first segment
    phase_sign = np.sign(amps[0]) if amps[0] != 0 else 1.0

    amps = np.abs(amps)

    # if len(amps) < 3:
    #     tau = times[-1] - times[0]
    #     return omega, tau, phase_sign

    # Exponential envelope: A*exp(-t/tau) -> ln(amp) = ln(A) - t/tau
    # Filter out zero or negative amplitudes to avoid log errors
    valid_mask = amps > 1e-15
    if not np.any(valid_mask) or np.sum(valid_mask) < 2:
        tau = times[-1] - times[0]
        return omega, tau, phase_sign

    times_valid = times[valid_mask]
    amps_valid = amps[valid_mask]
    slope, y_intercept = np.polyfit(times_valid, np.log(amps_valid), 1)
    tau = -1.0 / slope if slope < 0 else None
    # plt.figure()
    # plt.plot(times_valid, np.log(amps_valid), "o")
    # plt.plot(times_valid, slope * times_valid + y_intercept, "r-")
    # plt.text(
    #     0.05,
    #     0.95,
    #     f"Estimated tau: {tau:.3f}" if tau is not None else "Estimated tau: None",
    #     transform=plt.gca().transAxes,
    # )
    # plt.xlabel("Time")
    # plt.ylabel("Log Amplitude")
    # plt.title("Exponential Decay Fit")
    # plt.show()

    return omega, tau, phase_sign


def _amplitude_phase_to_positive(A, phi):
    """
    Convert (A, phi) to canonical form with A >= 0, using -A·cos(ωt + φ) = A·cos(ωt + φ + π).
    Phase is wrapped to [-π, π].
    """
    if A >= 0:
        return float(A), float(np.arctan2(np.sin(phi), np.cos(phi)))
    return float(-A), float(np.arctan2(np.sin(phi + np.pi), np.cos(phi + np.pi)))


def cosine(t, A, omega, phi, c):
    """
    Simple cosine function with amplitude, frequency, phase, and offset.

    Computes: y(t) = A·cos(ω·t + φ) + c

    Parameters
    ----------
    t : array_like
        Time values where function is evaluated.
    A : float
        Amplitude (peak deviation from offset).
    omega : float
        Angular frequency in rad/time_unit. Related to frequency f by ω = 2πf.
    phi : float
        Phase shift in radians. Positive values shift waveform left.
    c : float
        Vertical offset (DC component).

    Returns
    -------
    ndarray or float
        Function values at time points t.

    Examples
    --------
    >>> t = np.linspace(0, 1, 100)
    >>> y = cosine(t, A=2.0, omega=2*np.pi*5, phi=0, c=0.5)
    >>> # Creates 5 Hz sine wave with amplitude 2, offset 0.5
    """
    return A * np.cos(omega * t + phi) + c


def fit_cosine(
    t,
    y,
    y_err=None,
    fix_offset=None,
    fix_phase=False,
    return_cov=True,
    return_initial=False,
    verbose: bool = False,
):
    """
    Fit a simple cosine to data: y(t) = A·cos(ω·t + φ) + c.

    Uses the same initialization strategy as fit_exp_decay_cosine (omega from
    zero-crossing analysis, phase from initial values). Use this when there is
    no decay, or when fit_exp_decay_cosine has determined tau to be negligible.

    Parameters
    ----------
    t : array_like
        Time values (1D array), must be same length as y.
    y : array_like
        Measured signal values (1D array), must be same length as t.
    y_err : array_like or None, optional
        Standard deviations of y for weighted fitting. If None, unweighted.
    fix_offset : float or None, optional
        If provided, fixes the DC offset (c) to this value during fitting.
        If None (default), offset is fitted from data (initialized to median).
    fix_phase : bool or float, optional
        Phase handling: False (default) fit phase freely; float fixes phase (radians).
    return_cov : bool, optional
        If True (default), return covariance matrix from curve_fit.
        If False, return only fitted parameters.
    return_initial : bool, optional
        If True, also return dictionary of initial parameter guesses.
        If False (default), return only fitted results.

    Returns
    -------
    fit_params : dict
        Fitted parameters: 'A', 'omega', 'phi', 'offset' (no tau).
        A is always >= 0; if the fit would give A < 0, it is converted via
        (A, phi) -> (|A|, phi + π) so the same curve is represented.
    pcov : ndarray or None
        Covariance matrix from curve_fit. Only returned if return_cov=True.
    initial_params : dict
        Initial parameter guesses, same keys as fit_params.
        Only returned if return_initial=True.

    Raises
    ------
    ValueError
        If t and y have different lengths or fewer than 3 data points.

    See Also
    --------
    fit_exp_decay_cosine : Fit with optional exponential decay (tau).
    cosine : The model function being fitted.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    if len(t) != len(y):
        raise ValueError("t and y must have the same length")
    if len(t) < 3:
        raise ValueError("At least 3 data points required for fitting")

    c0 = (np.max(y) + np.min(y)) / 2 if fix_offset is None else float(fix_offset)
    yc = y - c0

    A0 = np.max(np.abs(yc))
    A0 = max(A0, 1e-12)

    omega0, _, phase_sign = _omega_tau_from_segments(t, yc, verbose=verbose)
    if not np.isfinite(omega0) or omega0 < 0:
        omega0 = 0.0

    fit_phase = isinstance(fix_phase, (float, int)) and not isinstance(fix_phase, bool)
    if fit_phase:
        phi0 = float(fix_phase)
        phi0 = np.arctan2(np.sin(phi0), np.cos(phi0))
    else:
        rss_list = []
        phi_list = [0, np.pi * 0.5, np.pi, np.pi * 1.5]
        for _phi in phi_list:
            rss = np.sum((yc - cosine(t, A0, omega0, _phi, c0)) ** 2)
            rss_list.append(rss)
        phi0 = phi_list[np.argmin(rss_list)]

    A0_c, phi0_c = _amplitude_phase_to_positive(A0, phi0)
    initial_params = dict(A=A0_c, omega=omega0, phi=phi0_c, offset=c0)

    # Build parameter lists for fitting (same order and append pattern as fit_exp_decay_cosine)
    p0 = []
    lbound = []
    ubound = []
    param_map = []

    # Amplitude (always fitted); may be negative, converted to A >= 0 at end via phase shift
    p0.append(A0)
    lbound.append(-np.inf)
    ubound.append(np.inf)
    param_map.append("A")

    # Frequency (always fitted)
    p0.append(omega0)
    lbound.append(0)
    ubound.append(np.inf)
    param_map.append("omega")

    # Phase (fitted unless fixed)
    if not fit_phase:
        p0.append(phi0)
        lbound.append(-2 * np.pi)
        ubound.append(2 * np.pi)
        param_map.append("phi")

    # Offset (fitted unless fixed)
    if fix_offset is None:
        p0.append(c0)
        ymin, ymax = float(np.min(y)), float(np.max(y))
        if ymax - ymin < 1e-12:
            lbound.append(ymin - 1e-6)
            ubound.append(ymax + 1e-6)
        else:
            lbound.append(ymin)
            ubound.append(ymax)
        param_map.append("offset")

    def wrapped(t_arr, *pars):
        params = dict(zip(param_map, pars))
        A = params["A"]
        omega = params["omega"]
        phi = params.get("phi", phi0)
        c = params.get("offset", c0)
        return cosine(t_arr, A, omega, phi, c)

    try:
        y_err = sanitize_y_err(y_err)
        popt, pcov = curve_fit(wrapped, t, y, p0=p0, sigma=y_err, bounds=(lbound, ubound))
    except RuntimeError:
        if verbose:
            print("Fit did not converge; returning initial guesses.")
        A_c, phi_c = _amplitude_phase_to_positive(A0, phi0)
        fit_params = dict(A=A_c, omega=omega0, phi=phi_c, offset=c0)
        if return_initial:
            initial_params = dict(A=A_c, omega=omega0, phi=phi_c, offset=c0)
            return (fit_params, None, initial_params) if return_cov else (fit_params, initial_params)
        return (fit_params, None) if return_cov else fit_params

    fitted_params = dict(zip(param_map, popt))
    A_c, phi_c = _amplitude_phase_to_positive(fitted_params["A"], fitted_params.get("phi", phi0))
    fit_params = dict(
        A=A_c,
        omega=fitted_params["omega"],
        phi=phi_c,
        offset=fitted_params.get("offset", c0),
    )
    if verbose:
        print(
            f"Fitted parameters (cosine): A={fit_params['A']:.3f}, omega={fit_params['omega']:.3f}, "
            f"phi={fit_params['phi']:.3f}, offset={fit_params['offset']:.3f}"
        )
    if return_initial:
        return (fit_params, pcov, initial_params) if return_cov else (fit_params, initial_params)
    return (fit_params, pcov) if return_cov else fit_params


def exp_decay_cosine(t, A, tau, omega, phi, c):
    """
    Exponentially decaying cosine function (damped oscillation).

    Computes: y(t) = A·exp(-t/τ)·cos(ω·t + φ) + c

    If tau is None or ≤0, returns simple cosine without decay envelope.

    Parameters
    ----------
    t : array_like
        Time values where function is evaluated. Should typically start at 0
        or be shifted so that decay begins at the first time point.
    A : float
        Initial amplitude (amplitude at t=0 before decay).
    tau : float or None
        Decay time constant. Amplitude decreases to 1/e (~37%) of initial
        value after time tau. If None or ≤0, no decay is applied.
    omega : float
        Angular frequency in rad/time_unit. Related to frequency f by ω = 2πf.
    phi : float
        Phase shift in radians. Positive values shift waveform left.
    c : float
        Vertical offset (DC component).

    Returns
    -------
    ndarray or float
        Function values at time points t.

    Notes
    -----
    This model describes underdamped harmonic oscillators, such as:
    - LC circuits with resistance
    - Mechanical oscillators with damping (springs, pendulums)
    - Qubit Rabi oscillations with T2* decay
    - Nuclear magnetic resonance (NMR) free induction decay

    The envelope amplitude follows: A_envelope(t) = A·exp(-t/τ)
    The quality factor Q relates to tau: Q ≈ ω·τ/2

    Examples
    --------
    >>> t = np.linspace(0, 10, 1000)
    >>> y = exp_decay_cosine(t, A=1.0, tau=3.0, omega=2*np.pi, phi=0, c=0)
    >>> # Damped 1 Hz oscillation with 3-second decay time
    """
    if tau:
        return np.exp(-t / tau) * cosine(t, A, omega, phi, c)
    else:
        return cosine(t, A, omega, phi, c)


def fit_exp_decay_cosine(t, y, y_err=None, fix_offset=None, fix_phase=False, return_cov=True, return_initial=False):
    """
    Fit exponentially decaying cosine to data using intelligent initialization.

    Fits the model: y(t) = A·exp(-t/τ)·cos(ω·t + φ) + c

    Uses sophisticated initialization strategy:
    1. Estimates ω from zero-crossing analysis
    2. Estimates τ from amplitude envelope decay
    3. Estimates φ from initial signal values and slope
    4. Adaptively includes/excludes τ based on data characteristics
    5. Refits if necessary when decay is negligible

    Parameters
    ----------
    t : array_like
        Time values (1D array), must be same length as y.
    y : array_like
        Measured signal values (1D array), must be same length as t.
    y_err : array_like or None, optional
        Standard deviations of y for weighted fitting. If None, unweighted.
    fix_offset : float or None, optional
        If provided, fixes the DC offset (c) to this value during fitting.
        If None (default), offset is fitted from data (initialized to median).
    fix_phase : bool or float, optional
        Phase handling: False (default) fit phase freely; float fixes phase (radians).
    return_cov : bool, optional
        If True (default), return covariance matrix from curve_fit.
        If False, return only fitted parameters.
    return_initial : bool, optional
        If True, also return dictionary of initial parameter guesses.
        If False (default), return only fitted results.

    Returns
    -------
    fit_params : dict
        Fitted parameters: 'A', 'tau' (or None), 'omega', 'phi', 'offset'.
        A is always >= 0; if the fit would give A < 0, it is converted via
        (A, phi) -> (|A|, phi + π) so the same curve is represented.
    pcov : ndarray or None
        Covariance matrix from curve_fit. Only returned if return_cov=True.
    initial_params : dict
        Initial parameter guesses, same keys as fit_params.
        Only returned if return_initial=True.

    Raises
    ------
    ValueError
        If t and y have different lengths or fewer than 3 data points.

    Notes
    -----
    Intelligent Fitting Features:
    - Automatically excludes tau if initial estimate is poor or too large
    - Refits without tau if fitted value indicates no decay in data range
    - Uses weighted fitting with exponential weights when tau is included
    - Handles pure cosine (tau=None) and damped cosine seamlessly
    - Prints convergence messages and parameter values

    Parameter Initialization:
    - ω: from zero-crossing median half-period
    - τ: from log-linear fit to amplitude envelope (None if poor fit)
    - A: from maximum absolute deviation from offset
    - φ: from initial values using atan2 for correct quadrant
    - c: from median (if not fixed)

    Bounds:
    - A: [0, ∞)
    - τ: [0, ∞) (excluded from fit if None)
    - ω: [0, ∞)
    - φ: [-2π, 2π]
    - c: [min(y), max(y)]

    Examples
    --------
    Basic fitting with all defaults:
    >>> t = np.linspace(0, 10, 200)
    >>> y_true = 0.8 * np.exp(-t/3) * np.cos(2*np.pi*t + 0.5) + 0.1
    >>> y_noisy = y_true + 0.02 * np.random.randn(len(t))
    >>> params, cov = fit_exp_decay_cosine(t, y_noisy)
    >>> print(f"Fitted tau: {params['tau']:.2f} s")

    With fixed offset and returning initial guess:
    >>> params, cov, initial = fit_exp_decay_cosine(
    ...     t, y_noisy, fix_offset=0.1, return_initial=True
    ... )
    >>> print(f"Initial omega: {initial['omega']:.3f}")
    >>> print(f"Fitted omega: {params['omega']:.3f}")

    See Also
    --------
    fit_cosine : Cosine-only fit (no decay).
    exp_decay_cosine : The model function being fitted.
    _omega_tau_from_segments : Parameter initialization method.
    plot_exp_decay_fit : Visualization of fit results.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    if len(t) != len(y):
        raise ValueError("t and y must have the same length")
    if len(t) < 3:
        raise ValueError("At least 3 data points required for fitting")

    c0 = np.median(y) if fix_offset is None else float(fix_offset)
    yc = y - c0

    # Amplitude initial guess
    A0 = np.max(np.abs(yc))
    A0 = max(A0, 1e-12)

    omega0, tau0, phase_sign = _omega_tau_from_segments(t, yc)
    if tau0 and tau0 > 5 * (t[-1] - t[0]):
        # practically no decay within data range
        tau0 = None
    include_tau = tau0 is not None and np.isfinite(tau0) and tau0 > 0
    if not include_tau:
        # Use cosine-only fit when tau is negligible
        result = fit_cosine(
            t,
            y,
            y_err=y_err,
            fix_offset=fix_offset,
            fix_phase=fix_phase,
            return_cov=return_cov,
            return_initial=return_initial,
        )
        if return_initial:
            if return_cov:
                fit_params, pcov, initial_params = result
            else:
                fit_params, initial_params = result
                pcov = None
        else:
            if return_cov:
                fit_params, pcov = result
            else:
                fit_params = result
                pcov = None
        fit_params["tau"] = None
        if return_initial:
            initial_params["tau"] = None
        print(
            f"Fitted parameters (cosine only): A={fit_params['A']:.3f}, tau=None, omega={fit_params['omega']:.3f}, phi={fit_params['phi']:.3f}, offset={fit_params['offset']:.3f}"
        )
        if return_initial:
            return (fit_params, pcov, initial_params) if return_cov else (fit_params, initial_params)
        return (fit_params, pcov) if return_cov else fit_params

    print(f"Initial parameters:  A={A0:.3f}, tau={tau0}, omega={omega0:.3f}, offset={c0:.3f}, phase_sign={phase_sign}")

    # Estimate phase from first value if requested; otherwise calculate from data
    fit_phase = isinstance(fix_phase, (float, int)) and not isinstance(fix_phase, bool)
    if fit_phase:
        phi0 = float(fix_phase)
        # Wrap phase to [-pi, pi]
        phi0 = np.arctan2(np.sin(phi0), np.cos(phi0))
    else:
        # Estimate phase from first few points
        # At t=0: y(0) = A*cos(phi) + c
        # At t=dt: y(dt) ≈ A*cos(omega*dt + phi) + c (ignoring decay for small dt)
        y0 = yc[0] - c0
        y1 = yc[1] - c0 if len(yc) > 1 else y0

        # cos(phi) = y0/A0
        cos_phi = np.clip(y0 / A0, -1, 1)

        # Estimate sin(phi) from the trend of first few points
        # If signal is increasing, sin is positive; if decreasing, sin is negative
        if len(yc) > 1:
            # For small dt: dy/dt ≈ -A*omega*sin(omega*dt + phi)*exp(-dt/tau) ≈ -A*omega*sin(phi)
            # Positive derivative at t=0 means sin(phi) < 0
            dy = y1 - y0
            sin_phi = -np.sign(dy) if abs(dy) > 1e-10 * A0 else 0
        else:
            sin_phi = 0

        # Use atan2 to get phase in correct quadrant
        phi0 = np.arctan2(sin_phi * np.sqrt(1 - cos_phi**2), cos_phi)

        # If the sign is ambiguous, use phase_sign from zero-crossing analysis
        if abs(dy) < 1e-10 * A0:
            if abs(cos_phi) > 0.5:  # Near 0 or pi
                phi0 = 0 if cos_phi > 0 else np.pi
            else:  # Near ±pi/2
                phi0 = np.pi / 2 * -phase_sign

    # Store initial parameters for optional return (canonical form: A >= 0)
    A0_c, phi0_c = _amplitude_phase_to_positive(A0, phi0)
    initial_params = dict(A=A0_c, tau=tau0, omega=omega0, phi=phi0_c, offset=c0)

    # Build parameter lists for fitting (same order and append pattern as fit_cosine)
    p0 = []
    lbound = []
    ubound = []
    param_map = []

    # Amplitude (always fitted); may be negative, converted to A >= 0 at end via phase shift
    p0.append(A0)
    lbound.append(-np.inf)
    ubound.append(np.inf)
    param_map.append("A")

    # Decay time constant (only when include_tau)
    if include_tau:
        p0.append(tau0)
        lbound.append(0)
        ubound.append(np.inf)
        param_map.append("tau")

    # Frequency (always fitted)
    if not np.isfinite(omega0) or omega0 < 0:
        omega0 = 0.0
    p0.append(omega0)
    lbound.append(0)
    ubound.append(np.inf)
    param_map.append("omega")

    # Phase (fitted unless fixed)
    if not fit_phase:
        p0.append(phi0)
        lbound.append(-2 * np.pi)
        ubound.append(2 * np.pi)
        param_map.append("phi")

    # Offset (fitted unless fixed)
    if fix_offset is None:
        p0.append(c0)
        ymin, ymax = float(np.min(y)), float(np.max(y))
        if ymax - ymin < 1e-12:
            lbound.append(ymin - 1e-6)
            ubound.append(ymax + 1e-6)
        else:
            lbound.append(ymin)
            ubound.append(ymax)
        param_map.append("offset")

    def wrapped(t, *pars):
        # Reconstruct full parameter list from fitted parameters
        params = dict(zip(param_map, pars))
        A = params.get("A", A0)
        tau = params.get("tau", tau0)
        omega = params.get("omega", omega0)
        phi = params.get("phi", phi0)
        c = params.get("offset", c0)
        if tau is None or (isinstance(tau, (float, int)) and tau <= 0) or (not include_tau):
            # No decay: pure cosine
            return cosine(t, A, omega, phi, c)
        # With decay
        t_shift = t - t[0]
        return exp_decay_cosine(t_shift, A, tau, omega, phi, c)

    # Normalize with respective tau to reduce effect on the fit
    if include_tau:
        if y_err is None:
            y_err = np.ones_like(y)
        sigma_weights = np.exp(t / tau0)
        # Normalize to avoid numerical issues
        sigma_weights = sigma_weights / np.min(sigma_weights)
        y_err = y_err * sigma_weights
    else:
        sigma_weights = None

    refit = True
    while refit:
        try:
            y_err = sanitize_y_err(y_err)
            popt, pcov = curve_fit(wrapped, t, y, p0=p0, sigma=y_err, bounds=(lbound, ubound))
        except RuntimeError:
            print("Fit did not converge; returning initial guesses.")
            fit_params = dict(A=A0_c, tau=tau0, omega=omega0, phi=phi0_c, offset=c0)
            if return_initial:
                return (fit_params, None, initial_params) if return_cov else (fit_params, initial_params)
            return (fit_params, None) if return_cov else fit_params

        fitted_params = dict(zip(param_map, popt))
        tau = fitted_params.get("tau")
        if tau is not None and tau > 5 * (t[-1] - t[0]):
            # Tau negligible: use cosine-only fit
            print(" Refitting without decay (tau) parameter due to large fitted tau.")
            result = fit_cosine(
                t,
                y,
                y_err=y_err,
                fix_offset=fix_offset,
                fix_phase=fix_phase,
                return_cov=return_cov,
                return_initial=return_initial,
            )
            if return_initial:
                if return_cov:
                    fit_params, pcov, initial_params = result
                else:
                    fit_params, initial_params = result
                    pcov = None
            else:
                if return_cov:
                    fit_params, pcov = result
                else:
                    fit_params = result
                    pcov = None
            fit_params["tau"] = None
            if return_initial:
                initial_params["tau"] = None
            print(
                f"Fitted parameters (cosine only): A={fit_params['A']:.3f}, tau=None, omega={fit_params['omega']:.3f}, phi={fit_params['phi']:.3f}, offset={fit_params['offset']:.3f}"
            )
            if return_initial:
                return (fit_params, pcov, initial_params) if return_cov else (fit_params, initial_params)
            return (fit_params, pcov) if return_cov else fit_params
        refit = False

    # Reconstruct full parameter set from fitted values; convert to A >= 0 via phase shift
    A_raw = fitted_params.get("A", A0)
    phi_raw = fitted_params.get("phi", phi0)
    A, phi = _amplitude_phase_to_positive(A_raw, phi_raw)
    tau = fitted_params.get("tau", None)
    omega = fitted_params.get("omega", omega0)
    c = fitted_params.get("offset", c0)

    fit_params = dict(A=A, tau=tau, omega=omega, phi=phi, offset=c)
    print(f"Fitted parameters: A={A:.3f}, tau={tau}, omega={omega:.3f}, phi={phi:.3f}, offset={c:.3f}")

    if return_initial:
        return (fit_params, pcov, initial_params) if return_cov else (fit_params, initial_params)
    return (fit_params, pcov) if return_cov else fit_params


def plot_exp_decay_fit(t, y, fit_params, initial_params=None, show_envelope=True, show_zero_crossings=True, ax=None):
    """
    Visualize exponentially decaying cosine fit with data, envelope, and crossings.

    Creates publication-quality plot showing measured data, fitted curve,
    optional initial guess, exponential envelope bounds, and zero crossings.
    Uses consistent color scheme for easy interpretation.

    Parameters
    ----------
    t : array_like
        Time values (1D), same as used for fitting.
    y : array_like
        Measured signal values (1D), same as used for fitting.
    fit_params : dict
        Fitted parameters from fit_exp_decay_cosine(). Must contain keys:
        'A', 'tau', 'omega', 'phi', 'offset'.
    initial_params : dict, optional
        Initial parameter guesses to overlay on plot. If provided, shows
        dashed green line of initial model for comparison. Should have same
        keys as fit_params. Default is None (don't show initial guess).
    show_envelope : bool, optional
        If True (default), plot exponential envelope boundaries as red dashed
        lines: ±A·exp(-t/τ) + c. Only shown when tau is finite and positive.
        If False, envelope is not plotted.
    show_zero_crossings : bool, optional
        If True (default), mark detected zero crossings with vertical gray
        dotted lines. Useful for verifying frequency estimation.
        If False, crossings are not marked.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None (default), creates new figure and axes.
        Use this to embed plot in larger figure layouts.

    Returns
    -------
    None
        Modifies the axes object in place. Use plt.show() to display if needed.

    Notes
    -----
    Color Scheme (standardized):
    - **Black dots**: Measured data points (markersize=10)
    - **Red solid line**: Fitted curve (linewidth=3, alpha=0.7)
    - **Green dashed line**: Initial guess (linewidth=2, alpha=0.5)
    - **Red dashed lines**: Exponential envelope bounds (linewidth=1)
    - **Gray dotted lines**: Zero crossings (linewidth=0.8)

    Plot Details:
    - Y-axis limited to [-1.05, 1.05] for consistent scaling
    - Legend automatically includes all plotted elements
    - X-label: "t", Y-label: "y"
    - Fitted curve uses 100 interpolated points for smooth display
    - Time shift applied so decay envelope starts at t[0]

    Envelope Behavior:
    - If tau is None or ≤0: no envelope shown (pure cosine)
    - Upper envelope: A·exp(-t/τ) + c
    - Lower envelope: -A·exp(-t/τ) + c

    Examples
    --------
    Basic usage after fitting:
    >>> params, cov = fit_exp_decay_cosine(t, y)
    >>> plot_exp_decay_fit(t, y, params)
    >>> plt.title("Qubit Rabi Oscillation with T2* Decay")
    >>> plt.show()

    Compare initial guess and fit:
    >>> params, cov, initial = fit_exp_decay_cosine(t, y, return_initial=True)
    >>> plot_exp_decay_fit(t, y, params, initial_params=initial)
    >>> plt.show()

    Embed in subplot:
    >>> fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    >>> plot_exp_decay_fit(t1, y1, params1, ax=axes[0, 0])
    >>> axes[0, 0].set_title("Qubit 1")
    >>> # ... plot other qubits in remaining axes

    Minimal plot (no envelope or crossings):
    >>> plot_exp_decay_fit(t, y, params, show_envelope=False,
    ...                    show_zero_crossings=False)

    See Also
    --------
    fit_exp_decay_cosine : Function that generates fit_params
    exp_decay_cosine : The model function being plotted
    """
    A = fit_params["A"]
    tau = getattr(fit_params, "tau", None)
    omega = fit_params["omega"]
    phi = fit_params["phi"]
    c = fit_params["offset"]

    t_fit = np.linspace(t[0], t[-1], 100)
    t_shift = t_fit - t[0]  # Shift time so decay starts at t[0]

    if tau is None or tau <= 0:
        # No decay, use simple cosine
        y_fit = A * np.cos(omega * t_fit + phi) + c
    else:
        y_fit = A * np.exp(-t_shift / tau) * np.cos(omega * t_fit + phi) + c

    if ax is None:
        plt.figure(figsize=(7, 4))
        ax = plt.gca()

    # Data: black
    ax.plot(t, y, "k.", markersize=10, label="Data")

    if initial_params is not None:
        A0 = initial_params["A"]
        tau0 = getattr(initial_params, "tau", None)
        omega0 = initial_params["omega"]
        phi0 = initial_params["phi"]
        c0 = initial_params["offset"]

        if tau0 is None or tau0 <= 0:
            y_init = A0 * np.cos(omega0 * t_fit + phi0) + c0
        else:
            y_init = A0 * np.exp(-t_shift / tau0) * np.cos(omega0 * t_fit + phi0) + c0

        ax.plot(t_fit, y_init, "g--", lw=2, label="Initial guess", alpha=0.5)

    ax.plot(t_fit, y_fit, "r-", lw=3, label="Fit", alpha=0.7)
    ax.axhline(c, color="k", ls="--", lw=1)

    if show_envelope and tau is not None and tau > 0:
        env = np.abs(A) * np.exp(-t_shift / tau) + c
        ax.plot(t_fit, env, "r--", lw=1, label="Envelope")
        ax.plot(t_fit, -env + 2 * c, "r--", lw=1)

        # oscillation without decay
        y_fit_no_decay = A * np.cos(omega * t_fit + phi) + c
        ax.plot(t_fit, y_fit_no_decay, "b-", lw=3, label="Oscillation", alpha=0.7)

    if show_zero_crossings:
        zt = _zero_crossings(t, y)
        for _z in zt:
            ax.axvline(_z, color="gray", ls=":", lw=0.8)

    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_ylim([-1.05, 1.05])
    ax.legend()
