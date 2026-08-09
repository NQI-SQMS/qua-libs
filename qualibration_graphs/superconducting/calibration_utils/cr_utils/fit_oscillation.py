from __future__ import annotations

from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from calibration_utils.common_utils.fitting_tools import fit_cosine, plot_exp_decay_fit


def recover_delta_omegas(
    params_x: dict[str, Any],
    params_y: dict[str, Any],
    params_z: dict[str, Any],
    t: np.ndarray,
    Xd: np.ndarray,
    Yd: np.ndarray,
    Zd: np.ndarray,
) -> Tuple[float, float, float, float]:
    Ax, Ay, Az = params_x["A"], params_y["A"], params_z["A"]
    std_x, std_y, std_z = np.std(Xd), np.std(Yd), np.std(Zd)
    omega = max(
        (params_x, std_x),
        (params_y, std_y),
        (params_z, std_z),
        key=lambda item: item[1],
    )[0]["omega"]
    try:
        taux, tauy, tauz = params_x["tau"], params_y["tau"], params_z["tau"]
        if any(tau is None for tau in [taux, tauy, tauz]):
            taux, tauy, tauz = None, None, None
    except KeyError:
        taux, tauy, tauz = None, None, None

    Ox = omega * np.sqrt(Ay / (Ax + Ay))
    Oy = omega * np.sqrt(Ax / (Ax + Ay))

    best_rss = np.inf
    best_params = None

    # Try all sign combinations
    for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        params = (0, sx * Ox, sy * Oy, taux, tauy, tauz)
        rss = residuals(params, t, Xd, Yd, Zd)
        rss = np.sum(rss**2)
        if rss < best_rss:
            best_rss = rss
            best_params = params

    return best_params[0], best_params[1], best_params[2], best_rss


###################################################
# Direct fit related
###################################################
def Bloch(
    t: np.ndarray,
    delta: float,
    Ox: float,
    Oy: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    omega_sq = delta**2 + Ox**2 + Oy**2
    omega = np.sqrt(omega_sq)
    inv_omega_sq = 1 / omega_sq
    cos_omega_t = np.cos(omega * t)
    sin_omega_t = np.sin(omega * t)

    X = inv_omega_sq * (delta * Ox * (1 - cos_omega_t) + omega * Oy * sin_omega_t)
    Y = inv_omega_sq * (delta * Oy * (1 - cos_omega_t) - omega * Ox * sin_omega_t)
    Z = inv_omega_sq * (delta**2 * (1 - cos_omega_t) + omega_sq * cos_omega_t)

    return X, Y, Z


def exp_decay_Bloch(
    t: np.ndarray,
    delta: float,
    Ox: float,
    Oy: float,
    taux: float = np.inf,
    tauy: float = np.inf,
    tauz: float = np.inf,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, Y, Z = Bloch(t, delta, Ox, Oy)

    if taux:
        X *= np.exp(-t / taux)
    if tauy:
        Y *= np.exp(-t / tauy)
    if tauz:
        Z *= np.exp(-t / tauz)

    return X, Y, Z


def model(
    params: Tuple[float, ...],
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(params) == 6:
        delta, Ox, Oy, taux, tauy, tauz = params
        return exp_decay_Bloch(t, delta, Ox, Oy, taux, tauy, tauz)
    elif len(params) == 3:
        delta, Ox, Oy = params
        return exp_decay_Bloch(t, delta, Ox, Oy)
    else:
        raise NotImplementedError("Parameters should either be 3 or 6.")


def residuals(
    params: Tuple[float, ...],
    t: np.ndarray,
    Xd: np.ndarray,
    Yd: np.ndarray,
    Zd: np.ndarray,
) -> np.ndarray:
    X, Y, Z = model(params, t)
    reg = []
    if len(params) == 6:
        pass
    return np.concatenate([(Xd - X), (Yd - Y), (Zd - Z), reg])


def fit_exp_decay_Bloch(
    t: np.ndarray,
    Xd: np.ndarray,
    Yd: np.ndarray,
    Zd: np.ndarray,
    plot: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    if plot:
        cosine_params_x, _, initial_params_x = fit_cosine(t, Xd, fix_offset=0, return_initial=True, verbose=verbose)
        cosine_params_y, _, initial_params_y = fit_cosine(t, Yd, fix_offset=0, return_initial=True, verbose=verbose)
        cosine_params_z, _, initial_params_z = fit_cosine(
            t, Zd, fix_offset=0, fix_phase=0, return_initial=True, verbose=verbose
        )
        initial_params = {
            "X": initial_params_x,
            "Y": initial_params_y,
            "Z": initial_params_z,
        }

        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        for i, (k, data, params) in enumerate(
            zip(["X", "Y", "Z"], [Xd, Yd, Zd], [cosine_params_x, cosine_params_y, cosine_params_z])
        ):
            plot_exp_decay_fit(
                t, data, params, initial_params[k], show_envelope=True, show_zero_crossings=True, ax=axs[i]
            )
            axs[i].set_ylabel(f"<{k}>")
        plt.show()
    else:
        cosine_params_x, _ = fit_cosine(t, Xd, fix_offset=0, verbose=verbose)
        cosine_params_y, _ = fit_cosine(t, Yd, fix_offset=0, verbose=verbose)
        cosine_params_z, _ = fit_cosine(t, Zd, fix_offset=0, fix_phase=0, verbose=verbose)

    # for k, params in zip(["X", "Y", "Z"], [cosine_params_x, cosine_params_y, cosine_params_z]):
    #     print(f"Fitted exponential cosine parameters for <{k}>:")
    #     for p_k, p_v in params.items():
    #         print(f"  {p_k}: {p_v}")
    #     print("")

    delta_guess, Ox_guess, Oy_guess, _diag = recover_delta_omegas(
        cosine_params_x, cosine_params_y, cosine_params_z, t, Xd, Yd, Zd
    )

    # # plot with param_guess
    # X_fit, Y_fit, Z_fit = exp_decay_Bloch(np.linspace(t[0], t[-1], 100), delta_guess, Ox_guess, Oy_guess)
    # fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    # axs[0].plot(t, Xd, "o", color="r", ms=3, label="X data")
    # axs[0].plot(np.linspace(t[0], t[-1], 100), X_fit, "r-", label="X fit")
    # axs[1].plot(t, Yd, "o", color="g", ms=3, label="Y data")
    # axs[1].plot(np.linspace(t[0], t[-1], 100), Y_fit, "g-", label="Y fit")
    # axs[2].plot(t, Zd, "o", color="b", ms=3, label="Z data")
    # axs[2].plot(np.linspace(t[0], t[-1], 100), Z_fit, "b-", label="Z fit")
    # plt.show()

    params_guess = (delta_guess, Ox_guess, Oy_guess)
    lbound = [-np.inf, -np.inf, -np.inf]
    ubound = [np.inf, np.inf, np.inf]
    params_guess = tuple(np.clip(params_guess, lbound, ubound))
    result = least_squares(
        residuals,
        x0=params_guess,
        bounds=(lbound, ubound),
        args=(t, Xd, Yd, Zd),
    )
    # print("Least squares fitting result: ", result, "\n")

    if result.success:
        return result.x
    else:
        raise ValueError(result.message)


def plot_Bloch_fit(
    t: np.ndarray,
    Xd: np.ndarray,
    Yd: np.ndarray,
    Zd: np.ndarray,
    fit_params: Tuple[float, ...],
    axs: Optional[np.ndarray] = None,
    color: str = "r",
    line_style: str = "-",
    alpha: float = 0.7,
    legend_label: str = "",
    ignore_decay: bool = False,
) -> None:
    if len(fit_params) == 3:
        delta, Ox, Oy = fit_params
        taux, tauy, tauz = None, None, None
    elif len(fit_params) == 6:
        delta, Ox, Oy, taux, tauy, tauz = fit_params
    t_fit = np.linspace(t[0], t[-1], 100)
    if ignore_decay:
        X_fit, Y_fit, Z_fit = exp_decay_Bloch(t_fit, delta, Ox, Oy)
    else:
        X_fit, Y_fit, Z_fit = exp_decay_Bloch(t_fit, delta, Ox, Oy, taux, tauy, tauz)

    fs = 16
    if axs is None:
        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    if legend_label:
        legend_label = legend_label.strip() + " "
    for i, (label, data, fit) in enumerate((("X", Xd, X_fit), ("Y", Yd, Y_fit), ("Z", Zd, Z_fit))):
        axs[i].plot(t, data, "o", color=color, ms=3, label=f"{legend_label} data")
        axs[i].plot(
            t_fit, fit, line_style, color=color, linewidth=3, alpha=alpha, label=f"{legend_label} fit"
        )  # line for fit
        axs[i].set_ylabel(f"<{label}>", fontsize=fs)
        axs[i].set_ylim([-1.05, 1.05])
        axs[i].tick_params(axis="both", which="major", labelsize=fs)
    # legend outside plot
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1)
    plt.xlabel("t (ns)", fontsize=fs)
