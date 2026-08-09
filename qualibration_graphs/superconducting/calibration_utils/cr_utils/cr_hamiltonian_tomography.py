from __future__ import annotations

import itertools
from typing import Callable, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

try:
    from fit_oscillation import fit_exp_decay_Bloch, plot_Bloch_fit
except ImportError:
    from .fit_oscillation import fit_exp_decay_Bloch, plot_Bloch_fit

CONTROL_STATES = ["0", "1"]  # control state: 0 or 1
TARGET_BASES = ["x", "y", "z"]  # target basiss x, y, z
PARAM_NAMES = ["delta", "omega_x", "omega_y"]
PAULI_2Q = ["IX", "IY", "IZ", "ZX", "ZY", "ZZ"]
PAULI_2Q_PLOT_STYLES = [
    {"color": "C0", "linestyle": "--", "marker": "."},
    {"color": "C1", "linestyle": "--", "marker": "."},
    {"color": "C2", "linestyle": "--", "marker": "."},
    {"color": "C0", "linestyle": "-", "marker": "x"},
    {"color": "C1", "linestyle": "-", "marker": "x"},
    {"color": "C2", "linestyle": "-", "marker": "x"},
]

# generate a set of points on sphere for initial values for delta, omega_x, omega_y
signss = itertools.product([-1, 1], repeat=3)
# pick a non polar point
p0_center = [np.cos(np.pi / 4), np.sin(np.pi / 4) * np.cos(np.pi / 4), np.sin(np.pi / 4) * np.sin(np.pi / 4)]
p0_centers = [np.array(signs) * p0_center for signs in signss]
# polar points
p0_poles = [np.array(p0) * sign for p0 in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] for sign in [-1, 1]]
# concatenate the two kinds
p0s = p0_centers + p0_poles
# scale the prepared points to expand the initial values
scales = np.arange(0.9, 1.15, 0.05)
# a set of initial values
P0s = [scale * p0 for p0 in p0s for scale in scales]


class CRHamiltonianTomographyFunctions:
    def __init__(self) -> None:
        """Initialize the class."""
        pass

    def _compute_omega_squared(
        self, d: Union[float, np.ndarray], mx: Union[float, np.ndarray], my: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Calculate the omega squared based on the parameters.

        :param d: delta.
        :param mx: omega X.
        :param my: omega y.

        :return: omega squared.
        """
        return d**2 + mx**2 + my**2

    def _compute_X(
        self,
        ts: np.ndarray,
        d: Union[float, np.ndarray],
        mx: Union[float, np.ndarray],
        my: Union[float, np.ndarray],
    ) -> np.ndarray:
        """
        Calculate expectation value of target <X> based on the parameters.

        :param ts: durations of CR drive.
        :param d: delta.
        :param mx: omega X.
        :param my: omega y.

        :return: <X>.
        """
        m2 = self._compute_omega_squared(d, mx, my)
        m = np.sqrt(m2)
        mt = m * ts
        return (d * mx * (1 - np.cos(mt)) + m * my * np.sin(mt)) / m2

    def _compute_Y(
        self,
        ts: np.ndarray,
        d: Union[float, np.ndarray],
        mx: Union[float, np.ndarray],
        my: Union[float, np.ndarray],
    ) -> np.ndarray:
        """
        Calculate expectation value of target <Y> based on the parameters.

        :param ts: durations of CR drive.
        :param d: delta.
        :param mx: omega X.
        :param my: omega y.

        :return: <Y>.
        """
        m2 = self._compute_omega_squared(d, mx, my)
        m = np.sqrt(m2)
        mt = m * ts
        return (d * my * (1 - np.cos(mt)) - m * mx * np.sin(mt)) / m2

    def _compute_Z(
        self,
        ts: np.ndarray,
        d: Union[float, np.ndarray],
        mx: Union[float, np.ndarray],
        my: Union[float, np.ndarray],
    ) -> np.ndarray:
        """
        Calculate expectation value of target <Z> based on the parameters.

        :param ts: durations of CR drive.
        :param d: delta.
        :param mx: omega X.
        :param my: omega y.

        :return: <Z>.
        """
        m2 = self._compute_omega_squared(d, mx, my)
        m = np.sqrt(m2)
        mt = m * ts
        cos_mt = np.cos(mt)
        return ((d**2) * (1 - cos_mt)) / m2 + cos_mt

    def compute_R(
        self,
        xyz0: dict[str, np.ndarray],
        xyz1: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Compute the root mean square of the sum of two sets of <X>, <Y>, and <Z> data.
        R = sqrt((X0 + X1) ** 2 + (Y0 + Y1) ** 2 (Z0 + Z1) ** 2)
        , where 0 (1) stands for control state = 0 (1).

        :param xyz0: Dictionary containing 'x', 'y', 'z' data for control state 0.
        :param xyz1: Dictionary containing 'x', 'y', 'z' data for control state 1.

        :return: Computed R value.
        """
        return np.sqrt((xyz0["x"] + xyz1["x"]) ** 2 + (xyz0["y"] + xyz1["y"]) ** 2 + (xyz0["z"] + xyz1["z"]) ** 2) / 2

    def compute_XYZ(
        self,
        ts: np.ndarray,
        d: Union[float, np.ndarray],
        mx: Union[float, np.ndarray],
        my: Union[float, np.ndarray],
        noise: float = 0,
        random_state: int = 0,
        clip: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Compute the expected values <X>, <Y>, <Z> for a given set of parameters and add noise if specified.

        :param ts: durations of CR drive.
        :param d: delta.
        :param mx: omega X.
        :param my: omega y.
        :param noise: Standard deviation of Gaussian noise to add to the data.
        :param random_state: Seed for the random number generator.
        :param clip: Boolean flag to clip the noisy data between -1 and 1.

        :return: Dictionary with keys 'x', 'y', 'z' containing the computed values.
        """
        xyz = {
            "x": self._compute_X(ts, d, mx, my),
            "y": self._compute_Y(ts, d, mx, my),
            "z": self._compute_Z(ts, d, mx, my),
        }
        if noise > 0:
            np.random.seed(random_state)
            for c in TARGET_BASES:
                xyz[c] += np.random.normal(scale=noise, size=xyz[c].shape)
                if clip:
                    xyz[c] = np.clip(xyz[c], -1.0, 1.0)

        return xyz


class CRHamiltonianTomographyAnalysis(CRHamiltonianTomographyFunctions):
    ts: np.ndarray
    data: np.ndarray
    data_dict: dict[str, dict[str, np.ndarray]]
    params_fitted: dict[str, list[float]]
    params_fitted_dict: dict[str, dict[str, Optional[float]]]
    interaction_coeffs_MHz: dict[str, Optional[float]]

    def __init__(self, ts: np.ndarray, data: np.ndarray) -> None:
        """
        CR Hamiltonian Tomography class.

        :param ts: durations of CR drive.
        :params data (np.ndarray):
            A 3-dimensional numpy array (len(ts) x len(TARGET_BASES) x len(CONTROL_STATES))
            where the first to durations of CR drive, the second to TARGET_BASES.
            the thir dimension corresponds to CONTROL_STATES.
        """
        self.ts = ts
        self.data = data
        self.data_dict = self.rearrange_data_ndarray2dict(data)
        self.params_fitted = {s: [] for s in CONTROL_STATES}
        self.params_fitted_dict = {s: {nm: None for nm in PARAM_NAMES} for s in CONTROL_STATES}
        self.interaction_coeffs_MHz = {p: None for p in PAULI_2Q}

    def rearrange_data_ndarray2dict(self, data: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        """
        Transforms a 3D numpy array into a nested dictionary format
        suitable for CRQuantumStateTomographyResults.

        :params data (np.ndarray): A 3-dimensional numpy array
            where the first dimension corresponds to CONTROL_STATES,
            the second to different measurement instances,
            and the third to TARGET_BASES.

        :returns: dict: A nested dictionary {control_state: {target_basis: data_array}}.

        Raises:
            ValueError: If the dimensions of the input do not match expected sizes
                based on CONTROL_STATES and TARGET_BASES.
        """
        if data.ndim != 3:
            raise ValueError("Input data must be a 3-dimensional array.")

        if data.shape[2] != len(CONTROL_STATES) or data.shape[1] != len(TARGET_BASES):
            raise ValueError("Dimensions of the input array must match the length of CONTROL_STATES and TARGET_BASES.")

        if data.shape[0] != self.ts.shape[0]:
            raise ValueError("Length of each tomographic data must be the same as the length of cr durations")

        return {st: {bss: data[:, j, i] for j, bss in enumerate(TARGET_BASES)} for i, st in enumerate(CONTROL_STATES)}

    def _bloch_vec_evolution(
        self,
        ts: np.ndarray,
        d: float,
        mx: float,
        my: float,
    ) -> np.ndarray:
        """
        Calculate the expected evolution of the Bloch vector basiss over time.

        :param ts: durations of CR drive.
        :param d, mx, my: Hamiltonian parameters.
        :return: Array of expected 'x', 'y', and 'z' basiss concatenated.
        """
        ts_len = len(ts) // len(TARGET_BASES)
        xyz = self.compute_XYZ(ts[:ts_len], *[d, mx, my])
        return np.hstack([xyz[c] for c in TARGET_BASES])

    def _fit_bloch_vec_evolution(
        self,
        xyz: dict[str, np.ndarray],
        p0: list[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit the model to the data using non-linear least squares.

        :param xyz: Measured data for the Bloch vector basiss.
        :param p0: Initial guess for the parameters.
        :return: Fitted parameters and the covariance of the parameters.
        """
        return curve_fit(
            f=self._bloch_vec_evolution,
            xdata=np.tile(self.ts, len(TARGET_BASES)),
            ydata=np.hstack([xyz[c] for c in TARGET_BASES]),
            p0=p0,
            method="trf",
        )

    def _find_dominant_frequency(self, data: np.ndarray) -> float:
        """
        Identify the dominant frequency in the provided data using Fourier transform.

        :param data: Time-series data from which to extract the frequency.
        :return: Dominant frequency value.
        """
        N = len(self.ts)
        dt = self.ts[1] - self.ts[0]
        freq = np.fft.fftfreq(N, dt)
        spectrum = np.abs(np.fft.fft(data - data.mean()))

        # Find peaks in the frequency spectrum (DC removed alraedy)
        peaks, _ = find_peaks(spectrum, prominence=0.1 * N)
        if len(peaks) == 0:
            print("the data should have more than 1 period")
        highest_peak_idx = np.argmax(spectrum[peaks])

        # Identify dominant frequency (peak frequency)
        return freq[peaks[highest_peak_idx]]

    def _pick_params_inits(self, xyz: dict[str, np.ndarray]) -> list[np.ndarray]:
        """
        Choose initial parameter estimates for the fitting process based on frequency analysis.

        :param xyz: Measured Bloch vector basiss.
        :return: Array of initial parameter guesses.
        """
        freq_inits = [self._find_dominant_frequency(data=xyz[c]) for c in TARGET_BASES]

        # pick the initial omega as median of estimated omega from x, y, z
        freq_init = np.median(np.array(freq_inits))

        # omega_init = initial value for sqrt(delta ** 2 + omega_x ** 2 + omega_y ** 2)
        return [2 * np.pi * freq_init * p0 for p0 in P0s]

    def fit_params(
        self,
        do_print: bool = True,
    ) -> CRHamiltonianTomographyAnalysis:
        """
        Fit the Hamiltonian parameters for each state and compute interaction rates.

        :param params_init: Initial parameter estimates (optional).
        :param _print: Boolean flag to control the printing of fitting results.
        :return: Self.
        """
        for st in CONTROL_STATES:
            t = self.ts
            Xd = self.data_dict[st]["x"]
            Yd = self.data_dict[st]["y"]
            Zd = self.data_dict[st]["z"]
            # parameters here are delta, Ox, Oy, taux, tauy, tauz
            params = fit_exp_decay_Bloch(t, Xd, Yd, Zd, plot=do_print, verbose=do_print)
            self.params_fitted[st] = params[:3]
            self.params_fitted_decay = params[3:]
            self.params_fitted_dict[st] = {nm: p for nm, p in zip(PARAM_NAMES, self.params_fitted[st])}

        # compute interaction rates based on the fitted params
        self.compute_interaction_rates()

        # print the fitted parameters and interaction coefficents
        if do_print:
            for st in CONTROL_STATES:
                ps = self.params_fitted[st]
                print(f"state = {st}: delta = {ps[0]:.3f}, omega_x = {ps[1]:.3f}, omega_y = {ps[2]:.3f}")
            for op in PAULI_2Q:
                print(f"{op}: {self.interaction_coeffs_MHz[op]:.3f} MHz")

        return self

    def compute_interaction_rates(self) -> None:
        """
        Compute the interaction coefficients from fitted Hamiltonian parameters.
        """
        # get the fitted params
        d0, mx0, my0 = self.params_fitted["0"]
        d1, mx1, my1 = self.params_fitted["1"]
        # compute the coefficients for each interaction terms
        self.interaction_coeffs_MHz["IX"] = 1e3 * (mx0 + mx1) / 2
        self.interaction_coeffs_MHz["IY"] = 1e3 * (my0 + my1) / 2
        self.interaction_coeffs_MHz["IZ"] = 1e3 * (d0 + d1) / 2
        self.interaction_coeffs_MHz["ZX"] = 1e3 * (mx0 - mx1) / 2
        self.interaction_coeffs_MHz["ZY"] = 1e3 * (my0 - my1) / 2
        self.interaction_coeffs_MHz["ZZ"] = 1e3 * (d0 - d1) / 2
        # In MHz, was in rad/ns before
        for key in self.interaction_coeffs_MHz:
            self.interaction_coeffs_MHz[key] /= 2 * np.pi

    def get_interaction_rates(self) -> dict[str, float]:
        """
        Get the computed interaction rates after fitting.

        :raises RuntimeError: If any interaction coefficient has not been computed yet.
        :return: Dictionary of interaction coefficients.
        """
        if any(value is None for value in self.interaction_coeffs_MHz.values()):
            raise RuntimeError("some of the interaction coefficients have not been computed yet.")
        return self.interaction_coeffs_MHz  # type: ignore[return-value]

    def plot_data(
        self,
        fig: Optional[plt.Figure] = None,
        axs: Optional[np.ndarray] = None,
        label: str = "",
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot the original measurement data along with the fitted data and interaction rates.

        :return: The matplotlib figure object containing the plots.
        """
        if fig is None:
            fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

        # x, y, z
        axs[0].set_title(label)
        for ax, bss in zip(axs, TARGET_BASES):
            ax.cla()
            v0 = self.data_dict["0"][bss]
            v1 = self.data_dict["1"][bss]
            ax.scatter(self.ts, v0, s=20, color="b", label="ctrl in |0>")
            ax.scatter(self.ts, v1, s=20, color="r", label="ctrl in |1>")
            ax.set_ylabel(f"<{bss}(t)>", fontsize=16)

        # plot "R"
        if len(axs) == 4:
            ax = axs[3]
            ax.cla()
            R = self.compute_R(self.data_dict["0"], self.data_dict["1"])
            ax.scatter(self.ts, R, color="k")
            ax.set_xlabel("Time (ns)", fontsize=16)
            ax.set_ylabel("R", fontsize=16)

        if show:
            plt.tight_layout()
            plt.show(block=False)

        return fig  # type: ignore[return-value]

    def plot_fit_result(
        self,
        fig: Optional[plt.Figure],
        axs: np.ndarray,
    ) -> plt.Figure:
        """
        Plot the original measurement data along with the fitted data and interaction rates.

        :return: The matplotlib figure object containing the plots.
        """

        if self.params_fitted["0"] is None or self.params_fitted["1"] is None:
            raise RuntimeError("fitting has not been done yet")

        if any(value is None for value in self.interaction_coeffs_MHz.values()):
            raise RuntimeError("some of the interaction coefficients have not been computed yet.")
        if fig is None:
            fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

        xyz0_fit = {}
        xyz1_fit = {}
        ts_fit = np.linspace(self.ts[0], self.ts[-1], 500)
        for ax, bss in zip(axs, TARGET_BASES):
            # v0 = self.data_dict["0"][bss]
            # v1 = self.data_dict["1"][bss]
            if bss == "x":
                eV0 = self._compute_X(ts_fit, *self.params_fitted["0"])
                eV1 = self._compute_X(ts_fit, *self.params_fitted["1"])
            elif bss == "y":
                eV0 = self._compute_Y(ts_fit, *self.params_fitted["0"])
                eV1 = self._compute_Y(ts_fit, *self.params_fitted["1"])
            elif bss == "z":
                eV0 = self._compute_Z(ts_fit, *self.params_fitted["0"])
                eV1 = self._compute_Z(ts_fit, *self.params_fitted["1"])
            xyz0_fit[bss] = eV0
            xyz1_fit[bss] = eV1

            # ax.scatter(self.ts, v0, s=20, color="b", label="ctrl in |0>")
            # ax.scatter(self.ts, v1, s=20, color="r", label="ctrl in |1>")
            ax.plot(ts_fit, eV0, lw=4.0, color="b", alpha=0.5)
            ax.plot(ts_fit, eV1, lw=4.0, color="r", alpha=0.5)
            ax.set_ylabel(f"<{bss}(t)>", fontsize=16)

            if bss == "x":
                ax.set_title("Pauli Expectation Value", fontsize=16)
                ax.legend(["0 meas", "1 meas", "0 fit", "1 fit"], fontsize=10, ncol=2)
            elif bss == "y":
                ax.set_title(
                    "IX = %.2f MHz, IY = %.2f MHz, IZ = %.2f MHz"
                    % (
                        self.interaction_coeffs_MHz["IX"],
                        self.interaction_coeffs_MHz["IY"],
                        self.interaction_coeffs_MHz["IZ"],
                    ),
                    fontsize=16,
                )
            elif bss == "z":
                ax.set_title(
                    "ZX = %.2f MHz, ZY = %.2f MHz, ZZ = %.2f MHz"
                    % (
                        self.interaction_coeffs_MHz["ZX"],
                        self.interaction_coeffs_MHz["ZY"],
                        self.interaction_coeffs_MHz["ZZ"],
                    ),
                    fontsize=16,
                )

        for ax in axs[:-1]:
            ax.hlines(y=0, xmin=self.ts[0], xmax=self.ts[-1], lw=1.0, color="k", alpha=0.2)
            ax.set_xlim((self.ts[0], self.ts[-1]))
            ax.set_ylim((-1.05, 1.05))

        # plot "R"
        ax = axs[3]
        R = self.compute_R(xyz0_fit, xyz1_fit)
        ax.plot(ts_fit, R, "k")
        ax.set_xlabel("Time (ns)", fontsize=16)
        ax.set_ylabel("R", fontsize=16)
        ax.hlines(
            y=[0, 1],
            xmin=self.ts[0],
            xmax=self.ts[-1],
            lw=1.0,
            color="k",
            alpha=0.2,
        )
        ax.set_ylim((-0.05, 1.05))

        # T_list = []
        # title = ""
        # for i in ["0", "1"]:
        #     d = self.params_fitted_dict[i]["delta"]
        #     mx = self.params_fitted_dict[i]["omega_x"]
        #     my = self.params_fitted_dict[i]["omega_y"]
        #     omega = np.sqrt(self._compute_omega_squared(d, mx, my))
        #     T_ns = 2 * np.pi / omega
        #     T_list.append(T_ns)
        #     title += f"$T_{i}$={T_ns:.1f} ns, "
        # title += f"$T_{{avg}}$={np.mean(T_list):.1f} ns"
        T_ZX = 1e3 / abs(self.interaction_coeffs_MHz["ZX"])
        title = f"$T_{{ZX90}}$={T_ZX * 0.25:.1f} ns"
        ax.set_title(title, fontsize=16)

        return fig  # type: ignore[return-value]


def plot_interaction_coeffs(
    coeffs: list[dict[str, Optional[float]]],
    xaxis: np.ndarray,
    xlabel: str = "amplitude",
    actual_xaxis: Optional[Callable[[Tuple[float, float]], Tuple[float, float]]] = None,
    fig: Optional[plt.Figure] = None,
) -> plt.Figure:
    """
    Plot the xaxis (amplitudes or phase) vs interaction rates.

    :param coeffs: List of interaction coefficient dicts (keys PAULI_2Q).
    :param xaxis: Sweep values for the primary x-axis.
    :param xlabel: Base label for the primary x-axis; " sweep" is appended.
    :param actual_xaxis: Optional callable (xlim) -> new_xlim to compute the second x-axis limits.
    :param fig: Optional matplotlib figure to draw into.
    :return: The matplotlib figure object containing the plots.
    """
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    else:
        ax = fig.axes[0]

    coeffs_array_dict = {p: np.array([coeff[p] for coeff in coeffs]) for p in PAULI_2Q}
    for p, styles in zip(PAULI_2Q, PAULI_2Q_PLOT_STYLES):
        filt = ~np.isnan(xaxis)
        ax.plot(xaxis[filt], coeffs_array_dict[p][filt], **styles)
    ax.axhline(y=0, color="k", alpha=0.3, linestyle="-")
    ax.set_xlabel(f"{xlabel} sweep")
    ax.set_ylabel("interaction coefficients [MHz]")
    ax.legend(PAULI_2Q)

    if actual_xaxis is not None:
        ax2 = ax.twiny()
        xlim = ax.get_xlim()
        new_xlim = actual_xaxis(xlim)
        ax2.set_xlim(new_xlim)
        ax2.set_xlabel(f"actual {xlabel}")

    return fig


def plot_cr_duration_vs_scan_param(
    data_c: np.ndarray,
    data_t: np.ndarray,
    ts_ns: np.ndarray,
    scan_param: np.ndarray,
    scan_param_name: str,
    axss: np.ndarray,
) -> None:
    data = 2 * [data_c] + 2 * [data_t]
    for i, (axs, bss) in enumerate(zip(axss, TARGET_BASES)):
        for j, (ax, dt, st) in enumerate(zip(axs, data, 2 * CONTROL_STATES)):
            ax.cla()
            Z = dt[:, :, i, j % 2]

            def pcolormesh_wrapper(Z):
                ax.pcolormesh(
                    ts_ns,
                    scan_param,
                    Z,
                    cmap="PiYG",
                    vmin=-1,
                    vmax=1,
                    shading="auto",
                )

            try:
                pcolormesh_wrapper(Z)
            except TypeError:
                # https://www.reddit.com/r/learnpython/comments/1ndeetm/how_do_i_change_dimensions_for_pcolormesh_in/
                pcolormesh_wrapper(Z.T)
            if i == 0 and j < 2:
                ax.set_title(f"Q_C={st}")
            if i == 0 and j >= 2:
                ax.set_title(f"Q_T with Q_C={st}")
            if j == 0:
                ax.set_ylabel(f"<{bss}>\n{scan_param_name}", fontsize=14)
            if i == 2:
                ax.set_xlabel(f"Total CR drive time (ns)", fontsize=14)
    plt.tight_layout()


def plot_crqst_result_2D(
    ts_ns: np.ndarray,
    data_c: np.ndarray,
    data_t: np.ndarray,
    fig: plt.Figure,
    axss: np.ndarray,
) -> plt.Figure:
    # control qubit
    fig = CRHamiltonianTomographyAnalysis(
        ts=ts_ns,
        data=data_c,
    ).plot_data(fig, axss[:, 0], label="control")
    # target qubit
    fig = CRHamiltonianTomographyAnalysis(
        ts=ts_ns,
        data=data_t,
    ).plot_data(fig, axss[:, 1], label="target")
    return fig


def plot_crqst_result_3D(
    ts_ns: np.ndarray,
    data_t: np.ndarray,
    title: str,
) -> plt.Figure:
    # Axes properties
    conf = {
        "projection": "3d",
        "proj_type": "persp",  # ortho or persp
        "box_aspect": (1, 1, 1),
        "elev": 20,
        "azim": 30,
        "roll": 0,
        "axisbelow": True,
        "facecolor": "w",
        "xticks": [],
        "yticks": [],
        "zticks": [],
    }

    # Create the figure and 3D subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title)

    # Create two 3D subplots
    ax1 = fig.add_subplot(121, projection=conf["projection"])
    ax2 = fig.add_subplot(122, projection=conf["projection"])
    ax1.set_title("Qc = 0")
    ax2.set_title("Qc = 1")

    # Generate data for the wireframe of the sphere
    u = np.linspace(0, 2 * np.pi, 100)  # Longitude values
    v = np.linspace(0, np.pi, 50)  # Latitude values
    x_sphere = np.outer(np.cos(u), np.sin(v))  # X coordinates of the sphere
    y_sphere = np.outer(np.sin(u), np.sin(v))  # Y coordinates of the sphere
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))  # Z coordinates of the sphere

    # Define the colors in the colormap
    colors = ["blue", "purple", "red"]
    # Create a colormap with a smooth transition between the specified colors
    cmap = mcolors.LinearSegmentedColormap.from_list("BlueRedPurple", colors)
    colors = cmap(np.linspace(0, 1, len(ts_ns)))

    for i, ax in enumerate([ax1, ax2]):
        # Plot the wireframe of the sphere
        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color="k", linewidth=1, alpha=0.05)

        # Get the data
        x, y, z = data_t[:, 0, i], data_t[:, 1, i], data_t[:, 2, i]

        # Plot the data
        ax.scatter(x, y, z, marker="o", color="k")

        # Quiver plot - arrows from the origin to each point (x, y, z)
        ax.quiver(
            np.zeros_like(x),
            np.zeros_like(y),
            np.zeros_like(z),  # Origins (all zeros)
            x,
            y,
            z,  # Directions (end points)
            color=colors,  # Color mapping based on t
            arrow_length_ratio=0.1,
            length=1.0,
            normalize=False,
        )

        # Label the axes
        ax.text(1.5, 0, 0.1, "X", color="k", fontsize=12)
        ax.text(0, 1.5, 0.1, "Y", color="k", fontsize=12)
        ax.text(0, 0.1, 1.5, "Z", color="k", fontsize=12)

        # x, y, z axes
        ax.quiver(0, 0, 0, 1.6, 0, 0, color="k", alpha=0.1, arrow_length_ratio=0.05)
        ax.quiver(0, 0, 0, 0, 1.6, 0, color="k", alpha=0.1, arrow_length_ratio=0.05)
        ax.quiver(0, 0, 0, 0, 0, 1.6, color="k", alpha=0.1, arrow_length_ratio=0.08)

        # Remove the x, y, z panes (spines)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Remove the x, y, z panes
        ax.xaxis.pane.set_edgecolor([1, 1, 1])
        ax.yaxis.pane.set_edgecolor([1, 1, 1])
        ax.zaxis.pane.set_edgecolor([1, 1, 1])

        # Remove the borders of the panes
        ax.xaxis.line.set_linewidth(0)
        ax.yaxis.line.set_linewidth(0)
        ax.zaxis.line.set_linewidth(0)

    return fig
