# %%
#!%load_ext autoreload
#!%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
import time
import xarray as xr

from cr_hamiltonian_tomography import *
from fit_oscillation import *

######################################
# Choose data file
######################################
# path = "C:\\Users\\SoonTeh\\Projects\\Repo\\CS_installations_all\\HI_23Oct2025\\data\\test\\2025-12-10\\#169_31b_CR_hamiltonian_tomography_vs_cr_drive_phase_115840\\ds_fit.h5"
path = "C:\\Users\\SoonTeh\\Projects\\Repo\\CS_installations_all\\HI_23Oct2025\\data\\test\\2025-12-10\\#217_31b_CR_hamiltonian_tomography_vs_cr_drive_phase_182347\\ds_fit.h5"
all_data = xr.load_dataset(path)
all_data

# %%
data = all_data.isel(qubit_pair=0, phase=0).sel(control_target="t")

######################################
# Choose the target qubit
######################################
ts = data.pulse_duration.data
bloch_data = data["bloch"].data

########################################
# Truncate data to max_t (only for new implementation)
#######################################
max_t = 40000
n = np.sum(ts <= max_t)
print(f"Using {n / ts.size:.2%} of data for fitting.")
t = ts[:n]

try:
    start = time.time()
    ###
    crht = CRHamiltonianTomographyAnalysis(ts=ts, data=bloch_data)
    crht.fit_params()
    ###
    end = time.time()
    print(f"Fitting time (s): {end - start:.3f} s")

    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    crht.plot_fit_result(fig, axs)
except:
    print("CR Hamiltonian Tomography fitting failed.")

fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
params_fit_dict_list = []
for cstate, color in zip([0, 1], ["b", "r"]):
    X = bloch_data[:, 0, cstate][:n]
    Y = bloch_data[:, 1, cstate][:n]
    Z = bloch_data[:, 2, cstate][:n]

    start = time.time()
    # the fitted parameters are (delta, Omega_x, Omega_y, taux, tauy, tauz)
    params_fit = fit_exp_decay_Bloch(t, X, Y, Z, refine_guess=False, plot=False)
    end = time.time()
    print(f"Fitting time (s): {end - start:.3f} s")
    params_fit_dict = {"delta": params_fit[0], "Omega_x": params_fit[1], "Omega_y": params_fit[2]}
    params_fit_dict_list.append(params_fit)

    plot_Bloch_fit(t, X, Y, Z, params_fit, axs, color=color, legend_label=f"Control State={cstate}")
    plot_Bloch_fit(
        t,
        X,
        Y,
        Z,
        params_fit,
        axs,
        color=color,
        line_style="--",
        alpha=0.3,
        legend_label=f"Control State={cstate} (decay)",
        ignore_decay=True,
    )
plt.show()


# %%
from fit_oscillation import fit_exp_decay_cosine, _omega_tau_from_segments

plt.plot(t, X, "k.", ms=5, label="Data")
plt.ylim(-1.1, 1.1)
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
print(_omega_tau_from_segments(t, X))
print(fit_exp_decay_cosine(t, X))

# %%
# Print fitted parameters for comparison
fs = 16

labels = ["delta", "Omega_x", "Omega_y"]
grouped = np.stack([crht.params_fitted["0"], params_fit_dict_list[0][:3]])  # (2, 3)

x = np.arange(len(labels))  # [0, 1, 2]
width = 0.35  # bar width

fig, axs = plt.subplots(1, 2, figsize=(8, 8), sharey=True)

# left panel: control = 0
axs[0].bar(x - width / 2, grouped[0], width, alpha=0.7, label="fitted")
axs[0].bar(x + width / 2, grouped[1], width, alpha=0.7, label="target")

axs[0].set_xticks(x)
axs[0].set_xticklabels(labels, fontsize=fs)
axs[0].set_title("Control State=0", fontsize=fs)
axs[0].axhline(0, color="k", linewidth=0.8)
axs[0].set_ylabel("Interaction (MHz)", fontsize=fs)
axs[0].legend(fontsize=fs)

# right panel: example for state=1
grouped1 = np.stack([crht.params_fitted["1"], params_fit_dict_list[1][:3]])
axs[1].bar(x - width / 2, grouped1[0], width, alpha=0.7, label="fitted")
axs[1].bar(x + width / 2, grouped1[1], width, alpha=0.7, label="target")
axs[1].set_xticks(x)
axs[1].set_xticklabels(labels, fontsize=fs)
axs[1].set_title("Control State=1", fontsize=fs)
axs[1].axhline(0, color="k", linewidth=0.8)
axs[1].legend(fontsize=fs)

fig.suptitle("CR Hamiltonian Tomography Fitted Parameters", fontsize=fs)
fig.tight_layout()
plt.show()

# %%
# path = "C:\\Users\\SoonTeh\\Projects\\Repo\\CS_installations_all\\HI_23Oct2025\\data\\test\\2025-12-10\\#169_31b_CR_hamiltonian_tomography_vs_cr_drive_phase_115840\\ds_fit.h5"
# path = "C:\\Users\\SoonTeh\\Projects\\Repo\\CS_installations_all\\HI_23Oct2025\\data\\test\\2025-12-10\\#216_31b_CR_hamiltonian_tomography_vs_cr_drive_phase_181112\\ds_fit.h5"
path = "C:\\Users\\SoonTeh\\Downloads\\get-data-for-yonatan\\2025-12-11\\#230_31b_CR_hamiltonian_tomography_vs_cr_drive_phase_140629\\ds_fit.h5"
all_data = xr.load_dataset(path)


# %%
coeffs = []
for phase in all_data.phase.values:
    data = all_data.isel(qubit_pair=0).sel(control_target="t", phase=phase)

    ######################################
    # Choose the target qubit
    ######################################
    ts = data.pulse_duration.data
    bloch_data = data["bloch"].data

    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    try:
        crht = CRHamiltonianTomographyAnalysis(ts=ts, data=bloch_data)
        crht.fit_params()
        crht.plot_fit_result(fig, axs)
    except:
        crht.interaction_coeffs_MHz = {p: None for p in PAULI_2Q}
    coeffs.append(crht.interaction_coeffs_MHz)

# %%
fig_summary = plot_interaction_coeffs(coeffs, all_data.phase.values, xlabel="cr drive phase")

# %%
phase = all_data.phase.values[5]
data = all_data.isel(qubit_pair=0).sel(control_target="t", phase=phase)

######################################
# Choose the target qubit
######################################
ts = data.pulse_duration.data
bloch_data = data["bloch"].data

fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
try:
    crht = CRHamiltonianTomographyAnalysis(ts=ts, data=bloch_data)
    crht.fit_params()
    crht.plot_data(fig, axs)
    crht.plot_fit_result(fig, axs)
except:
    pass
# plt.show()

for ax in axs:
    ax.set_xlim((0, ts[-1]))
