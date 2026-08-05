# %%
#!%load_ext autoreload
#!%autoreload 2
# %%

import os
import pickle as pkl

import hvplot.xarray  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.mplot3d import Axes3D
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from sequence_tools import pre_generate_sequence_interleaved

number_of_sequences = 10
depth_list = [0] + list(range(1, 30, 2))
if False:
    # if os.path.exists("saved_sequence.pkl"):
    with open("saved_sequence.pkl", "rb") as f:
        full_sequence_list = pkl.load(f)
else:
    sequence_list, len_list, full_list = pre_generate_sequence_interleaved(
        number_of_sequences, depth_list, interleaved_instruct=None, exclude_cnot=False
    )
    full_sequence_list = full_list
    with open("saved_sequence.pkl", "wb") as f:
        pkl.dump(full_sequence_list, f)

assert len(full_sequence_list) == number_of_sequences * len(depth_list)

# %%
ins_len_list = [len(seq) for seq in full_sequence_list]
wrap = len(depth_list)
ins_len_arr = np.array(ins_len_list).reshape(-1, wrap).T
plt.plot(depth_list, ins_len_arr, "o-")
plt.ylim(ymin=0)
plt.xlabel("Sequence index")
plt.ylabel("Number of instructions in sequence")

# %%
sim = AerSimulator()
shots = 20000
states = ["00", "01", "10", "11"]


def instruction_to_circuit(instruction_list, zerr=0.1):
    qc = QuantumCircuit(2, 2)  # 2 classical bits for both qubits
    for inst in instruction_list:
        if inst[1] == "01":
            qc.cx(0, 1)
            # CNOT decomposition into [Z(-pi/2) x I] * [I x X(-pi/2)] * ZX(pi/2)
            ## IX(-pi/2)
            # qc.rx(-np.pi / 2, 1)
            ## CR + ZI
            # qc.rzx(np.pi / 2, 0, 1)
            # qc.rz(zerr * 2 * np.pi, 0)
            ## ZI(-pi/2)
            # qc.rz(-np.pi / 2, 0)
            # qc.rx(np.pi / 2, 0)
            # qc.ry(np.pi / 2, 0)
            # qc.rx(-np.pi / 2, 0)
        else:
            qubit = int(inst[1])
            gate = inst[0]
            if gate == "I":
                continue
            elif gate == "x90":
                qc.rx(np.pi / 2, qubit)
            elif gate == "-x90":
                qc.rx(-np.pi / 2, qubit)
            elif gate == "x180":
                qc.rx(np.pi, qubit)
            elif gate == "y90":
                qc.ry(np.pi / 2, qubit)
            elif gate == "-y90":
                qc.ry(-np.pi / 2, qubit)
            elif gate == "y180":
                qc.ry(np.pi, qubit)
        qc.barrier()
    return qc


# %%
# num_instr_list = []
# full_prob_list = []
# rand_indices = np.array(range(10)) + 10
# # rand_indices = np.random.randint(len(full_sequence_list), size=5)
# for rand_idx in rand_indices:
#     instruction_list = full_sequence_list[rand_idx]  # change index to test different sequences
#     zerr_list = np.linspace(0, 0.5, 20)
#     prob_list = []
#     for zerr in zerr_list:
#         qc = instruction_to_circuit(instruction_list, zerr=zerr)
#         qc.barrier()
#         qc.measure(0, 1)
#         qc.measure(1, 0)
#         # print(qc.draw())
#         result = sim.run(qc, shots=shots).result()
#         counts = result.get_counts()
#         # plot counts
#         # plt.figure(figsize=(6, 4))
#         probs = {}
#         for state in states:
#             if state not in counts:
#                 probs[state] = 0
#             else:
#                 probs[state] = counts[state] / shots
#         # plt.bar(probs.keys(), probs.values())
#         # plt.xlabel(r"Measurement outcome $|\, q0, q1 \rangle$")
#         # plt.ylabel("Probability")
#         prob_list.append(probs)

#     num_instr_list.append(len(instruction_list))
#     full_prob_list.append(prob_list)
#     # assert np.isclose(probs["00"], 1.0)


# %%
# max_cols = 4
# n = len(num_instr_list)
# ncols = min(max_cols, n)
# nrows = np.ceil(n / max_cols).astype(int)


# fig = plt.figure(figsize=(5.5 * ncols, 4.8 * nrows), constrained_layout=True)
# with plt.rc_context({"font.size": 14}):
#     dx = 0.3
#     dy = (zerr_list[1] - zerr_list[0]) * 0.5

#     cmap = plt.cm.viridis
#     row_colors = cmap(np.linspace(0, 1, len(zerr_list)))
#     colors = np.repeat(row_colors[:, None, :], len(states), axis=1).reshape(-1, 4)

#     for i, (num_instr, prob_list) in enumerate(zip(num_instr_list, full_prob_list)):
#         ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")

#         P = np.array([[d[s] for s in states] for d in prob_list])  # (len(zerr_list), len(states))
#         X, Y = np.meshgrid(np.arange(len(states)), zerr_list)

#         ax.bar3d(
#             X.ravel() - dx / 2,
#             Y.ravel() - dy / 2,
#             np.zeros(P.size),
#             dx,
#             dy,
#             P.ravel(),
#             alpha=0.8,
#             color=colors,
#             shade=True,
#         )

#         ax.set_title(f"N instr: {num_instr}", pad=10)
#         ax.set_xlabel("State", labelpad=8)
#         ax.set_ylabel(r"Residual ZI in CR ($2\pi$)", labelpad=8)
#         ax.set_zlabel("Prob.", labelpad=6)
#         ax.set_xticks(np.arange(len(states)))
#         ax.set_xticklabels(states)

#     plt.tight_layout()
#     plt.show()

# # %%
full_prob_list = []
full_prob_mean = {k: [] for k in states}
full_prob_std = {k: [] for k in states}
num_instr_mean = []

zerr = 0.0
rep = ins_len_arr.shape[1]
for i in range(wrap):
    prob_list = []
    num_instr_list = []
    for j in range(rep):
        instruction_list = full_sequence_list[i + wrap * j]
        ins_len = len(instruction_list)  # change index to test different sequences
        qc = instruction_to_circuit(instruction_list, zerr=zerr)
        qc.barrier()
        qc.measure(0, 1)
        qc.measure(1, 0)
        # print(qc.draw())
        result = sim.run(qc, shots=shots).result()
        counts = result.get_counts()
        # plot counts
        # plt.figure(figsize=(6, 4))
        probs = {}
        for state in states:
            if state not in counts:
                probs[state] = 0
            else:
                probs[state] = counts[state] / shots
        # plt.bar(probs.keys(), probs.values())
        # plt.xlabel(r"Measurement outcome $|\, q0, q1 \rangle$")
        # plt.ylabel("Probability")
        prob_list.append(probs)
        num_instr_list.append(ins_len)

    full_prob_list.append(prob_list)
    for state in states:
        vals = np.array([prob_list[r][state] for r in range(len(prob_list))])
        full_prob_mean[state].append(vals.mean())
        full_prob_std[state].append(vals.std())
    num_instr_mean.append(np.mean(num_instr_list))
    print(f"Done for instr idx: {i} with avg num instr: {num_instr_mean[-1]:.1f}")
    # assert np.isclose(probs["00"], 1.0)

# %%


## %%
for state in states:
    plt.errorbar(
        depth_list,
        full_prob_mean[state],
        yerr=full_prob_std[state],
        fmt="o-",
        label=f"State |{state}>",
        capsize=5,
    )
plt.ylim(-0.05, 1.05)
plt.axhline(0.25, color="k", linestyle="--", label="Fully mixed")
plt.xlabel("Clifford depth")
plt.ylabel("Measured probability")
plt.title(f"Simulated RB outcome vs Clifford depth\n(Residual ZI in CR = {zerr:.2f} x 2pi)")
plt.legend()

ax = plt.gca()
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(ax.get_xticks())
# interpolate num_instr_mean to match depth_list ticks
num_instr_mean_arr = np.array(num_instr_mean)
num_instr_mean_interp = np.interp(ax.get_xticks(), depth_list, num_instr_mean_arr)
ax2.set_xticks(ax.get_xticks())
ax2.set_xticklabels([f"{n:.1f}" for n in num_instr_mean_interp])
ax2.set_xlabel("Average number of instructions")
plt.show()

# %%
# compare with actual data

data_path1 = r"C:\Users\SoonTeh\Downloads\get-data-for-yonatan\2025-12-12\#303_60a_two_qubit_randomized_benchmarking_193603\ds_proc.h5"
# data_path1 = r"C:\Users\SoonTeh\Downloads\get-data-for-yonatan\2025-12-12\#299_60a_two_qubit_randomized_benchmarking_192110\ds_proc.h5"
data_path2 = r"C:\Users\SoonTeh\Downloads\get-data-for-yonatan\2025-12-12\#300_60a_two_qubit_randomized_benchmarking_192443\ds_proc.h5"
titles = ["Standard RB", "CNOT Interleaved RB"]
for data_path, title in zip([data_path1, data_path2], titles):
    data = xr.load_dataset(data_path)
    cmap = plt.get_cmap("tab10")
    g = data.state.plot.scatter(
        x="depths",
        row="qubit_pair",
        col="measured_state",
        ylim=(0, 1),
        hue="nb_of_sequences",
        cmap=cmap,
        figsize=(12, 6),
    )

    for ax in g.axs.flat:
        ax.axhline(0.25, color="r", linestyle="--")

    g.fig.suptitle(title, fontsize=20)

    # leave room on the right for the colorbar
    g.fig.subplots_adjust(top=0.85, right=0.77, wspace=0.25, hspace=0.25)

    plt.show()

# %%
data.hvplot.scatter(
    x="nb_of_sequences",
    # x="depths",
    row="qubit_pair",
    col="measured_state",
    ylim=(0, 1),
    # hue="nb_of_sequences",
    # cmap=cmap,
    figsize=(12, 6),
)

# %%
data_path1 = r"C:\Users\SoonTeh\Downloads\get-data-for-yonatan\2025-12-12\#305_32c_CR_cancel_phase_erroramplification_200533\ds_raw.h5"
data_path2 = r"C:\Users\SoonTeh\Downloads\get-data-for-yonatan\2025-12-12\#304_32c_CR_cancel_phase_erroramplification_200201\ds_raw.h5"

# %%
ds = xr.load_dataset(data_path1)
qubit_pair = ds.qubit_pair.values[0]

# diverging cmap
cmap = plt.get_cmap("RdBu")
sub = ds.sel(qubit_pair=qubit_pair)[["state_c", "state_t"]]

da = sub.to_array(dim="which_state")  # dims now include which_state ∈ {state_c, state_t}

da.plot(
    row="which_state",  # <- state_c vs state_t
    col="control_state",  # <- your existing facet
    x="phase",
    y="n_cr",
    cmap=cmap,
    vmin=0,
    vmax=1,
)

# %% test decomposition of CNOT by transpiling
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RXGate, RYGate, RZGate, RZXGate
from qiskit.quantum_info import Operator
from qiskit.transpiler import InstructionProperties, Target

target = Target()

# Add 1q gates once, with properties for each qubit
target.add_instruction(RXGate(0), {(0,): InstructionProperties(), (1,): InstructionProperties()})
target.add_instruction(RYGate(0), {(0,): InstructionProperties(), (1,): InstructionProperties()})
target.add_instruction(RZGate(0), {(0,): InstructionProperties(), (1,): InstructionProperties()})

# Add fixed-angle 2q entangler
target.add_instruction(RZXGate(np.pi / 2), {(0, 1): InstructionProperties()})

qc = QuantumCircuit(2)
# qc.cx(0, 1)
qc.h(0)
# theta = 1.2
# qc.rz(-theta, 0)

tqc = transpile(qc, target=target, optimization_level=3)
print(tqc)


# %%
qc = QuantumCircuit(2)
# qc.rzx(np.pi / 2, 0, 1)
# qc.rz(-np.pi / 2, 0)
qc.rx(np.pi / 2, 0)
# qc.ry(theta, 0)
qc.rx(-np.pi / 2, 0)

print(qc.draw())

Operator(qc).equiv(Operator(tqc))
