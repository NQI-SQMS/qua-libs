# %%
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# %%


# ---- Rx/Ry-only basis prep/undo ----
def z_to_basis_rxry(qc, q, basis):
    basis = basis.lower()
    if basis == "z":
        return
    if basis == "x":
        qc.ry(np.pi / 2, q)
        qc.rx(np.pi, q)
        return
    if basis == "y":
        qc.rx(-np.pi / 2, q)
        return
    raise ValueError("basis must be 'x','y', or 'z'")


def basis_to_z_rxry(qc, q, basis):
    basis = basis.lower()
    if basis == "z":
        return
    if basis == "x":
        qc.rx(-np.pi, q)
        qc.ry(-np.pi / 2, q)
        return
    if basis == "y":
        qc.rx(np.pi / 2, q)
        return
    raise ValueError("basis must be 'x','y', or 'z'")


def prepare_pauli_eigenstate_rxry(qc, q, basis, sign):
    if sign not in ["+", "-"]:
        raise ValueError("sign must be '+' or '-'")
    if sign == "-":
        qc.rx(np.pi, q)  # |1>
    z_to_basis_rxry(qc, q, basis)


def zx90_n_then_measure_target_z(n, control_basis="z", control_sign="+", target_basis="z", target_sign="+"):
    qc = QuantumCircuit(2, 1)  # 1 classical bit for target

    # prep eigenstates
    prepare_pauli_eigenstate_rxry(qc, 0, control_basis, control_sign)
    prepare_pauli_eigenstate_rxry(qc, 1, target_basis, target_sign)

    # n ZX(90)
    theta = np.pi / 2
    for _ in range(n):
        qc.rzx(theta, 0, 1)

    # rotate back to Z
    basis_to_z_rxry(qc, 0, control_basis)
    basis_to_z_rxry(qc, 1, target_basis)

    # measure target qubit (q1) in Z -> c0
    qc.measure(1, 0)
    return qc


# ---- sweep n and measure ----
sim = AerSimulator()
shots = 20000

control_basis, control_sign = "z", "+"  # edit
target_basis, target_sign = "x", "-"  # edit

import matplotlib.pyplot as plt

p0_list, p1_list = [], []
n_list = list(range(10))

for n in n_list:
    qc = zx90_n_then_measure_target_z(
        n, control_basis=control_basis, control_sign=control_sign, target_basis=target_basis, target_sign=target_sign
    )
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    p0 = counts.get("0", 0) / shots
    p1 = counts.get("1", 0) / shots

    p0_list.append(p0)
    p1_list.append(p1)

plt.figure(figsize=(6, 4))
plt.plot(n_list, p0_list, marker="o", label="P(target=0)")
plt.plot(n_list, p1_list, marker="o", label="P(target=1)")
plt.xlabel("Number of ZX(90) gates")
plt.ylabel("Probability")
plt.legend()
plt.title(f"Control: {control_sign}{control_basis}, Target: {target_sign}{target_basis}")

# %%
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

sim = AerSimulator()
shots = 20000

qc = QuantumCircuit(2, 2)  # 2 classical bits for both qubits
qc.rx(np.pi, 0)
# qc.ry(np.pi / 2, 0)
# qc.ry(np.pi / 2, 1)
for i in range(2):
    qc.rzx(np.pi / 2, 0, 1)
# qc.ry(-np.pi / 2, 0)
# qc.ry(-np.pi / 2, 1)
qc.barrier()
qc.measure(0, 1)
qc.measure(1, 0)
print(qc.draw())
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()
# plot counts
plt.figure(figsize=(6, 4))
probs = {}
for state in ["00", "01", "10", "11"]:
    if state not in counts:
        probs[state] = 0
    else:
        probs[state] = counts[state] / shots
plt.bar(probs.keys(), probs.values())
plt.xlabel(r"Measurement outcome $|\, q0, q1 \rangle$")
plt.ylabel("Probability")

# %%
# convert to matrix
from qiskit.quantum_info import Operator

qc = QuantumCircuit(2)
qc.ry(np.pi / 2, 0)
qc.rx(2 * np.pi, 0)
qc.ry(-np.pi / 2, 0)
U = Operator(qc).data  # qc is a QuantumCircuit
fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
# diverging cmap
cmap = plt.get_cmap("RdBu")
ax[0].imshow(U.real, cmap=cmap, vmin=-1, vmax=1)
ax[0].set_title("Real part")
ax[1].imshow(U.imag, cmap=cmap, vmin=-1, vmax=1)
ax[1].set_title("Imaginary part")
ax[0].set_xticks(range(4))
ax[0].set_yticks(range(4))
ax[1].set_xticks(range(4))
ax[1].set_yticks(range(4))
plt.colorbar(ax[1].imshow(U.imag, cmap=cmap, vmin=-1, vmax=1), ax=ax, orientation="vertical", shrink=0.8)
