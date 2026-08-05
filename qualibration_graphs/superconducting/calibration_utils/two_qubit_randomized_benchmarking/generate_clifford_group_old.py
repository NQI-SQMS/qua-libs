# %%
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford


def cliff_from_circ(circ):
    return Clifford(circ)


def key_from_cliff(c: Clifford):
    # hashable canonical key from boolean tableau
    return c.tableau.astype(np.uint8).tobytes()


import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford


def key_from_cliff(c: Clifford):
    return c.tableau.astype(np.uint8).tobytes()


def generate_clifford_group(generators):
    """
    generators: iterable of qiskit.quantum_info.Clifford (or QuantumCircuit)
    returns: list[Clifford] = group closure of generators
    auto-detects n qubits from first generator.
    """
    gens = []
    for g in generators:
        gens.append(g if isinstance(g, Clifford) else Clifford(g))
    if not gens:
        raise ValueError("No generators provided.")

    n = gens[0].num_qubits
    if any(g.num_qubits != n for g in gens):
        raise ValueError("All generators must act on the same number of qubits.")

    identity = Clifford(QuantumCircuit(n))
    seen = {key_from_cliff(identity): identity}
    frontier = [identity]

    while frontier:
        g = frontier.pop()
        for h in gens:
            gh = g.compose(h)
            k = key_from_cliff(gh)
            if k not in seen:
                seen[k] = gh
                frontier.append(gh)

    return list(seen.values())


# --- generators of C_2 ---
gens = []
for q in [0, 1]:
    qc_h = QuantumCircuit(2)
    qc_h.h(q)
    gens.append(cliff_from_circ(qc_h))
    qc_s = QuantumCircuit(2)
    qc_s.s(q)
    gens.append(cliff_from_circ(qc_s))

qc_cx01 = QuantumCircuit(2)
qc_cx01.cx(0, 1)
gens.append(cliff_from_circ(qc_cx01))
# qc_cx10 = QuantumCircuit(2)
# qc_cx10.cx(1, 0)
# gens.append(cliff_from_circ(qc_cx10))

cliffords_2q = generate_clifford_group(gens)
assert len(cliffords_2q) == 11520, f"got {len(cliffords_2q)}"

# %%
# --- generators of C_1 ---
gens = []

qc_h = QuantumCircuit(1)
qc_h.h(0)
gens.append(cliff_from_circ(qc_h))
qc_s = QuantumCircuit(1)
qc_s.s(0)
gens.append(cliff_from_circ(qc_s))

cliffords_1q = generate_clifford_group(gens)
assert len(cliffords_1q) == 24, f"got {len(cliffords_1q)}"

# %% Clifford decompositions
import numpy as np
from collections import deque
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford


def key_from_cliff(c: Clifford):
    return c.tableau.astype(np.uint8).tobytes()


def tensor_1q_into_2q(c1: Clifford, which: int):
    """Return 2q Clifford = c1 on qubit 'which', identity on other."""
    qc = QuantumCircuit(2)
    # use exact circuit for 1q Clifford
    sub = c1.to_circuit()
    qc.compose(sub, qubits=[which], inplace=True)
    return Clifford(qc)


def generate_2q_from_1q_with_words(cliffords_1q):
    """
    Returns:
      cliffords_2q: list[Clifford]
      words: dict[key -> list of steps]
    Each step is either:
      ("local", i0, i1)  meaning apply C1[i0] on q0 and C1[i1] on q1
      ("cx", ctrl, tgt)
    """
    # lookup for 1q cliffords by key (for compression / identity)
    lookup_1q = {key_from_cliff(c): i for i, c in enumerate(cliffords_1q)}
    id1_key = key_from_cliff(Clifford(QuantumCircuit(1)))
    id1_idx = lookup_1q[id1_key]

    # build 2q generators with attached "step labels"
    gens = []

    # local generators: c ⊗ I and I ⊗ c for all 1q cliffords
    for i, c in enumerate(cliffords_1q):
        g0 = tensor_1q_into_2q(c, 0)
        gens.append((g0, ("local", i, id1_idx)))

        g1 = tensor_1q_into_2q(c, 1)
        gens.append((g1, ("local", id1_idx, i)))

    # entangling generators
    for ctrl, tgt in [(0, 1)]:
        # for ctrl, tgt in [(0, 1), (1, 0)]:
        qc = QuantumCircuit(2)
        qc.cx(ctrl, tgt)
        gens.append((Clifford(qc), ("cx", ctrl, tgt)))

    # BFS closure while storing parents
    identity = Clifford(QuantumCircuit(2))
    id_key = key_from_cliff(identity)

    parent = {id_key: None}  # key -> parent_key
    move = {id_key: None}  # key -> step label used from parent to here

    q = deque([identity])

    while q:
        g = q.popleft()
        g_key = key_from_cliff(g)
        for h, step in gens:
            gh = g.compose(h)
            k = key_from_cliff(gh)
            if k not in parent:
                parent[k] = g_key
                move[k] = step
                q.append(gh)

    # backtrack to form words
    def word_for(key):
        w = []
        cur = key
        while parent[cur] is not None:
            w.append(move[cur])
            cur = parent[cur]
        w.reverse()
        return w

    words = {k: word_for(k) for k in parent.keys()}
    cliffords_2q = [None] * len(parent)
    for idx, k in enumerate(parent.keys()):
        # reconstruct Clifford objects in same order as keys
        # easiest: store them again from BFS parent dict
        cliffords_2q[idx] = None
    # instead just return dict of key->Clifford for convenience:
    # rebuild by re-running over stored keys is harder; so keep objects during BFS:
    # We'll do it properly:
    key_to_cliff = {}
    # re-run BFS to fill key_to_cliff using parent (small size anyway)
    q = deque([identity])
    key_to_cliff[id_key] = identity
    while q:
        g = q.popleft()
        g_key = key_from_cliff(g)
        for h, step in gens:
            gh = g.compose(h)
            k = key_from_cliff(gh)
            if k in parent and k not in key_to_cliff:
                key_to_cliff[k] = gh
                q.append(gh)

    cliffords_2q = list(key_to_cliff.values())
    return cliffords_2q, words, key_to_cliff


def compress_word(word, cliffords_1q):
    """
    Combine consecutive local steps into a single local step.
    """
    lookup_1q = {key_from_cliff(c): i for i, c in enumerate(cliffords_1q)}
    id1_idx = lookup_1q[key_from_cliff(Clifford(QuantumCircuit(1)))]

    out = []
    cur0 = id1_idx
    cur1 = id1_idx

    def flush():
        nonlocal cur0, cur1
        if cur0 != id1_idx or cur1 != id1_idx:
            out.append(("local", cur0, cur1))
            cur0 = id1_idx
            cur1 = id1_idx

    # precompute multiplication table for 1q cliffords
    mul = {}
    for i, a in enumerate(cliffords_1q):
        for j, b in enumerate(cliffords_1q):
            k = key_from_cliff(a.compose(b))
            mul[(i, j)] = lookup_1q[k]

    for step in word:
        if step[0] == "local":
            _, i0, i1 = step
            cur0 = mul[(cur0, i0)]
            cur1 = mul[(cur1, i1)]
        else:
            flush()
            out.append(step)
    flush()
    return out


cliffords_2q, words, key_to_cliff = generate_2q_from_1q_with_words(cliffords_1q)
assert len(cliffords_2q) == 11520, f"got {len(cliffords_2q)}"

# decomposition for a specific 2q Clifford c2:
# k = key_from_cliff(c2)
# w = words[k]
# w_compact = compress_word(w, cliffords_1q)
# print(w_compact)


# %% physical gate decomposition

gates_1q = {
    "I": QuantumCircuit(1),
    "x90": QuantumCircuit(1).rx(np.pi / 2, 0),
    "-x90": QuantumCircuit(1).rx(-np.pi / 2, 0),
    "x180": QuantumCircuit(1).rx(np.pi, 0),
    "y90": QuantumCircuit(1).ry(np.pi / 2, 0),
    "-y90": QuantumCircuit(1).ry(-np.pi / 2, 0),
    "y180": QuantumCircuit(1).ry(np.pi, 0),
}

gates_2q = {
    "CNOT": QuantumCircuit(2).cx(0, 1),
}

import numpy as np
from collections import deque
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford


def key_from_cliff(c: Clifford):
    return c.tableau.astype(np.uint8).tobytes()


def decompose_1q_cliffords(cliffords_1q, generators):
    """
    cliffords_1q: list[Clifford] to decompose (e.g., your 24 elements)
    generators: iterable of (name, Clifford or QuantumCircuit) for 1q gateset
                e.g. [("x90", qc_rx_pi2), ("z90", qc_rz_pi2), ...]
    returns: dict[index -> list of generator names]
             giving a shortest word for each Clifford.
    """
    gens = []
    for name, g in generators:
        cg = g if isinstance(g, Clifford) else Clifford(g)
        if cg.num_qubits != 1:
            raise ValueError("Generators must be 1-qubit Cliffords.")
        gens.append((name, cg))

    # BFS over the group generated by 'gens'
    identity = Clifford(QuantumCircuit(1))
    id_key = key_from_cliff(identity)

    parent = {id_key: None}
    move = {id_key: None}
    q = deque([identity])

    while q:
        g = q.popleft()
        g_key = key_from_cliff(g)
        for name, h in gens:
            gh = g.compose(h)
            k = key_from_cliff(gh)
            if k not in parent:
                parent[k] = g_key
                move[k] = name
                q.append(gh)

    # backtrack a word for any reachable Clifford
    def word_for(c):
        k = key_from_cliff(c)
        if k not in parent:
            raise ValueError("Target Clifford not generated by this gateset.")
        w = []
        cur = k
        while parent[cur] is not None:
            w.append(move[cur])
            cur = parent[cur]
        w.reverse()
        return w

    return {i: word_for(c) for i, c in enumerate(cliffords_1q)}


angles = {
    "x90": np.pi / 2,
    "x180": np.pi,
    "-x90": -np.pi / 2,
    "-x180": -np.pi,
    "y90": np.pi / 2,
    "y180": np.pi,
    "-y90": -np.pi / 2,
    "-y180": -np.pi,
    "z90": np.pi / 2,
    "z180": np.pi,
    "-z90": -np.pi / 2,
    "-z180": -np.pi,
}

gens_1q = []
for name, ang in angles.items():
    qc = QuantumCircuit(1)
    axis = name[0]
    if axis == "-":
        axis = name[1]
    getattr(qc, f"r{axis}")(ang, 0)  # rx/ry/rz
    gens_1q.append((name, qc))

words_1q = decompose_1q_cliffords(cliffords_1q, gens_1q)
average_length = np.mean([len(w) for w in words_1q.values()])
print(f"Average 1q Clifford decomposition length: {average_length:.2f} gates")
