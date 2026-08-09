# %%
import numpy as np
import tqdm

from collections import deque
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford


# ---------- basic helpers ----------
def cliff(circ_or_cliff):
    """Ensure Clifford object."""
    return circ_or_cliff if isinstance(circ_or_cliff, Clifford) else Clifford(circ_or_cliff)


def key_from_cliff(c: Clifford) -> bytes:
    """Hashable canonical key from stabilizer tableau."""
    return c.tableau.astype(np.uint8).tobytes()


# ---------- generic group closure ----------
def generate_clifford_group(generators):
    """
    generators: iterable of Clifford or QuantumCircuit (all same n)
    returns: list[Clifford] group closure
    """
    gens = [cliff(g) for g in generators]
    if not gens:
        raise ValueError("No generators provided.")
    n = gens[0].num_qubits
    if any(g.num_qubits != n for g in gens):
        raise ValueError("All generators must have same num_qubits.")

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


# ---------- 1q Clifford generation ----------
def one_qubit_cliffords_from_HS():
    qc_h = QuantumCircuit(1)
    qc_h.h(0)
    qc_s = QuantumCircuit(1)
    qc_s.s(0)
    C1 = generate_clifford_group([qc_h, qc_s])
    assert len(C1) == 24
    return C1


# ---------- 2q generation from C1 with word-tracking ----------
def tensor_1q_into_2q(c1: Clifford, which: int) -> Clifford:
    """Two-qubit Clifford = c1 on qubit `which`, identity on other."""
    qc2 = QuantumCircuit(2)
    qc2.compose(c1.to_circuit(), qubits=[which], inplace=True)
    return Clifford(qc2)


def generate_2q_from_1q_with_words(cliffords_1q, include_cx10=False):
    """
    Build C2 from {C1⊗I, I⊗C1, CX} while storing a decomposition word.

    Returns:
      cliffords_2q : list[Clifford]
      words        : dict[key -> list[step]]
      key_to_cliff : dict[key -> Clifford]
    step is either:
      ("local", i0, i1)  meaning C1[i0] on q0 and C1[i1] on q1
      ("cx", ctrl, tgt)
    """
    lookup_1q = {key_from_cliff(c): i for i, c in enumerate(cliffords_1q)}
    id1_idx = lookup_1q[key_from_cliff(Clifford(QuantumCircuit(1)))]

    gens = []
    for i, c in enumerate(cliffords_1q):
        gens.append((tensor_1q_into_2q(c, 0), ("local", i, id1_idx)))
        gens.append((tensor_1q_into_2q(c, 1), ("local", id1_idx, i)))

    cx_pairs = [(0, 1)] + ([(1, 0)] if include_cx10 else [])
    for ctrl, tgt in cx_pairs:
        qc = QuantumCircuit(2)
        qc.cx(ctrl, tgt)
        gens.append((Clifford(qc), ("cx", ctrl, tgt)))

    identity = Clifford(QuantumCircuit(2))
    id_key = key_from_cliff(identity)

    parent = {id_key: None}  # key -> parent_key
    move = {id_key: None}  # key -> step from parent to key
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

    def word_for(k):
        w = []
        cur = k
        while parent[cur] is not None:
            w.append(move[cur])
            cur = parent[cur]
        w.reverse()
        return w

    words = {k: word_for(k) for k in parent}

    # recover Clifford objects (using same allowed transitions)
    key_to_cliff = {id_key: identity}
    q = deque([identity])
    while q:
        g = q.popleft()
        g_key = key_from_cliff(g)
        for h, _ in gens:
            gh = g.compose(h)
            k = key_from_cliff(gh)
            if k in parent and k not in key_to_cliff:
                key_to_cliff[k] = gh
                q.append(gh)

    cliffords_2q = list(key_to_cliff.values())
    return cliffords_2q, words, key_to_cliff


def compress_word(word, cliffords_1q):
    """Combine consecutive local steps into one local step."""
    lookup_1q = {key_from_cliff(c): i for i, c in enumerate(cliffords_1q)}
    id1_idx = lookup_1q[key_from_cliff(Clifford(QuantumCircuit(1)))]

    # multiplication table on C1
    mul = {}
    for i, a in enumerate(cliffords_1q):
        for j, b in enumerate(cliffords_1q):
            mul[(i, j)] = lookup_1q[key_from_cliff(a.compose(b))]

    out = []
    cur0, cur1 = id1_idx, id1_idx

    def flush():
        nonlocal cur0, cur1
        if cur0 != id1_idx or cur1 != id1_idx:
            out.append(("local", cur0, cur1))
            cur0, cur1 = id1_idx, id1_idx

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


# ---------- shortest 1q decompositions in a given gateset ----------
def make_1q_rotation_gate(axis: str, angle: float) -> QuantumCircuit:
    qc = QuantumCircuit(1)
    getattr(qc, f"r{axis}")(angle, 0)  # rx/ry/rz
    return qc


def decompose_1q_cliffords(cliffords_1q, generators):
    """
    generators: iterable of (name, QuantumCircuit|Clifford), all 1q Clifford
    returns: dict[i -> list[name]] shortest words for each C1[i]
    """
    gens = [(name, cliff(g)) for name, g in generators]
    if any(g.num_qubits != 1 for _, g in gens):
        raise ValueError("All generators must be 1-qubit Cliffords.")

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

    def word_for(c):
        k = key_from_cliff(c)
        if k not in parent:
            raise ValueError("Target not generated by this gateset.")
        w = []
        cur = k
        while parent[cur] is not None:
            w.append(move[cur])
            cur = parent[cur]
        w.reverse()
        return w

    return {i: word_for(c) for i, c in enumerate(cliffords_1q)}


# %%
# 1q Cliffords
cliffords_1q = one_qubit_cliffords_from_HS()

# 2q Cliffords with decomposition words from C1 and CX(0,1)
cliffords_2q, words_2q, key_to_cliff_2q = generate_2q_from_1q_with_words(cliffords_1q)
assert len(cliffords_2q) == 11520

# %% compress 2q words
compressed_words_2q = {
    k: compress_word(w, cliffords_1q) for k, w in tqdm.tqdm(words_2q.items(), desc="Compressing 2q words")
}

original_lengths = np.array([len(w) for w in words_2q.values()], dtype=float)
compressed_lengths = np.array([len(w) for w in compressed_words_2q.values()], dtype=float)

# avoid 0/0 by treating any 0 length as 1 for the ratio
den = np.where(compressed_lengths == 0, 1.0, compressed_lengths)
num = np.where(original_lengths == 0, 1.0, original_lengths)

den_mean = np.mean(den)
num_mean = np.mean(num)
avg_compress = num_mean / den_mean
print(f"Average 2q word compression factor: {num_mean:.2f}/{den_mean:.2f}={avg_compress:.2f}")


# %%
# physical 1q gateset decomposition
angles = {
    "x90": np.pi / 2,
    "-x90": -np.pi / 2,
    "x180": np.pi,
    "-x180": -np.pi,
    "y90": np.pi / 2,
    "-y90": -np.pi / 2,
    "y180": np.pi,
    "-y180": -np.pi,
    "z90": np.pi / 2,
    "-z90": -np.pi / 2,
    "z180": np.pi,
    "-z180": -np.pi,
}
gens_phys_1q = [(name, make_1q_rotation_gate(name.lstrip("-")[0], ang)) for name, ang in angles.items()]

words_1q = decompose_1q_cliffords(cliffords_1q, gens_phys_1q)
avg_len = np.mean([len(w) for w in words_1q.values()])
print(f"Average 1q Clifford decomposition length: {avg_len:.2f} gates")

# %%
from qiskit import QuantumCircuit


# --- build physical 1q gate circuits ONCE (for optional circuit materialization) ---
def make_1q_rot(axis, angle):
    qc = QuantumCircuit(1)
    getattr(qc, f"r{axis}")(angle, 0)
    return qc


gates_1q_circs = {
    "x90": make_1q_rot("x", np.pi / 2),
    "-x90": make_1q_rot("x", -np.pi / 2),
    "x180": make_1q_rot("x", np.pi),
    "-x180": make_1q_rot("x", -np.pi),
    "y90": make_1q_rot("y", np.pi / 2),
    "-y90": make_1q_rot("y", -np.pi / 2),
    "y180": make_1q_rot("y", np.pi),
    "-y180": make_1q_rot("y", -np.pi),
    "z90": make_1q_rot("z", np.pi / 2),
    "-z90": make_1q_rot("z", -np.pi / 2),
    "z180": make_1q_rot("z", np.pi),
    "-z180": make_1q_rot("z", -np.pi),
    # identity not needed for counts/circuits; skip unless you explicitly use it
}

cx01 = QuantumCircuit(2)
cx01.cx(0, 1)


# --- lift 2q words to physical words by direct mapping ---
def lift_2q_word_to_physical(word2q, words_1q):
    """
    word2q: list of ("local", i0, i1) or ("cx", ctrl, tgt)
    words_1q: dict[i -> list[str]] physical 1q gate names
    """
    phys = []
    for step in word2q:
        if step[0] == "local":
            _, i0, i1 = step
            phys.append(("local_phys", words_1q[i0], words_1q[i1]))
        else:
            phys.append(step)  # cx
    return phys


def physical_word_to_circuit(phys_word, gates_1q_circs, cx_circ=cx01):
    qc = QuantumCircuit(2)
    for step in phys_word:
        if step[0] == "local_phys":
            _, g0, g1 = step
            for name in g0:
                qc.compose(gates_1q_circs[name], [0], inplace=True)
            for name in g1:
                qc.compose(gates_1q_circs[name], [1], inplace=True)
        else:
            _, ctrl, tgt = step
            if (ctrl, tgt) == (0, 1):
                qc.compose(cx_circ, [0, 1], inplace=True)
            else:
                qc.cx(ctrl, tgt)
    return qc


# choose which 2q words to lift (compressed is usually what you want)
source_words_2q = compressed_words_2q  # or words_2q

phys_words_2q = {k: lift_2q_word_to_physical(w, words_1q) for k, w in source_words_2q.items()}

# --- stats without redundancy ---
total = len(phys_words_2q)
one_q_count = 0
cx_count = 0

for pw in phys_words_2q.values():
    for step in pw:
        if step[0] == "local_phys":
            one_q_count += len(step[1]) + len(step[2])
        else:
            cx_count += 1

print(f"Average physical gates per 2q Clifford: {one_q_count / total:.2f} 1q, {cx_count / total:.2f} CX")

# --- optional: materialize circuits only if you need them ---
phys_circs_2q = {k: physical_word_to_circuit(pw, gates_1q_circs) for k, pw in phys_words_2q.items()}

# %%
# physical 2q gateset decomposition
from qiskit import QuantumCircuit


# --- build physical 1q gate circuits ONCE (for optional circuit materialization) ---
def make_1q_rot(axis, angle):
    qc = QuantumCircuit(1)
    getattr(qc, f"r{axis}")(angle, 0)
    return qc


gates_1q_circs = {
    "x90": make_1q_rot("x", np.pi / 2),
    "-x90": make_1q_rot("x", -np.pi / 2),
    "x180": make_1q_rot("x", np.pi),
    "-x180": make_1q_rot("x", -np.pi),
    "y90": make_1q_rot("y", np.pi / 2),
    "-y90": make_1q_rot("y", -np.pi / 2),
    "y180": make_1q_rot("y", np.pi),
    "-y180": make_1q_rot("y", -np.pi),
    "z90": make_1q_rot("z", np.pi / 2),
    "-z90": make_1q_rot("z", -np.pi / 2),
    "z180": make_1q_rot("z", np.pi),
    "-z180": make_1q_rot("z", -np.pi),
    # identity not needed for counts/circuits; skip unless you explicitly use it
}

cx01 = QuantumCircuit(2)
cx01.cx(0, 1)


# --- lift 2q words to physical words by direct mapping ---
def lift_2q_word_to_physical(word2q, words_1q):
    """
    word2q: list of ("local", i0, i1) or ("cx", ctrl, tgt)
    words_1q: dict[i -> list[str]] physical 1q gate names
    """
    phys = []
    for step in word2q:
        if step[0] == "local":
            _, i0, i1 = step
            phys.append(("local_phys", words_1q[i0], words_1q[i1]))
        else:
            phys.append(step)  # cx
    return phys


def physical_word_to_circuit(phys_word, gates_1q_circs, cx_circ=cx01):
    qc = QuantumCircuit(2)
    for step in phys_word:
        if step[0] == "local_phys":
            _, g0, g1 = step
            for name in g0:
                qc.compose(gates_1q_circs[name], [0], inplace=True)
            for name in g1:
                qc.compose(gates_1q_circs[name], [1], inplace=True)
        else:
            _, ctrl, tgt = step
            if (ctrl, tgt) == (0, 1):
                qc.compose(cx_circ, [0, 1], inplace=True)
            else:
                qc.cx(ctrl, tgt)
    return qc


# choose which 2q words to lift (compressed is usually what you want)
source_words_2q = compressed_words_2q  # or words_2q

phys_words_2q = {k: lift_2q_word_to_physical(w, words_1q) for k, w in source_words_2q.items()}

# --- stats without redundancy ---
total = len(phys_words_2q)
one_q_count = 0
cx_count = 0

for pw in phys_words_2q.values():
    for step in pw:
        if step[0] == "local_phys":
            one_q_count += len(step[1]) + len(step[2])
        else:
            cx_count += 1

print(f"Average physical gates per 2q Clifford: {one_q_count / total:.2f} 1q, {cx_count / total:.2f} CX")

# --- optional: materialize circuits only if you need them ---
phys_circs_2q = {k: physical_word_to_circuit(pw, gates_1q_circs) for k, pw in phys_words_2q.items()}


total_cliffords = len(phys_words_2q)
avg_local = gate_count["local_phys"] / total_cliffords
avg_cx = gate_count["cx"] / total_cliffords
print(f"Average physical gates per 2q Clifford: {avg_local:.2f} 1q, {avg_cx:.2f} CX")

# %%
import pickle as pkl

instruct_list = pkl.load(open("2q_Clifford_gen_CNOT_instruct.pkl", "rb"))
unitary_list = pkl.load(open("2q_Clifford_gen_CNOT_unitary.pkl", "rb"))

gate_count = {"local": 0, "CNOT": 0}
for ins in instruct_list:
    for step in ins:
        if step[0] == "CNOT":
            gate_count["CNOT"] += 1
        elif step[0] == "I":
            continue
        else:
            gate_count["local"] += 1

total_cliffords = len(instruct_list)
avg_local = gate_count["local"] / total_cliffords
avg_cnot = gate_count["CNOT"] / total_cliffords
print(f"Average gates per 2q Clifford: {avg_local:.2f} 1q, {avg_cnot:.2f} CNOT")

decomposition_info = {
    "avg_num_1q_gate": avg_local,
    "avg_num_CNOT_gate": avg_cnot,
}
pkl.dump(decomposition_info, open("2q_Clifford_gen_CNOT_stats.pkl", "wb"))
