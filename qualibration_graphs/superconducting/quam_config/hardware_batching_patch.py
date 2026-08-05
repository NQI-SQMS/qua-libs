"""Patch qualibration_libs' multiplexed batching to respect this cluster's FEM resource limits.

`get_qubits(node).batch()` / `get_qubit_pairs(node).batch()`, when `multiplexed=True`, normally put
*every* selected qubit (or qubit pair) into one single simultaneous QUA measurement group, with no
awareness of how many physical elements a single FEM board can demodulate at once. Beyond a handful
of simultaneous elements this fails at `qm.execute()` time with
`FailedToExecuteJobException: Failed to allocate resources` (confirmed on this cluster's MW-FEM,
con1 slot 3: 4 qubits multiplexed together succeeds, 6 does not).

Importing this module (done once, from quam_config/__init__.py) replaces the shared chunking
function used by every calibration node, so all of them automatically get safe multiplexed batch
sizes without being edited individually.

Not every node has the same per-qubit resource cost, though: a node that does real-time frequency
*and* real-time amplitude-scaled pulses on every multiplexed element (e.g.
03c_qubit_spectroscopy_vs_power) is much more resource-hungry than one that just plays a static
readout pulse, and can hit the ceiling at a lower qubit count. For those cases, wrap the
`get_qubits(node)` / `get_qubit_pairs(node)` call in a specific node with the
`max_simultaneous_multiplexed(n)` context manager below to temporarily tighten the cap just for
that call.
"""
from contextlib import contextmanager
from typing import List, Optional

import qualibration_libs.parameters.experiment as _experiment
from qualibration_libs.core import BatchableList

# Empirically confirmed safe limit: 4 qubits multiplexed together works, 6 does not.
# Qubit *pairs* touch roughly twice as many elements per item (two qubits' resonators + drives),
# so they get half the headroom.
MAX_SIMULTANEOUS_QUBITS = 4
MAX_SIMULTANEOUS_QUBIT_PAIRS = max(1, MAX_SIMULTANEOUS_QUBITS // 2)

_override_max_group_size: Optional[int] = None


@contextmanager
def max_simultaneous_multiplexed(n: int):
    """Temporarily tighten the safe multiplexed-batch size for one get_qubits()/get_qubit_pairs() call.

    Use this inside a specific node's create_qua_program when that node's QUA program is more
    resource-hungry per qubit than a plain readout sweep (e.g. it does real-time frequency *and*
    amplitude sweeps on every multiplexed element), so the shared default cap isn't safe for it.
    """
    global _override_max_group_size
    previous = _override_max_group_size
    _override_max_group_size = n
    try:
        yield
    finally:
        _override_max_group_size = previous


def _chunk_indices(n: int, chunk_size: int) -> List[List[int]]:
    return [list(range(i, min(i + chunk_size, n))) for i in range(0, n, chunk_size)]


def _make_batchable_list_from_multiplexed(items: List, multiplexed: bool) -> BatchableList:
    if not multiplexed:
        return BatchableList(items, [[i] for i in range(len(items))])

    if _override_max_group_size is not None:
        chunk_size = _override_max_group_size
    else:
        is_qubit_pairs = bool(items) and not hasattr(items[0], "resonator")
        chunk_size = MAX_SIMULTANEOUS_QUBIT_PAIRS if is_qubit_pairs else MAX_SIMULTANEOUS_QUBITS

    return BatchableList(items, _chunk_indices(len(items), chunk_size))


_experiment._make_batchable_list_from_multiplexed = _make_batchable_list_from_multiplexed
