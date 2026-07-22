"""Utility helpers for declaring QUA variables sized by an arbitrary count.

Use ``declare_iq_variables`` when the number of readout channels differs from
the number of qubits registered in the machine -- for example when iterating
over qubit **pairs** rather than individual qubits.
"""

from typing import List, Tuple

from qm.qua import declare, declare_stream, fixed


def declare_iq_variables(
    count: int,
) -> Tuple[List, List, List, List]:
    """Declare *count* I/Q QUA variables and their corresponding streams.

    Returns ``(I, I_st, Q, Q_st)`` where each element is a list of length
    *count*.  This is intended as a drop-in replacement for the I/Q portion of
    ``machine.declare_qua_variables()`` when the caller needs arrays whose
    length matches the number of qubit pairs (or any other entity count) rather
    than the number of qubits.
    """
    I = [declare(fixed) for _ in range(count)]
    I_st = [declare_stream() for _ in range(count)]
    Q = [declare(fixed) for _ in range(count)]
    Q_st = [declare_stream() for _ in range(count)]
    return I, I_st, Q, Q_st
