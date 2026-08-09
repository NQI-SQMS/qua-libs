from typing import Literal, Sequence

from qm.qua import *
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from quam_builder.architecture.superconducting.qubit_pair import AnyTransmonPair


def get_cr_elements(qp: AnyTransmonPair):
    qc = qp.qubit_control
    qt = qp.qubit_target
    cr = qp.cross_resonance
    cr_elems = [qc.xy.name, qt.xy.name, cr.name]
    return qc, qt, cr, cr_elems


def _qp_name(control: AnyTransmon, target: AnyTransmon) -> str:
    return f"q{control.name[1:]}-{target.name[1:]}"


def _get_qp(
    qc: AnyTransmon,
    qt: AnyTransmon,
    qubit_pairs: Sequence[AnyTransmonPair],
    *,
    reverse: bool | None = None,
):
    """Resolve the physical qubit pair for a logical CNOT from *qc* to *qt*.

    Parameters
    ----------
    reverse
        ``False`` — pair name ``{qc}-{qt}`` (native / direct orientation).
        ``True`` — pair name ``{qt}-{qc}`` (reverse orientation).
        ``None`` — direct if present in *qubit_pairs*, otherwise reverse.

    Returns
    -------
    qp, qc, qt, cr, cr_elems, use_reverse
        After resolution, *qc* / *qt* are the pair's native control and target.
        *use_reverse* is ``True`` when the reverse pair name was used (Hadamard sandwich needed).
    """
    qubit_pairs_by_name = {qp.name: qp for qp in qubit_pairs}
    direct_name = _qp_name(qc, qt)
    reverse_name = _qp_name(qt, qc)

    if reverse is None:
        use_reverse = direct_name not in qubit_pairs_by_name
        qp_name = reverse_name if use_reverse else direct_name
    else:
        use_reverse = reverse
        qp_name = reverse_name if reverse else direct_name

    if qp_name not in qubit_pairs_by_name:
        raise ValueError(f"No '{qp_name}' in active qubit pairs {list(qubit_pairs_by_name.keys())}")

    qp = qubit_pairs_by_name[qp_name]
    qc, qt, cr, cr_elems = get_cr_elements(qp)
    return qp, qc, qt, cr, cr_elems, use_reverse


def _cnot_direct(
    qp: AnyTransmonPair,
    qc: AnyTransmon,
    qt: AnyTransmon,
    cr_elems: list,
    qubit_pairs: Sequence[AnyTransmonPair],
    virtual_z: bool = False,
    cr_type: str = "direct+echo",
    wf_type: str = "flattop",
):
    """CNOT on a resolved direct-orientation pair (no Hadamard sandwich)."""
    # CNOT decomposition into [Z(-pi/2) x I] * [I x X(-pi/2)] * ZX(pi/2)
    qt.xy.play("-x90")  # X(-pi/2)
    if virtual_z:
        virtual_z_2pi(qc, qubit_pairs, 0.25)  # Z(-pi/2)
    else:
        # Z(theta) = X(-x90) Y(-theta) X(x90)
        qc.xy.play("x90")
        qc.xy.play("y90")
        qc.xy.play("-x90")
    align(*cr_elems)
    qp.apply("cr", cr_type=cr_type, wf_type=wf_type)
    align(*cr_elems)


def _cnot_reverse(
    qp: AnyTransmonPair,
    qc: AnyTransmon,
    qt: AnyTransmon,
    cr_elems: list,
    qubit_pairs: Sequence[AnyTransmonPair],
    virtual_z: bool = False,
    cr_type: str = "direct+echo",
    wf_type: str = "flattop",
):
    """CNOT on a resolved reverse-orientation pair (Hadamard sandwich)."""
    # Sandwich Hadamard on control and target to flip logical direction
    hadamard_decomposition_type = "YZ" if virtual_z else "XY"
    align(*cr_elems)
    hadamard(qc, qubit_pairs, decomposition_type=hadamard_decomposition_type)
    hadamard(qt, qubit_pairs, decomposition_type=hadamard_decomposition_type)

    # CNOT decomposition into [Z(-pi/2) x I] * [I x X(-pi/2)] * ZX(pi/2)
    qt.xy.play("-x90")  # X(-pi/2)
    if virtual_z:
        virtual_z_2pi(qc, qubit_pairs, 0.25)  # Z(-pi/2)
    else:
        # Z(theta) = X(-x90) Y(-theta) X(x90)
        qc.xy.play("x90")
        qc.xy.play("y90")
        qc.xy.play("-x90")
    align(*cr_elems)
    qp.apply("cr", cr_type=cr_type, wf_type=wf_type)

    align(*cr_elems)
    hadamard(qc, qubit_pairs, decomposition_type=hadamard_decomposition_type)
    hadamard(qt, qubit_pairs, decomposition_type=hadamard_decomposition_type)
    align(*cr_elems)


def cnot(
    qc: AnyTransmon,
    qt: AnyTransmon,
    qubit_pairs: Sequence[AnyTransmonPair],
    virtual_z: bool = False,
    cr_type: str = "direct+echo",
    wf_type: str = "flattop",
    forced_reverse: bool = False,
):
    """Auto-select direct vs reverse pair — equivalent to the original ``cnot`` implementation."""
    cnot_kwargs = dict(virtual_z=virtual_z, cr_type=cr_type, wf_type=wf_type)
    qp, qc, qt, _cr, cr_elems, use_reverse = _get_qp(qc, qt, qubit_pairs, reverse=True if forced_reverse else None)
    if use_reverse:
        _cnot_reverse(qp, qc, qt, cr_elems, qubit_pairs, **cnot_kwargs)
    else:
        _cnot_direct(qp, qc, qt, cr_elems, qubit_pairs, **cnot_kwargs)


def swap(qc: AnyTransmon, qt: AnyTransmon, qubit_pairs: Sequence[AnyTransmonPair], **kwargs):
    cnot(qc, qt, qubit_pairs, **kwargs)
    cnot(qt, qc, qubit_pairs, **kwargs)
    cnot(qc, qt, qubit_pairs, **kwargs)


def hadamard(q: AnyTransmon, qubit_pairs: Sequence[AnyTransmonPair], decomposition_type: Literal["XY", "YZ"] = "XY"):
    if decomposition_type == "XY":
        q.xy.play("y90")
        q.xy.play("x180")
    if decomposition_type == "YZ":
        q.xy.play("-y90")
        virtual_z_2pi(q, qubit_pairs, 0.5)


def virtual_z_2pi(q: AnyTransmon, qubit_pairs: Sequence[AnyTransmonPair], angle_2pi: float):
    elems = [q.xy.name]
    for qp in qubit_pairs:
        if qp.qubit_target.name == q.name:
            elems.append(qp.cross_resonance.name)
    for elem in elems:
        frame_rotation_2pi(-angle_2pi, elem)
