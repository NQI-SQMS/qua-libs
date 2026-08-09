"""Resolve the CZ avoided-crossing branch (|20> vs |02>) for a qubit pair.

In BOTH branches the CONTROL qubit is the one flux-moved, so the amplitude
conversion always uses qubit_control.freq_vs_flux_01_quad_term. Only the
anharmonicity term in the detuning and which qubit transiently reaches |2>
(and is therefore GEF-read / used for the conditioning + leakage) depend on
the branch:

    "20" (|11>-|20>, CONTROL -> |2>): detuning = f_control - f_target - A_control ; excited = control
    "02" (|11>-|02>, TARGET  -> |2>): detuning = f_control - f_target + A_target  ; excited = target

(anharmonicity A is stored as a positive magnitude in this state.)

Lives in the chevron_cz package because there is no shared `common` utils
folder; the other CZ nodes import it as
`from calibration_utils.chevron_cz import resolve_cz_branch`.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

CZBranch = Literal["20", "02"]
# What callers may pass: a concrete branch, or "auto" to pick the reachable one.
CZBranchArg = Literal["20", "02", "auto"]


@dataclass
class CZBranchSpec:
    """Branch-resolved roles/values for a qubit pair's CZ calibration."""

    excited_qubit: object  # qubit transiently in |2> (GEF readout + conditioning + leakage)
    other_qubit: object  # qubit that stays in {g, e} (Ramsey / phase tomography + g-e readout)
    detuning: float  # Hz; amount the control qubit must be detuned
    quad_term_control: float  # always qubit_control.freq_vs_flux_01_quad_term
    pulse_amplitude: float  # sqrt(-detuning / quad_term_control)
    branch: CZBranch


def _resolve_single_branch(qubit_pair, cz_branch: CZBranch) -> CZBranchSpec:
    """Resolve one explicit branch ("20" or "02").

    pulse_amplitude comes out NaN when that branch is not physically reachable, i.e.
    -detuning/quad < 0 -- the avoided crossing lies on the side the control qubit
    cannot reach by flux-tuning (flux only lowers the frequency from the sweet spot).
    """
    qc, qt = qubit_pair.qubit_control, qubit_pair.qubit_target
    f_c, f_t = qc.xy.RF_frequency, qt.xy.RF_frequency
    quad = qc.freq_vs_flux_01_quad_term  # control is flux-moved in both branches

    if cz_branch == "20":
        # |11>-|20>: control's 1->2 meets target's 0->1; control ends in |2>.
        detuning = f_c - f_t - qc.anharmonicity
        excited, other = qc, qt
    elif cz_branch == "02":
        # |11>-|02>: control's 0->1 meets target's 1->2; target ends in |2>.
        detuning = f_c - f_t + qt.anharmonicity
        excited, other = qt, qc
    else:
        raise ValueError(f"Invalid cz_branch: {cz_branch!r} (expected '20' or '02')")

    with np.errstate(invalid="ignore"):  # unreachable branch -> NaN amplitude, silently
        pulse_amplitude = float(np.sqrt(-detuning / quad))
    return CZBranchSpec(
        excited_qubit=excited,
        other_qubit=other,
        detuning=float(detuning),
        quad_term_control=float(quad),
        pulse_amplitude=pulse_amplitude,
        branch=cz_branch,
    )


def resolve_cz_branch(
    qubit_pair,
    cz_branch: Optional[CZBranchArg] = None,
    max_amplitude: Optional[float] = None,
) -> CZBranchSpec:
    """Resolve branch-dependent roles and the flux-pulse amplitude for a qubit pair.

    Args:
        qubit_pair: a FluxTunableTransmonPair with .qubit_control / .qubit_target.
        cz_branch: "20" for the |11>-|20> branch (control -> |2>), "02" for the
            |11>-|02> branch (target -> |2>). None (default) inherits the branch from
            qubit_pair.extras["cz_branch"] (written by the chevron / coupler zero-point
            node), falling back to "20". "auto" picks the physically reachable branch
            purely from the stored frequencies -- no measurement needed.
        max_amplitude: optional upper bound on |pulse_amplitude|, used only by the "auto"
            selection; a branch whose flux amplitude exceeds it is treated as unreachable
            (a real-but-clipping amplitude is as unusable as a NaN one). Ignored for an
            explicit "20"/"02" request.

    "auto" logic: try both branches; a branch is reachable when its pulse_amplitude is
    finite (-detuning/quad >= 0) and within max_amplitude. Among the reachable branches,
    return the one with the SMALLER flux amplitude (the near avoided crossing). Raise if
    neither is reachable (usually means the control/target roles are wrong for this pair).
    Pass "20"/"02" explicitly to force one.

    Returns:
        CZBranchSpec with the excited/other qubit objects and the detuning /
        pulse-amplitude needed by the downstream CZ nodes.
    """
    if cz_branch is None:
        cz_branch = qubit_pair.extras.get("cz_branch", "20")

    if cz_branch == "auto":
        reachable = []
        for br in ("20", "02"):
            spec = _resolve_single_branch(qubit_pair, br)
            if np.isfinite(spec.pulse_amplitude) and (
                max_amplitude is None or abs(spec.pulse_amplitude) <= max_amplitude
            ):
                reachable.append(spec)
        if not reachable:
            name = getattr(qubit_pair, "name", "?")
            raise ValueError(
                f"{name}: no reachable CZ branch -- both |20> and |02> give a NaN or "
                f"out-of-range flux amplitude (max_amplitude={max_amplitude}). "
                f"Check the control/target role assignment for this pair."
            )
        return min(reachable, key=lambda s: abs(s.pulse_amplitude))

    return _resolve_single_branch(qubit_pair, cz_branch)
