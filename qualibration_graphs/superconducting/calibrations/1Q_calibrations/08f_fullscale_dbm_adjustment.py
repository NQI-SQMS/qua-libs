"""Full-scale dBm adjustment for readout resonators and qubit x180/x90 pulses.

A pure state-update node (no QUA program) that re-targets the OPX-output
``full_scale_power_dbm`` for the resonator readout outputs and for the qubit XY
drive outputs, while compensating the pulse amplitudes so the physically emitted
power is unchanged -- only the DAC head-room (and hence the noise/SNR trade-off)
changes.

Why it helps:
    - Resonator: lowering the full-scale (e.g. -11 dBm) shrinks the DAC full-scale
      voltage, so the readout pulse uses a larger fraction of full-scale at the same
      physical power -> better effective resolution / less readout noise.
    - Qubit: raising the full-scale (e.g. 16 dBm) gives more head-room to the x180
      drive, allowing shorter/stronger pi pulses without clipping.

Shared physical ports:
    ``full_scale_power_dbm`` lives on the OPX output *port*, not the qubit. When
    several qubits are wired to the same physical port (e.g. a multiplexed readout
    feedline), they must all end up with the *same* full-scale value. Qubits are
    grouped by the physical port their channel resolves to (``opx_output.get_reference()``),
    and for each group the node solves for the lowest full-scale (starting at the
    requested target, stepping up in 3 dB increments) that keeps every member's
    pulse amplitude within [-1, 1]. This is applied identically to resonator and XY
    outputs, so it also works (as a no-op) on chips where XY drives are not shared.

Optional pi-pulse resize:
    If ``pi_pulse_length_ns`` is set, the x180 (and x90) pulses are resized to that
    length. ``sigma`` is recomputed as ``length / 5`` (matching this repo's DRAG
    convention, see ``BaseTransmon.sigma_time_factor``), and the amplitude is
    rescaled inversely proportional to the length change so the pulse area
    (rotation angle) -- and hence the pi rotation -- is preserved.

State update:
    - Resonator: readout amplitude and opx_output.full_scale_power_dbm
    - Qubit: x180/x90 amplitude and opx_output.full_scale_power_dbm, and
      (optionally) x180/x90 length and sigma
"""

# %% {Imports}
from typing import Optional

import numpy as np

from qualibrate import NodeParameters, QualibrationNode
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from quam_builder.tools.power_tools import calculate_voltage_scaling_factor

from quam_config import Quam

_FULL_SCALE_STEP_DBM = 3
# MW-FEM full-scale power grid used throughout this repo (see quam_builder.tools.power_tools).
_ALLOWED_FULL_SCALE_DBM = np.arange(-11, 17, _FULL_SCALE_STEP_DBM)


class NodeSpecificParameters(RunnableParameters):
    resonator_full_scale_power_dbm: float = -11
    """Target full-scale power (dBm) for resonator readout OPX outputs."""
    qubit_full_scale_power_dbm: float = 16
    """Target full-scale power (dBm) for qubit XY OPX outputs."""
    pi_pulse_length_ns: Optional[int] = None
    """If set, resize x180/x90 to this length (ns). sigma is recomputed as length / 5,
    and amplitude is rescaled inversely proportional to length to preserve the pulse area."""


class Parameters(NodeParameters, CommonNodeParameters, NodeSpecificParameters):
    pass


# %% {Description}
description = """
        FULLSCALE DBM ADJUSTMENT
Adjust the full-scale power of the resonator and qubit XY OPX outputs while keeping
the emitted power constant (amplitudes are rescaled). Qubits sharing a physical output
port (e.g. a multiplexed readout feedline) are grouped so that the same full-scale value
is safely applied to all of them. Optionally resize the x180/x90 pi pulses to a new
length, rescaling amplitude inversely to preserve the pulse area. No QUA program runs.

State update:
    - Resonator: readout amplitude and opx_output.full_scale_power_dbm
    - Qubit: x180/x90 amplitude and opx_output.full_scale_power_dbm, and
      (optionally) x180/x90 length and sigma
"""


node = QualibrationNode[Parameters, Quam](
    name="08f_fullscale_dbm_adjustment",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging."""
    pass


# %% {Create_QUA_program} (no QUA program is run for this node)
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    pass


@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    pass


@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    pass


@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    pass


@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    pass


@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    pass


def _group_by_opx_output(qubits, channel_attr: str) -> list:
    """Group qubits by the physical OPX output port of the given channel.

    ``full_scale_power_dbm`` lives on the port, so qubits whose channel (e.g.
    ``resonator`` or ``xy``) resolves to the same physical port must be re-targeted
    together.
    """
    groups: dict = {}
    for q in qubits:
        channel = getattr(q, channel_attr)
        key = channel.opx_output.get_reference()
        groups.setdefault(key, []).append(q)
    return list(groups.values())


def _resolve_group_full_scale_dbm(group, channel_attr: str, operation_name: str, target_dbm: float) -> float:
    """Lowest full_scale_power_dbm (>= target_dbm, in 3 dB steps) that keeps every
    group member's ``operation_name`` pulse amplitude within [-1, 1] once re-targeted.
    """
    phys_powers_dbm = []
    for q in group:
        channel = getattr(q, channel_attr)
        full_scale = channel.opx_output.full_scale_power_dbm
        amplitude = channel.operations[operation_name].amplitude
        phys_powers_dbm.append(full_scale + 20 * np.log10(abs(amplitude)))

    fs = target_dbm
    while any(
        calculate_voltage_scaling_factor(fixed_power_dBm=fs, target_power_dBm=p) > 1 for p in phys_powers_dbm
    ):
        fs += _FULL_SCALE_STEP_DBM

    if fs not in _ALLOWED_FULL_SCALE_DBM:
        raise ValueError(
            f"No full_scale_power_dbm in {_ALLOWED_FULL_SCALE_DBM.tolist()} dBm accommodates the "
            f"'{operation_name}' pulse for qubits {[q.name for q in group]} (would require >= {fs} dBm)."
        )
    return float(fs)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Re-target OPX full-scale power for readout/XY outputs (grouped by shared physical
    port) and, if requested, resize the x180/x90 pi pulses -- rescaling amplitudes so the
    physically emitted power/rotation is preserved.
    """
    machine = node.machine
    resonator_target = node.parameters.resonator_full_scale_power_dbm
    qubit_target = node.parameters.qubit_full_scale_power_dbm
    pi_length = node.parameters.pi_pulse_length_ns
    qubits = list(machine.qubits.values())

    for target_dbm in (resonator_target, qubit_target):
        if target_dbm not in _ALLOWED_FULL_SCALE_DBM:
            raise ValueError(
                f"full_scale_power_dbm targets must be one of {_ALLOWED_FULL_SCALE_DBM.tolist()} dBm "
                f"(the search steps up from the target in 3 dB increments), got {target_dbm}."
            )

    with node.record_state_updates():
        # --- Resonator readout outputs: group qubits sharing a (possibly multiplexed) feedline ---
        for group in _group_by_opx_output(qubits, "resonator"):
            old_fs = group[0].resonator.opx_output.full_scale_power_dbm
            fs = _resolve_group_full_scale_dbm(group, "resonator", "readout", resonator_target)
            rescaling = calculate_voltage_scaling_factor(fixed_power_dBm=fs, target_power_dBm=old_fs)
            for q in group:
                readout_op = q.resonator.operations["readout"]
                readout_op.amplitude *= rescaling
                assert readout_op.amplitude <= 1
            for q in group:
                q.resonator.opx_output.full_scale_power_dbm = fs

        # --- Qubit XY drive outputs (x180/x90): same shared-port logic, generalized ---
        for group in _group_by_opx_output(qubits, "xy"):
            old_fs = group[0].xy.opx_output.full_scale_power_dbm
            fs = _resolve_group_full_scale_dbm(group, "xy", "x180", qubit_target)
            rescaling = calculate_voltage_scaling_factor(fixed_power_dBm=fs, target_power_dBm=old_fs)
            for q in group:
                x180 = q.xy.operations["x180"]
                x180.amplitude *= rescaling
                assert x180.amplitude <= 1
                if "x90" in q.xy.operations:
                    x90 = q.xy.operations["x90"]
                    x90.amplitude *= rescaling
                    assert x90.amplitude <= 1
            for q in group:
                q.xy.opx_output.full_scale_power_dbm = fs

        # --- Optional pi-pulse resize: length/sigma/amplitude, preserving the pulse area ---
        if pi_length is not None:
            new_length = int(pi_length)
            if new_length % 4 != 0:
                raise ValueError(f"pi_pulse_length_ns must be a multiple of 4 ns, got {new_length}.")
            new_sigma = max(4, new_length // 5)

            for q in qubits:
                x180 = q.xy.operations["x180"]
                length_ratio = x180.length / new_length
                x180.length = new_length
                x180.sigma = new_sigma
                x180.amplitude *= length_ratio
                assert x180.amplitude <= 1

                if "x90" in q.xy.operations:
                    x90 = q.xy.operations["x90"]
                    try:
                        x90.length = new_length
                        x90.sigma = new_sigma
                    except (ValueError, KeyError, AttributeError):
                        pass  # length/sigma are QUAM references to x180 and propagate automatically
                    x90.amplitude *= length_ratio
                    assert x90.amplitude <= 1


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
