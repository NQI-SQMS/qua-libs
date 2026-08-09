"""Qubit spectroscopy versus drive power.

Drive a saturation pulse across a detuning x drive-power sweep, fit the
power-broadened qubit line per power, pick the optimal power just below the
power-broadening onset, and fit the 2-photon EF line for the anharmonicity.

The saturation-pulse DURATION across the amplitude sweep is set by ``duration_mode``:

  * ``fixed``          – one duration for every amplitude (``operation_len_in_ns``
                         if set, else the stored operation length). The classic,
                         simplest scheme, and the default.
  * ``constant_angle`` – the duration is scaled inversely with the drive amplitude
                         so the coherent rotation angle Omega*t is held constant
                         across the amplitude axis: t(a) = base_len_ns * a_max / a,
                         clamped to ``max_len_ns``. The high-amplitude end uses the
                         short ``base_len_ns`` (never lands on an accidental 2*pi
                         nutation node), the low-amplitude end is long but has small
                         Omega so it is still safe. This removes the coherent
                         Rabi-nutation fringing the fixed scheme can show at high
                         power when the duration happens to be a pi-pulse multiple.

WHY NO T1/T2/pi PARAMETERS: this is one of the earliest qubit calibrations, where
the pi-pulse, T1 and T2 are all still unknown — so the duration is NEVER derived
from them. Both modes are controlled by direct nanosecond lengths and amplitude
ratios only. A self-contained fringe diagnostic (peak height vs power) flags
coherent nutation from the data shape alone and is plotted.

State update (only for qubits whose fit passes the gates):
    - qubit.f_01 & qubit.xy.RF_frequency
    - qubit.xy drive power at the optimal power (full_scale_power_dbm on the 1 dB grid + amplitude)
    - (optional) qubit.anharmonicity (from the 2-photon EF fit)
"""

# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from qualibrate.core.utils.node.record_state_update import record_state_update, update_machine_attribute
from quam_config import Quam
from calibration_utils.qubit_spectroscopy_vs_amplitude import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    detect_amplitude_fringe,
    log_fringe_results,
    plot_raw_data_with_fit,
    plot_raw_data_amp_linear,
    plot_raw_data_no_fit,
    plot_peak_height_vs_power,
)
from quam_builder.tools.power_tools import calculate_voltage_scaling_factor
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates


# MW-FEM full-scale power grid: the installed quam_builder set_output_power rejects the 1 dB grid,
# so we BYPASS it and set full_scale_power_dbm + operations[op].amplitude directly.
_FULL_SCALE_DBM_MIN, _FULL_SCALE_DBM_MAX = -11, 16


def _on_grid_full_scale_dbm(power_in_dbm: float) -> int:
    """Smallest 1 dB-grid full-scale power >= ``power_in_dbm``, clamped to the allowed range."""
    return int(min(max(int(np.ceil(power_in_dbm)), _FULL_SCALE_DBM_MIN), _FULL_SCALE_DBM_MAX))


def _fs_and_amp(power_in_dbm: float, max_amp: float):
    """Full-scale (1 dB grid) + waveform amplitude realising ``power_in_dbm`` with amplitude <= ``max_amp``."""
    fs = _on_grid_full_scale_dbm(power_in_dbm - 20.0 * np.log10(max_amp))
    amp = 10 ** ((power_in_dbm - fs) / 20.0)
    return fs, amp


def _amp_holder_op(xy, op_name: str) -> str:
    """Operation name whose ``.amplitude`` literally backs ``operations[op_name]`` (resolve gate aliases)."""
    try:
        ref = xy.operations[op_name].get_reference("amplitude")
        parts = str(ref).strip("#/").split("/")
        if "operations" in parts:
            return parts[parts.index("operations") + 1]
    except Exception:
        pass
    return op_name


def _saturation_durations_cc(amps: np.ndarray, node: QualibrationNode):
    """constant_angle clock-cycle durations per amplitude: t(a)=base_len_ns*a_max/a, clamped to max_len_ns.

    Returned in clock cycles (ns // 4). Floor at base_cc (a_max gives the shortest pulse) and cap at max_cc.
    """
    amp_max = float(np.max(amps))
    base_cc = max(int(round(node.parameters.base_len_ns / 4)), 4)
    max_cc = max(int(round(node.parameters.max_len_ns / 4)), base_cc)
    durs = np.round(base_cc * amp_max / np.asarray(amps, dtype=float)).astype(int)
    return np.clip(durs, base_cc, max_cc)


# %% {Node initialisation}
description = """
        QUBIT SPECTROSCOPY VERSUS DRIVE POWER
A saturation pulse drives the qubit across a detuning sweep, repeated for a range
of drive powers. duration_mode='fixed' (default) uses one duration for all amplitudes;
duration_mode='constant_angle' scales the duration inversely with amplitude so the
coherent rotation angle stays constant across the sweep (no pi-pulse-multiple fringing).
A fringe diagnostic (peak height vs power) is plotted.

State update:
    - qubit.f_01 & qubit.xy.RF_frequency
    - qubit.xy drive power at the optimal power (full_scale_power_dbm on the 1 dB grid + amplitude)
    - qubit.anharmonicity (optional, from the 2-photon EF fit)
"""


node = QualibrationNode[Parameters, Quam](
    name="08b_qubit_spectroscopy_vs_power",
    description=description,
    parameters=Parameters(),
    machine = Quam.load()
)

@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging."""
    # node.parameters.qubits = ["q1"]
    # node.parameters.duration_mode = "constant_angle"
    pass


# # Instantiate the QUAM class from the state file
# node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program (duration set per duration_mode)."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    operation = node.parameters.operation
    duration_mode = node.parameters.duration_mode

    # Set the qubit-drive power baseline to the top of the sweep; reverted at the end of the node.
    node.namespace["tracked_xy"] = []
    fs_base, amp_base = _fs_and_amp(node.parameters.max_power_dbm, node.parameters.max_amp)
    for qubit in qubits:
        with tracked_updates(qubit.xy, auto_revert=False, dont_assign_to_none=True) as xy:
            xy.opx_output.full_scale_power_dbm = fs_base
            xy.operations[operation].amplitude = amp_base
            node.namespace["tracked_xy"].append(xy)

    n_avg = node.parameters.num_shots
    # Drive-amplitude sweep (pre-factor of the saturation amplitude) - geometric in V == linear in dBm
    amp_min = calculate_voltage_scaling_factor(node.parameters.max_power_dbm, node.parameters.min_power_dbm)
    amps = np.geomspace(amp_min, 1, node.parameters.num_power_points)
    power_dbm = np.linspace(
        node.parameters.min_power_dbm,
        node.parameters.max_power_dbm,
        node.parameters.num_power_points,
    )
    # constant_angle: per-amplitude clock-cycle durations (Omega*t held constant across the sweep)
    durs_cc = _saturation_durations_cc(amps, node) if duration_mode == "constant_angle" else None
    if durs_cc is not None:
        node.namespace["durations_ns"] = (durs_cc * 4).tolist()  # provenance for the saved data

    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, +span // 2, step)

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "drive frequency", "units": "Hz"}),
        "power": xr.DataArray(power_dbm, attrs={"long_name": "drive power", "units": "dBm"}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        df = declare(int)   # drive frequency detuning
        a = declare(fixed)  # drive amplitude pre-factor
        dur = declare(int)  # saturation duration in clock cycles (constant_angle mode)

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency)

                    if duration_mode == "constant_angle":
                        # amplitude AND its paired (inverse-scaled) duration sweep together
                        with for_each_((a, dur), (amps.tolist(), durs_cc.tolist())):
                            for i, qubit in multiplexed_qubits.items():
                                qubit.xy.play(operation, amplitude_scale=a, duration=dur)
                            align()
                            for i, qubit in multiplexed_qubits.items():
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                qubit.resonator.wait(node.machine.depletion_time * u.ns)
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])
                            align()
                    else:  # "fixed": one duration for every amplitude
                        with for_each_(a, amps):
                            for i, qubit in multiplexed_qubits.items():
                                duration = (
                                    node.parameters.operation_len_in_ns
                                    if node.parameters.operation_len_in_ns is not None
                                    else qubit.xy.operations[operation].length
                                )
                                qubit.xy.play(operation, amplitude_scale=a, duration=duration // 4)
                            align()
                            for i, qubit in multiplexed_qubits.items():
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                qubit.resonator.wait(node.machine.depletion_time * u.ns)
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])
                            align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    debug = False
    if debug:
        from pathlib import Path
        from qm import generate_qua_script
        file_name = Path(__file__).stem
        with open(Path(__file__).parent.parent / f"{file_name}_debug.py", 'w') as sourceFile:
            print(generate_qua_script(node.namespace["qua_program"], config), file=sourceFile)
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the QUA program and fetch the raw dataset."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    # load_from_id rebuilds node.parameters from the SAVED run, which would revert any
    # re-fit / analysis knob the user changed to re-analyse loaded data. Snapshot the
    # user's current values for those knobs (+ load_data_id) and restore them after load.
    _refit_keep = {k: getattr(node.parameters, k) for k in (
        "load_data_id",
        "update_anharmonicity",
        "update_pulses_amplitude",
        "auto_tune",
        "fit_ef_transition",
        "power_below_saturation_db",
        "ge_low_power_first",
    )}
    node.load_from_id(node.parameters.load_data_id)
    for _k, _v in _refit_keep.items():
        setattr(node.parameters, _k, _v)
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Per-power peak fit + power-broadening optimal-power selection + fringe diagnostic."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    log_fitted_results(node.results["fit_results"], log_callable=node.log)

    # Fringe diagnostic (data-shape only; flags coherent-nutation fringing of the fixed scheme)
    node.results["fringe"] = detect_amplitude_fringe(node.results["ds_raw"], node)
    log_fringe_results(node.results["fringe"], log_callable=node.log)

    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Four figures: raw (no fit), power (dBm) with fit, linear drive-amplitude (V) with fit, and the fringe diagnostic."""
    fig_raw = plot_raw_data_no_fit(node.results["ds_raw"], node.namespace["qubits"])
    fig_power = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    fig_amp = plot_raw_data_amp_linear(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    fig_fringe = plot_peak_height_vs_power(node.results["ds_raw"], node.namespace["qubits"], node.results["fringe"])
    plt.show()
    node.results["figures"] = {
        "power_dbm_raw": fig_raw,
        "power_dbm": fig_power,
        "amplitude_linear": fig_amp,
        "fringe_diagnostic": fig_fringe,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Write the fitted GE frequency, optimal drive power (1 dB grid) and measured anharmonicity to QUAM.

    The adaptive duration only changes how the sweep is acquired, not how the optimal power /
    frequency / anharmonicity are written back.
    """
    for tracked_xy in node.namespace.get("tracked_xy", []):
        tracked_xy.revert_changes()

    op = node.parameters.operation
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            fit_result = node.results["fit_results"][q.name]
            q.f_01 = fit_result["frequency"]
            q.xy.RF_frequency = fit_result["frequency"]

            ref_fs = q.xy.opx_output.get_reference(attr="full_scale_power_dbm")
            old_fs = int(q.xy.opx_output.full_scale_power_dbm)
            new_fs, amp_opt = _fs_and_amp(fit_result["optimal_power"], node.parameters.max_amp)
            if op == "saturation":
                q.xy.operations["saturation"].amplitude = amp_opt
                if node.parameters.operation_len_in_ns is not None:
                    q.xy.operations["saturation"].length = int(node.parameters.operation_len_in_ns)
            else:
                holder = _amp_holder_op(q.xy, op)
                q.xy.operations[holder].amplitude = amp_opt
                scale = 10 ** ((old_fs - new_fs) / 20.0)
                for companion in ("x90", "saturation"):
                    h = _amp_holder_op(q.xy, companion)
                    if h != holder:
                        q.xy.operations[h].amplitude = float(q.xy.operations[h].amplitude * scale)
            if new_fs != old_fs:
                attr_key = update_machine_attribute(node.machine, ref_fs, old_fs)
                record_state_update(node, ref_fs, str(attr_key), old_fs, new_fs)

            if node.parameters.update_pulses_amplitude and np.isfinite(fit_result["saturation_amp"]):
                q.xy.operations["saturation"].amplitude = fit_result["saturation_amp"]

            if (node.parameters.update_anharmonicity and fit_result["ef_success"]
                    and np.isfinite(fit_result["anharmonicity_fitted"])):
                # The EF fit measures the MAGNITUDE; preserve the state's sign convention
                # (it varies across states — some store anharmonicity negative). Default to
                # positive when there is no prior value.
                stored = fit_result.get("anharmonicity_stored")
                sign = -1.0 if (stored is not None and np.isfinite(stored) and stored < 0) else 1.0
                q.anharmonicity = sign * float(abs(fit_result["anharmonicity_fitted"]))


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save node results."""
    node.save()
