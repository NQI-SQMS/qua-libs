
# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.snz_b_over_a import decompose_t_phi_eff, snz_factory
from calibration_utils.snz_jazz2_n import (
    FitResults,
    Parameters,
    coerce_to_even,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from qm.qua import *
from qualang_tools.bakery import baking
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam


# %% {Initialisation}
description = """
        JAZZ2-N SNZ AMPLITUDE / t_phi_eff SCAN

This node calibrates the SNZ (Sudden Net-Zero) bipolar flux pulse using the
JAZZ2-N protocol (arXiv:2402.18926v3, Appendix I.1, Fig. 13(b)). The pulse
sequence is identical to 33c_JAZZ2-N.py except that every "Z" inside the
JAZZ2-N train is replaced by a BAKED SNZ waveform on the control qubit's
flux line, and the repetition N is fixed to a single user-chosen value (N
= 2k, paper convention):

    x90(control) & x90(target)
    SNZ(amp_scale, t_phi_eff)                              (initial Z)
    [ X_pi(control) & X_pi(target) -- SNZ(amp_scale, t_phi_eff) ] x (2N + 1)
    x90(control) & x90(target)
    measure(control), measure(target) -> p00 = (1 - state_c) * (1 - state_t)

With the X_pi refocusing pulses, the joint ground-state probability evolves
as

    P_|00>(amp, t_phi_eff, N) = (1 - cos((N + 1) * theta_CZ(amp, t_phi_eff))) / 2

up to leakage. The node sweeps N over the set {N_min, N_min + 2, ..., N_max}
(paper convention N = 2k) and the optimum search is performed on the
N-averaged map

    <P_|00>>_N(amp, t_phi_eff) = mean_N P_|00>(amp, t_phi_eff, N) .

Since each P_|00>(.; N) is reduced both by an imperfect conditional phase
AND by population transfer out of the computational subspace (leakage),
the averaged map carries the same signature: the (amp, t_phi_eff) point
that MAXIMISES <P_|00>>_N simultaneously calibrates the SNZ angle and
minimises leakage. Averaging over several N regularises the principal
peak via the Dirichlet kernel (see 33c_JAZZ2-N for the analytical form).
Setting N_min == N_max disables averaging and recovers the single-N
behaviour.

The sweep axes are (amplitude_scale, t_phi_eff, N); one SNZ waveform is
baked per (qubit_pair, t_phi_eff) value using ``snz_factory`` from
calibration_utils.snz_b_over_a. The amplitude scale is then applied at
runtime via QUA's ``amp_array``. The N axis is the innermost QUA loop:
for each (amp, t_phi_eff) grid point all N values are run in immediate
succession to minimise drift between samples that will be averaged.

Prerequisites:
    - Calibrated single-qubit gates (x90, x180) for both qubits.
    - Calibrated, state-discriminating readout for BOTH qubits.
    - A CZ macro (cz_SNZ recommended) providing nominal amplitude A and
      flat duration; the flat duration is split equally between the two
      SNZ lobes.

State update (cz_SNZ macro only):
    - qp.macros["cz_SNZ"].flux_pulse_qubit.amplitude (fitted optimum, V).
    - qp.macros["cz_SNZ"].flux_pulse_qubit.t_phi_eff (fitted optimum, ns).
The update is applied only when the qubit pair carries a cz_SNZ macro AND
the discrete argmax landed in the interior of the swept grid (so the
quadratic refinement is meaningful). The ``operation`` parameter controls
which macro seeds the baking nominal A / flat_length but does NOT change
where the fit results are written: only cz_SNZ is updated.
"""

node = QualibrationNode[Parameters, Quam](
    name="39b_JAZZ2_N_SNZ",
    description=description,
    parameters=Parameters(),
    machine = Quam.load()
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow local debug parameter overrides when running directly from IDE."""
    node.parameters.qubit_pairs = ["qA6-qD3"]
    node.parameters.operation = "cz_SNZ"
    node.parameters.N_min = 0
    node.parameters.N_max = 4
    node.parameters.amp_range = 0.05
    node.parameters.amp_step = 0.0002
    node.parameters.t_phi_eff_min = 5.0
    node.parameters.t_phi_eff_max = 6.5
    node.parameters.t_phi_eff_step = 0.1
    node.parameters.padding = 10
    node.parameters.num_shots = 20
    # node.parameters.load_data_id = None


# node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Bake SNZ waveforms and build the JAZZ2-N QUA program."""
    unit(coerce_to_integer=True)

    if not node.parameters.use_state_discrimination:
        raise RuntimeError(
            "JAZZ2-N SNZ reads the joint P_|00> of the qubit pair and therefore requires "
            "use_state_discrimination = True."
        )

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    # Coerce N_min / N_max to the nearest even integer (paper convention N = 2k), swap if reversed,
    # and log if coerced. The QUA inner loop steps by 2 (i.e. every valid N value is visited).
    n_min_req = int(node.parameters.N_min)
    n_max_req = int(node.parameters.N_max)
    n_min = coerce_to_even(n_min_req)
    n_max = coerce_to_even(n_max_req)
    if n_min > n_max:
        n_min, n_max = n_max, n_min
    if n_min != n_min_req:
        node.log(f"N_min {n_min_req} coerced to nearest even value: {n_min}.")
    if n_max != n_max_req:
        node.log(f"N_max {n_max_req} coerced to nearest even value: {n_max}.")
    n_values = np.arange(n_min, n_max + 1, 2, dtype=int)

    operation = node.parameters.operation
    padding = node.parameters.padding
    n_avg = node.parameters.num_shots

    t_phi_eff_values = np.arange(
        node.parameters.t_phi_eff_min,
        node.parameters.t_phi_eff_max,
        node.parameters.t_phi_eff_step,
    )
    amplitudes = np.arange(
        1 - node.parameters.amp_range,
        1 + node.parameters.amp_range,
        node.parameters.amp_step,
    )

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "amplitude": xr.DataArray(amplitudes, attrs={"long_name": "relative amplitude scale", "units": "a.u."}),
        "t_phi_eff": xr.DataArray(t_phi_eff_values, attrs={"long_name": "effective idle time", "units": "ns"}),
        "N": xr.DataArray(n_values, attrs={"long_name": "repetition N = 2k"}),
    }

    # ---- Bake one waveform per (qubit_pair, t_phi_eff). ----
    baked_config = node.machine.generate_config()
    baked_snz = {}
    for qp in qubit_pairs:
        A_nominal = qp.macros[operation].flux_pulse_qubit.amplitude
        if operation == "cz_SNZ":
            length = qp.macros[operation].flux_pulse_qubit.flat_length
        else:
            length = qp.macros[operation].flux_pulse_qubit.length
        baked_snz[qp.name] = []
        for j, tpe in enumerate(t_phi_eff_values):
            t_phi, ratio = decompose_t_phi_eff(float(tpe))
            wf = snz_factory(A_nominal, ratio, length, t_phi, padding)
            with baking(baked_config, padding_method="right") as b:
                b.add_op(f"snz_{qp.name}_{j}", qp.qubit_control.z.name, wf.tolist())
                b.play(f"snz_{qp.name}_{j}", qp.qubit_control.z.name)
            baked_snz[qp.name].append(b)

    node.namespace["baked_config"] = baked_config
    node.namespace["baked_snz"] = baked_snz

    num_tpe = len(t_phi_eff_values)

    def play_baked_snz(qp_name: str, channel_name: str, idx_var, amp_var):
        """Run the baked SNZ at runtime-selected ``idx_var`` with amplitude scaling ``amp_var``."""
        with switch_(idx_var):
            for j in range(num_tpe):
                with case_(j):
                    baked_snz[qp_name][j].run(amp_array=[(channel_name, amp_var)])

    num_N = len(n_values)

    # ---- QUA program ----
    with program() as node.namespace["qua_program"]:
        n = declare(int)
        a = declare(fixed)
        idx = declare(int)
        n_op = declare(int)
        count = declare(int)
        n_st = declare_stream()
        state_c = [declare(int) for _ in range(num_qubit_pairs)]
        state_t = [declare(int) for _ in range(num_qubit_pairs)]
        p00 = [declare(int) for _ in range(num_qubit_pairs)]
        p00_st = [declare_stream() for _ in range(num_qubit_pairs)]

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(a, amplitudes)):
                    with for_(idx, 0, idx < num_tpe, idx + 1):
                        # Innermost: sweep n_op = 2*N + 1 (i.e. paper-N = 2k) so all N values
                        # for a given (amp, t_phi_eff) point are taken back-to-back, minimising
                        # drift between samples that will be averaged in post-processing.
                        with for_(n_op, n_min, n_op <= n_max, n_op + 2):
                            for ii, qp in multiplexed_qubit_pairs.items():
                                qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                                qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                                qp.align()
                                reset_frame(qp.qubit_target.xy.name)
                                reset_frame(qp.qubit_control.xy.name)

                                # Boundary X_{pi/2} X_{pi/2} (both qubits).
                                qp.qubit_control.xy.play("x90")
                                qp.qubit_target.xy.play("x90")
                                qp.align()

                                # First SNZ (the "Z" preceding the (pi-Z)^(2N+1) pattern).
                                play_baked_snz(qp.name, qp.qubit_control.z.name, idx, a)
                                qp.align()

                                qp.qubit_control.xy.play("x180")
                                qp.qubit_target.xy.play("x180")
                                qp.align()

                                # First SNZ (the "Z" preceding the (pi-Z)^(2N+1) pattern).
                                play_baked_snz(qp.name, qp.qubit_control.z.name, idx, a)

                                # (X_pi X_pi, SNZ) x (2N + 1).
                                with if_(n_op > 1):
                                    with for_(count, 1, count <= n_op, count + 1):
                                        qp.qubit_control.xy.play("x180")
                                        qp.qubit_target.xy.play("x180")
                                        qp.align()
                                        play_baked_snz(qp.name, qp.qubit_control.z.name, idx, a)
                                        qp.qubit_control.xy.frame_rotation_2pi(0.5)
                                        qp.qubit_target.xy.frame_rotation_2pi(0.5)
                                        qp.qubit_control.xy.play("x180")
                                        qp.qubit_target.xy.play("x180")
                                        qp.align()
                                        play_baked_snz(qp.name, qp.qubit_control.z.name, idx, a)
                                        qp.qubit_control.xy.frame_rotation_2pi(-0.5)
                                        qp.qubit_target.xy.frame_rotation_2pi(-0.5)

                                qp.align()
                                # Boundary X_{pi/2} X_{pi/2} (both qubits).
                                qp.qubit_control.xy.play("x90")
                                qp.qubit_control.xy.play("x180")
                                qp.qubit_target.xy.play("x90")
                                qp.qubit_target.xy.play("x180")
                                qp.align()

                                qp.qubit_control.readout_state(state_c[ii])
                                qp.qubit_target.readout_state(state_t[ii])
                                assign(p00[ii], (1 - state_c[ii]) * (1 - state_t[ii]))
                                save(p00[ii], p00_st[ii])

            align()

        with stream_processing():
            n_st.save("n")
            for ii in range(num_qubit_pairs):
                (
                    p00_st[ii]
                    .buffer(num_N)
                    .buffer(num_tpe)
                    .buffer(len(amplitudes))
                    .average()
                    .save(f"p00{ii + 1}")
                )


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program using the baked config."""
    qmm = node.machine.connect()
    config = node.namespace["baked_config"]
    debug = False
    if debug:
        from pathlib import Path
        from qm import generate_qua_script
        file_name = Path(__file__).stem
        with open(Path(__file__).parent.parent / f"{file_name}_debug.py", 'w') as sourceFile:
            print(generate_qua_script(node.namespace["qua_program"], config), file=sourceFile)
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict(), "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the QUA program using the baked config and fetch raw data."""
    qmm = node.machine.connect()
    config = node.namespace["baked_config"]
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


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Process the raw data, find the (amp, t_phi_eff) optimum and set node outcomes."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {
        qp_name: ("successful" if fr.success else "failed") for qp_name, fr in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the JAZZ2-N SNZ P_|00> map with the fitted optimum per qubit pair."""
    fig = plot_raw_data_with_fit(node.results["ds_fit"], node.namespace["qubit_pairs"])
    plt.show()
    node.results["figures"] = {"jazz2_n_snz_scan": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the cz_SNZ macro's amplitude and t_phi_eff for each successful qubit pair.

    The update writes the absolute optimal amplitude (Volts) and the
    continuous-valued optimal t_phi_eff (ns) onto ``cz_SNZ``'s flux pulse.
    The SNZPulse class decomposes ``t_phi_eff`` into integer t_phi samples
    and B/A internally at config-generation time, so writing the float
    ``optimal_t_phi_eff`` is sufficient. Pairs without a ``cz_SNZ`` macro
    or with a failed fit are skipped (and the reason is logged).
    """
    fit_results = node.results["fit_results"]
    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            if node.outcomes[qp.name] == "failed":
                node.log(f"Skipping state update for {qp.name}: fit did not converge to an interior optimum.")
                continue
            if "cz_SNZ" not in qp.macros:
                node.log(f"Skipping state update for {qp.name}: no 'cz_SNZ' macro present on this pair.")
                continue
            snz_pulse = qp.macros["cz_SNZ"].flux_pulse_qubit
            snz_pulse.amplitude = float(fit_results[qp.name]["optimal_amplitude"])
            snz_pulse.t_phi_eff = float(fit_results[qp.name]["optimal_t_phi_eff"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the calibration results."""
    node.save()


# %%
