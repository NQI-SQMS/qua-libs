# SPDX-License-Identifier: EUPL-1.2
# Copyright (C) 2026 Q.M Technologies Ltd. / Soon Teh
# Copyright (C) 2026 Q.M Technologies Ltd. / Hiroyuki Inoue
# Copyright (C) 2026 RIKEN / András Gunyhó
# Licensed under the EUPL v1.2.
# See: https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12

# %%
#!%load_ext autoreload
#!%autoreload 2
# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from tqdm import tqdm

from calibration_utils.cr_utils import *
from calibration_utils.data_process_utils import *
from calibration_utils.two_qubit_randomized_benchmarking import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from calibration_utils.two_qubit_randomized_benchmarking.sequence_tools import (
    pre_generate_sequence_interleaved,
)
from calibration_utils.common_utils.plotting_tools import patch_fig_info
from quam_config import Quam

# %% {Initialisation}
description = """
        TWO-QUBIT INTERLEAVED RANDOMIZED BENCHMARKING (CZ)
"""


class CZParameters(Parameters):
    operation: str = "cz_bipolar"
    """Name of the calibrated CZ macro to benchmark via qp.macros[operation].apply(). Default 'cz_bipolar'."""


def get_cz_elements(qp):
    """CZ analogue of get_cr_elements: list the two qubits' xy drives for align/reset_frame.

    The CZGate macro (qp.macros[operation].apply()) self-aligns its own flux/z line and applies
    its own virtual-Z corrections, so only the xy drives are listed here (a flux-tunable CZ pair
    has no .cross_resonance element, unlike the CR version)."""
    qc = qp.qubit_control
    qt = qp.qubit_target
    cz_elems = [qc.xy.name, qt.xy.name]
    return qc, qt, cz_elems


node = QualibrationNode[CZParameters, Quam](
    name="60c_cz_two_qubit_randomized_benchmarking_input_stream",
    description=description,
    parameters=CZParameters(),
    machine=Quam.load(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.multiplexed = True
    node.parameters.multiplexed = True  # same as 31z: all pairs in one batch, sequential inner loop
    # node.parameters.qubit_pairs = ["q40-41", "q42-43"]
    # node.parameters.qubit_pairs = ["q0-2"]
    node.parameters.qubit_pairs = ["q0-2", "q2-0"]
    # node.parameters.qubit_pairs = [
    #     "q0-2",
    #     "q2-0",
    #     # "q0-1",
    #     "q1-0",
    #     # "q1-3",
    #     "q3-1",
    #     "q8-10",
    #     "q10-8",
    #     # "q8-9",
    #     "q9-8",
    #     # "q9-11",
    #     "q11-9",
    #     # "q9-12",
    #     "q12-9",
    #     # "q12-14",
    #     "q14-12",
    #     # "q12-13",
    #     "q13-12",
    #     # "q13-15",
    #     "q15-13",
    #     # "q2-3",
    #     # "q3-2",
    #     # "q10-11",
    #     # "q11-10",
    #     # "q11-14",
    #     # "q14-11",
    #     # "q14-15",
    #     # "q15-14",
    # ]
    node.parameters.operation = "cz_bipolar"
    node.parameters.max_circuit_depth = 20
    node.parameters.num_intervals = 15
    # node.parameters.simulate = False

    # 299-303
    # node.parameters.load_data_id = 299
    # node.parameters.load_data_id = 78
    # node.parameters.load_data_id = 173
    # Dec
    # node.parameters.load_data_id = 163


# Instantiate the QUAM class from the state file
# node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)

    # reversing the qubit pairs
    if node.parameters.forced_reverse:
        assert node.parameters.qubit_pairs is not None, "Must specify qubit pairs when using forced_reverse"
        new_qubit_pairs = []
        for qp_name in node.parameters.qubit_pairs:
            qc_name, qt_name = qp_name.split("-")
            qc_name = "q" + qc_name[1:]
            qt_name = "q" + qt_name
            qp_name_reversed = f"q{qt_name[1:]}-{qc_name[1:]}"
            new_qubit_pairs.append(qp_name_reversed)
        node.parameters.qubit_pairs = new_qubit_pairs

    # Get the active qubits from the node and organize them by batches
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    num_of_sequences = node.parameters.num_random_sequences  # Number of random sequences
    n_avg = node.parameters.num_shots  # Number of averaging loops for each random sequence

    operation = node.parameters.operation

    max_circuit_depth = node.parameters.max_circuit_depth
    intervals = node.parameters.num_intervals
    if node.parameters.interval == "linear":
        depths = np.linspace(0, max_circuit_depth, intervals, dtype=int)
    elif node.parameters.interval == "logarithmic":
        depths = np.geomspace(1, max_circuit_depth + 1, intervals, dtype=int) - 1
    depths = np.unique(depths)
    print("Circuit depths:", depths)
    num_depths = len(depths)
    seed = node.parameters.seed  # Pseudo-random number generator seed
    strict_timing = node.parameters.use_strict_timing

    def play_sequence(sequence_list, start, length, qp):
        qc, qt, cz_elems = get_cz_elements(qp)

        i = declare(int)
        with for_(i, start, i < start + length, i + 1):
            align(*cz_elems)
            with switch_(sequence_list[i], unsafe=True):
                with case_(0):
                    # identity
                    # align(*cz_elems)
                    wait(4, *cz_elems)
                with case_(1):
                    qc.xy.play("x90")
                with case_(2):
                    qc.xy.play("-x90")
                with case_(3):
                    qc.xy.play("x180")
                with case_(4):
                    qc.xy.play("y90")
                with case_(5):
                    qc.xy.play("-y90")
                with case_(6):
                    qc.xy.play("y180")
                with case_(7):
                    qt.xy.play("x90")
                with case_(8):
                    qc.xy.play("x90")
                    qt.xy.play("x90")
                with case_(9):
                    qc.xy.play("-x90")
                    qt.xy.play("x90")
                with case_(10):
                    qc.xy.play("x180")
                    qt.xy.play("x90")
                with case_(11):
                    qc.xy.play("y90")
                    qt.xy.play("x90")
                with case_(12):
                    qc.xy.play("-y90")
                    qt.xy.play("x90")
                with case_(13):
                    qc.xy.play("y180")
                    qt.xy.play("x90")
                with case_(14):
                    qt.xy.play("-x90")
                with case_(15):
                    qc.xy.play("x90")
                    qt.xy.play("-x90")
                with case_(16):
                    qc.xy.play("-x90")
                    qt.xy.play("-x90")
                with case_(17):
                    qc.xy.play("x180")
                    qt.xy.play("-x90")
                with case_(18):
                    qc.xy.play("y90")
                    qt.xy.play("-x90")
                with case_(19):
                    qc.xy.play("-y90")
                    qt.xy.play("-x90")
                with case_(20):
                    qc.xy.play("y180")
                    qt.xy.play("-x90")
                with case_(21):
                    qt.xy.play("x180")
                with case_(22):
                    qc.xy.play("x90")
                    qt.xy.play("x180")
                with case_(23):
                    qc.xy.play("-x90")
                    qt.xy.play("x180")
                with case_(24):
                    qc.xy.play("x180")
                    qt.xy.play("x180")
                with case_(25):
                    qc.xy.play("y90")
                    qt.xy.play("x180")
                with case_(26):
                    qc.xy.play("-y90")
                    qt.xy.play("x180")
                with case_(27):
                    qc.xy.play("y180")
                    qt.xy.play("x180")
                with case_(28):
                    qt.xy.play("y90")
                with case_(29):
                    qc.xy.play("x90")
                    qt.xy.play("y90")
                with case_(30):
                    qc.xy.play("-x90")
                    qt.xy.play("y90")
                with case_(31):
                    qc.xy.play("x180")
                    qt.xy.play("y90")
                with case_(32):
                    qc.xy.play("y90")
                    qt.xy.play("y90")
                with case_(33):
                    qc.xy.play("-y90")
                    qt.xy.play("y90")
                with case_(34):
                    qc.xy.play("y180")
                    qt.xy.play("y90")
                with case_(35):
                    qt.xy.play("-y90")
                with case_(36):
                    qc.xy.play("x90")
                    qt.xy.play("-y90")
                with case_(37):
                    qc.xy.play("-x90")
                    qt.xy.play("-y90")
                with case_(38):
                    qc.xy.play("x180")
                    qt.xy.play("-y90")
                with case_(39):
                    qc.xy.play("y90")
                    qt.xy.play("-y90")
                with case_(40):
                    qc.xy.play("-y90")
                    qt.xy.play("-y90")
                with case_(41):
                    qc.xy.play("y180")
                    qt.xy.play("-y90")
                with case_(42):
                    qt.xy.play("y180")
                with case_(43):
                    qc.xy.play("x90")
                    qt.xy.play("y180")
                with case_(44):
                    qc.xy.play("-x90")
                    qt.xy.play("y180")
                with case_(45):
                    qc.xy.play("x180")
                    qt.xy.play("y180")
                with case_(46):
                    qc.xy.play("y90")
                    qt.xy.play("y180")
                with case_(47):
                    qc.xy.play("-y90")
                    qt.xy.play("y180")
                with case_(48):
                    qc.xy.play("y180")
                    qt.xy.play("y180")
                with case_(49):
                    # CNOT = (I (x) H_target) . CZ . (I (x) H_target), with the target Hadamard
                    # realized as the XY decomposition (y90; x180) -- CZ-safe (no cross_resonance
                    # frame ops). QUA plays first-to-last in time = right-to-left in the operator
                    # product, so H wraps the CZ on both sides. The played unitary equals the CX
                    # baked into the Clifford pkl, so inverse-finding stays valid (no regeneration).
                    qt.xy.play("y90")
                    qt.xy.play("x180")
                    align(*cz_elems)
                    qp.macros[operation].apply()
                    align(*cz_elems)
                    qt.xy.play("y90")
                    qt.xy.play("x180")

    def qubit_pair_reset(qp):
        qc = qp.qubit_control
        qt = qp.qubit_target
        # Reset the qubits to the ground state
        qc.reset(
            node.parameters.reset_type,
            node.parameters.simulate,
            log_callable=node.log,
        )
        qt.reset(
            node.parameters.reset_type,
            node.parameters.simulate,
            log_callable=node.log,
        )

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "nb_of_sequences": xr.DataArray(np.arange(num_of_sequences), attrs={"long_name": "Number of sequences"}),
        "depths": xr.DataArray(depths, attrs={"long_name": "Number of Clifford gates"}),
        "nb_of_shots": xr.DataArray(np.arange(n_avg), attrs={"long_name": "Number of shots"}),
    }

    # should be 16_000 variables distributed across everything, i.e. all the arrays and variables
    # using less to avoid accidental overflow
    node.namespace["max_sequence_length"] = 10000
    if "sequence_list" not in node.results:
        np.random.seed(seed=seed)
        interleaved_instruct = [("CNOT", "01")] if node.parameters.interleaved_CNOT else None
        sequence_list, len_list, full_list = pre_generate_sequence_interleaved(
            num_of_sequences,
            depths,
            interleaved_instruct=interleaved_instruct,
            exclude_cnot=node.parameters.exclude_CNOT_from_clifford,
        )
        sequence_len = len(sequence_list)
        print(f"Total decomposed Clifford sequence length: {sequence_len}")
        single_max = max(len_list)
        print(f"Max single sequence length: {single_max}")
        node.namespace["max_sequence_length"] -= len(len_list)
        safe_depth = int(max_circuit_depth / single_max * node.namespace["max_sequence_length"])
        assert single_max < node.namespace["max_sequence_length"], (
            f"Potentially exceed OPX memory limit: a maximum of {single_max} sequence used in one of the sequences, try depth <= {safe_depth}"
        )
        # assert sequence_len < node.namespace["max_sequence_length"], f"Exceed OPX memory limit: {sequence_len}"

        # node.results["full_sequence_list"] = full_list
        # save as ndarray
        node.results["sequence_list"] = np.array(sequence_list)
        node.results["len_list"] = np.array(len_list)
    else:
        print(" >> Using pre-loaded sequence list.")
        sequence_list = node.results["sequence_list"]
        len_list = node.results["len_list"]

    with program() as node.namespace["qua_program"]:
        state = [declare(int) for _ in range(num_qubit_pairs)]
        state_c = [declare(int) for _ in range(num_qubit_pairs)]
        state_t = [declare(int) for _ in range(num_qubit_pairs)]
        state_st = [declare_stream() for _ in range(num_qubit_pairs)]
        state_c_st = [declare_stream() for _ in range(num_qubit_pairs)]
        state_t_st = [declare_stream() for _ in range(num_qubit_pairs)]

        input_stream_end = declare_input_stream(bool, name="end", value=False)
        sequence_list_qua = declare_input_stream(int, size=node.namespace["max_sequence_length"], name="sequence")
        len_list_qua = declare_input_stream(int, size=len(len_list), name="len_list")
        start = declare(int, value=0)
        n = [declare(int, value=0) for _ in range(len(qubit_pairs.batch()))]
        seq_idx = declare(int, value=0)  # shared across batches, like 31z
        idx = declare(int, value=0)
        idx_st = declare_stream()

        pause()

        # Reset explicitly
        reset_global_phase()

        with while_(~input_stream_end):
            advance_input_stream(input_stream_end)
            advance_input_stream(sequence_list_qua)
            advance_input_stream(len_list_qua)

            # reset to 0 for each advance_input_stream
            assign(start, 0)

            for batch_num, multiplexed_qubit_pairs in enumerate(qubit_pairs.batch()):
                # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
                for qp in multiplexed_qubit_pairs.values():
                    node.machine.initialize_qpu(target=qp.qubit_control)
                    node.machine.initialize_qpu(target=qp.qubit_target)

                # QUA for_ loop over the random sequences
                with for_(seq_idx, 0, seq_idx < len(len_list), seq_idx + 1):
                    with if_(len_list_qua[seq_idx] > 0):
                        with for_(n[batch_num], 0, n[batch_num] < n_avg, n[batch_num] + 1):
                            for i, qp in multiplexed_qubit_pairs.items():
                                qc, qt, cz_elems = get_cz_elements(qp)
                                reset_frame(qc.xy.name, qt.xy.name)

                                # Initialize the qubits
                                qubit_pair_reset(qp)
                                align(*cz_elems)

                                # Manipulate the qubits
                                # The strict_timing ensures that the sequence will be played without gaps
                                if strict_timing:
                                    with strict_timing_():
                                        # Play the random sequence of desired depth
                                        play_sequence(
                                            sequence_list_qua,
                                            start,
                                            len_list_qua[seq_idx],
                                            qp,
                                        )
                                else:
                                    play_sequence(
                                        sequence_list_qua,
                                        start,
                                        len_list_qua[seq_idx],
                                        qp,
                                    )
                                align(qc.resonator.name, qt.resonator.name, *cz_elems)

                                # Readout the qubits
                                qc.readout_state(state_c[i])
                                qt.readout_state(state_t[i])
                                # state is 0 when |00>
                                # P(|00>) =/= P(|0X>)P(|X0>)
                                assign(
                                    state[i],
                                    Cast.to_int(~((state_c[i] == 0) & (state_t[i] == 0))),
                                )
                                save(state[i], state_st[i])
                                save(state_c[i], state_c_st[i])
                                save(state_t[i], state_t_st[i])
                                align(qc.resonator.name, qt.resonator.name, *cz_elems)

                        assign(start, start + len_list_qua[seq_idx])

                        # Save the counter for the progress bar (first batch only, like 31z)
                        if batch_num == 0:
                            save(idx, idx_st)
                            assign(idx, idx + 1)

        with stream_processing():
            idx_st.save("iteration")
            for i in range(num_qubit_pairs):
                state_st[i].buffer(n_avg).buffer(num_depths).buffer(num_of_sequences).save(f"state{i + 1}")
                state_c_st[i].buffer(n_avg).buffer(num_depths).buffer(num_of_sequences).save(f"state_c{i + 1}")
                state_t_st[i].buffer(n_avg).buffer(num_depths).buffer(num_of_sequences).save(f"state_t{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()

    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""

    def group_by_limit(sequence_list, len_list, limit, sequence_padding=None, len_padding=None):
        """
        Returns grouped sequence_list and len_list such that sum of lengths in each group <= limit.
        if padding is chosen, sequence_list is padded to length limit with sequence_padding,
        and len_list is padded to original length with len_padding.
        """
        assert all(0 < l <= limit for l in len_list), "Each length must be in (0, limit]."
        assert len(sequence_list) == sum(len_list), "sequence_list must match sum(len_list)."

        if isinstance(sequence_list, np.ndarray):
            sequence_list = sequence_list.tolist()
        if isinstance(len_list, np.ndarray):
            len_list = len_list.tolist()
        grouped_len_list = []
        grouped_sequence_list = []

        i = 0  # index in len_list
        pos = 0  # index in sequence_list
        n = len(len_list)

        while i < n:
            cur_lens = []
            cur_seq = []
            cur_sum = 0

            while i < n and cur_sum + len_list[i] <= limit:
                l = len_list[i]
                cur_lens.append(l)
                cur_seq.extend(sequence_list[pos : pos + l])
                pos += l
                cur_sum += l
                i += 1

            grouped_len_list.append(cur_lens)
            grouped_sequence_list.append(cur_seq)

        # pad to satisfy fixed size input streams
        if sequence_padding is not None:
            grouped_sequence_list = [gs + [sequence_padding] * (limit - len(gs)) for gs in grouped_sequence_list]
        if len_padding is not None:
            len_len_list = len(len_list)
            grouped_len_list = [gl + [len_padding] * (len_len_list - len(gl)) for gl in grouped_len_list]

        return grouped_sequence_list, grouped_len_list

    grouped_sequence_list, grouped_len_list = group_by_limit(
        node.results["sequence_list"],
        node.results["len_list"],
        node.namespace["max_sequence_length"],
        # the padding are needed because of the fixed size input streams
        sequence_padding=-1,  # default should ignore this
        len_padding=0,  # minimum length need to be at least 1, but should be ignored anyhow
    )

    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()

    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])

        for i, (sl_chunk, ll_chunk) in tqdm(
            enumerate(zip(grouped_sequence_list, grouped_len_list)),
            desc="Uploading sequences",
            total=len(grouped_len_list),
        ):
            if i == len(grouped_len_list) - 1:
                job.push_to_input_stream("end", True)
            else:
                job.push_to_input_stream("end", False)
            job.push_to_input_stream("sequence", sl_chunk)
            job.push_to_input_stream("len_list", ll_chunk)
        job.resume()
        print("Job resumed")

        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher["iteration"],
                node.parameters.num_random_sequences * len(node.namespace["sweep_axes"]["depths"]),
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())

    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_proc"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_proc"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_raw, fig_fit = plot_raw_data_with_fit(
        node.results["ds_proc"], node.namespace["qubit_pairs"], node.results["ds_fit"]
    )
    patch_fig_info(node)
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "2QRB": fig_fit,
        "all_state_raw": fig_raw,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for i, qp in enumerate(node.namespace["qubit_pairs"]):
            if node.outcomes[qp.name] == "failed":
                continue

            if node.parameters.interleaved_CNOT:
                extras_name = "2QRB_p_interleaved_CNOT"
            else:
                extras_name = "2QRB_p"
            qp.extras[f"{extras_name}"] = node.results["fit_results"][qp.name]["p"]
            qp.extras[f"{extras_name}_sem"] = node.results["fit_results"][qp.name]["p_sem"]
            qp.extras["epg"] = node.results["fit_results"][qp.name]["error_per_gate"]
            qp.extras["epg_eval_method"] = node.results["fit_results"][qp.name]["epg_eval_method"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
