# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import NodeParameters, QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam


# %% {Description}
description = """
        IQ BLOBS GEF
This sequence involves measuring the state of the resonator 'N' times, first after thermalization (with the qubit in
the |g> state), then after applying a x180 (pi) pulse to the qubit (bringing the qubit to the |e> state) and finally
after applying a x180 (pi) pulse plus an EF_180 pulse (bringing the qubit to the |f> state).
The resulting IQ blobs are displayed, and the data is processed to determine:
    - The centers of the |g>, |e> and |f> state IQ blobs.
    - The readout confusion matrix, which is also influenced by the x180 and EF_180 pulses fidelities.

Prerequisites:
    - Having calibrated the readout parameters (nodes 02a, 02b and/or 02c).
    - Having calibrated the qubit x180 pulse parameters.
    - Having calibrated the qubit EF_180 pulse parameters.

State update:
    - The gef centers positions: qubit.resonator.gef_centers
"""


class Parameters(NodeParameters):
    """Parameters for the IQ blobs GEF calibration node.

    Attributes:
        coupler_or_qubit_name (str): Name of the qubit or coupler to be calibrated.
        duration_in_ns (int): Duration of the pulse in nanoseconds.
        wait_time_in_ns (int): Waiting time between experiments in nanoseconds.
        simulation_duration_ns (int): Simulation duration in nanoseconds.
        amplitude_in_V (float): Amplitude of the waveform in Volts.
        use_waveform_report (bool): Whether to use the waveform report for debugging.
    """

    coupler_or_qubit_name: str = "q7"
    duration_in_ns: int = 1000
    wait_time_in_ns: int = 1000
    simulation_duration_ns: int = 10_000
    amplitude_in_V: float = 0.3
    use_waveform_report: bool = False


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="99_filter_plot",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
    machine = Quam.load()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""
    # You can get type hinting in your IDE by typing node.parameters
    node.parameters.coupler_or_qubit_name = "q6-7"
    pass


# Instantiate the QUAM class from the state file
# node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action()
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """
    Create the sweep axes and generate the QUA program from the pulse sequence and the
    node parameters.
    """

    try:
        flux_element = node.machine.qubits[node.parameters.coupler_or_qubit_name].z
    except:
        flux_element = node.machine.qubit_pairs[node.parameters.coupler_or_qubit_name].coupler

    with program() as node.namespace["qua_program"]:

        with infinite_loop_():
            flux_element.play(
                "const",
                amplitude_scale=node.parameters.amplitude_in_V / flux_element.operations["const"].amplitude,
                duration=node.parameters.duration_in_ns // 4,
            )
            flux_element.wait(node.parameters.wait_time_in_ns // 4)


# %% {Simulate}
@node.run_action()
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    debug = False
    if debug:
        from pathlib import Path
        from qm import generate_qua_script
        file_name = Path(__file__).stem
        with open(Path(__file__).parent.parent / f"{file_name}_debug.py", 'w') as sourceFile:
            print(generate_qua_script(node.namespace["qua_program"], config), file=sourceFile)
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %%
node.save()
# %%
