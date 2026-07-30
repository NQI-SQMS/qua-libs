"""Wiring example: OPX+ + Octave with one qubit, one cavity (alice + bob), and sideband drives.

RF output pin assignment (Octave oct1):
  RF1 → readout resonator (shared, FDM across qubits)
  RF2 → sideband drives    (f0g1, shared across pairs via FDM; triggered)
  RF3 → qubit XY drives    (shared, FDM across qubits; triggered)
  RF4 → cavity mode drives (alice + bob shared port, FDM; triggered)
  RF5 → unused (set always_off by default)

The sideband and cavity channels share their respective Octave RF outputs across
all modes/pairs via frequency-division multiplexing, matching the lab convention
used in DR3-Run011.
"""
import matplotlib.pyplot as plt
from qualang_tools.wirer.wirer.channel_specs import octave_spec
from qualang_tools.wirer import Instruments, allocate_wiring, visualize

from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.builder.qop_connectivity.cavity_connectivity import CavityConnectivity
from quam_builder.builder.superconducting import build_quam
from quam_config import Quam

########################################################################################################################
# %%                                              Define static parameters
########################################################################################################################
host_ip = "127.0.0.1"        # QOP IP address
port = None                   # QOP Port
cluster_name = "Cluster_1"   # Name of the cluster
calibration_db_path = None    # "/path/to/some/config/folder"

########################################################################################################################
# %%                                      Define the available instrument setup
########################################################################################################################
instruments = Instruments()
instruments.add_opx_plus(controllers=[1])
instruments.add_octave(indices=1)

########################################################################################################################
# %%                           Define qubits, cavities, and qubit-cavity pair IDs
########################################################################################################################
qubits = [1]

# cavity_id + mode names uniquely identify each mode; pair IDs link each qubit to each mode
CAVITY_ID = "c1"
CAVITY_MODES = ["alice", "bob"]
# pair_ids format: "{qubit_name}_{mode_name}"  (qubit_name = "q{index}")
SIDEBAND_PAIR_IDS = [f"q{q}_{mode}" for q in qubits for mode in CAVITY_MODES]

########################################################################################################################
# %%                                 Define any custom/hardcoded channel addresses
########################################################################################################################
# RF1: multiplexed readout for all qubits
q_res_ch    = octave_spec(index=1, rf_out=1, rf_in=1)
# RF2: sideband drive (f0g1), shared across all qubit-cavity pairs via FDM
sideband_ch = octave_spec(index=1, rf_out=2)
# RF3: qubit XY drives, shared across all qubits via FDM
xy_ch       = octave_spec(index=1, rf_out=3)
# RF4: cavity mode drives (alice + bob), shared via FDM
cavity_ch   = octave_spec(index=1, rf_out=4)

########################################################################################################################
# %%                 Allocate the wiring to the connectivity object based on the available instruments
########################################################################################################################
connectivity = CavityConnectivity()

# Standard transmon lines
connectivity.add_resonator_line(qubits=qubits, constraints=q_res_ch)
connectivity.add_qubit_drive_lines(qubits=qubits, constraints=xy_ch, triggered=True)

# Cavity mode drives (alice and bob on shared RF4)
connectivity.add_cavity_mode_drive_lines(
    CAVITY_ID, CAVITY_MODES, constraints=cavity_ch, triggered=True
)

# Sideband drives for every qubit-cavity pair (all on shared RF2)
connectivity.add_cavity_sideband_lines(
    SIDEBAND_PAIR_IDS, constraints=sideband_ch, triggered=True
)

# Allocate physical channels
allocate_wiring(connectivity, instruments)

# View wiring schematic — cavity and sideband elements appear alongside qubits
visualize(connectivity.elements, available_channels=instruments.available_channels)
plt.show(block=True)

########################################################################################################################
# %%                                   Build the wiring and QUAM
########################################################################################################################
user_input = input("Do you want to save the updated QUAM? (y/n)")
if user_input.lower() == "y":
    machine = Quam()
    # Build wiring.json and initialise the QUAM
    build_quam_wiring(connectivity, host_ip, cluster_name, machine)

    # Reload QUAM, build the QUAM object (state.json) with cavity + sideband components
    machine = Quam.load()
    build_quam(machine, calibration_db_path)
