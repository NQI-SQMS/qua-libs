"""Wiring example: OPX1000 (LF-FEM + MW-FEM) with one qubit, one cavity (alice + bob),
and a sideband drive.

MW-FEM slot 1 port assignment:
  Port 1 → readout resonator (in + out)
  Port 2 → qubit XY drive
  Port 3 → sideband drive    (f0g1; shareable with port 2 if MW-FEM coupling requires it)
  Port 4 → cavity mode drive (alice + bob shared, FDM; shareable)

Note on MW-FEM port coupling: on some MW-FEM revisions adjacent ports are
electrically coupled, which requires both ports to be declared shareable.
Set shareable=True on the mw_fem_spec if the QM SDK raises a port-coupling error.
"""
import matplotlib.pyplot as plt
from qualang_tools.wirer.wirer.channel_specs import mw_fem_spec
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
cluster_name = "Cluster_OPX1000"  # Name of the cluster
calibration_db_path = None    # "/path/to/some/config/folder"

########################################################################################################################
# %%                                      Define the available instrument setup
########################################################################################################################
instruments = Instruments()
instruments.add_mw_fem(controller=1, slots=[1])

########################################################################################################################
# %%                           Define qubits, cavities, and qubit-cavity pair IDs
########################################################################################################################
qubits = [1]

CAVITY_ID = "c1"
CAVITY_MODES = ["alice", "bob"]
SIDEBAND_PAIR_IDS = [f"q{q}_{mode}" for q in qubits for mode in CAVITY_MODES]

########################################################################################################################
# %%                                 Define any custom/hardcoded channel addresses
########################################################################################################################
# MW-FEM slot 1 port map
rr_ch       = mw_fem_spec(con=1, slot=1, in_port=1, out_port=1)   # Port 1: readout
xy_ch       = mw_fem_spec(con=1, slot=1, in_port=None, out_port=2) # Port 2: XY drive
sideband_ch = mw_fem_spec(con=1, slot=1, in_port=None, out_port=3) # Port 3: sideband
cavity_ch   = mw_fem_spec(con=1, slot=1, in_port=None, out_port=4) # Port 4: cavity (shareable)

########################################################################################################################
# %%                 Allocate the wiring to the connectivity object based on the available instruments
########################################################################################################################
connectivity = CavityConnectivity()

# Standard transmon lines
connectivity.add_resonator_line(qubits=qubits, constraints=rr_ch)
connectivity.add_qubit_drive_lines(qubits=qubits, constraints=xy_ch)

# Cavity mode drives (alice and bob share port 4 via FDM)
connectivity.add_cavity_mode_drive_lines(
    CAVITY_ID, CAVITY_MODES, constraints=cavity_ch, triggered=True
)

# Sideband drives for every qubit-cavity pair (share port 3 via FDM)
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
