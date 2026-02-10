from typing import Dict
from quam.core import quam_dataclass
from quam_builder.architecture.superconducting.qpu import FixedFrequencyQuam, FluxTunableQuam


# Define the QUAM class that will be used in all calibration nodes
# Should inherit from either FixedFrequencyQuam or FluxTunableQuam
@quam_dataclass
class Quam(FixedFrequencyQuam):
    # abs(IQ) amplitudes in Volts, per qubit
    resonator_amplitudes: Dict[str, Dict[str, float]] = None
