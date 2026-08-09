from typing import ClassVar, Literal

from qualang_tools.bakery import baking
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for the chevron CZ (11-02) calibration."""

    num_shots: int = 40
    """Number of shots to perform. Default is 100."""
    max_time_in_ns: int = 120
    """Upper bound (exclusive) of the flux-pulse duration sweep in nanoseconds. Default is 160."""
    amp_range: float = 0.1
    """Half-width of the amplitude sweep around the nominal value (sweeps 1 - amp_range to 1 + amp_range). Default is 0.1."""
    amp_step: float = 0.006
    """Step size for the amplitude sweep. Default is 0.003."""
    use_state_discrimination: bool = True
    """Whether to use state discrimination for readout. Default is True."""
    cz_branch: Literal["20", "02"] = "20"
    """CZ avoided-crossing branch: '20' = |11>-|20> (control -> |2>), '02' = |11>-|02> (target -> |2>).
    Selects the detuning anharmonicity term and which qubit is GEF-read. Default '20' preserves prior behaviour."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Main parameters class for the chevron CZ (11-02) calibration node."""

    targets_name: ClassVar[str] = "qubit_pairs"


def baked_waveform(qubit, baked_config, base_level: float = 0.5, max_samples: int = 16):
    """Create truncated baked waveforms for the chevron CZ calibration.

    Generates a list of baking objects, each containing an incrementally longer flux pulse
    (1..max_samples samples) at the specified base_level. Each baked pulse is registered
    as an operation named "flux_pulse{i}" on the provided qubit z line.

    Args:
        qubit: The qubit object whose z line is used.
        baked_config: A mutable QM configuration object to which baked operations are added.
        base_level (float): The constant waveform level used for the short baked segments.
        max_samples (int): The maximum number of samples (and thus baked variants) to generate.

    Returns:
        List of baking objects, index i corresponds to pulse length i+1 samples.
    """
    pulse_segments = []
    waveform = [base_level] * max_samples
    for i in range(1, max_samples + 1):
        with baking(baked_config, padding_method="right") as b:
            wf = waveform[:i]
            b.add_op(f"flux_pulse{i}", qubit.z.name, wf)
            b.play(f"flux_pulse{i}", qubit.z.name)
        pulse_segments.append(b)
    return pulse_segments
