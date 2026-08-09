"""Parameter definitions for cryoscope experiment."""


from typing import Optional

from qualang_tools.bakery import baking
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


def baked_waveform(config, waveform_amp: float, qubit, max_length: int = 16):
    """Create baked pulse segments with 1ns granularity up to ``max_length`` ns.

    This mirrors the previous inline implementation inside ``12b_cryoscope.py`` and is
    extracted here so it can be shared / unit tested. Each index ``i`` (1..max_length)
    produces a baking object that plays a constant waveform of ``i`` ns with amplitude
    ``waveform_amp`` on the qubit flux line.

    Parameters
    ----------
    config : dict
        Configuration dictionary (typically produced by ``machine.generate_config()``)
        that the baking context mutates.
    waveform_amp : float
        The absolute amplitude to use for the flux pulse.
    qubit : Any
        QUAM qubit object containing the ``z`` element name.
    max_length : int, optional
        Maximum pulse length in ns to bake (default 16 to keep within baking memory limits).

    Returns
    -------
    list
        A list of baking objects; element ``i-1`` corresponds to a pulse of length ``i`` ns.
    """
    pulse_segments = []
    # Create the base waveform (1ns resolution). Represent as list of samples.
    waveform = [waveform_amp] * max_length
    for i in range(1, max_length + 1):  # inclusive
        with baking(config, padding_method="right") as b:
            wf = waveform[:i]
            b.add_op(f"flux_pulse{i}", qubit.z.name, wf)
            b.play(f"flux_pulse{i}", qubit.z.name)
        pulse_segments.append(b)
    return pulse_segments


class NodeSpecificParameters(RunnableParameters):
    """Cryoscope-specific parameters for flux line characterization."""

    num_shots: int = 5000
    """Number of averages to perform. Default is 5000."""
    reset_type: str = "active"
    """Type of reset to perform: 'active' or 'thermal'."""
    
    detuning_target_in_mhz: int = 300
    """Target detuning from sweetspot for the cryoscope pulse in MHz. Default is 300."""
    cryoscope_len: int = 240
    """Length of the cryoscope operation in nanoseconds. Default is 240."""
    num_frames: int = 17
    """Number of frames to use in the cryoscope experiment. Default is 17."""
    n_exponentials: int = 2
    """Number of exponential components in IIR to fit in the cryoscope flux step response model ``y(t) = a_dc + Σ a_i exp(-t/tau_i)``."""
    use_fir: bool = False
    """Run FIR analysis after IIR. Default False."""
    fir_max_taps: int = 48
    """Upper bound for forward and inverse FIR length."""

    update_iir: bool = False
    """Push IIR exponential filter into state on this run."""
    update_fir: bool = False
    """Push FIR feedforward filter into state on this run."""

    # Dispersion-curve-based amplitude (Path 1: spectroscopy, Path 2: Ramsey vs flux)
    use_spectroscopy_data: bool = False
    """Use qubit spectroscopy vs Z-flux run to compute cryoscope flux amplitude (Path 1)."""
    spectroscopy_run_id: Optional[int] = None
    """Run ID of the qubit spectroscopy vs Z-flux dataset (required when use_spectroscopy_data=True)."""
    use_ramsey_data: bool = False
    """Use Ramsey vs Z-flux data to compute cryoscope flux amplitude."""
    ramsey_run_id: Optional[int] = None
    """Run ID of a previous Ramsey vs Z-flux calibration experiment."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameters for cryoscope calibration node."""
