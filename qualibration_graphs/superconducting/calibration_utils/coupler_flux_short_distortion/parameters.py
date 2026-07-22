"""Parameter definitions for coupler flux short distortion (cryoscope) — new module.

Extends the original coupler_flux_short_distortion parameters with
three-path amplitude cascade (spectroscopy / Ramsey / fixed fallback),
mirroring 20e_qubit_flux_short_distortion for the coupler flux line.
"""

from typing import ClassVar, List, Literal, Optional

from qualang_tools.bakery import baking
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


def baked_coupler_waveform(config, waveform_amp: float, coupler, max_length: int = 16):
    """Create baked pulse segments with 1ns granularity up to ``max_length`` ns for coupler flux.

    Parameters
    ----------
    config : dict
        Configuration dictionary (typically produced by ``machine.generate_config()``)
        that the baking context mutates.
    waveform_amp : float
        The absolute amplitude to use for the coupler flux pulse.
    coupler : Any
        QUAM coupler object.
    max_length : int, optional
        Maximum pulse length in ns to bake (default 16 to keep within baking memory limits).

    Returns
    -------
    list
        A list of baking objects; element ``i-1`` corresponds to a pulse of length ``i`` ns.
    """
    pulse_segments = []
    waveform = [waveform_amp] * max_length
    for i in range(1, max_length + 1):
        with baking(config, padding_method="right") as b:
            wf = waveform[:i]
            b.add_op(f"coupler_flux_pulse{i}", coupler.name, wf)
            b.play(f"coupler_flux_pulse{i}", coupler.name)
        pulse_segments.append(b)
    return pulse_segments


class NodeSpecificParameters(RunnableParameters):
    """Specific parameters for coupler flux short distortion (cryoscope) characterization.

    Extends 22b with a three-path cascade for computing the cryoscope flux amplitude:

      Path 1 — Coupler spectroscopy (use_spectroscopy_data=True, spectroscopy_run_id set):
          Loads the freq-vs-coupler-flux curve from a qubit-spectroscopy-vs-coupler-flux
          run (node 10) and inverts it to find the coupler flux amplitude that achieves
          the target detuning.

      Path 2 — Ramsey vs coupler flux (use_ramsey_data=True, ramsey_run_id set):
          Loads the freq-vs-coupler-flux curve from a Ramsey calibration run and performs
          the same inversion.

      Path 3 — Fixed amplitude fallback:
          Uses coupler_flux_amplitude_in_v directly.  Always available; least accurate.
    """

    num_shots: int = 5000
    """Number of averages to perform. Default is 5000."""
    cryoscope_len: int = 240
    """Length of the cryoscope operation in nanoseconds. Default is 240."""
    num_frames: int = 17
    """Number of frames to use in the cryoscope experiment. Default is 17."""
    exponential_fit_time_fractions: List[float] = [0.5, 0.01]
    """List of time fractions for the exponential fit. Default is [0.5, 0.01]."""
    n_exponentials: int = 2
    """Number of exponential components in the IIR fit of the cryoscope flux step response model
    ``y(t) = a_dc + Sum a_i exp(-t/tau_i)``. Default is 2."""
    update_iir: bool = False
    """Push IIR exponential filter into state on this run."""
    update_fir: bool = False
    """Push FIR feedforward filter into state on this run."""
    measure_qubit: Literal["control", "target"] = "target"
    """Which qubit in the pair to measure: 'control' or 'target'. Default is 'target'."""

    # --- Three-path amplitude cascade ---
    detuning_target_in_mhz: int = 300
    """Target qubit detuning in MHz produced by the coupler flux pulse.

    Signed: a positive value targets a qubit frequency **above** the
    decouple-point frequency, a negative value targets one **below**. Which
    sign (and how large a magnitude) is reachable depends on the coverage of
    the spectroscopy/Ramsey dispersion curve (Path 1/2); if the resulting
    target frequency is outside the curve range the cascade falls back to the
    fixed Path 3 amplitude (coupler_flux_amplitude_in_v)."""

    use_spectroscopy_data: bool = False
    """Path 1: load qubit-spectroscopy-vs-coupler-flux data (node 10) to compute coupler amplitude from detuning_target_in_mhz. Default is False."""
    spectroscopy_run_id: Optional[int] = None
    """Run ID of the qubit-spectroscopy-vs-coupler-flux dataset (node 10). Default is None."""

    use_ramsey_data: bool = False
    """Path 2: load Ramsey-vs-coupler-flux data to compute coupler amplitude from detuning_target_in_mhz. Default is False."""
    ramsey_run_id: Optional[int] = None
    """Run ID of the Ramsey-vs-coupler-flux dataset."""

    coupler_flux_amplitude_in_v: float = 0.1
    """Path 3 fallback: amplitude of the coupler flux pulse in Volts. Default is 0.1V."""

    coupler_flux_branch: Literal["auto", "left", "right"] = "auto"
    """Which side of the decouple point to pick the coupler-flux crossing when
    resolving detuning_target_in_mhz from a dispersion curve (Path 1/2).
    'auto' picks the crossing nearest the decouple offset and warns if multiple
    crossings exist; 'left'/'right' restrict to flux below/above the decouple
    offset. Default 'auto'."""

    # --- FIR filter analysis ---
    use_fir: bool = False
    """Run FIR analysis after IIR. Default False."""
    fir_max_taps: int = 48
    """Upper bound for forward and inverse FIR length."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for coupler flux short distortion (22c) calibration node."""

    targets_name: ClassVar[str] = "qubit_pairs"
