"""Parameter definitions for coupler flux long distortion calibration (dataload variant).

Same as coupler_flux_distortion parameters but uses flux-domain sweep parameters
(flux_amp_for_detuning_in_volt, flux_sweep_*_volt) and explicit run-ID-based
dispersion curve loading (instead of reading from qubit.extras).
"""

from typing import ClassVar, Literal, Optional, Union

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Specific parameters for coupler flux long distortion characterization (dataload variant)."""

    num_shots: int = 30
    """Number of shots to acquire."""
    operation: str = "x180"
    """Operation to excite the qubit."""
    operation_amplitude_factor: float = 1.0
    """Amplitude factor for the operation."""
    duration_in_ns: int = 8000
    """Maximum duration of the sequence."""
    time_axis: Literal["linear", "log"] = "log"
    """Time axis for the operation."""
    time_step_in_ns: int = 48
    """Time step in nanoseconds. For linear time axis."""
    time_step_num: int = 100
    """Number of time steps. Used for log time axis."""
    min_wait_time_in_ns: int = 32
    """Minimum wait time in nanoseconds."""
    detuning_in_mhz: float = -5.0
    """Signed detuning from the qubit frequency at the coupler's decouple_offset in MHz (positive = above, negative = below); reference frequency is read from the loaded dispersion curve at decouple_offset (state.json). Default is -5.0."""
    frequency_span_in_mhz: float = 10.0
    """Total frequency sweep width in MHz centered on the detuning point; covers [detuning - span/2, detuning + span/2]. Default is 10.0."""
    frequency_step_in_mhz: float = 0.5
    """Frequency step in MHz for the uniform spectroscopy sweep passed to QUA from_array. Default is 0.5."""
    n_exponentials: int = 3
    """Number of exponential components to fit in the flux step response model."""
    update_state: bool = False
    """Whether to update the state. CAUTION: assumes fitting will be acceptable."""
    update_state_from_GUI: bool = False
    """Whether to update the state from the GUI, select when fitting is successful."""
    coupler_flux_amplitude_in_v: float = 0.1
    """Legacy fallback coupler flux amplitude in V (superseded by flux_amp_for_detuning_in_volt); retained for backward compatibility with older saved data. Default is 0.1."""
    measure_qubit: Literal["control", "target"] = "target"
    """Which qubit to measure: 'control' or 'target'. Default is 'target'."""
    buffer_during_operation_in_ns: int = 600
    """Buffer time in ns during the operation to avoid turn-off transient overlapping with the XY pulse."""
    buffer_after_operation_in_ns: int = 600
    """Buffer time in ns after the operation to keep readout clean from the flux turn-off artifact."""
    use_ramsey_vs_flux_data: bool = True
    """When True, load Ramsey-vs-coupler-flux for freq->flux conversion; uses ramsey_vs_flux_run_id if set, else extras per coupler."""
    ramsey_vs_flux_run_id: Optional[int] = None
    """Run ID of a prior node-10 / node-21a (Ramsey vs coupler flux) experiment.
    If None, the run ID is read per qubit pair from
    ``qubit.extras[f"{coupler.name}_dispersion_load_id"]`` (written by those nodes).
    Set explicitly to use one run ID for every pair."""
    use_spectroscopy_data: bool = False
    """Use qubit spectroscopy vs coupler flux raw data (node 10) for freq->flux conversion; takes priority over ramsey_vs_flux_run_id when both are set. Default is False."""
    spectroscopy_run_id: Optional[int] = None
    """Run ID of a prior node-10 (qubit spectroscopy vs coupler flux) experiment."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for coupler flux long distortion calibration (dataload variant)."""

    targets_name: ClassVar[str] = "qubit_pairs"
