from typing import Literal, Optional
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    fock_level: int = 0
    """Which sideband transition to calibrate (0=f0g1, 1=f1g2, ...). Internally prepares |fock_level+1⟩ photons."""
    num_shots: int = 200
    """Number of averages."""
    mode_name: str = "alice"
    """Which cavity mode: 'alice' or 'bob'."""
    frequency_span_in_mhz: float = 5.0
    """Frequency span to sweep around the resonator IF [MHz]. Use a wider range than qubit spectroscopy
    since the resonator cross-Kerr shift per photon is smaller and less well known a priori."""
    frequency_step_in_mhz: float = 0.05
    """Step size for the frequency sweep [MHz]."""
    chi2_threshold: float = 3.0
    """Residual chi-squared threshold for the Lorentzian dip fit. Lower to reject marginal fits."""
    cavity_reset_type: Literal["thermal", "active_sideband"] = "thermal"
    """How to reset the cavity at the end of each shot.
    'thermal'         - wait thermalization_time_factor × T1 (passive decay).
    'active_sideband' - cascade sideband π-pulses to actively remove photons."""
    cavity_active_cooling_fock_n: int = 1
    """Starting Fock level for active sideband cooling (only used when
    cavity_reset_type='active_sideband').  Set to 1 for thermal state cooling."""
    sideband_pulse_duration_ns: Optional[int] = None
    """Override the sideband pulse flat-top duration [ns] during active cooling.
    When None, the calibrated pi_flat_top_length_ns from pair.transitions is used."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
