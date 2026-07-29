from typing import Literal, Optional
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    fock_level: int = 0
    """Which |n⟩→|n+1⟩ sideband transition to calibrate (0=f0g1, 1=f1g2, ...)."""
    num_shots: int = 100
    """Number of averages to perform."""
    frequency_span_in_mhz: float = 10.0
    """Frequency span to sweep around the centre RF frequency [MHz]."""
    frequency_step_in_mhz: float = 0.05
    """Step size for the frequency sweep [MHz]."""
    operation: str = "saturation"
    """Pulse name on the sideband channel to play during the sweep."""
    operation_amplitude_factor: float = 1.0
    """Pre-factor applied to the operation amplitude."""
    operation_len_in_ns: Optional[int] = None
    """Override pulse length [ns]. None → use the pulse's own length."""
    mode_name: str = "alice"
    """Which cavity mode: 'alice' or 'bob'."""
    min_dip_fraction: float = 0.1
    """Minimum dip depth as a fraction of the state-population range."""
    use_state_discrimination: bool = True
    """True → measure qubit state. False → measure raw I/Q."""
    use_displaced_threshold: bool = False
    """When True and use_state_discrimination is True, use pair.ge_iq_threshold_displaced
    instead of the vacuum readout threshold (calibrated by node 26j)."""
    use_confusion_matrix_correction: bool = False
    """Apply ge confusion matrix correction to averaged state probabilities."""
    cavity_reset_type: Literal["thermal", "active_sideband"] = "thermal"
    """How to reset the cavity at the end of each shot.
    'thermal'         - wait thermalization_time_factor × T1 (passive decay).
    'active_sideband' - cascade sideband π-pulses to actively remove photons;
                        requires calibrated sideband operations (nodes 26 / 26b)."""
    cavity_active_cooling_fock_n: int = 1
    """Starting Fock level for active sideband cooling (only used when
    cavity_reset_type='active_sideband').  Set to 1 for thermal state cooling."""
    sideband_pulse_duration_ns: Optional[int] = None
    """Override the sideband pulse flat-top duration [ns] during active cooling.
    When None, the calibrated pi_flat_top_length_ns from pair.transitions is used.
    Must be a multiple of 4 ns."""
    use_theoretical_frequency_estimate: bool = False
    """If True, seed the sweep center from the theoretical formula (2·f_ge + anharmonicity − f_cav for k=0,
    or f0g1 − k·|chi| for k>0). If False (default), use the RF_frequency saved in the state."""
    update_chi_if_absent: bool = False
    """If True and fock_level > 0, derive chi from the measured sideband shift
    (chi = (f_fkgk+1 − f_f0g1) / k) and save it to the pair state when chi is currently absent (None)."""
    use_gaussian_fit: bool = False
    """If True, fit the sideband dip with a Gaussian instead of the default peaks_dips (Lorentzian-based)
    detector. The Gaussian gives a cleaner fit when the linewidth is set by the pulse bandwidth rather
    than the natural decay rate. The extracted center frequency and FWHM are used identically."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
