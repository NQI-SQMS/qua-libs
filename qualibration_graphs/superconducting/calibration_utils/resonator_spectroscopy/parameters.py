from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters

from typing import Optional


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 30.0
    """Span of frequencies to sweep in MHz. Default is 30 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""
    readout_power_dbm: Optional[float] = None
    """Readout power in dBm for the spectroscopy sweep.
    If None, the current QUAM state power is used unchanged.
    The QUAM state is reverted to its original value after the node finishes."""
    max_amp: float = 0.1
    """Maximum readout pulse amplitude (OPX units, 0–0.5).
    Only used when readout_power_dbm is set. Default is 0.1."""
    save_readout_amplitude: bool = True
    """When True (default) and readout_power_dbm is set, permanently save the calibrated
    readout power/amplitude to the QUAM state after a successful run.
    Set to False to keep the QUAM state readout power unchanged (e.g. when using this node
    only for frequency calibration and the power is set just to improve the SNR)."""
    chi2_threshold: float = 3.0
    """Residual chi-squared threshold for the Lorentzian dip fit.
    chi2 = SS_res / ((N - 4) * amp²) where N = number of frequency points, P = 4 free
    parameters (center, FWHM, amplitude, offset).  chi2 ≤ threshold → real dip detected;
    chi2 > threshold → residuals dominate the dip depth, likely fitting noise.
    Default 3.0. Lower (e.g. 1.5) to reject marginal fits; raise if noisy data."""
    run_circle_fit: bool = False
    """When True, run the Probst circle-fit pipeline after the Lorentzian dip fit
    to extract Q_loaded, Q_internal, Q_external, kappa_Hz, and phi0 from the IQ circle.
    Results are stored in QUAM (q.resonator.Q_loaded, etc.) at the end of the node.
    Geometry is controlled by circle_fit_geometry. Default False."""
    circle_fit_geometry: str = "transmission"
    """Measurement geometry used by the circle fit.
    'transmission' — hanger/notch S21 (two-port, signal dips at resonance).
    'reflection'   — one-port S11 (signal loops around a circle centred away from origin).
    Default 'transmission'."""



class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
