"""Parameter definitions for resonator spectroscopy calibration."""

from typing import List, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Resonator spectroscopy specific parameters."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 30.0
    """Span of frequencies to sweep in MHz. Default is 30 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""

    # --- v2 fit gates ---
    min_dip_snr: float = 6.0
    """FREQUENCY gate: minimum statistical significance of the dip (baseline-subtracted
    prominence / per-point noise sigma) for the resonator to count as found. This is the
    only gate on the frequency write; R²/FWHM/contrast only guard `success_shape`
    (the reported linewidth), never the frequency."""
    dip_dominance: float = 2.0
    """A window is flagged `ambiguous` when the second-most-prominent dip is within this
    factor of the top one (several comparable resonators in one window — typical for wide
    bring-up scans catching feedline neighbours). Downstream should verify the pick
    against the expected frequency or the vs-power punch-out behaviour."""

    # --- Bring-up span escalation (no-dip retry) ---
    escalate_on_no_dip: bool = False
    """When True and no significant dip is found for some qubit, the node re-measures
    those qubits with the span doubled (up to `max_escalation_span_in_mhz`), re-centering
    the readout LO when the wider sweep would push the IF past its limit. Fresh-from-fab
    bring-up mode; leave False for routine tracking scans."""
    max_escalation_span_in_mhz: float = 800.0
    """Span ceiling (MHz) for the no-dip escalation ladder. Default ±400 MHz."""

    # --- Re-fit overrides (used together with load_data_id) ---
    re_fit_resonators: Optional[List[str]] = None
    """Qubit names to re-fit with a manually specified window, e.g. ["qA1", "qD3"].
    Must be the same length as re_fit_centers_ghz and re_fit_span_mhz."""
    re_fit_centers_ghz: Optional[List[float]] = None
    """Absolute RF center frequency (GHz) for each qubit in re_fit_resonators."""
    re_fit_span_mhz: Optional[List[float]] = None
    """Fit span (MHz) around the center for each qubit in re_fit_resonators."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameter class for resonator spectroscopy calibration node."""
